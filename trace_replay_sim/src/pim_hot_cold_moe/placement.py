"""Placement state/policies for MVP-0.

The original ``static_hot_cold`` policy ranks experts within each layer by a
configurable demand score, places the top hot fraction on GPU, the bottom cold
fraction on PIM, and keeps the middle band on BOTH.

``adaptive_extra_budget`` keeps the same fixed-after-warmup placement contract,
but learns each layer's GPU/PIM primary residency from warmup oracle-regret
signals and spends a simple per-layer extra-copy budget on BOTH placements.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence, runtime_checkable

from .interfaces import PredictionFrame, ReplayStep
from .scheduler import combine_gpu_offload_latency, offload_units_for_token_count, oracle_layer_split, scheduler_config_from_dict

if TYPE_CHECKING:  # pragma: no cover - import-only typing helper
    from .model_hardware import HardwareModel, ModelSpec
    from .scheduler import SchedulerConfig

Residency = Literal["GPU", "PIM", "BOTH"]
RESIDENCY_VALUES: set[str] = {"GPU", "PIM", "BOTH"}
PLACEMENT_MODES: set[str] = {"static_hot_cold", "adaptive_extra_budget", "all_gpu"}

Histogram = dict[int, dict[int, float]]


@dataclass(frozen=True)
class PlacementConfig:
    """Validated placement-policy configuration.

    MVP-0 defaults come from the project-owner decision for the initial
    fraction policy: top 50% hot experts per layer are GPU-only, bottom 25%
    cold experts are PIM-only, and the remaining 25% are on BOTH.

    ``warmup_weight`` and ``prediction_weight`` combine mean warmup demand and
    predicted mean demand.
    """

    mode: str = "static_hot_cold"
    hot_fraction: float = 0.50
    cold_fraction: float = 0.25
    warmup_weight: float = 1.0
    prediction_weight: float = 1.0
    extra_copy_fraction: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in PLACEMENT_MODES:
            raise ValueError(f"unsupported placement mode {self.mode!r}; choose from {sorted(PLACEMENT_MODES)}")
        hot_fraction = _fraction(self.hot_fraction, "placement.hot_fraction")
        cold_fraction = _fraction(self.cold_fraction, "placement.cold_fraction")
        if hot_fraction + cold_fraction > 1.0 + 1e-12:
            raise ValueError("placement hot_fraction + cold_fraction must be <= 1.0")
        warmup_weight = _non_negative_float(self.warmup_weight, "placement.warmup_weight")
        prediction_weight = _non_negative_float(self.prediction_weight, "placement.prediction_weight")
        extra_copy_fraction = _fraction(self.extra_copy_fraction, "placement.extra_copy_fraction")
        if mode == "static_hot_cold" and warmup_weight + prediction_weight <= 0.0:
            raise ValueError("static_hot_cold requires warmup_weight or prediction_weight to be positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "hot_fraction", hot_fraction)
        object.__setattr__(self, "cold_fraction", cold_fraction)
        object.__setattr__(self, "warmup_weight", warmup_weight)
        object.__setattr__(self, "prediction_weight", prediction_weight)
        object.__setattr__(self, "extra_copy_fraction", extra_copy_fraction)

    @property
    def both_fraction(self) -> float:
        """Return the configured middle-band fraction assigned to BOTH."""

        return max(0.0, 1.0 - self.hot_fraction - self.cold_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "hot_fraction": self.hot_fraction,
            "cold_fraction": self.cold_fraction,
            "both_fraction": self.both_fraction,
            "warmup_weight": self.warmup_weight,
            "prediction_weight": self.prediction_weight,
            "extra_copy_fraction": self.extra_copy_fraction,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PlacementState:
    """Stateful model-wide expert residency map.

    ``residency[(layer_id, expert_id)]`` is one of ``GPU``, ``PIM``, or
    ``BOTH``.
    """

    version: int
    residency: dict[tuple[int, int], Residency]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        version = _non_negative_int(self.version, "placement version")
        normalized_residency: dict[tuple[int, int], Residency] = {}
        for raw_key, raw_value in self.residency.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError(f"placement residency key must be (layer, expert), got {raw_key!r}")
            layer_id = _non_negative_int(raw_key[0], "placement layer")
            expert_id = _non_negative_int(raw_key[1], "placement expert")
            value = _normalize_residency(raw_value)
            normalized_residency[(layer_id, expert_id)] = value
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "residency", dict(sorted(normalized_residency.items())))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def get(self, layer: int, expert: int) -> Residency:
        """Return residency for one ``(layer, expert)`` pair."""

        key = (_non_negative_int(layer, "layer"), _non_negative_int(expert, "expert"))
        try:
            return self.residency[key]
        except KeyError as exc:
            raise KeyError(f"placement has no residency for layer={key[0]}, expert={key[1]}") from exc

    def count_by_residency(self) -> dict[str, int]:
        counts = {"GPU": 0, "PIM": 0, "BOTH": 0}
        for value in self.residency.values():
            counts[value] += 1
        return counts

    def report_residency(self) -> dict[str, Any]:
        """Return a compact residency summary."""
        counts = self.count_by_residency()
        return {
            "version": self.version,
            "expert_count": len(self.residency),
            "residency_counts": counts,
        }

    def clone_with_updates(self, updates: dict[tuple[int, int], Residency]) -> "PlacementState":
        """Return a new state with changed residency and incremented version.

        ``updates`` may only target experts already present in this model-wide
        state. The current simulation keeps placement fixed after warmup; this
        helper exists for explicit callers that need a new placement version.
        """

        normalized_updates: dict[tuple[int, int], Residency] = {}
        for raw_key, raw_value in updates.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError(f"placement update key must be (layer, expert), got {raw_key!r}")
            key = (
                _non_negative_int(raw_key[0], "placement update layer"),
                _non_negative_int(raw_key[1], "placement update expert"),
            )
            if key not in self.residency:
                raise KeyError(f"cannot update unknown placement entry layer={key[0]}, expert={key[1]}")
            normalized_updates[key] = _normalize_residency(raw_value)

        new_residency = dict(self.residency)
        new_residency.update(normalized_updates)
        metadata = dict(self.metadata)
        metadata["updated_entries"] = [
            {"layer": layer_id, "expert": expert_id, "residency": residency}
            for (layer_id, expert_id), residency in sorted(normalized_updates.items())
        ]
        return _state_from_residency_and_metadata(
            new_residency,
            metadata=metadata,
            version=self.version + 1,
        )


@runtime_checkable
class PlacementPolicy(Protocol):
    """Common placement-policy interface used by the simulation engine."""

    def initialize(
        self,
        warmup_stats: Any,
        prediction: PredictionFrame | MappingABC[Any, Any] | None,
        model: "ModelSpec",
        hardware: "HardwareModel",
        scheduler_config: "SchedulerConfig | MappingABC[str, Any] | None" = None,
    ) -> PlacementState:
        ...


class AllGpuPlacementPolicy:
    """Baseline where every expert is GPU-resident."""

    def __init__(self, config: PlacementConfig | MappingABC[str, Any] | None = None):
        cfg = placement_config_from_dict(config)
        if cfg.mode != "all_gpu":
            cfg = PlacementConfig(
                mode="all_gpu",
                hot_fraction=cfg.hot_fraction,
                cold_fraction=cfg.cold_fraction,
                warmup_weight=cfg.warmup_weight,
                prediction_weight=cfg.prediction_weight,
                extra_copy_fraction=cfg.extra_copy_fraction,
                metadata=cfg.metadata,
            )
        self.config = cfg

    def initialize(
        self,
        warmup_stats: Any,
        prediction: PredictionFrame | MappingABC[Any, Any] | None,
        model: "ModelSpec",
        hardware: "HardwareModel",
        scheduler_config: "SchedulerConfig | MappingABC[str, Any] | None" = None,
    ) -> PlacementState:
        del warmup_stats, prediction, scheduler_config
        residency: dict[tuple[int, int], Residency] = {
            (layer_id, expert_id): "GPU"
            for layer_id in range(model.num_layers)
            for expert_id in range(model.experts_for_layer(layer_id))
        }
        metadata = _base_state_metadata(model, hardware)
        metadata.update(
            {
                "policy": "all_gpu",
                "placement_config": self.config.to_dict(),
                "classification_method": "all_experts_gpu_resident_baseline",
            }
        )
        return _make_placement_state(residency, model=model, hardware=hardware, metadata=metadata)


class StaticHotColdPlacementPolicy:
    """MVP static fraction split: hot GPU, cold PIM, middle BOTH."""

    def __init__(self, config: PlacementConfig | MappingABC[str, Any] | None = None):
        cfg = placement_config_from_dict(config)
        if cfg.mode != "static_hot_cold":
            cfg = PlacementConfig(
                mode="static_hot_cold",
                hot_fraction=cfg.hot_fraction,
                cold_fraction=cfg.cold_fraction,
                warmup_weight=cfg.warmup_weight,
                prediction_weight=cfg.prediction_weight,
                extra_copy_fraction=cfg.extra_copy_fraction,
                metadata=cfg.metadata,
            )
        self.config = cfg

    def initialize(
        self,
        warmup_stats: Any,
        prediction: PredictionFrame | MappingABC[Any, Any] | None,
        model: "ModelSpec",
        hardware: "HardwareModel",
        scheduler_config: "SchedulerConfig | MappingABC[str, Any] | None" = None,
    ) -> PlacementState:
        del scheduler_config
        warmup_histogram = warmup_stats_to_histogram(warmup_stats)
        prediction_histogram = prediction_to_histogram(prediction)
        scores = placement_scores(
            warmup_histogram=warmup_histogram,
            prediction_histogram=prediction_histogram,
            model=model,
            config=self.config,
        )

        residency: dict[tuple[int, int], Residency] = {}
        layer_summaries: dict[str, dict[str, Any]] = {}
        for layer_id in range(model.num_layers):
            expert_count = model.experts_for_layer(layer_id)
            layer_scores = scores.get(layer_id, {})
            hot_count = int(math.floor(expert_count * self.config.hot_fraction))
            cold_count = int(math.floor(expert_count * self.config.cold_fraction))
            ranked = sorted(
                range(expert_count),
                key=lambda expert_id: (-layer_scores.get(expert_id, 0.0), expert_id),
            )
            hot_experts = set(ranked[:hot_count])
            cold_experts = set(ranked[expert_count - cold_count :]) if cold_count else set()
            # Config validation guarantees no overlap by count. Tied scores are
            # resolved by expert id so the fraction policy always produces
            # deterministic exact counts for a fixed model/config.
            overlap = hot_experts & cold_experts
            if overlap:  # defensive guard against future config changes
                raise RuntimeError(f"hot/cold placement sets overlapped in layer {layer_id}: {sorted(overlap)}")

            for expert_id in range(expert_count):
                if expert_id in hot_experts:
                    residency[(layer_id, expert_id)] = "GPU"
                elif expert_id in cold_experts:
                    residency[(layer_id, expert_id)] = "PIM"
                else:
                    residency[(layer_id, expert_id)] = "BOTH"

            values = [layer_scores.get(expert_id, 0.0) for expert_id in range(expert_count)]
            layer_summaries[str(layer_id)] = {
                "experts": expert_count,
                "hot_count": hot_count,
                "cold_count": cold_count,
                "both_count": expert_count - hot_count - cold_count,
                "min_score": min(values) if values else 0.0,
                "max_score": max(values) if values else 0.0,
                "score_sum": sum(values),
            }

        metadata = _base_state_metadata(model, hardware)
        metadata.update(
            {
                "policy": "static_hot_cold",
                "placement_config": self.config.to_dict(),
                "classification_method": "mvp_fraction_rank_per_layer",
                "classification_note": (
                    "Ranks experts within each layer by blended warmup/predicted demand. "
                    "Top hot_fraction -> GPU, bottom cold_fraction -> PIM, middle -> BOTH."
                ),
                "score_metadata": {
                    "warmup_weight": self.config.warmup_weight,
                    "prediction_weight": self.config.prediction_weight,
                    "warmup_histogram_layers": sorted(warmup_histogram),
                    "prediction_histogram_layers": sorted(prediction_histogram),
                },
                "layer_summaries": layer_summaries,
            }
        )
        return _make_placement_state(residency, model=model, hardware=hardware, metadata=metadata)


class AdaptiveExtraBudgetPlacementPolicy:
    """Warmup-learned placement with a fixed per-layer extra-copy budget.

    The only policy knob is ``extra_copy_fraction``.  Warmup replay is used to
    estimate, for every expert, the layer-latency regret of being GPU-only or
    PIM-only relative to a per-step oracle split that ignores placement.  Each
    expert receives one primary copy on its lower-regret side, then each layer
    spends ``round(experts_per_layer * extra_copy_fraction)`` second copies on
    the experts with largest measured opposite-side regret.

    This is still placement, not migration: the returned ``PlacementState`` is
    fixed after warmup unless a separate migration policy changes it later.
    """

    def __init__(self, config: PlacementConfig | MappingABC[str, Any] | None = None):
        cfg = placement_config_from_dict(config)
        if cfg.mode != "adaptive_extra_budget":
            cfg = PlacementConfig(
                mode="adaptive_extra_budget",
                hot_fraction=cfg.hot_fraction,
                cold_fraction=cfg.cold_fraction,
                warmup_weight=cfg.warmup_weight,
                prediction_weight=cfg.prediction_weight,
                extra_copy_fraction=cfg.extra_copy_fraction,
                metadata=cfg.metadata,
            )
        self.config = cfg

    def initialize(
        self,
        warmup_stats: Any,
        prediction: PredictionFrame | MappingABC[Any, Any] | None,
        model: "ModelSpec",
        hardware: "HardwareModel",
        scheduler_config: "SchedulerConfig | MappingABC[str, Any] | None" = None,
    ) -> PlacementState:
        del prediction
        latency_combine_mode = scheduler_config_from_dict(scheduler_config).latency_combine_mode
        warmup_steps = _replay_steps_from_warmup_stats(warmup_stats)
        loss_if_gpu_only: dict[tuple[int, int], float] = {}
        loss_if_pim_only: dict[tuple[int, int], float] = {}
        active_events: dict[tuple[int, int], int] = {}
        oracle_gpu_events: dict[tuple[int, int], int] = {}
        oracle_pim_events: dict[tuple[int, int], int] = {}

        def add_loss(table: dict[tuple[int, int], float], key: tuple[int, int], value: float) -> None:
            table[key] = table.get(key, 0.0) + value

        def add_count(table: dict[tuple[int, int], int], key: tuple[int, int], value: int = 1) -> None:
            table[key] = table.get(key, 0) + value

        for step in warmup_steps:
            if not isinstance(step, ReplayStep):
                raise ValueError("adaptive_extra_budget placement requires ReplayStep warmup entries")
            for layer_id, active in _validated_active_items_by_layer(step, model).items():
                if not active:
                    continue
                split = oracle_layer_split(
                    dict(active),
                    model,
                    hardware,
                    layer_id=layer_id,
                    latency_combine_mode=latency_combine_mode,
                )
                pim_experts = set(split.pim_experts)
                gpu_active = split.active_experts - split.cold_experts
                offload_units = split.pim_quanta
                oracle_layer_latency = split.total_latency_s

                for expert_id, token_count in active:
                    key = (layer_id, expert_id)
                    expert_offload_units = offload_units_for_token_count(token_count, hardware)
                    add_count(active_events, key)
                    if expert_id in pim_experts:
                        add_count(oracle_pim_events, key)
                        forced_gpu_latency = _adaptive_layer_latency(
                            gpu_active + 1,
                            max(0, offload_units - expert_offload_units),
                            model,
                            hardware,
                            layer_id,
                            latency_combine_mode=latency_combine_mode,
                        )
                        add_loss(loss_if_gpu_only, key, max(0.0, forced_gpu_latency - oracle_layer_latency))
                    else:
                        add_count(oracle_gpu_events, key)
                        forced_pim_latency = _adaptive_layer_latency(
                            max(0, gpu_active - 1),
                            offload_units + expert_offload_units,
                            model,
                            hardware,
                            layer_id,
                            latency_combine_mode=latency_combine_mode,
                        )
                        add_loss(loss_if_pim_only, key, max(0.0, forced_pim_latency - oracle_layer_latency))

        residency: dict[tuple[int, int], Residency] = {}
        layer_summaries: dict[str, dict[str, Any]] = {}
        for layer_id in range(model.num_layers):
            expert_count = model.experts_for_layer(layer_id)
            experts = list(range(expert_count))
            extra_copies = min(expert_count, int(round(expert_count * self.config.extra_copy_fraction)))
            layer_gpu_events = sum(oracle_gpu_events.get((layer_id, expert_id), 0) for expert_id in experts)
            layer_pim_events = sum(oracle_pim_events.get((layer_id, expert_id), 0) for expert_id in experts)
            default_pim = layer_pim_events > layer_gpu_events

            primary_gpu_count = 0
            primary_pim_count = 0
            redundancy_candidates: list[tuple[float, int, int]] = []

            for expert_id in experts:
                key = (layer_id, expert_id)
                gpu_only_loss = loss_if_gpu_only.get(key, 0.0)
                pim_only_loss = loss_if_pim_only.get(key, 0.0)
                if active_events.get(key, 0) == 0:
                    choose_pim = default_pim
                else:
                    choose_pim = pim_only_loss < gpu_only_loss

                if choose_pim:
                    residency[key] = "PIM"
                    primary_pim_count += 1
                    opposite_loss = pim_only_loss
                else:
                    residency[key] = "GPU"
                    primary_gpu_count += 1
                    opposite_loss = gpu_only_loss
                redundancy_candidates.append((opposite_loss, active_events.get(key, 0), expert_id))

            # Spend the full budget.  When measured losses tie, warmup activity
            # is only a deterministic tie-breaker; it is not another user knob.
            for _opposite_loss, _activity, expert_id in sorted(
                redundancy_candidates,
                key=lambda item: (-item[0], -item[1], item[2]),
            )[:extra_copies]:
                residency[(layer_id, expert_id)] = "BOTH"

            final_counts = {"GPU": 0, "PIM": 0, "BOTH": 0}
            for expert_id in experts:
                final_counts[residency[(layer_id, expert_id)]] += 1
            layer_summaries[str(layer_id)] = {
                "experts": expert_count,
                "extra_copy_fraction": self.config.extra_copy_fraction,
                "extra_copies": extra_copies,
                "primary_gpu_count": primary_gpu_count,
                "primary_pim_count": primary_pim_count,
                "gpu_only_count": final_counts["GPU"],
                "pim_only_count": final_counts["PIM"],
                "both_count": final_counts["BOTH"],
                "warmup_observed_experts": sum(1 for expert_id in experts if active_events.get((layer_id, expert_id), 0)),
                "oracle_gpu_events": layer_gpu_events,
                "oracle_pim_events": layer_pim_events,
                "total_loss_if_gpu_only": sum(loss_if_gpu_only.get((layer_id, expert_id), 0.0) for expert_id in experts),
                "total_loss_if_pim_only": sum(loss_if_pim_only.get((layer_id, expert_id), 0.0) for expert_id in experts),
            }

        metadata = _base_state_metadata(model, hardware)
        metadata.update(
            {
                "policy": "adaptive_extra_budget",
                "placement_config": self.config.to_dict(),
                "classification_method": "warmup_oracle_regret_primary_plus_extra_copy_fraction",
                "classification_note": (
                    "Learns per-expert GPU-only and PIM-only regret from warmup oracle splits. "
                    "Chooses the lower-regret primary side per expert and spends a fixed per-layer "
                    "extra-copy budget on the highest opposite-side regrets."
                ),
                "score_metadata": {
                    "warmup_steps": len(warmup_steps),
                    "hardware_ratio": hardware.hardware_ratio(),
                    "offload_backend": hardware.offload_backend,
                    "offload_label": hardware.offload_label(),
                    "pim_granularity": hardware.pim_granularity,
                    "oracle_source": "scheduler.oracle_layer_split",
                    "latency_combine_mode": latency_combine_mode,
                },
                "layer_summaries": layer_summaries,
            }
        )
        if not warmup_steps:
            metadata.setdefault("warnings", []).append(
                "adaptive_extra_budget saw zero warmup steps; placement fell back to deterministic layer defaults"
            )
        return _make_placement_state(residency, model=model, hardware=hardware, metadata=metadata)


def placement_config_from_dict(data: MappingABC[str, Any] | PlacementConfig | None = None) -> PlacementConfig:
    """Create ``PlacementConfig`` from a root or ``placement:`` config mapping."""

    if data is None:
        return PlacementConfig()
    if isinstance(data, PlacementConfig):
        return data
    cfg = data.get("placement", data)
    if not isinstance(cfg, MappingABC):
        raise ValueError("placement config must be a mapping")

    known = {
        "mode",
        "policy",
        "hot_fraction",
        "cold_fraction",
        "warmup_weight",
        "prediction_weight",
        "extra_copy_fraction",
        "metadata",
    }
    extra_keys = sorted(str(k) for k in cfg if k not in known)
    if extra_keys:
        raise ValueError(f"unknown placement config fields: {extra_keys}")
    metadata = dict(cfg.get("metadata") or {})

    return PlacementConfig(
        mode=str(cfg.get("mode", cfg.get("policy", "static_hot_cold"))),
        hot_fraction=float(cfg.get("hot_fraction", 0.50)),
        cold_fraction=float(cfg.get("cold_fraction", 0.25)),
        warmup_weight=float(cfg.get("warmup_weight", 1.0)),
        prediction_weight=float(cfg.get("prediction_weight", 1.0)),
        extra_copy_fraction=float(cfg.get("extra_copy_fraction", 0.25)),
        metadata=metadata,
    )


def make_placement_policy(config: MappingABC[str, Any] | PlacementConfig | None = None) -> PlacementPolicy:
    """Instantiate an MVP placement policy from config."""

    cfg = placement_config_from_dict(config)
    if cfg.mode == "static_hot_cold":
        return StaticHotColdPlacementPolicy(cfg)
    if cfg.mode == "adaptive_extra_budget":
        return AdaptiveExtraBudgetPlacementPolicy(cfg)
    if cfg.mode == "all_gpu":
        return AllGpuPlacementPolicy(cfg)
    raise ValueError(f"unsupported placement mode: {cfg.mode}")


def placement_layer_fraction_rows(state: PlacementState, model: "ModelSpec") -> list[dict[str, Any]]:
    """Return one CSV-friendly placement-fraction row per layer.

    Fractions are computed from final residency:
    ``GPU`` and ``PIM`` are single-resident experts, while ``BOTH`` experts
    count as resident on both sides and as one extra copy.
    """

    layer_metadata = state.metadata.get("layer_summaries", {})
    if not isinstance(layer_metadata, MappingABC):
        layer_metadata = {}

    rows: list[dict[str, Any]] = []
    for layer_id in range(model.num_layers):
        expert_count = model.experts_for_layer(layer_id)
        counts = {"GPU": 0, "PIM": 0, "BOTH": 0}
        for expert_id in range(expert_count):
            counts[state.get(layer_id, expert_id)] += 1

        gpu_resident_count = counts["GPU"] + counts["BOTH"]
        pim_resident_count = counts["PIM"] + counts["BOTH"]
        total_copy_count = counts["GPU"] + counts["PIM"] + (2 * counts["BOTH"])
        row: dict[str, Any] = {
            "layer_id": layer_id,
            "experts": expert_count,
            "gpu_only_count": counts["GPU"],
            "pim_only_count": counts["PIM"],
            "both_count": counts["BOTH"],
            "gpu_only_fraction": _safe_fraction(counts["GPU"], expert_count),
            "pim_only_fraction": _safe_fraction(counts["PIM"], expert_count),
            "both_fraction": _safe_fraction(counts["BOTH"], expert_count),
            "gpu_resident_count": gpu_resident_count,
            "pim_resident_count": pim_resident_count,
            "gpu_resident_fraction": _safe_fraction(gpu_resident_count, expert_count),
            "pim_resident_fraction": _safe_fraction(pim_resident_count, expert_count),
            "total_copy_count": total_copy_count,
            "total_copy_fraction": _safe_fraction(total_copy_count, expert_count),
            "extra_copy_count": counts["BOTH"],
            "extra_copy_fraction": _safe_fraction(counts["BOTH"], expert_count),
        }

        meta = layer_metadata.get(str(layer_id), {})
        if isinstance(meta, MappingABC):
            for key in (
                "primary_gpu_count",
                "primary_pim_count",
                "warmup_observed_experts",
                "oracle_gpu_events",
                "oracle_pim_events",
                "total_loss_if_gpu_only",
                "total_loss_if_pim_only",
            ):
                if key in meta:
                    row[key] = meta[key]
            if "extra_copies" in meta:
                row["configured_extra_copies"] = meta["extra_copies"]
            if "extra_copy_fraction" in meta:
                row["configured_extra_copy_fraction"] = meta["extra_copy_fraction"]
        rows.append(row)
    return rows


def _replay_steps_from_warmup_stats(warmup_stats: Any) -> list[ReplayStep]:
    if warmup_stats is None:
        return []
    if isinstance(warmup_stats, ReplayStep):
        return [warmup_stats]
    if isinstance(warmup_stats, IterableABC) and not isinstance(warmup_stats, (str, bytes, MappingABC)):
        steps = list(warmup_stats)
        if not all(isinstance(step, ReplayStep) for step in steps):
            raise ValueError("adaptive_extra_budget placement requires ReplayStep warmup entries")
        return steps
    raise ValueError("adaptive_extra_budget placement requires warmup_stats as ReplayStep or iterable[ReplayStep]")


def _validated_active_items_by_layer(step: ReplayStep, model: "ModelSpec") -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for raw_layer_id, raw_counts in sorted(step.layer_expert_counts.items(), key=lambda item: int(item[0])):
        layer_id = _non_negative_int(raw_layer_id, "warmup layer")
        if layer_id >= model.num_layers:
            raise ValueError(f"warmup layer {layer_id} exceeds model.num_layers={model.num_layers}")
        if not isinstance(raw_counts, MappingABC):
            raise ValueError(f"warmup layer {layer_id} must map expert->token_count")
        expert_count = model.experts_for_layer(layer_id)
        active: list[tuple[int, int]] = []
        for raw_expert_id, raw_token_count in sorted(raw_counts.items(), key=lambda item: int(item[0])):
            expert_id = _non_negative_int(raw_expert_id, f"warmup layer {layer_id} expert")
            if expert_id >= expert_count:
                raise ValueError(f"warmup expert {expert_id} exceeds layer {layer_id} expert_count={expert_count}")
            token_count = _non_negative_int(raw_token_count, f"warmup layer {layer_id} expert {expert_id} token_count")
            if token_count > 0:
                active.append((expert_id, token_count))
        if active:
            out[layer_id] = active
    return out


def _adaptive_layer_latency(
    gpu_active: int,
    offload_units: int,
    model: "ModelSpec",
    hardware: "HardwareModel",
    layer_id: int,
    *,
    latency_combine_mode: str = "max",
) -> float:
    gpu_time_s = hardware.gpu_expert_time(gpu_active, model, layer_id=layer_id)
    offload_time_s = hardware.offload_expert_time(offload_units, model, layer_id=layer_id)
    return combine_gpu_offload_latency(
        gpu_time_s,
        offload_time_s,
        latency_combine_mode=latency_combine_mode,
    )


def warmup_stats_to_histogram(warmup_stats: Any) -> Histogram:
    """Normalize warmup stats into mean demand per decode step.

    Accepted inputs are ``None``, a single ``ReplayStep``, an iterable of
    ``ReplayStep`` objects, a ``PredictionFrame``, or an already-shaped
    ``{layer: {expert: demand}}`` mapping. Iterable warmup steps are averaged so
    the scale is comparable with ``PredictionFrame`` values, which are also mean
    predicted demand per step.
    """

    if warmup_stats is None:
        return {}
    if isinstance(warmup_stats, ReplayStep):
        return _histogram_from_replay_step(warmup_stats)
    if isinstance(warmup_stats, PredictionFrame):
        return prediction_to_histogram(warmup_stats)
    if isinstance(warmup_stats, MappingABC):
        return _histogram_from_mapping(warmup_stats)
    if isinstance(warmup_stats, IterableABC) and not isinstance(warmup_stats, (str, bytes)):
        total: Histogram = {}
        count = 0
        for item in warmup_stats:
            _add_scaled(total, warmup_stats_to_histogram(item), 1.0)
            count += 1
        if count == 0:
            return {}
        return _scale_histogram(total, 1.0 / count)
    raise ValueError(f"unsupported warmup_stats type for placement: {type(warmup_stats).__name__}")


def prediction_to_histogram(prediction: PredictionFrame | MappingABC[Any, Any] | None) -> Histogram:
    """Normalize a prediction object into ``{layer: {expert: demand}}``."""

    if prediction is None:
        return {}
    if isinstance(prediction, PredictionFrame):
        return _copy_histogram(prediction.per_layer_histograms)
    if isinstance(prediction, MappingABC):
        return _histogram_from_mapping(prediction)
    raise ValueError(f"unsupported prediction type for placement: {type(prediction).__name__}")


def placement_scores(
    *,
    warmup_histogram: MappingABC[int, MappingABC[int, float]],
    prediction_histogram: MappingABC[int, MappingABC[int, float]],
    model: "ModelSpec",
    config: PlacementConfig | MappingABC[str, Any] | None = None,
) -> Histogram:
    """Return per-layer per-expert placement demand scores."""

    cfg = placement_config_from_dict(config)
    scores: Histogram = {}
    for layer_id in range(model.num_layers):
        layer_scores: dict[int, float] = {}
        warmup_layer = _normalize_layer_scores(warmup_histogram.get(layer_id, {}))
        prediction_layer = _normalize_layer_scores(prediction_histogram.get(layer_id, {}))
        for expert_id in range(model.experts_for_layer(layer_id)):
            value = (
                cfg.warmup_weight * float(warmup_layer.get(expert_id, 0.0))
                + cfg.prediction_weight * float(prediction_layer.get(expert_id, 0.0))
            )
            layer_scores[expert_id] = value
        scores[layer_id] = layer_scores
    return scores


def _normalize_layer_scores(layer_counts: MappingABC[int, float]) -> dict[int, float]:
    total = sum(max(0.0, float(value)) for value in layer_counts.values())
    if total <= 0.0:
        return {}
    return {
        int(expert_id): max(0.0, float(value)) / total
        for expert_id, value in layer_counts.items()
        if float(value) > 0.0
    }


def placement_state_to_dict(state: PlacementState, *, include_residency: bool = True) -> dict[str, Any]:
    """Return a JSON-serializable placement-state dictionary."""

    payload: dict[str, Any] = {
        "version": state.version,
        "metadata": state.metadata,
        "residency_report": state.report_residency(),
    }
    if include_residency:
        nested: dict[str, dict[str, Residency]] = {}
        for (layer_id, expert_id), residency in state.residency.items():
            nested.setdefault(str(layer_id), {})[str(expert_id)] = residency
        payload["residency"] = nested
    return payload


def _make_placement_state(
    residency: MappingABC[tuple[int, int], Residency],
    *,
    model: "ModelSpec",
    hardware: "HardwareModel",
    metadata: MappingABC[str, Any] | None = None,
    version: int = 0,
) -> PlacementState:
    state_metadata = _base_state_metadata(model, hardware)
    state_metadata.update(dict(metadata or {}))
    return _state_from_residency_and_metadata(residency, metadata=state_metadata, version=version)


def _state_from_residency_and_metadata(
    residency: MappingABC[tuple[int, int], Residency],
    *,
    metadata: MappingABC[str, Any],
    version: int,
) -> PlacementState:
    normalized_residency: dict[tuple[int, int], Residency] = {}
    for (layer_id, expert_id), value in residency.items():
        normalized_residency[
            (
                _non_negative_int(layer_id, "placement layer"),
                _non_negative_int(expert_id, "placement expert"),
            )
        ] = _normalize_residency(value)
    state_metadata = dict(metadata)
    return PlacementState(
        version=version,
        residency=normalized_residency,
        metadata=state_metadata,
    )


def _base_state_metadata(model: "ModelSpec", hardware: "HardwareModel") -> dict[str, Any]:
    experts_per_layer = {
        str(layer_id): model.experts_for_layer(layer_id)
        for layer_id in range(model.num_layers)
    }
    return {
        "model_id": model.model_id,
        "hardware_id": hardware.hardware_id,
        "num_layers": model.num_layers,
        "experts_per_layer": experts_per_layer,
    }


def _histogram_from_replay_step(step: ReplayStep) -> Histogram:
    return {
        int(layer_id): {
            int(expert_id): float(count)
            for expert_id, count in sorted(layer_counts.items())
            if float(count) != 0.0
        }
        for layer_id, layer_counts in sorted(step.layer_expert_counts.items())
        if layer_counts
    }


def _histogram_from_mapping(value: MappingABC[Any, Any]) -> Histogram:
    raw: Any
    if "per_layer_histograms" in value:
        raw = value["per_layer_histograms"]
    elif "layer_expert_counts" in value:
        raw = value["layer_expert_counts"]
    else:
        raw = value
    if not isinstance(raw, MappingABC):
        raise ValueError("histogram mapping must be {layer: {expert: value}}")
    return _copy_histogram(raw)


def _copy_histogram(histogram: MappingABC[Any, Any]) -> Histogram:
    out: Histogram = {}
    for raw_layer_id, raw_layer_counts in sorted(histogram.items(), key=lambda item: int(item[0])):
        layer_id = _non_negative_int(raw_layer_id, "histogram layer")
        if not isinstance(raw_layer_counts, MappingABC):
            raise ValueError(f"histogram layer {layer_id} must map expert->value")
        layer_counts: dict[int, float] = {}
        for raw_expert_id, raw_value in sorted(raw_layer_counts.items(), key=lambda item: int(item[0])):
            expert_id = _non_negative_int(raw_expert_id, f"histogram layer {layer_id} expert")
            value = float(raw_value)
            if value != 0.0:
                layer_counts[expert_id] = value
        if layer_counts:
            out[layer_id] = layer_counts
    return out


def _add_scaled(target: Histogram, source: MappingABC[int, MappingABC[int, float]], scale: float) -> None:
    for layer_id, layer_counts in source.items():
        out_layer = target.setdefault(int(layer_id), {})
        for expert_id, value in layer_counts.items():
            out_layer[int(expert_id)] = out_layer.get(int(expert_id), 0.0) + float(value) * scale


def _scale_histogram(histogram: MappingABC[int, MappingABC[int, float]], scale: float) -> Histogram:
    out: Histogram = {}
    _add_scaled(out, histogram, scale)
    return {
        layer_id: {
            expert_id: value
            for expert_id, value in sorted(layer_counts.items())
            if value != 0.0
        }
        for layer_id, layer_counts in sorted(out.items())
        if layer_counts
    }


def _normalize_residency(value: Any) -> Residency:
    text = str(value).upper()
    if text not in RESIDENCY_VALUES:
        raise ValueError(f"invalid residency {value!r}; choose from {sorted(RESIDENCY_VALUES)}")
    return text  # type: ignore[return-value]


def _fraction(value: Any, field_name: str) -> float:
    out = _non_negative_float(value, field_name)
    if out > 1.0:
        raise ValueError(f"{field_name} must be in [0.0, 1.0]")
    return out


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    return 0.0 if denominator == 0.0 else float(numerator) / denominator


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative float") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative float")
    return out


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return out


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .model_hardware import load_hardware_model, load_model_spec

    model = load_model_spec(args.model_config)
    hardware = load_hardware_model(args.hardware_config)
    config = PlacementConfig(
        mode=args.mode,
        hot_fraction=args.hot_fraction,
        cold_fraction=args.cold_fraction,
        warmup_weight=args.warmup_weight,
        prediction_weight=args.prediction_weight,
        extra_copy_fraction=args.extra_copy_fraction,
    )

    warmup_steps: list[ReplayStep] = []
    prediction: PredictionFrame | None = None
    if args.trace_pack:
        from .prediction import make_predictor
        from .trace_pack import ReplayStream, TracePack

        with TracePack.open(args.trace_pack) as pack:
            stream = ReplayStream(
                pack,
                seed=args.seed,
                max_batch_size=args.max_batch_size,
                warmup_steps=args.warmup_steps,
                eval_steps=1,
                benchmarks=[args.benchmark] if args.benchmark else None,
                limit_per_benchmark=args.limit_per_benchmark,
            )
            warmup_steps = list(stream.warmup_steps())
            eval_steps = list(stream.eval_steps())
            next_step_id = (
                eval_steps[0].step_id
                if eval_steps
                else (warmup_steps[-1].step_id + 1 if warmup_steps else 0)
            )
            predictor = make_predictor(
                {
                    "type": "heuristic",
                    "horizon": args.horizon,
                    "window_size": args.window_size,
                    "decay": args.decay,
                    "seed": args.seed,
                },
                model=model,
            )
            predictor.initialize(warmup_steps)
            prediction = predictor.predict(next_step_id, args.horizon)

    policy = make_placement_policy(config)
    state = policy.initialize(warmup_steps, prediction, model, hardware)
    print(
        _json_dumps(
            {
                "config": config.to_dict(),
                "residency_report": state.report_residency(),
                "metadata": {
                    "policy": state.metadata.get("policy"),
                    "classification_method": state.metadata.get("classification_method"),
                    "score_metadata": state.metadata.get("score_metadata"),
                },
            }
        )
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Placement utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="initialize placement and print residency summary")
    smoke.add_argument("--model-config", required=True)
    smoke.add_argument("--hardware-config", required=True)
    smoke.add_argument("--mode", choices=sorted(PLACEMENT_MODES), default="static_hot_cold")
    smoke.add_argument("--hot-fraction", type=float, default=0.50)
    smoke.add_argument("--cold-fraction", type=float, default=0.25)
    smoke.add_argument("--warmup-weight", type=float, default=1.0)
    smoke.add_argument("--prediction-weight", type=float, default=1.0)
    smoke.add_argument("--extra-copy-fraction", type=float, default=0.25)
    smoke.add_argument("--trace-pack", "--pack", default=None, dest="trace_pack", help="optional TracePack path for warmup+heuristic prediction smoke")
    smoke.add_argument("--benchmark", default=None, help="single benchmark when TracePack has multiple benchmarks")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--max-batch-size", type=int, default=1)
    smoke.add_argument("--warmup-steps", type=int, default=0)
    smoke.add_argument("--limit-per-benchmark", type=int, default=None)
    smoke.add_argument("--horizon", type=int, default=1)
    smoke.add_argument("--window-size", type=int, default=32)
    smoke.add_argument("--decay", type=float, default=1.0)
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
