"""Scheduler and aggregate MoE latency model for MVP-0.

The scheduler consumes actual routed counts for one decode step, respects the
current placement map for normal policies, and estimates only MoE expert-memory
latency.  It reuses the report-equation helpers from ``model_hardware`` instead
of maintaining separate math.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence

from .interfaces import ReplayStep

if TYPE_CHECKING:  # pragma: no cover - import-only typing helper
    from .model_hardware import HardwareModel, ModelSpec
    from .placement import PlacementState

Device = Literal["GPU", "PIM"]
SchedulerPolicyName = Literal["placement_greedy", "gpu_baseline", "oracle_split"]
LatencyCombineMode = Literal["max", "sum"]

SCHEDULER_POLICIES: set[str] = {"placement_greedy", "gpu_baseline", "oracle_split"}
DEVICES: set[str] = {"GPU", "PIM"}
LATENCY_COMBINE_MODES: set[str] = {"max", "sum"}


@dataclass(frozen=True)
class SchedulerConfig:
    """Validated scheduler configuration.

    ``oracle_comparison`` adds a report-style oracle split to decision metadata
    for gap metrics.
    """

    policy: str = "placement_greedy"
    oracle_comparison: bool = False
    latency_combine_mode: str = "max"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = str(self.policy).lower()
        if policy not in SCHEDULER_POLICIES:
            raise ValueError(f"unsupported scheduler policy {self.policy!r}; choose from {sorted(SCHEDULER_POLICIES)}")
        latency_combine_mode = normalize_latency_combine_mode(self.latency_combine_mode)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "latency_combine_mode", latency_combine_mode)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "oracle_comparison": self.oracle_comparison,
            "latency_combine_mode": self.latency_combine_mode,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class WorkAssignment:
    """One active expert's scheduled device for a decode step."""

    layer: int
    expert: int
    token_count: int
    device: Device
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _non_negative_int(self.layer, "assignment layer"))
        object.__setattr__(self, "expert", _non_negative_int(self.expert, "assignment expert"))
        object.__setattr__(self, "token_count", _positive_int(self.token_count, "assignment token_count"))
        object.__setattr__(self, "device", _normalize_device(self.device, "assignment device"))
        object.__setattr__(self, "reason", str(self.reason))

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "token_count": self.token_count,
            "device": self.device,
            "reason": self.reason,
        }


def _trusted_work_assignment(layer: int, expert: int, token_count: int, device: Device, reason: str) -> WorkAssignment:
    """Construct an internal assignment after scheduler-side validation."""

    assignment = object.__new__(WorkAssignment)
    object.__setattr__(assignment, "layer", layer)
    object.__setattr__(assignment, "expert", expert)
    object.__setattr__(assignment, "token_count", token_count)
    object.__setattr__(assignment, "device", device)
    object.__setattr__(assignment, "reason", reason)
    return assignment


@dataclass(frozen=True)
class ScheduleDecision:
    """Scheduler output and aggregate latency estimate for one decode step."""

    step_id: int
    assignments: list[WorkAssignment]
    gpu_time_s: list[float]
    pim_time_s: list[float]
    total_latency_s: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _non_negative_int(self.step_id, "schedule step_id"))
        assignments = list(self.assignments or [])
        for assignment in assignments:
            if not isinstance(assignment, WorkAssignment):
                raise ValueError("schedule assignments must be WorkAssignment objects")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "gpu_time_s", [_non_negative_float(v, "gpu_time_s") for v in self.gpu_time_s])
        object.__setattr__(self, "pim_time_s", [_non_negative_float(v, "pim_time_s") for v in self.pim_time_s])
        object.__setattr__(self, "total_latency_s", _non_negative_float(self.total_latency_s, "total_latency_s"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "gpu_time_s": self.gpu_time_s,
            "pim_time_s": self.pim_time_s,
            "total_latency_s": self.total_latency_s,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OracleLayerSplit:
    """Report-style latency-minimizing split for one layer."""

    layer: int
    active_experts: int
    cold_experts: int
    pim_quanta: int
    gpu_time_s: float
    pim_time_s: float
    total_latency_s: float
    pim_experts: tuple[int, ...]
    gpu_experts: tuple[int, ...]
    boundary_token_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "active_experts": self.active_experts,
            "cold_experts": self.cold_experts,
            "pim_quanta": self.pim_quanta,
            "gpu_time_s": self.gpu_time_s,
            "pim_time_s": self.pim_time_s,
            "total_latency_s": self.total_latency_s,
            "pim_experts": list(self.pim_experts),
            "gpu_experts": list(self.gpu_experts),
            "boundary_token_count": self.boundary_token_count,
        }


@dataclass(frozen=True)
class _LayerSchedule:
    assignments: list[WorkAssignment]
    gpu_active_experts: int
    pim_active_experts: int
    pim_quanta: int
    gpu_time_s: float
    pim_time_s: float
    layer_latency_s: float
    active_residency_counts: dict[str, int]
    token_count_by_device: dict[str, int]


class Scheduler:
    """MVP aggregate scheduler.

    ``placement_greedy`` is the default vertical-slice policy: GPU-only experts
    run on GPU, PIM-only experts run on PIM, and BOTH experts use a
    residency-aware report-style global split over the active BOTH set.
    """

    def __init__(self, config: SchedulerConfig | MappingABC[str, Any] | None = None):
        self.config = scheduler_config_from_dict(config)

    def schedule(
        self,
        replay_step: ReplayStep,
        placement: "PlacementState",
        model: "ModelSpec",
        hardware: "HardwareModel",
    ) -> ScheduleDecision:
        counts_by_layer = _validated_active_counts(replay_step, model)
        assignments: list[WorkAssignment] = []
        gpu_time_s = [0.0 for _ in range(model.num_layers)]
        pim_time_s = [0.0 for _ in range(model.num_layers)]
        layer_latency_s = [0.0 for _ in range(model.num_layers)]
        gpu_active_by_layer = [0 for _ in range(model.num_layers)]
        pim_quanta_by_layer = [0 for _ in range(model.num_layers)]
        offload_active_by_layer = [0 for _ in range(model.num_layers)]
        active_residency_counts = {"GPU": 0, "PIM": 0, "BOTH": 0}
        token_count_by_device = {"GPU": 0, "PIM": 0}

        for layer_id in range(model.num_layers):
            layer_counts = counts_by_layer.get(layer_id, {})
            if not layer_counts:
                continue
            layer_schedule = self._schedule_layer(layer_id, layer_counts, placement, model, hardware)
            assignments.extend(layer_schedule.assignments)
            gpu_time_s[layer_id] = layer_schedule.gpu_time_s
            pim_time_s[layer_id] = layer_schedule.pim_time_s
            layer_latency_s[layer_id] = layer_schedule.layer_latency_s
            gpu_active_by_layer[layer_id] = layer_schedule.gpu_active_experts
            pim_quanta_by_layer[layer_id] = layer_schedule.pim_quanta
            offload_active_by_layer[layer_id] = layer_schedule.pim_active_experts
            for key, value in layer_schedule.active_residency_counts.items():
                active_residency_counts[key] += value
            for key, value in layer_schedule.token_count_by_device.items():
                token_count_by_device[key] += value

        assignments.sort(key=lambda item: (item.layer, item.expert))
        total_latency_s = sum(layer_latency_s)
        assignment_counts = _assignment_counts(assignments)
        metadata: dict[str, Any] = {
            "policy": self.config.policy,
            "scheduler_config": self.config.to_dict(),
            "latency_combine_mode": self.config.latency_combine_mode,
            "placement_version": getattr(placement, "version", None),
            "layer_latency_s": layer_latency_s,
            "gpu_active_experts_by_layer": gpu_active_by_layer,
            "pim_quanta_by_layer": pim_quanta_by_layer,
            "offload_backend": hardware.offload_backend,
            "offload_label": hardware.offload_label(),
            "offload_active_experts_by_layer": offload_active_by_layer,
            "offload_time_s": pim_time_s,
            "assignment_counts": assignment_counts,
            "token_count_by_device": token_count_by_device,
            "active_residency_counts": active_residency_counts,
            "total_gpu_time_s": sum(gpu_time_s),
            "total_pim_time_s": sum(pim_time_s),
            "total_offload_time_s": sum(pim_time_s),
            "max_layer_imbalance_s": max((abs(g - p) for g, p in zip(gpu_time_s, pim_time_s)), default=0.0),
        }
        if self.config.oracle_comparison:
            oracle = oracle_schedule_decision(
                replay_step,
                model,
                hardware,
                latency_combine_mode=self.config.latency_combine_mode,
            )
            metadata.update(
                {
                    "oracle_comparison_enabled": True,
                    "oracle_total_latency_s": oracle.total_latency_s,
                    "oracle_gap_s": total_latency_s - oracle.total_latency_s,
                    "oracle_gap_ratio": _safe_ratio(total_latency_s, oracle.total_latency_s),
                    "oracle_layer_latency_s": oracle.metadata["layer_latency_s"],
                    "oracle_layer_splits": oracle.metadata["oracle_layer_splits"],
                }
            )
        else:
            metadata["oracle_comparison_enabled"] = False

        return ScheduleDecision(
            step_id=replay_step.step_id,
            assignments=assignments,
            gpu_time_s=gpu_time_s,
            pim_time_s=pim_time_s,
            total_latency_s=total_latency_s,
            metadata=metadata,
        )

    def _schedule_layer(
        self,
        layer_id: int,
        layer_counts: MappingABC[int, int],
        placement: "PlacementState",
        model: "ModelSpec",
        hardware: "HardwareModel",
    ) -> _LayerSchedule:
        if self.config.policy == "oracle_split":
            return _layer_schedule_from_oracle_split(
                layer_id,
                layer_counts,
                model,
                hardware,
                latency_combine_mode=self.config.latency_combine_mode,
            )

        assignments: list[WorkAssignment] = []
        both: list[tuple[int, int]] = []
        gpu_active = 0
        pim_active = 0
        pim_quanta = 0
        active_residency_counts = {"GPU": 0, "PIM": 0, "BOTH": 0}
        token_count_by_device = {"GPU": 0, "PIM": 0}
        for expert_id, token_count in sorted(layer_counts.items()):
            if self.config.policy == "gpu_baseline":
                device, reason = "GPU", "gpu_baseline_ignore_placement"
                active_residency_counts["GPU"] += 1
            else:
                residency = placement.get(layer_id, expert_id)
                active_residency_counts[residency] += 1
                if residency == "GPU":
                    device, reason = "GPU", "gpu_resident"
                elif residency == "PIM":
                    device, reason = "PIM", "pim_resident"
                else:
                    both.append((expert_id, token_count))
                    continue
            assignment = _trusted_work_assignment(layer_id, expert_id, token_count, device, reason)
            assignments.append(assignment)
            if device == "GPU":
                gpu_active += 1
            else:
                pim_active += 1
                pim_quanta += _offload_units(token_count, hardware)
            token_count_by_device[device] += token_count

        if self.config.policy == "placement_greedy" and both:
            both_sorted = sorted(both, key=lambda item: (item[1], item[0]))
            both_count = len(both_sorted)
            prefix_quanta = _prefix_offload_units(both_sorted, hardware)
            best_cold_count = _best_prefix_split(
                split_count=both_count,
                base_gpu_active=gpu_active,
                base_pim_quanta=pim_quanta,
                prefix_pim_quanta=prefix_quanta,
                model=model,
                hardware=hardware,
                layer_id=layer_id,
                latency_combine_mode=self.config.latency_combine_mode,
            )

            for idx, (expert_id, token_count) in enumerate(both_sorted):
                if idx < best_cold_count:
                    device, reason = "PIM", "both_global_optimal_split_pim"
                else:
                    device, reason = "GPU", "both_global_optimal_split_gpu"
                assignments.append(_trusted_work_assignment(layer_id, expert_id, token_count, device, reason))
                if device == "GPU":
                    gpu_active += 1
                else:
                    pim_active += 1
                    pim_quanta += _offload_units(token_count, hardware)
                token_count_by_device[device] += token_count

        gpu_time_s, pim_time_s, layer_latency_s = _latency_from_work(
            gpu_active,
            pim_quanta,
            model,
            hardware,
            layer_id,
            latency_combine_mode=self.config.latency_combine_mode,
        )
        return _LayerSchedule(
            assignments=assignments,
            gpu_active_experts=gpu_active,
            pim_active_experts=pim_active,
            pim_quanta=pim_quanta,
            gpu_time_s=gpu_time_s,
            pim_time_s=pim_time_s,
            layer_latency_s=layer_latency_s,
            active_residency_counts=active_residency_counts,
            token_count_by_device=token_count_by_device,
        )


def scheduler_config_from_dict(data: MappingABC[str, Any] | SchedulerConfig | None = None) -> SchedulerConfig:
    """Create ``SchedulerConfig`` from a root or ``scheduler:`` config mapping."""

    if data is None:
        return SchedulerConfig()
    if isinstance(data, SchedulerConfig):
        return data
    cfg = data.get("scheduler", data)
    if not isinstance(cfg, MappingABC):
        raise ValueError("scheduler config must be a mapping")

    known = {
        "policy",
        "type",
        "oracle_comparison",
        "enable_oracle_comparison",
        "latency_combine_mode",
        "metadata",
    }
    metadata = dict(cfg.get("metadata") or {})
    extra = {str(k): v for k, v in cfg.items() if k not in known}
    if extra:
        metadata.setdefault("extra_config", extra)

    return SchedulerConfig(
        policy=str(cfg.get("policy", cfg.get("type", "placement_greedy"))),
        oracle_comparison=bool(cfg.get("oracle_comparison", cfg.get("enable_oracle_comparison", False))),
        latency_combine_mode=str(cfg.get("latency_combine_mode", "max")),
        metadata=metadata,
    )


def make_scheduler(config: MappingABC[str, Any] | SchedulerConfig | None = None) -> Scheduler:
    """Instantiate an MVP scheduler from config."""

    return Scheduler(config)


def oracle_layer_split(
    layer_counts: MappingABC[int, int],
    model: "ModelSpec",
    hardware: "HardwareModel",
    *,
    layer_id: int = 0,
    latency_combine_mode: str = "max",
) -> OracleLayerSplit:
    """Return the report-equation latency-minimizing split for one layer.

    Active experts are sorted by ``(token_count, expert_id)``.  If multiple
    split widths have exactly equal latency, the smaller PIM width is selected
    to keep the oracle deterministic and avoid unnecessary PIM work.
    """

    active = [(expert_id, _positive_int(count, "layer token count")) for expert_id, count in layer_counts.items() if count]
    active.sort(key=lambda item: (item[1], item[0]))
    active_count = len(active)
    prefix_quanta = _prefix_offload_units(active, hardware)
    best_n = _best_prefix_split(
        split_count=active_count,
        base_gpu_active=0,
        base_pim_quanta=0,
        prefix_pim_quanta=prefix_quanta,
        model=model,
        hardware=hardware,
        layer_id=layer_id,
        latency_combine_mode=latency_combine_mode,
    )
    gpu_time_s, pim_time_s, total_latency_s = _latency_from_work(
        active_count - best_n,
        prefix_quanta[best_n],
        model,
        hardware,
        layer_id,
        latency_combine_mode=latency_combine_mode,
    )
    pim_items = active[:best_n]
    gpu_items = active[best_n:]
    return OracleLayerSplit(
        layer=layer_id,
        active_experts=active_count,
        cold_experts=best_n,
        pim_quanta=prefix_quanta[best_n],
        gpu_time_s=gpu_time_s,
        pim_time_s=pim_time_s,
        total_latency_s=total_latency_s,
        pim_experts=tuple(expert_id for expert_id, _ in pim_items),
        gpu_experts=tuple(expert_id for expert_id, _ in gpu_items),
        boundary_token_count=(pim_items[-1][1] if pim_items else None),
    )


def oracle_schedule_decision(
    replay_step: ReplayStep,
    model: "ModelSpec",
    hardware: "HardwareModel",
    *,
    latency_combine_mode: str = "max",
) -> ScheduleDecision:
    """Return a report-style oracle schedule independent of placement."""

    latency_combine_mode = normalize_latency_combine_mode(latency_combine_mode)
    counts_by_layer = _validated_active_counts(replay_step, model)
    assignments: list[WorkAssignment] = []
    gpu_time_s = [0.0 for _ in range(model.num_layers)]
    pim_time_s = [0.0 for _ in range(model.num_layers)]
    layer_latency_s = [0.0 for _ in range(model.num_layers)]
    gpu_active_by_layer = [0 for _ in range(model.num_layers)]
    pim_quanta_by_layer = [0 for _ in range(model.num_layers)]
    offload_active_by_layer = [0 for _ in range(model.num_layers)]
    splits: list[dict[str, Any]] = []

    for layer_id in range(model.num_layers):
        layer_counts = counts_by_layer.get(layer_id, {})
        if not layer_counts:
            continue
        split = oracle_layer_split(
            layer_counts,
            model,
            hardware,
            layer_id=layer_id,
            latency_combine_mode=latency_combine_mode,
        )
        splits.append(split.to_dict())
        layer_assignments, _ = _assignments_from_oracle_split(layer_id, layer_counts, split)
        assignments.extend(layer_assignments)
        gpu_time_s[layer_id] = split.gpu_time_s
        pim_time_s[layer_id] = split.pim_time_s
        layer_latency_s[layer_id] = split.total_latency_s
        gpu_active_by_layer[layer_id] = split.active_experts - split.cold_experts
        pim_quanta_by_layer[layer_id] = split.pim_quanta
        offload_active_by_layer[layer_id] = split.cold_experts

    total_latency_s = sum(layer_latency_s)
    return ScheduleDecision(
        step_id=replay_step.step_id,
        assignments=assignments,
        gpu_time_s=gpu_time_s,
        pim_time_s=pim_time_s,
        total_latency_s=total_latency_s,
        metadata={
            "policy": "oracle_split",
            "latency_combine_mode": latency_combine_mode,
            "oracle_comparison_enabled": False,
            "layer_latency_s": layer_latency_s,
            "gpu_active_experts_by_layer": gpu_active_by_layer,
            "pim_quanta_by_layer": pim_quanta_by_layer,
            "offload_backend": hardware.offload_backend,
            "offload_label": hardware.offload_label(),
            "offload_active_experts_by_layer": offload_active_by_layer,
            "offload_time_s": pim_time_s,
            "total_offload_time_s": sum(pim_time_s),
            "assignment_counts": _assignment_counts(assignments),
            "oracle_layer_splits": splits,
            "comparison_only_ignores_placement": True,
        },
    )


def schedule_decision_to_dict(decision: ScheduleDecision) -> dict[str, Any]:
    """Compatibility helper for metrics/reporting code."""

    return decision.to_dict()


def _layer_schedule_from_oracle_split(
    layer_id: int,
    layer_counts: MappingABC[int, int],
    model: "ModelSpec",
    hardware: "HardwareModel",
    *,
    latency_combine_mode: str = "max",
) -> _LayerSchedule:
    split = oracle_layer_split(
        layer_counts,
        model,
        hardware,
        layer_id=layer_id,
        latency_combine_mode=latency_combine_mode,
    )
    assignments, token_count_by_device = _assignments_from_oracle_split(layer_id, layer_counts, split)
    return _LayerSchedule(
        assignments=assignments,
        gpu_active_experts=split.active_experts - split.cold_experts,
        pim_active_experts=split.cold_experts,
        pim_quanta=split.pim_quanta,
        gpu_time_s=split.gpu_time_s,
        pim_time_s=split.pim_time_s,
        layer_latency_s=split.total_latency_s,
        active_residency_counts={"GPU": 0, "PIM": 0, "BOTH": 0},
        token_count_by_device=token_count_by_device,
    )


def _assignments_from_oracle_split(
    layer_id: int,
    layer_counts: MappingABC[int, int],
    split: OracleLayerSplit,
) -> tuple[list[WorkAssignment], dict[str, int]]:
    pim_experts = set(split.pim_experts)
    assignments: list[WorkAssignment] = []
    token_count_by_device = {"GPU": 0, "PIM": 0}
    for expert_id, token_count in sorted(layer_counts.items()):
        device: Device = "PIM" if expert_id in pim_experts else "GPU"
        reason = "oracle_cold_split" if device == "PIM" else "oracle_gpu_split"
        assignments.append(_trusted_work_assignment(layer_id, expert_id, token_count, device, reason))
        token_count_by_device[device] += token_count
    return assignments, token_count_by_device


def _validated_active_counts(replay_step: ReplayStep, model: "ModelSpec") -> dict[int, dict[int, int]]:
    if not isinstance(replay_step, ReplayStep):
        raise ValueError("scheduler expects a ReplayStep")
    out: dict[int, dict[int, int]] = {}
    for raw_layer, raw_counts in sorted(replay_step.layer_expert_counts.items(), key=lambda item: int(item[0])):
        layer_id = _non_negative_int(raw_layer, "replay layer")
        if layer_id >= model.num_layers:
            raise ValueError(f"replay layer {layer_id} is outside model layer range [0, {model.num_layers})")
        if not isinstance(raw_counts, MappingABC):
            raise ValueError(f"replay layer {layer_id} counts must map expert->count")
        layer_out: dict[int, int] = {}
        for raw_expert, raw_count in sorted(raw_counts.items(), key=lambda item: int(item[0])):
            expert_id = _non_negative_int(raw_expert, f"replay layer {layer_id} expert")
            if expert_id >= model.experts_for_layer(layer_id):
                raise ValueError(
                    f"replay expert {expert_id} in layer {layer_id} exceeds model experts_per_layer={model.experts_for_layer(layer_id)}"
                )
            count = _non_negative_int(raw_count, f"replay layer {layer_id} expert {expert_id} count")
            if count:
                layer_out[expert_id] = count
        if layer_out:
            out[layer_id] = layer_out
    return out


def _assignment_counts(assignments: Sequence[WorkAssignment]) -> dict[str, int]:
    counts = {"GPU": 0, "PIM": 0}
    for assignment in assignments:
        counts[assignment.device] += 1
    return counts


def _pim_quanta(token_count: int, hardware: "HardwareModel") -> int:
    return (token_count + hardware.pim_granularity - 1) // hardware.pim_granularity


def offload_units_for_token_count(token_count: int, hardware: "HardwareModel") -> int:
    """Return scheduler-consistent offload units for one active expert."""

    if hardware.is_remote_memory_backend():
        return 1
    return _pim_quanta(token_count, hardware)


def _offload_units(token_count: int, hardware: "HardwareModel") -> int:
    return offload_units_for_token_count(token_count, hardware)


def _prefix_offload_units(sorted_items: Sequence[tuple[int, int]], hardware: "HardwareModel") -> list[int]:
    prefix = [0]
    for _, token_count in sorted_items:
        prefix.append(prefix[-1] + _offload_units(token_count, hardware))
    return prefix


def normalize_latency_combine_mode(value: Any) -> LatencyCombineMode:
    mode = str(value).lower()
    if mode not in LATENCY_COMBINE_MODES:
        raise ValueError(f"latency_combine_mode must be one of {sorted(LATENCY_COMBINE_MODES)}, got {value!r}")
    return mode  # type: ignore[return-value]


def combine_gpu_offload_latency(
    gpu_time_s: float,
    offload_time_s: float,
    *,
    latency_combine_mode: str = "max",
) -> float:
    mode = normalize_latency_combine_mode(latency_combine_mode)
    if mode == "sum":
        return gpu_time_s + offload_time_s
    return max(gpu_time_s, offload_time_s)


def _best_prefix_split(
    *,
    split_count: int,
    base_gpu_active: int,
    base_pim_quanta: int,
    prefix_pim_quanta: Sequence[int],
    model: "ModelSpec",
    hardware: "HardwareModel",
    layer_id: int,
    latency_combine_mode: str = "max",
) -> int:
    latency_combine_mode = normalize_latency_combine_mode(latency_combine_mode)
    gpu_unit_s = hardware.gpu_expert_time(1, model, layer_id=layer_id)
    pim_unit_s = hardware.offload_expert_time(1, model, layer_id=layer_id)
    best_cold_count = 0
    best_latency = math.inf
    for cold_count in range(split_count + 1):
        gpu_active = base_gpu_active + (split_count - cold_count)
        pim_quanta = base_pim_quanta + prefix_pim_quanta[cold_count]
        latency = combine_gpu_offload_latency(
            gpu_active * gpu_unit_s,
            pim_quanta * pim_unit_s,
            latency_combine_mode=latency_combine_mode,
        )
        if latency < best_latency:
            best_cold_count = cold_count
            best_latency = latency
    return best_cold_count


def _latency_from_work(
    gpu_active: int,
    pim_quanta: int,
    model: "ModelSpec",
    hardware: "HardwareModel",
    layer_id: int,
    *,
    latency_combine_mode: str = "max",
) -> tuple[float, float, float]:
    gpu_time_s = hardware.gpu_expert_time(gpu_active, model, layer_id=layer_id)
    pim_time_s = hardware.offload_expert_time(pim_quanta, model, layer_id=layer_id)
    return gpu_time_s, pim_time_s, combine_gpu_offload_latency(
        gpu_time_s,
        pim_time_s,
        latency_combine_mode=latency_combine_mode,
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _normalize_device(value: Any, field_name: str) -> Device:
    text = str(value).upper()
    if text not in DEVICES:
        raise ValueError(f"{field_name} must be one of {sorted(DEVICES)}, got {value!r}")
    return text  # type: ignore[return-value]


def _positive_int(value: Any, field_name: str) -> int:
    out = _non_negative_int(value, field_name)
    if out <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
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
    if str(value).strip() != str(out) and not (isinstance(value, float) and value.is_integer()):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if out < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return out


def _non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite non-negative float")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite non-negative float") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative float")
    return out


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .model_hardware import load_hardware_model, load_model_spec
    from .placement import make_placement_policy
    from .trace_pack import ReplayStream, TracePack

    model = load_model_spec(args.model_config)
    hardware = load_hardware_model(args.hardware_config)
    scheduler = make_scheduler(
        {
            "scheduler": {
                "policy": args.policy,
                "oracle_comparison": not args.disable_oracle_comparison,
            }
        }
    )
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
        placement = make_placement_policy({"placement": {"mode": "all_gpu"}}).initialize(warmup_steps, None, model, hardware)
        eval_steps = list(stream.eval_steps())
        if not eval_steps:
            raise ValueError("no eval step available after warmup")
        decision = scheduler.schedule(eval_steps[0], placement, model, hardware)
    print(_json_dumps(decision.to_dict()))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheduler and latency utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="schedule one eval step from a TracePack")
    smoke.add_argument("--trace-pack", "--pack", required=True, dest="trace_pack")
    smoke.add_argument("--model-config", required=True)
    smoke.add_argument("--hardware-config", required=True)
    smoke.add_argument("--benchmark")
    smoke.add_argument("--limit-per-benchmark", type=int)
    smoke.add_argument("--seed", type=int, default=123)
    smoke.add_argument("--max-batch-size", type=int, default=16)
    smoke.add_argument("--warmup-steps", type=int, default=0)
    smoke.add_argument("--policy", default="placement_greedy", choices=sorted(SCHEDULER_POLICIES))
    smoke.add_argument("--disable-oracle-comparison", action="store_true")
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
