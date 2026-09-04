"""Prediction-guided expert migration policies.

The implemented dynamic policy is intentionally small: it spends a model-wide
budget of paired overlap exchanges and never performs direct GPU<->PIM primary
swaps.  Each pair promotes one single-resident expert to BOTH and demotes one
BOTH expert to a single side in the same layer, so the per-layer extra-copy
count chosen by placement remains invariant during evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence, runtime_checkable

from .scheduler import combine_gpu_offload_latency

if TYPE_CHECKING:  # pragma: no cover - import-only typing helper
    from .model_hardware import HardwareModel, ModelSpec
    from .placement import PlacementState, Residency

MigrationMode = Literal["none", "predictive_budgeted_regret"]
BudgetPairsPerStep = int | Literal["auto"]
MIGRATION_MODES: set[str] = {"none", "predictive_budgeted_regret"}
_SINGLE_RESIDENCIES = {"GPU", "PIM"}
_LATENCY_COMBINE_MODE = "max"
_AUTO_BUDGETS_BY_SLUG = {
    "qwen3_235b_a22b_fp8": 100,
    "deepseek_r1_awq": 50,
    "kimi_k2_thinking": 50,
    "llama_4_maverick_17b_128e_instruct": 20,
}


@dataclass(frozen=True)
class MigrationConfig:
    """Validated migration config.

    User-facing knobs are intentionally limited to ``mode`` and
    ``budget_pairs_per_step``.  ``auto`` resolves to the project-owner default
    budget for the active model.
    """

    mode: MigrationMode | str = "none"
    budget_pairs_per_step: BudgetPairsPerStep | str = "auto"

    def __post_init__(self) -> None:
        mode = _normalize_mode(self.mode)
        budget = _normalize_budget_pairs(self.budget_pairs_per_step)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "budget_pairs_per_step", budget)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "budget_pairs_per_step": self.budget_pairs_per_step}


@dataclass(frozen=True)
class MigrationAction:
    """One legal overlap-exchange pair.

    A pair always performs:
    - one single-resident expert ``GPU/PIM -> BOTH``;
    - one currently-BOTH expert ``BOTH -> GPU/PIM`` in the same layer.
    """

    layer: int
    promote_expert: int
    promote_from: str
    demote_expert: int
    demote_to: str
    score_s: float
    promote_gain_s: float = 0.0
    demote_gain_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _non_negative_int(self.layer, "migration action layer"))
        object.__setattr__(self, "promote_expert", _non_negative_int(self.promote_expert, "migration promote expert"))
        object.__setattr__(self, "demote_expert", _non_negative_int(self.demote_expert, "migration demote expert"))
        promote_from = _normalize_residency_name(self.promote_from, "migration promote_from")
        demote_to = _normalize_residency_name(self.demote_to, "migration demote_to")
        if promote_from == "BOTH":
            raise ValueError("migration promotion must start from GPU or PIM")
        if demote_to == "BOTH":
            raise ValueError("migration demotion must end at GPU or PIM")
        object.__setattr__(self, "promote_from", promote_from)
        object.__setattr__(self, "demote_to", demote_to)
        object.__setattr__(self, "score_s", float(self.score_s))
        object.__setattr__(self, "promote_gain_s", float(self.promote_gain_s))
        object.__setattr__(self, "demote_gain_s", float(self.demote_gain_s))

    @property
    def kind(self) -> str:
        return f"{self.promote_from}->BOTH+BOTH->{self.demote_to}"

    @property
    def changed_entries(self) -> int:
        return 2

    def updates(self) -> dict[tuple[int, int], "Residency"]:
        return {
            (self.layer, self.promote_expert): "BOTH",  # type: ignore[return-value]
            (self.layer, self.demote_expert): self.demote_to,  # type: ignore[return-value]
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "kind": self.kind,
            "score_s": self.score_s,
            "promote": {
                "expert": self.promote_expert,
                "from": self.promote_from,
                "to": "BOTH",
                "gain_s": self.promote_gain_s,
            },
            "demote": {
                "expert": self.demote_expert,
                "from": "BOTH",
                "to": self.demote_to,
                "gain_s": self.demote_gain_s,
            },
        }


@dataclass(frozen=True)
class MigrationPlan:
    """Migration decisions for one control step."""

    control_step: int
    actions: tuple[MigrationAction, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_step", _non_negative_int(self.control_step, "migration control_step"))
        object.__setattr__(self, "actions", tuple(self.actions or ()))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def is_empty(self) -> bool:
        return len(self.actions) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_step": self.control_step,
            "actions": [action.to_dict() for action in self.actions],
            "metadata": self.metadata,
        }


@runtime_checkable
class MigrationPolicy(Protocol):
    """Common migration hook interface used by the simulation engine."""

    def should_run(self, step_id: int, metrics_snapshot: Any | None = None) -> bool:
        ...

    def plan(
        self,
        placement: "PlacementState",
        prediction: Any,
        model: "ModelSpec | None" = None,
        hardware: "HardwareModel | None" = None,
    ) -> MigrationPlan:
        ...

    def apply(
        self,
        placement: "PlacementState",
        plan: MigrationPlan,
        model: "ModelSpec | None" = None,
        hardware: "HardwareModel | None" = None,
    ) -> "PlacementState":
        ...


class NoneMigrationPolicy:
    """Disabled migration policy: never request a placement change."""

    policy_name = "none"

    def __init__(self, config: MigrationConfig | MappingABC[str, Any] | None = None):
        self.config = migration_config_from_dict(config)
        if self.config.mode != "none":
            raise ValueError("NoneMigrationPolicy requires mode 'none'")

    def should_run(self, step_id: int, metrics_snapshot: Any | None = None) -> bool:
        del metrics_snapshot
        _non_negative_int(step_id, "migration step_id")
        return False

    def plan(
        self,
        placement: "PlacementState",
        prediction: Any,
        model: "ModelSpec | None" = None,
        hardware: "HardwareModel | None" = None,
    ) -> MigrationPlan:
        del model, hardware
        metadata = {
            "mode": "none",
            "policy": self.policy_name,
            "placement_version": getattr(placement, "version", None),
        }
        return MigrationPlan(control_step=_control_step_from_prediction(prediction), metadata=metadata)

    def apply(
        self,
        placement: "PlacementState",
        plan: MigrationPlan,
        model: "ModelSpec | None" = None,
        hardware: "HardwareModel | None" = None,
    ) -> "PlacementState":
        del plan, model, hardware
        return placement


@dataclass(frozen=True)
class _Candidate:
    gain_s: float
    layer: int
    expert: int
    before: str
    after: str


class PredictiveBudgetedRegretMigrationPolicy:
    """Simple prediction-guided overlap exchange policy.

    The policy uses the already-active predictor output for the next control
    step.  It only moves second copies: every selected pair promotes one
    single-resident expert to BOTH and demotes one BOTH expert to a single side
    in the same layer.  This stages primary side changes through BOTH and keeps
    the placement policy's per-layer overlap budget exactly fixed.
    """

    policy_name = "predictive_budgeted_regret"

    def __init__(self, config: MigrationConfig | MappingABC[str, Any] | None = None):
        self.config = migration_config_from_dict(config)
        if self.config.mode != "predictive_budgeted_regret":
            raise ValueError("PredictiveBudgetedRegretMigrationPolicy requires mode 'predictive_budgeted_regret'")

    def should_run(self, step_id: int, metrics_snapshot: Any | None = None) -> bool:
        del metrics_snapshot
        _non_negative_int(step_id, "migration step_id")
        return True

    def plan(
        self,
        placement: "PlacementState",
        prediction: Any,
        model: "ModelSpec | None" = None,
        hardware: "HardwareModel | None" = None,
    ) -> MigrationPlan:
        if model is None or hardware is None:
            raise ValueError("predictive migration requires model and hardware")
        control_step = _control_step_from_prediction(prediction)
        budget, budget_source = _resolve_budget_pairs(self.config.budget_pairs_per_step, model)
        gpu_values, pim_values = _predicted_side_values(prediction, model, hardware)

        pair_candidates: list[MigrationAction] = []
        candidate_pairs_by_layer: dict[str, int] = {}
        both_before = _both_counts_by_layer(placement, model)
        for layer_id in range(model.num_layers):
            promotions, demotions = _layer_overlap_candidates(
                layer_id,
                placement,
                model,
                gpu_values.get(layer_id, {}),
                pim_values.get(layer_id, {}),
                both_budget=both_before[layer_id],
            )
            layer_pairs = _pair_layer_candidates(promotions, demotions)
            if layer_pairs:
                candidate_pairs_by_layer[str(layer_id)] = len(layer_pairs)
            pair_candidates.extend(layer_pairs)

        pair_candidates.sort(
            key=lambda action: (
                -action.score_s,
                action.layer,
                action.promote_expert,
                action.demote_expert,
                action.promote_from,
                action.demote_to,
            )
        )
        positive_candidates = [action for action in pair_candidates if action.score_s > 0.0]
        selected = tuple(positive_candidates[:budget])
        kind_counts = Counter(action.kind for action in selected)
        selected_by_layer = Counter(str(action.layer) for action in selected)
        metadata = {
            "mode": self.config.mode,
            "policy": self.policy_name,
            "placement_version": getattr(placement, "version", None),
            "budget_unit": "paired_overlap_exchange",
            "budget_pairs_per_step": budget,
            "budget_source": budget_source,
            "selected_pairs": len(selected),
            "changed_entries": 2 * len(selected),
            "positive_candidate_pairs": len(positive_candidates),
            "skipped_budget_pairs": max(0, len(positive_candidates) - len(selected)),
            "candidate_pairs_by_layer_nonzero": candidate_pairs_by_layer,
            "selected_pairs_by_layer_nonzero": dict(selected_by_layer),
            "kind_counts": dict(kind_counts),
            "latency_value_source": "prediction_frame",
            "latency_combine_mode": _LATENCY_COMBINE_MODE,
            "overlap_invariant": "preserve_current_per_layer_BOTH_count",
            "direct_primary_swaps": False,
        }
        return MigrationPlan(control_step=control_step, actions=selected, metadata=metadata)

    def apply(
        self,
        placement: "PlacementState",
        plan: MigrationPlan,
        model: "ModelSpec | None" = None,
        hardware: "HardwareModel | None" = None,
    ) -> "PlacementState":
        del hardware
        if plan.is_empty:
            return placement
        if model is None:
            raise ValueError("applying predictive migration requires model")
        both_before = _both_counts_by_layer(placement, model)
        updates: dict[tuple[int, int], "Residency"] = {}
        for action in plan.actions:
            if placement.get(action.layer, action.promote_expert) != action.promote_from:
                raise RuntimeError("migration plan is stale: promotion source no longer matches placement")
            if placement.get(action.layer, action.demote_expert) != "BOTH":
                raise RuntimeError("migration plan is stale: demotion source no longer matches placement")
            for key, residency in action.updates().items():
                if key in updates:
                    raise RuntimeError(f"migration plan updates placement entry {key} more than once")
                updates[key] = residency
        updated = placement.clone_with_updates(updates)
        both_after = _both_counts_by_layer(updated, model)
        if both_after != both_before:
            raise RuntimeError(
                "predictive migration violated per-layer BOTH-count invariant: "
                f"before={both_before}, after={both_after}"
            )
        # Keep a compact breadcrumb on the new placement version for debug dumps.
        updated.metadata["last_migration"] = {
            "control_step": plan.control_step,
            "mode": self.config.mode,
            "selected_pairs": len(plan.actions),
            "changed_entries": len(updates),
        }
        return updated


def migration_config_from_dict(data: MappingABC[str, Any] | MigrationConfig | None = None) -> MigrationConfig:
    """Create ``MigrationConfig`` from a root or ``migration:`` config mapping."""

    if data is None:
        return MigrationConfig()
    if isinstance(data, MigrationConfig):
        return data
    cfg = data.get("migration", data)
    if not isinstance(cfg, MappingABC):
        raise ValueError("migration config must be a mapping")

    known = {"mode", "type", "policy", "budget_pairs_per_step"}
    extra = sorted(str(key) for key in cfg if key not in known)
    if extra:
        raise ValueError(f"unsupported migration config keys: {extra}")
    return MigrationConfig(
        mode=str(cfg.get("mode", cfg.get("type", cfg.get("policy", "none")))),
        budget_pairs_per_step=cfg.get("budget_pairs_per_step", "auto"),
    )


def make_migration_policy(config: MappingABC[str, Any] | MigrationConfig | None = None) -> MigrationPolicy:
    """Instantiate a migration policy from config."""

    cfg = migration_config_from_dict(config)
    if cfg.mode == "none":
        return NoneMigrationPolicy(cfg)
    if cfg.mode == "predictive_budgeted_regret":
        return PredictiveBudgetedRegretMigrationPolicy(cfg)
    raise ValueError(f"unsupported migration mode: {cfg.mode}")


def empty_migration_plan(
    control_step: int,
    metadata: MappingABC[str, Any] | None = None,
) -> MigrationPlan:
    """Return an explicit empty plan for callers that skip disabled migration."""

    plan_metadata = {"mode": "none"}
    plan_metadata.update(dict(metadata or {}))
    return MigrationPlan(control_step=control_step, metadata=plan_metadata)


def migration_plan_to_dict(plan: MigrationPlan) -> dict[str, Any]:
    """Return a JSON-serializable migration-plan dictionary."""

    return plan.to_dict()


def default_budget_pairs_for_model(model: "ModelSpec") -> int:
    """Return the project default paired-swap budget for a model."""

    slug = _model_slug(model)
    if slug in _AUTO_BUDGETS_BY_SLUG:
        return _AUTO_BUDGETS_BY_SLUG[slug]
    # Conservative fallback for toy/new configs. Real models should be added to
    # the explicit table above rather than silently getting a large budget.
    return 10


def _layer_overlap_candidates(
    layer_id: int,
    placement: "PlacementState",
    model: "ModelSpec",
    gpu_value_by_expert: MappingABC[int, float],
    pim_value_by_expert: MappingABC[int, float],
    *,
    both_budget: int,
) -> tuple[list[_Candidate], list[_Candidate]]:
    expert_count = model.experts_for_layer(layer_id)
    target = _target_residency_for_layer_fixed_overlap(
        gpu_value_by_expert,
        pim_value_by_expert,
        expert_count,
        both_budget,
    )
    promotions: list[_Candidate] = []
    demotions: list[_Candidate] = []
    for expert_id in range(expert_count):
        current = placement.get(layer_id, expert_id)
        desired = target[expert_id]
        next_state = _next_overlap_only_state(current, desired)
        if next_state is None:
            continue
        gpu_value = float(gpu_value_by_expert.get(expert_id, 0.0))
        pim_value = float(pim_value_by_expert.get(expert_id, 0.0))
        gain_s = _state_value(next_state, gpu_value, pim_value) - _state_value(current, gpu_value, pim_value)
        candidate = _Candidate(
            gain_s=float(gain_s),
            layer=layer_id,
            expert=expert_id,
            before=current,
            after=next_state,
        )
        if current != "BOTH" and next_state == "BOTH":
            # A zero-gain promotion is cold tie churn, and cannot make a
            # positive pair unless the demotion is impossible-positive. Skip it.
            if gain_s > 0.0:
                promotions.append(candidate)
        elif current == "BOTH" and next_state != "BOTH":
            demotions.append(candidate)
        else:  # pragma: no cover - _next_overlap_only_state prevents this.
            raise RuntimeError(f"non-overlap migration candidate generated: {current}->{next_state}")
    promotions.sort(key=lambda item: (-item.gain_s, item.layer, item.expert, item.before, item.after))
    demotions.sort(key=lambda item: (-item.gain_s, item.layer, item.expert, item.before, item.after))
    return promotions, demotions


def _pair_layer_candidates(promotions: Sequence[_Candidate], demotions: Sequence[_Candidate]) -> list[MigrationAction]:
    actions: list[MigrationAction] = []
    for promote, demote in zip(promotions, demotions):
        score_s = promote.gain_s + demote.gain_s
        if score_s <= 0.0:
            continue
        actions.append(
            MigrationAction(
                layer=promote.layer,
                promote_expert=promote.expert,
                promote_from=promote.before,
                demote_expert=demote.expert,
                demote_to=demote.after,
                score_s=score_s,
                promote_gain_s=promote.gain_s,
                demote_gain_s=demote.gain_s,
            )
        )
    return actions


def _target_residency_for_layer_fixed_overlap(
    gpu_value_by_expert: MappingABC[int, float],
    pim_value_by_expert: MappingABC[int, float],
    expert_count: int,
    both_budget: int,
) -> dict[int, str]:
    both_budget = max(0, min(expert_count, int(both_budget)))
    target: dict[int, str] = {}
    second_copy_scores: list[tuple[float, int]] = []
    for expert_id in range(expert_count):
        gpu_value = float(gpu_value_by_expert.get(expert_id, 0.0))
        pim_value = float(pim_value_by_expert.get(expert_id, 0.0))
        if gpu_value >= pim_value:
            target[expert_id] = "GPU"
            second_copy_scores.append((pim_value, expert_id))
        else:
            target[expert_id] = "PIM"
            second_copy_scores.append((gpu_value, expert_id))
    for _score, expert_id in sorted(second_copy_scores, key=lambda item: (-item[0], item[1]))[:both_budget]:
        target[expert_id] = "BOTH"
    return target


def _next_overlap_only_state(current: str, desired: str) -> str | None:
    current = _normalize_residency_name(current, "current residency")
    desired = _normalize_residency_name(desired, "desired residency")
    if current == desired:
        return None
    if current in _SINGLE_RESIDENCIES and desired == "BOTH":
        return "BOTH"
    if current == "BOTH" and desired in _SINGLE_RESIDENCIES:
        return desired
    if current in _SINGLE_RESIDENCIES and desired in _SINGLE_RESIDENCIES:
        # Stage primary-side changes through BOTH; never do GPU<->PIM directly.
        return "BOTH"
    raise ValueError(f"unsupported migration transition {current}->{desired}")


def _predicted_side_values(
    prediction: Any,
    model: "ModelSpec",
    hardware: "HardwareModel",
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    histograms = _prediction_histograms(prediction)
    gpu_values: dict[int, dict[int, float]] = {}
    pim_values: dict[int, dict[int, float]] = {}
    for layer_id in range(model.num_layers):
        expert_count = model.experts_for_layer(layer_id)
        raw_hist = histograms.get(layer_id, {})
        scores = _clean_layer_scores(raw_hist, expert_count)
        active = [expert_id for expert_id, score in scores.items() if score > 0.0]
        gpu_layer = {expert_id: 0.0 for expert_id in range(expert_count)}
        pim_layer = {expert_id: 0.0 for expert_id in range(expert_count)}
        if active:
            total_score = sum(scores[expert_id] for expert_id in active)
            total_tokens = float(max(len(active), int(getattr(model, "top_k", 1) or 1)))
            predicted_items: list[tuple[int, float, int]] = []
            for expert_id in active:
                if total_score > 0.0:
                    predicted_tokens = total_tokens * scores[expert_id] / total_score
                else:  # pragma: no cover - active implies positive total_score.
                    predicted_tokens = total_tokens / float(len(active))
                predicted_quanta = _predicted_offload_units(predicted_tokens, hardware)
                predicted_items.append((expert_id, predicted_tokens, predicted_quanta))
            predicted_items.sort(key=lambda item: (item[1], item[0]))

            prefix_quanta = [0]
            for _expert_id, _predicted_tokens, predicted_quanta in predicted_items:
                prefix_quanta.append(prefix_quanta[-1] + predicted_quanta)
            gpu_unit_s = hardware.gpu_expert_time(1, model, layer_id=layer_id)
            offload_unit_s = hardware.offload_expert_time(1, model, layer_id=layer_id)
            best_cold_count = 0
            best_latency_s = math.inf
            for cold_count in range(len(predicted_items) + 1):
                gpu_active = len(predicted_items) - cold_count
                offload_quanta = prefix_quanta[cold_count]
                latency_s = combine_gpu_offload_latency(
                    gpu_active * gpu_unit_s,
                    offload_quanta * offload_unit_s,
                    latency_combine_mode=_LATENCY_COMBINE_MODE,
                )
                if latency_s < best_latency_s:
                    best_latency_s = latency_s
                    best_cold_count = cold_count

            gpu_active = len(predicted_items) - best_cold_count
            offload_quanta = prefix_quanta[best_cold_count]
            for index, (expert_id, _predicted_tokens, predicted_quanta) in enumerate(predicted_items):
                if index < best_cold_count:
                    forced_gpu_latency_s = combine_gpu_offload_latency(
                        (gpu_active + 1) * gpu_unit_s,
                        max(0, offload_quanta - predicted_quanta) * offload_unit_s,
                        latency_combine_mode=_LATENCY_COMBINE_MODE,
                    )
                    pim_layer[expert_id] = max(0.0, forced_gpu_latency_s - best_latency_s)
                else:
                    forced_pim_latency_s = combine_gpu_offload_latency(
                        max(0, gpu_active - 1) * gpu_unit_s,
                        (offload_quanta + predicted_quanta) * offload_unit_s,
                        latency_combine_mode=_LATENCY_COMBINE_MODE,
                    )
                    gpu_layer[expert_id] = max(0.0, forced_pim_latency_s - best_latency_s)
        gpu_values[layer_id] = gpu_layer
        pim_values[layer_id] = pim_layer
    return gpu_values, pim_values


def _predicted_offload_units(predicted_tokens: float, hardware: "HardwareModel") -> int:
    if hardware.is_remote_memory_backend():
        return 1
    return max(1, int(math.ceil(max(0.0, float(predicted_tokens)) / float(hardware.pim_granularity))))


def _prediction_histograms(prediction: Any) -> dict[int, dict[int, float]]:
    if hasattr(prediction, "per_layer_histograms"):
        raw = getattr(prediction, "per_layer_histograms")
    elif hasattr(prediction, "per_layer_expert_histograms"):
        raw = getattr(prediction, "per_layer_expert_histograms")
    elif isinstance(prediction, MappingABC):
        raw = prediction.get("per_layer_histograms", prediction.get("per_layer_expert_histograms", {}))
    else:
        raise ValueError("migration planning requires prediction per_layer_histograms")
    if not isinstance(raw, MappingABC):
        raise ValueError("prediction per_layer_histograms must be a mapping")
    out: dict[int, dict[int, float]] = {}
    for raw_layer, raw_counts in raw.items():
        layer_id = _non_negative_int(raw_layer, "prediction layer")
        if not isinstance(raw_counts, MappingABC):
            raise ValueError(f"prediction layer {layer_id} histogram must be a mapping")
        out[layer_id] = _clean_layer_scores(raw_counts, expert_count=None)
    return out


def _clean_layer_scores(raw_counts: MappingABC[Any, Any], expert_count: int | None) -> dict[int, float]:
    out: dict[int, float] = {}
    for raw_expert, raw_score in raw_counts.items():
        expert_id = _non_negative_int(raw_expert, "prediction expert")
        if expert_count is not None and expert_id >= expert_count:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction score for expert {expert_id} must be numeric") from exc
        if math.isfinite(score) and score > 0.0:
            out[expert_id] = score
    return out


def _both_counts_by_layer(placement: "PlacementState", model: "ModelSpec") -> dict[int, int]:
    counts: dict[int, int] = {}
    for layer_id in range(model.num_layers):
        counts[layer_id] = sum(
            1 for expert_id in range(model.experts_for_layer(layer_id)) if placement.get(layer_id, expert_id) == "BOTH"
        )
    return counts


def _state_value(residency: str, gpu_value_s: float, pim_value_s: float) -> float:
    residency = _normalize_residency_name(residency, "residency")
    if residency == "GPU":
        return float(gpu_value_s)
    if residency == "PIM":
        return float(pim_value_s)
    return float(gpu_value_s) + float(pim_value_s)


def _resolve_budget_pairs(config_value: BudgetPairsPerStep | str, model: "ModelSpec") -> tuple[int, str]:
    if str(config_value).lower() == "auto":
        return default_budget_pairs_for_model(model), "auto_model_default"
    return _positive_int(config_value, "migration.budget_pairs_per_step"), "config"


def _normalize_budget_pairs(value: Any) -> BudgetPairsPerStep:
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return _positive_int(value, "migration.budget_pairs_per_step")


def _model_slug(model: "ModelSpec") -> str:
    candidates = [
        model.metadata.get("short_name") if isinstance(getattr(model, "metadata", None), MappingABC) else None,
        getattr(model, "model_id", ""),
    ]
    text = " ".join(str(item).lower() for item in candidates if item)
    if "qwen3" in text or "qwen/qwen3" in text:
        return "qwen3_235b_a22b_fp8"
    if "deepseek" in text:
        return "deepseek_r1_awq"
    if "kimi" in text:
        return "kimi_k2_thinking"
    if "llama" in text and "maverick" in text:
        return "llama_4_maverick_17b_128e_instruct"
    return str(candidates[0] or candidates[-1] or "unknown").lower()


def _normalize_mode(value: Any) -> str:
    mode = str(value).lower()
    aliases = {
        "pbrm": "predictive_budgeted_regret",
        "predictive": "predictive_budgeted_regret",
        "predictive_overlap_exchange": "predictive_budgeted_regret",
    }
    mode = aliases.get(mode, mode)
    if mode not in MIGRATION_MODES:
        raise ValueError(f"unsupported migration mode {value!r}; choose from {sorted(MIGRATION_MODES)}")
    return mode


def _normalize_residency_name(value: Any, field_name: str) -> str:
    text = str(value).upper()
    if text not in {"GPU", "PIM", "BOTH"}:
        raise ValueError(f"{field_name} must be one of ['BOTH', 'GPU', 'PIM'], got {value!r}")
    return text


def _control_step_from_prediction(prediction: Any) -> int:
    if hasattr(prediction, "step_id"):
        return _non_negative_int(getattr(prediction, "step_id"), "prediction step_id")
    if isinstance(prediction, MappingABC) and "step_id" in prediction:
        return _non_negative_int(prediction["step_id"], "prediction step_id")
    raise ValueError("migration planning requires a prediction object or mapping with step_id")


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
    if isinstance(value, str) and value.strip() != str(out):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if out < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return out


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _cmd_smoke(args: argparse.Namespace) -> int:
    config = MigrationConfig(mode=args.mode, budget_pairs_per_step=args.budget_pairs_per_step)
    policy = make_migration_policy(config)
    payload: dict[str, Any] = {
        "config": config.to_dict(),
        "should_run": policy.should_run(args.step_id, {}),
    }
    if config.mode == "none":
        plan = policy.plan(None, {"step_id": args.step_id}, None, None)  # type: ignore[arg-type]
        payload["plan"] = plan.to_dict()
        payload["apply_returned_same_object"] = policy.apply(None, plan) is None  # type: ignore[arg-type]
    else:
        payload["note"] = "predictive smoke requires placement, prediction, model, and hardware from a simulation run"
    print(_json_dumps(payload))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migration policy utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="emit a migration-policy smoke payload")
    smoke.add_argument("--mode", choices=sorted(MIGRATION_MODES), default="none")
    smoke.add_argument("--budget-pairs-per-step", default="auto")
    smoke.add_argument("--step-id", type=int, default=0)
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
