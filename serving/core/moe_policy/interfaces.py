"""Small shared interfaces and compatibility re-exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-only typing helper
    from .model_hardware import HardwareModel, ModelSpec

LayerExpertHistogram = dict[int, float]
PerLayerExpertHistograms = dict[int, LayerExpertHistogram]

__all__ = [
    "ReplayStep",
    "PredictionFrame",
    "PlacementState",
    "Residency",
    "MigrationPlan",
    "MigrationPolicy",
    "WorkAssignment",
    "ScheduleDecision",
    "Scheduler",
    "LayerExpertHistogram",
    "PerLayerExpertHistograms",
    "ModelSpec",
    "HardwareModel",
]


@dataclass(frozen=True)
class ReplayStep:
    """One decode forward produced by continuous-batch replay.

    Attributes match the system-level contract. ``request_positions`` are
    zero-based decode positions: raw JSON output token 1 is replay position 0.
    """

    step_id: int
    active_request_ids: list[str]
    layer_expert_counts: dict[int, dict[int, int]]
    request_positions: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionFrame:
    """Near-future per-layer expert-demand prediction.

    ``per_layer_histograms[layer_id]`` is that layer's independent expert
    histogram: ``{expert_id: predicted_score}``. MVP predictors normalize each
    layer to expert proportions so forecasts are batch-size invariant. The
    same numeric expert ID in two different layers refers to two different
    experts and is never mixed by downstream ranking/classification code.
    Missing experts are interpreted as zero demand by downstream code.
    """

    step_id: int
    horizon: int
    per_layer_histograms: PerLayerExpertHistograms
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def per_layer_expert_histograms(self) -> PerLayerExpertHistograms:
        """Clearer alias for the spec field ``per_layer_histograms``."""

        return self.per_layer_histograms


def __getattr__(name: str) -> Any:
    """Lazily preserve ``pim_hot_cold_moe.interfaces`` import compatibility."""

    if name in {"ModelSpec", "HardwareModel"}:
        from .model_hardware import HardwareModel, ModelSpec

        return {"ModelSpec": ModelSpec, "HardwareModel": HardwareModel}[name]
    if name in {"PlacementState", "Residency"}:
        from .placement import PlacementState, Residency

        return {"PlacementState": PlacementState, "Residency": Residency}[name]
    if name in {"WorkAssignment", "ScheduleDecision", "Scheduler"}:
        from .scheduler import ScheduleDecision, Scheduler, WorkAssignment

        return {"WorkAssignment": WorkAssignment, "ScheduleDecision": ScheduleDecision, "Scheduler": Scheduler}[name]
    if name in {"MigrationPlan", "MigrationPolicy"}:
        from .migration import MigrationPlan, MigrationPolicy

        return {
            "MigrationPlan": MigrationPlan,
            "MigrationPolicy": MigrationPolicy,
        }[name]
    raise AttributeError(name)
