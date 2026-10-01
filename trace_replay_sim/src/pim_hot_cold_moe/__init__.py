"""PIM hot/cold MoE simulation package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .interfaces import LayerExpertHistogram, PerLayerExpertHistograms, PredictionFrame, ReplayStep

if TYPE_CHECKING:  # pragma: no cover - import-only typing helper
    from .model_hardware import HardwareModel, ModelSpec

__all__ = [
    "HardwareModel",
    "LayerExpertHistogram",
    "ModelSpec",
    "PerLayerExpertHistograms",
    "PlacementConfig",
    "PlacementPolicy",
    "PlacementState",
    "MigrationConfig",
    "MigrationPlan",
    "MigrationPolicy",
    "NoneMigrationPolicy",
    "PredictiveBudgetedRegretMigrationPolicy",
    "ExperimentConfig",
    "MetricsCollector",
    "MetricsConfig",
    "metrics_config_from_dict",
    "SimulationEngine",
    "SimulationResult",
    "resolve_experiment_configs",
    "run_simulations_from_config",
    "build_trace_from_config",
    "run_sweep_from_config",
    "summarize_runs",
    "Scheduler",
    "SchedulerConfig",
    "ScheduleDecision",
    "WorkAssignment",
    "PredictionFrame",
    "Predictor",
    "PredictorConfig",
    "ReplayStep",
    "Residency",
    "TracePack",
    "ReplayStream",
    "build_trace_pack",
    "build_trace_packs",
    "discover_raw_traces",
    "make_predictor",
    "make_placement_policy",
    "make_migration_policy",
    "make_scheduler",
    "placement_config_from_dict",
    "migration_config_from_dict",
    "scheduler_config_from_dict",
    "migration_plan_to_dict",
    "schedule_decision_to_dict",
    "empty_migration_plan",
    "load_hardware_model",
    "load_model_spec",
    "predictor_config_from_dict",
    "validate_trace_metadata",
]


def __getattr__(name: str) -> Any:
    if name in {"MetricsCollector", "MetricsConfig", "metrics_config_from_dict"}:
        from . import metrics

        return getattr(metrics, name)
    if name in {
        "ExperimentConfig",
        "SimulationEngine",
        "SimulationResult",
        "resolve_experiment_configs",
        "run_simulations_from_config",
    }:
        from . import simulation

        return getattr(simulation, name)
    if name in {"build_trace_from_config", "run_sweep_from_config", "summarize_runs"}:
        from . import experiments

        return getattr(experiments, name)
    if name in {"TracePack", "ReplayStream", "build_trace_pack", "build_trace_packs", "discover_raw_traces"}:
        from . import trace_pack

        return getattr(trace_pack, name)
    if name in {
        "HardwareModel",
        "ModelSpec",
        "load_hardware_model",
        "load_model_spec",
        "validate_trace_metadata",
    }:
        from . import model_hardware

        return getattr(model_hardware, name)
    if name in {"Predictor", "PredictorConfig", "make_predictor", "predictor_config_from_dict"}:
        from . import prediction

        return getattr(prediction, name)
    if name in {
        "PlacementConfig",
        "PlacementPolicy",
        "PlacementState",
        "Residency",
        "make_placement_policy",
        "placement_config_from_dict",
    }:
        from . import placement

        return getattr(placement, name)
    if name in {
        "MigrationConfig",
        "MigrationPlan",
        "MigrationPolicy",
        "NoneMigrationPolicy",
        "PredictiveBudgetedRegretMigrationPolicy",
        "empty_migration_plan",
        "make_migration_policy",
        "migration_config_from_dict",
        "migration_plan_to_dict",
    }:
        from . import migration

        return getattr(migration, name)
    if name in {
        "Scheduler",
        "SchedulerConfig",
        "ScheduleDecision",
        "WorkAssignment",
        "make_scheduler",
        "scheduler_config_from_dict",
        "schedule_decision_to_dict",
    }:
        from . import scheduler

        return getattr(scheduler, name)
    raise AttributeError(name)
