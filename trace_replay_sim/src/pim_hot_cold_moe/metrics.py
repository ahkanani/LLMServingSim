"""Slim metrics and reporting for MVP-0 simulations.

Default output is summary-level only: ``metrics.json`` and ``summary.md``.
Debug mode can additionally write the compact per-step timeline that the
current simulator populates.  The module deliberately avoids placeholder
metrics for planned-but-dropped work.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from .interfaces import PredictionFrame, ReplayStep
from .scheduler import scheduler_config_from_dict

if TYPE_CHECKING:  # pragma: no cover
    from .model_hardware import HardwareModel, ModelSpec
    from .placement import PlacementState
    from .scheduler import ScheduleDecision

MetricsLevel = Literal["summary", "debug"]
METRICS_LEVELS = {"summary", "debug"}
_BYTES_PER_GB = 1_000_000_000.0
_TIMELINE_COLUMNS = [
    "step_id",
    "active_request_count",
    "latency_ms",
    "gpu_time_ms",
    "pim_time_ms",
    "histogram_prediction_error",
    "hot_miss_load_rate",
]


@dataclass(frozen=True)
class MetricsConfig:
    """Validated reporting config.

    ``summary`` is the fast default. ``debug`` keeps a compact per-step
    timeline, intentionally without request IDs, JSON blobs, oracle details,
    or duplicated policy config.
    """

    level: MetricsLevel = "summary"

    def __post_init__(self) -> None:
        level = str(self.level).lower()
        if level not in METRICS_LEVELS:
            raise ValueError(f"unsupported metrics level {self.level!r}; choose from {sorted(METRICS_LEVELS)}")
        object.__setattr__(self, "level", level)

    @property
    def write_timeline(self) -> bool:
        return self.level == "debug"

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level}


class MetricsCollector:
    """Collect small run metrics without changing simulation behavior."""

    def __init__(
        self,
        *,
        run_id: str,
        benchmark: str | None,
        trace_pack_path: str | Path,
        trace_manifest: Mapping[str, Any],
        model: "ModelSpec",
        hardware: "HardwareModel",
        config: Any,
    ):
        self.run_id = run_id
        self.benchmark = benchmark
        self.trace_pack_path = str(Path(trace_pack_path).expanduser().resolve())
        self.trace_manifest_summary = _trace_manifest_summary(trace_manifest)
        self.model = model
        self.hardware = hardware
        self.config = config
        self.metrics_config = metrics_config_from_dict(getattr(config, "metrics_config", {}))
        self.warmup: dict[str, Any] = {}
        self._latencies_ms: list[float] = []
        self._gpu_times_ms: list[float] = []
        self._pim_times_ms: list[float] = []
        self._prediction_errors: list[float] = []
        self._hot_miss_rates: list[float] = []
        self._oracle_gaps_ms: list[float] = []
        self._oracle_gap_ratios: list[float] = []
        self._migration_selected_pairs: list[float] = []
        self._migration_changed_entries: list[float] = []
        self._migration_positive_candidate_pairs: list[float] = []
        self._migration_skipped_budget_pairs: list[float] = []
        self._migration_kind_counts: dict[str, int] = {}
        self._migration_budget_pairs_per_step: int | None = None
        self._migration_mode: str | None = None
        self._timeline_rows: list[dict[str, Any]] = []
        self._placement_report: dict[str, Any] = {}
        self._memory_report: dict[str, Any] = {}
        self._cost_report: dict[str, Any] = {}
        self._offload_report: dict[str, Any] = {}
        self._last_step_id: int | None = None

    def record_warmup(self, warmup_stats: Iterable[ReplayStep], placement: "PlacementState") -> None:
        steps = list(warmup_stats)
        self._record_placement_reports(placement)
        self.warmup = {"step_count": len(steps)}

    def record_step(
        self,
        replay_step: ReplayStep,
        prediction: PredictionFrame,
        placement: "PlacementState",
        decision: "ScheduleDecision",
    ) -> None:
        latency_ms = _seconds_to_ms(float(decision.total_latency_s))
        gpu_time_ms = _seconds_to_ms(float(sum(decision.gpu_time_s)))
        pim_time_ms = _seconds_to_ms(float(sum(decision.pim_time_s)))
        pred_error = _histogram_prediction_error(replay_step, prediction, self.model)
        hot_miss_rate = _hot_miss_load_rate(replay_step, placement, self.hardware)
        oracle_gap_s = decision.metadata.get("oracle_gap_s")
        oracle_gap_ratio = decision.metadata.get("oracle_gap_ratio")

        self._last_step_id = int(replay_step.step_id)
        self._latencies_ms.append(latency_ms)
        self._gpu_times_ms.append(gpu_time_ms)
        self._pim_times_ms.append(pim_time_ms)
        self._prediction_errors.append(pred_error)
        self._hot_miss_rates.append(hot_miss_rate)
        if oracle_gap_s is not None:
            self._oracle_gaps_ms.append(_seconds_to_ms(float(oracle_gap_s)))
        if oracle_gap_ratio is not None:
            self._oracle_gap_ratios.append(float(oracle_gap_ratio))
        self._record_placement_reports(placement)

        if self.metrics_config.write_timeline:
            self._timeline_rows.append(
                {
                    "step_id": replay_step.step_id,
                    "active_request_count": len(replay_step.active_request_ids),
                    "latency_ms": latency_ms,
                    "gpu_time_ms": gpu_time_ms,
                    "pim_time_ms": pim_time_ms,
                    "histogram_prediction_error": pred_error,
                    "hot_miss_load_rate": hot_miss_rate,
                }
            )

    def record_migration(self, plan: Any) -> None:
        metadata = dict(getattr(plan, "metadata", {}) or {})
        selected = float(metadata.get("selected_pairs", len(getattr(plan, "actions", ()) or ())))
        changed = float(metadata.get("changed_entries", 2 * int(selected)))
        positive = float(metadata.get("positive_candidate_pairs", selected))
        skipped = float(metadata.get("skipped_budget_pairs", max(0.0, positive - selected)))
        self._migration_selected_pairs.append(selected)
        self._migration_changed_entries.append(changed)
        self._migration_positive_candidate_pairs.append(positive)
        self._migration_skipped_budget_pairs.append(skipped)
        if metadata.get("budget_pairs_per_step") is not None:
            self._migration_budget_pairs_per_step = int(metadata["budget_pairs_per_step"])
        if metadata.get("mode") is not None:
            self._migration_mode = str(metadata["mode"])
        kind_counts = metadata.get("kind_counts", {})
        if isinstance(kind_counts, Mapping):
            for kind, count in kind_counts.items():
                self._migration_kind_counts[str(kind)] = self._migration_kind_counts.get(str(kind), 0) + int(count)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "eval_step_count": len(self._latencies_ms),
            "last_step_id": self._last_step_id,
            "last_latency_ms": self._latencies_ms[-1] if self._latencies_ms else None,
            "mean_latency_ms": _mean(self._latencies_ms),
        }

    def finalize(self, output_dir: str | Path, *, extra_artifacts: Mapping[str, str] | None = None) -> dict[str, Any]:
        run_dir = Path(output_dir).expanduser().resolve()
        metrics_path = run_dir / "metrics.json"
        summary_path = run_dir / "summary.md"
        artifact_paths: dict[str, str] = dict(extra_artifacts or {})
        if self.metrics_config.write_timeline:
            timeline_path = run_dir / "timeline.parquet"
            _write_parquet(self._timeline_rows, timeline_path)
            artifact_paths["timeline_path"] = str(timeline_path)

        metrics = self._aggregate(metrics_path=metrics_path, summary_path=summary_path, artifact_paths=artifact_paths)
        metrics_path.write_text(_json_dumps(metrics), encoding="utf-8")
        summary_path.write_text(_summary_markdown(metrics), encoding="utf-8")
        return {
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "metrics": metrics,
            **artifact_paths,
        }

    def _aggregate(self, *, metrics_path: Path, summary_path: Path, artifact_paths: Mapping[str, str]) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "paths": {
                "trace_pack_path": self.trace_pack_path,
                "config_path": str(self.config.config_path),
                "metrics_path": str(metrics_path),
                "summary_path": str(summary_path),
                **artifact_paths,
            },
            "model_id": self.model.model_id,
            "hardware_id": self.hardware.hardware_id,
            "trace": self.trace_manifest_summary,
            "metrics_config": self.metrics_config.to_dict(),
            "scheduler": scheduler_config_from_dict(self.config.scheduler_config).to_dict(),
            "warmup": self.warmup,
            "evaluation": {
                "step_count": len(self._latencies_ms),
                "last_step_id": self._last_step_id,
                "max_batch_size": self.config.max_batch_size,
            },
            "latency_ms": _distribution(self._latencies_ms),
            "gpu_time_ms": _distribution(self._gpu_times_ms),
            "pim_time_ms": _distribution(self._pim_times_ms),
            "offload_time_ms": _distribution(self._pim_times_ms),
            "histogram_prediction_error": _distribution(self._prediction_errors),
            "hot_miss_load_rate": _distribution(self._hot_miss_rates),
            "oracle_gap_ms": _distribution(self._oracle_gaps_ms),
            "oracle_gap_ratio": _distribution(self._oracle_gap_ratios),
            "migration": self._migration_report(),
            "placement": self._placement_report,
            "offload": self._offload_report,
            "memory": self._memory_report,
            "cost": self._cost_report,
            "warnings": self._warnings(),
        }

    def _migration_report(self) -> dict[str, Any]:
        return {
            "mode": self._migration_mode or "none",
            "budget_unit": "paired_overlap_exchange" if self._migration_mode else None,
            "budget_pairs_per_step": self._migration_budget_pairs_per_step,
            "selected_pairs": _distribution(self._migration_selected_pairs),
            "changed_entries": _distribution(self._migration_changed_entries),
            "positive_candidate_pairs": _distribution(self._migration_positive_candidate_pairs),
            "skipped_budget_pairs": _distribution(self._migration_skipped_budget_pairs),
            "kind_counts": dict(sorted(self._migration_kind_counts.items())),
        }

    def _record_placement_reports(self, placement: "PlacementState") -> None:
        self._placement_report = _slim_placement_report(placement.report_residency())
        footprint = _expert_memory_footprint(placement, self.model, self.hardware)
        self._memory_report = footprint["memory"]
        self._cost_report = footprint["cost"]
        self._offload_report = footprint["offload"]

    def _warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.config.eval_steps is not None and len(self._latencies_ms) < int(self.config.eval_steps):
            warnings.append("evaluation ended before replay.eval_steps; trace stream drained early")
        return warnings


def metrics_config_from_dict(data: Mapping[str, Any] | MetricsConfig | None = None) -> MetricsConfig:
    """Create ``MetricsConfig`` from a root or ``metrics:`` config mapping."""

    if data is None:
        return MetricsConfig()
    if isinstance(data, MetricsConfig):
        return data
    cfg = data.get("metrics", data)
    if not isinstance(cfg, Mapping):
        raise ValueError("metrics config must be a mapping")
    allowed = {"level"}
    unknown = sorted(str(key) for key in cfg.keys() if str(key) not in allowed)
    if unknown:
        raise ValueError(f"unsupported metrics config keys: {unknown}; supported keys: {sorted(allowed)}")
    return MetricsConfig(level=cast(MetricsLevel, str(cfg.get("level", "summary"))))


def _trace_manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    requests = manifest.get("requests", [])
    model_meta = manifest.get("model_metadata", {}) if isinstance(manifest.get("model_metadata", {}), Mapping) else {}
    benchmarks = sorted({str(item.get("benchmark")) for item in requests if isinstance(item, Mapping) and item.get("benchmark")})
    return {
        "schema_version": manifest.get("schema_version"),
        "trace_pack_path": manifest.get("trace_pack_path"),
        "request_count": len(requests) if isinstance(requests, Sequence) else None,
        "benchmarks": benchmarks,
        "model_id": model_meta.get("model_id"),
        "num_layers": model_meta.get("num_layers"),
        "top_k": model_meta.get("top_k"),
    }


def _slim_placement_report(residency: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "residency_counts": residency.get("residency_counts", {}),
    }


def _expert_memory_footprint(
    placement: "PlacementState",
    model: "ModelSpec",
    hardware: "HardwareModel",
) -> dict[str, dict[str, Any]]:
    gpu_only_bytes = 0
    offload_only_bytes = 0
    both_bytes = 0
    for (layer_id, _), residency in placement.residency.items():
        weight_bytes = model.expert_weight_bytes_for_layer(layer_id)
        if residency == "GPU":
            gpu_only_bytes += weight_bytes
        elif residency == "PIM":
            offload_only_bytes += weight_bytes
        elif residency == "BOTH":
            both_bytes += weight_bytes
        else:  # pragma: no cover - PlacementState validates this.
            raise ValueError(f"unsupported residency for memory accounting: {residency}")

    baseline_hbm_bytes = model.total_expert_weight_bytes()
    hbm_expert_bytes = gpu_only_bytes + both_bytes
    offload_expert_bytes = offload_only_bytes + both_bytes
    hbm_saved_bytes = max(0, baseline_hbm_bytes - hbm_expert_bytes)
    hbm_expert_gb_per_gpu = _bytes_to_gb(hbm_expert_bytes) / hardware.gpu_count
    hbm_capacity_gb_each = hardware.hbm_capacity_gb_each
    fits_hbm_capacity = (
        None
        if hbm_capacity_gb_each is None
        else hbm_expert_gb_per_gpu <= float(hbm_capacity_gb_each)
    )
    is_remote = hardware.is_remote_memory_backend()

    memory = {
        "baseline_all_gpu_hbm_bytes": baseline_hbm_bytes,
        "baseline_all_gpu_hbm_gb": _bytes_to_gb(baseline_hbm_bytes),
        "gpu_only_bytes": gpu_only_bytes,
        "gpu_only_gb": _bytes_to_gb(gpu_only_bytes),
        "offload_only_bytes": offload_only_bytes,
        "offload_only_gb": _bytes_to_gb(offload_only_bytes),
        "both_bytes": both_bytes,
        "both_gb": _bytes_to_gb(both_bytes),
        "hbm_expert_bytes": hbm_expert_bytes,
        "hbm_expert_gb": _bytes_to_gb(hbm_expert_bytes),
        "offload_expert_bytes": offload_expert_bytes,
        "offload_expert_gb": _bytes_to_gb(offload_expert_bytes),
        "remote_expert_bytes": offload_expert_bytes if is_remote else None,
        "remote_expert_gb": _bytes_to_gb(offload_expert_bytes) if is_remote else None,
        "hbm_saved_bytes": hbm_saved_bytes,
        "hbm_saved_gb": _bytes_to_gb(hbm_saved_bytes),
        "hbm_saved_fraction": _safe_ratio_float(hbm_saved_bytes, baseline_hbm_bytes),
        "hbm_expert_gb_per_gpu": hbm_expert_gb_per_gpu,
        "hbm_capacity_gb_each": hbm_capacity_gb_each,
        "fits_hbm_capacity": fits_hbm_capacity,
        "both_counted_in_hbm_and_offload": True,
    }

    cost: dict[str, Any] = {
        "enabled": hardware.hbm_cost_per_gb is not None and hardware.remote_cost_per_gb is not None,
        "hbm_cost_per_gb": hardware.hbm_cost_per_gb,
        "remote_cost_per_gb": hardware.remote_cost_per_gb,
    }
    if hardware.hbm_cost_per_gb is not None and hardware.remote_cost_per_gb is not None:
        hbm_cost_per_gb = float(hardware.hbm_cost_per_gb)
        remote_cost_per_gb = float(hardware.remote_cost_per_gb)
        baseline_cost = memory["baseline_all_gpu_hbm_gb"] * hbm_cost_per_gb
        design_cost = (
            memory["hbm_expert_gb"] * hbm_cost_per_gb
            + memory["offload_expert_gb"] * remote_cost_per_gb
        )
        cost.update(
            {
                "baseline_hbm_cost": baseline_cost,
                "remote_memory_design_cost": design_cost,
                "cost_saved": baseline_cost - design_cost,
                "cost_saved_fraction": 1.0 - design_cost / baseline_cost if baseline_cost else None,
            }
        )

    offload = {
        "backend": hardware.offload_backend,
        "label": hardware.offload_label(),
        "is_remote_memory": is_remote,
        "remote_count": hardware.remote_count,
        "remote_bw_tbps_each": hardware.remote_bw_tbps_each,
        "total_remote_bw_tbps": hardware.total_remote_bw_tbps() if is_remote else None,
        "remote_capacity_gb_each": hardware.remote_capacity_gb_each,
    }
    return {"memory": memory, "cost": cost, "offload": offload}


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / _BYTES_PER_GB


def _safe_ratio_float(numerator: int | float, denominator: int | float) -> float | None:
    return None if float(denominator) == 0.0 else float(numerator) / float(denominator)


def _histogram_prediction_error(replay_step: ReplayStep, prediction: PredictionFrame, model: "ModelSpec") -> float:
    """Return mean per-layer total variation distance.

    Actual and predicted histograms are normalized to per-layer expert
    distributions. For each layer, the metric is
    ``0.5 * sum_expert(abs(pred_prop - true_prop))``. This is the fraction of
    probability mass assigned to the wrong expert bins, bounded in ``[0, 1]``.
    The reported scalar is the mean across non-empty active/predicted layers.
    """
    actual = replay_step.layer_expert_counts
    predicted = prediction.per_layer_histograms
    errors: list[float] = []
    for layer_id in range(model.num_layers):
        actual_props = _normalize_layer(actual.get(layer_id, {}))
        predicted_props = _normalize_layer(predicted.get(layer_id, {}))
        if not actual_props and not predicted_props:
            continue
        expert_count = model.experts_for_layer(layer_id)
        l1_error = 0.0
        for expert_id in range(expert_count):
            l1_error += abs(predicted_props.get(expert_id, 0.0) - actual_props.get(expert_id, 0.0))
        errors.append(0.5 * l1_error)
    return _mean(errors) or 0.0


def _normalize_layer(layer_counts: Mapping[int, float] | Mapping[int, int]) -> dict[int, float]:
    total = sum(max(0.0, float(value)) for value in layer_counts.values())
    if total <= 0.0:
        return {}
    return {
        int(expert_id): max(0.0, float(value)) / total
        for expert_id, value in layer_counts.items()
        if float(value) > 0.0
    }


def _hot_miss_load_rate(
    replay_step: ReplayStep,
    placement: "PlacementState",
    hardware: "HardwareModel",
) -> float:
    hot_threshold = int(hardware.pim_granularity)
    total_tokens = 0
    missed_tokens = 0
    for layer_id, counts in replay_step.layer_expert_counts.items():
        for expert_id, token_count in counts.items():
            token_count = int(token_count)
            if token_count <= 0:
                continue
            total_tokens += token_count
            if token_count > hot_threshold and placement.get(int(layer_id), int(expert_id)) == "PIM":
                missed_tokens += token_count
    return 0.0 if total_tokens == 0 else missed_tokens / total_tokens


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("debug timeline output requires pyarrow") from exc
    table = pa.Table.from_pylist([dict(row) for row in rows]) if rows else pa.table({name: [] for name in _TIMELINE_COLUMNS})
    pq.write_table(table, path)


def _seconds_to_ms(value: float) -> float:
    return float(value) * 1000.0


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "total": sum(data) if data else 0.0,
        "mean": _mean(data),
        "p50": _percentile(data, 50),
        "p90": _percentile(data, 90),
        "p99": _percentile(data, 99),
        "max": max(data) if data else None,
    }


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    data = sorted(float(value) for value in values)
    index = max(0, min(len(data) - 1, math.ceil((percentile / 100.0) * len(data)) - 1))
    return data[index]


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _summary_markdown(metrics: Mapping[str, Any]) -> str:
    latency = metrics.get("latency_ms", {})
    prediction = metrics.get("histogram_prediction_error", {})
    hot_miss = metrics.get("hot_miss_load_rate", {})
    oracle_gap = metrics.get("oracle_gap_ratio", {})
    placement = metrics.get("placement", {})
    memory = metrics.get("memory", {})
    cost = metrics.get("cost", {})
    offload = metrics.get("offload", {})
    scheduler = metrics.get("scheduler", {})
    migration = metrics.get("migration", {})
    lines = [
        f"# Simulation Summary: {metrics.get('run_id')}",
        "",
        f"- Benchmark: `{metrics.get('benchmark')}`",
        f"- Eval steps: {metrics.get('evaluation', {}).get('step_count', 0)} after "
        f"{metrics.get('warmup', {}).get('step_count', 0)} warmup steps",
        f"- Model / hardware: `{metrics.get('model_id')}` / `{metrics.get('hardware_id')}`",
        f"- Scheduler latency combine mode: `{scheduler.get('latency_combine_mode', 'max')}`",
        f"- Latency mean / p50 / p90 / p99: {_ms(latency.get('mean'))} / {_ms(latency.get('p50'))} / "
        f"{_ms(latency.get('p90'))} / {_ms(latency.get('p99'))}",
        f"- Histogram prediction error mean: {_number(prediction.get('mean'))}",
        f"- Hot-miss load-rate mean: {_number(hot_miss.get('mean'))}",
        f"- Oracle gap ratio mean: {_number(oracle_gap.get('mean'))}"
        if oracle_gap.get("count", 0)
        else "- Oracle gap ratio mean: n/a",
        f"- Migration mode / mean selected pairs: `{migration.get('mode', 'none')}` / "
        f"{_number(migration.get('selected_pairs', {}).get('mean'))}",
        f"- Residency counts: {placement.get('residency_counts', {})}",
        f"- Offload backend: `{offload.get('backend', 'n/a')}` ({offload.get('label', 'n/a')})",
        f"- HBM expert memory: {_gb(memory.get('hbm_expert_gb'))} total, "
        f"{_gb(memory.get('hbm_expert_gb_per_gpu'))} per GPU",
        f"- HBM saved vs all-GPU expert baseline: {_gb(memory.get('hbm_saved_gb'))} "
        f"({_percent(memory.get('hbm_saved_fraction'))})",
        f"- Remote/offload expert memory: {_gb(memory.get('offload_expert_gb'))}",
    ]
    if memory.get("hbm_capacity_gb_each") is not None:
        lines.append(
            f"- HBM capacity check: {_gb(memory.get('hbm_capacity_gb_each'))} per GPU, "
            f"fits={memory.get('fits_hbm_capacity')}"
        )
    if cost.get("enabled"):
        lines.append(f"- Relative cost saved vs all-HBM: {_percent(cost.get('cost_saved_fraction'))}")
    warnings = metrics.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def _ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f} ms"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _gb(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f} GB"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value
