"""Decode-only MVP simulation engine.

This module owns the vertical-slice control loop that stitches together the
TracePack replay, model/hardware, predictor, placement, migration, scheduler,
and metrics/reporting modules.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .interfaces import ReplayStep
from .metrics import MetricsCollector
from .migration import make_migration_policy
from .model_hardware import (
    HardwareModel,
    ModelSpec,
    hardware_model_from_dict,
    load_hardware_model,
    load_mapping_config,
    load_model_spec,
    model_spec_from_dict,
    validate_trace_metadata,
)
from .placement import make_placement_policy, placement_layer_fraction_rows, placement_state_to_dict
from .prediction import make_predictor, predictor_config_from_dict
from .scheduler import make_scheduler
from .trace_pack import ReplayStream, TracePack


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved single-benchmark simulation config."""

    run_id: str
    output_dir: Path
    trace_pack_path: Path
    benchmark: str | None
    replay_seed: int = 0
    max_batch_size: int = 1
    warmup_steps: int = 0
    eval_steps: int | None = None
    limit_per_benchmark: int | None = None
    predictor_config: dict[str, Any] = field(default_factory=dict)
    placement_config: dict[str, Any] = field(default_factory=dict)
    migration_config: dict[str, Any] = field(default_factory=dict)
    scheduler_config: dict[str, Any] = field(default_factory=dict)
    metrics_config: dict[str, Any] = field(default_factory=dict)
    original_config_path: str | None = None
    source_config: dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "trace_pack_path", Path(self.trace_pack_path).expanduser().resolve())
        object.__setattr__(self, "replay_seed", int(self.replay_seed))
        object.__setattr__(self, "max_batch_size", _positive_int(self.max_batch_size, "replay.max_batch_size"))
        object.__setattr__(self, "warmup_steps", _non_negative_int(self.warmup_steps, "replay.warmup_steps"))
        if self.eval_steps is not None:
            object.__setattr__(self, "eval_steps", _non_negative_int(self.eval_steps, "replay.eval_steps"))
        if self.limit_per_benchmark is not None:
            object.__setattr__(
                self,
                "limit_per_benchmark",
                _positive_int(self.limit_per_benchmark, "replay.limit_per_benchmark"),
            )
        for name in (
            "predictor_config",
            "placement_config",
            "migration_config",
            "scheduler_config",
            "metrics_config",
            "source_config",
        ):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping/dict")
            object.__setattr__(self, name, dict(value))

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def predictor_horizon(self) -> int:
        return predictor_config_from_dict(self.predictor_config).horizon

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run": {
                "run_id": self.run_id,
                "output_dir": str(self.output_dir),
                "overwrite": self.overwrite,
                "source_config_path": self.original_config_path,
            },
            "trace": {
                "trace_pack_path": str(self.trace_pack_path),
                "benchmark": self.benchmark,
            },
            "replay": {
                "seed": self.replay_seed,
                "max_batch_size": self.max_batch_size,
                "warmup_steps": self.warmup_steps,
                "eval_steps": self.eval_steps,
                "limit_per_benchmark": self.limit_per_benchmark,
            },
            "predictor": self.predictor_config,
            "placement": self.placement_config,
            "scheduler": self.scheduler_config,
            "metrics": self.metrics_config,
        }
        if self.migration_config:
            payload["migration"] = self.migration_config
        return payload


@dataclass(frozen=True)
class SimulationResult:
    """Paths written by one simulation run."""

    run_id: str
    config_path: str
    metrics_path: str
    summary_path: str
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_path": self.config_path,
            "metrics_path": self.metrics_path,
            "summary_path": self.summary_path,
            "artifacts": self.artifacts,
        }



class SimulationEngine:
    """Decode-only control loop for one benchmark/run."""

    def __init__(
        self,
        config: ExperimentConfig,
        trace_pack: TracePack,
        model: ModelSpec,
        hardware: HardwareModel,
        predictor: Any,
        placement_policy: Any,
        migration_policy: Any,
        scheduler: Any,
        metrics: MetricsCollector,
    ):
        self.config = config
        self.trace_pack = trace_pack
        self.model = model
        self.hardware = hardware
        self.predictor = predictor
        self.placement_policy = placement_policy
        self.migration_policy = migration_policy
        self.scheduler = scheduler
        self.metrics = metrics

    def run(self) -> SimulationResult:
        run_dir = self._prepare_run_dir()
        _write_yaml(self.config.to_dict(), self.config.config_path)

        stream = ReplayStream(
            self.trace_pack,
            seed=self.config.replay_seed,
            max_batch_size=self.config.max_batch_size,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
            benchmarks=[self.config.benchmark] if self.config.benchmark else None,
            limit_per_benchmark=self.config.limit_per_benchmark,
        )
        selected_request_ids = stream.selected_request_ids()
        warmup_steps, eval_steps = _split_stream_once(
            stream,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
        )
        if not eval_steps:
            raise ValueError("simulation has no eval steps after applying warmup/eval replay windows")

        if hasattr(self.predictor, "set_oracle_steps"):
            self.predictor.set_oracle_steps(eval_steps)
        prefill_context = None
        predictor_cfg = getattr(self.predictor, "config", None)
        if bool(getattr(predictor_cfg, "use_prefill_context", False)):
            prefill_context = self.trace_pack.prefill_context(selected_request_ids)
        self.predictor.initialize(warmup_steps, prefill_context)

        horizon = self.config.predictor_horizon
        first_prediction = self.predictor.predict(eval_steps[0].step_id, horizon)
        scheduler_config = getattr(self.scheduler, "config", self.config.scheduler_config)
        placement = self.placement_policy.initialize(
            warmup_steps,
            first_prediction,
            self.model,
            self.hardware,
            scheduler_config=scheduler_config,
        )
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        placement_path = artifacts_dir / "placement_initial.json"
        placement_path.write_text(_json_dumps(placement_state_to_dict(placement)), encoding="utf-8")
        placement_layer_fractions_path = artifacts_dir / "placement_layer_fractions.csv"
        _write_csv(placement_layer_fraction_rows(placement, self.model), placement_layer_fractions_path)
        self.metrics.record_warmup(warmup_steps, placement)

        for index, step in enumerate(eval_steps):
            prediction = first_prediction if index == 0 else self.predictor.predict(step.step_id, horizon)
            if self.migration_policy.should_run(step.step_id, self.metrics.snapshot()):
                plan = self.migration_policy.plan(
                    placement,
                    prediction,
                    self.model,
                    self.hardware,
                )
                placement = self.migration_policy.apply(placement, plan, self.model, self.hardware)
                self.metrics.record_migration(plan)
            decision = self.scheduler.schedule(step, placement, self.model, self.hardware)
            self.metrics.record_step(step, prediction, placement, decision)
            self.predictor.update(step)

        finalized = self.metrics.finalize(
            run_dir,
            extra_artifacts={
                "placement_initial_path": str(placement_path),
                "placement_layer_fractions_path": str(placement_layer_fractions_path),
            },
        )
        artifact_paths = {
            key: value
            for key, value in finalized.items()
            if key.endswith("_path") and key not in {"metrics_path", "summary_path"}
        }
        return SimulationResult(
            run_id=self.config.run_id,
            config_path=str(self.config.config_path),
            metrics_path=finalized["metrics_path"],
            summary_path=finalized["summary_path"],
            artifacts=artifact_paths,
        )

    def _prepare_run_dir(self) -> Path:
        run_dir = self.config.run_dir
        if run_dir.exists():
            if not self.config.overwrite:
                raise FileExistsError(f"run directory already exists: {run_dir}; choose a new run_id or set overwrite")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def run_simulations_from_config(
    config_path: str | Path,
    *,
    benchmarks: Sequence[str] | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[SimulationResult]:
    """Load an experiment config and run one independent simulation per benchmark."""

    source_path = Path(config_path).expanduser().resolve()
    source = load_mapping_config(source_path)
    configs = resolve_experiment_configs(
        source,
        config_path=source_path,
        benchmarks=benchmarks,
        run_id=run_id,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    model = _load_model(source, source_path.parent)
    hardware = _load_hardware(source, source_path.parent)

    results: list[SimulationResult] = []
    for cfg in configs:
        if not cfg.trace_pack_path.is_dir():
            raise FileNotFoundError(f"TracePack not found: {cfg.trace_pack_path}")
        with TracePack.open(cfg.trace_pack_path) as trace_pack:
            validate_trace_metadata(model, trace_pack.model_metadata())
            if cfg.benchmark is not None and cfg.benchmark not in trace_pack.benchmarks():
                raise ValueError(
                    f"benchmark {cfg.benchmark!r} not found in TracePack; available benchmarks: {trace_pack.benchmarks()}"
                )
            predictor = make_predictor(cfg.predictor_config, model=model)
            placement_policy = make_placement_policy(cfg.placement_config)
            migration_policy = make_migration_policy(cfg.migration_config)
            scheduler = make_scheduler(cfg.scheduler_config)
            metrics = MetricsCollector(
                run_id=cfg.run_id,
                benchmark=cfg.benchmark,
                trace_pack_path=cfg.trace_pack_path,
                trace_manifest=trace_pack.manifest(),
                model=model,
                hardware=hardware,
                config=cfg,
            )
            result = SimulationEngine(
                cfg,
                trace_pack,
                model,
                hardware,
                predictor,
                placement_policy,
                migration_policy,
                scheduler,
                metrics,
            ).run()
            results.append(result)
    return results


def resolve_experiment_configs(
    source: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    benchmarks: Sequence[str] | None = None,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[ExperimentConfig]:
    """Resolve a root config into single-benchmark configs without opening traces."""

    base_dir = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
    project_root = _project_root_for_config(config_path)
    run_cfg = _mapping(source.get("run", {}), "run")
    trace_cfg = _mapping(source.get("trace", {}), "trace")
    replay_cfg = _mapping(source.get("replay", {}), "replay")

    selected_benchmarks = _selected_benchmarks(benchmarks, trace_cfg)
    if not selected_benchmarks:
        selected_benchmarks = [None]
    base_run_id = run_id or run_cfg.get("run_id")
    run_count = len(selected_benchmarks)
    out_dir = _resolve_output_path(output_dir or run_cfg.get("output_dir", "runs"), project_root)
    effective_overwrite = bool(overwrite or run_cfg.get("overwrite", False))
    seed = int(replay_cfg.get("seed", run_cfg.get("seed", 0)))

    predictor_cfg = dict(_mapping(source.get("predictor", {}), "predictor"))
    predictor_cfg.setdefault("type", "heuristic")
    placement_cfg = dict(_mapping(source.get("placement", {}), "placement"))
    placement_cfg.setdefault("mode", "static_hot_cold")
    migration_cfg = dict(_mapping(source.get("migration", {}), "migration"))
    scheduler_cfg = dict(_mapping(source.get("scheduler", {}), "scheduler"))
    scheduler_cfg.setdefault("policy", "placement_greedy")
    metrics_cfg = dict(_mapping(source.get("metrics", {}), "metrics"))

    configs: list[ExperimentConfig] = []
    for benchmark in selected_benchmarks:
        resolved_run_id = _resolve_run_id(base_run_id, benchmark, run_count)
        trace_pack_path = _trace_pack_path_for_benchmark(
            trace_cfg,
            benchmark=benchmark,
            benchmark_count=run_count,
            base_dir=base_dir,
        )
        configs.append(
            ExperimentConfig(
                run_id=resolved_run_id,
                output_dir=out_dir,
                trace_pack_path=trace_pack_path,
                benchmark=benchmark,
                replay_seed=seed,
                max_batch_size=replay_cfg.get("max_batch_size", 1),
                warmup_steps=replay_cfg.get("warmup_steps", 0),
                eval_steps=replay_cfg.get("eval_steps"),
                limit_per_benchmark=replay_cfg.get("limit_per_benchmark"),
                predictor_config=predictor_cfg,
                placement_config=placement_cfg,
                migration_config=migration_cfg,
                scheduler_config=scheduler_cfg,
                metrics_config=metrics_cfg,
                original_config_path=str(Path(config_path).expanduser().resolve()) if config_path else None,
                source_config=dict(source),
                overwrite=effective_overwrite,
            )
        )
    return configs


def _split_stream_once(
    stream: ReplayStream,
    *,
    warmup_steps: int,
    eval_steps: int | None,
) -> tuple[list[ReplayStep], list[ReplayStep]]:
    warmup: list[ReplayStep] = []
    evaluation: list[ReplayStep] = []
    for index, step in enumerate(stream):
        metadata = dict(step.metadata)
        if index < warmup_steps:
            metadata.update({"phase": "warmup", "warmup_steps": warmup_steps})
            warmup.append(_replace_step_metadata(step, metadata))
            continue
        if eval_steps is not None and len(evaluation) >= eval_steps:
            break
        metadata.update({"phase": "eval", "warmup_steps": warmup_steps, "eval_steps": eval_steps})
        evaluation.append(_replace_step_metadata(step, metadata))
    return warmup, evaluation


def _replace_step_metadata(step: ReplayStep, metadata: Mapping[str, Any]) -> ReplayStep:
    return ReplayStep(
        step_id=step.step_id,
        active_request_ids=list(step.active_request_ids),
        layer_expert_counts=step.layer_expert_counts,
        request_positions=dict(step.request_positions),
        metadata=dict(metadata),
    )


def _load_model(source: Mapping[str, Any], base_dir: Path) -> ModelSpec:
    cfg = _mapping(source.get("model", {}), "model")
    if "config_path" in cfg:
        return load_model_spec(_resolve_path(cfg["config_path"], base_dir))
    if "path" in cfg:
        return load_model_spec(_resolve_path(cfg["path"], base_dir))
    if "model_id" in cfg:
        return model_spec_from_dict({"model": cfg})
    raise ValueError("model config requires config_path/path or inline model_id fields")


def _load_hardware(source: Mapping[str, Any], base_dir: Path) -> HardwareModel:
    cfg = _mapping(source.get("hardware", {}), "hardware")
    if "config_path" in cfg:
        return load_hardware_model(_resolve_path(cfg["config_path"], base_dir))
    if "path" in cfg:
        return load_hardware_model(_resolve_path(cfg["path"], base_dir))
    if "hardware_id" in cfg:
        return hardware_model_from_dict({"hardware": cfg})
    raise ValueError("hardware config requires config_path/path or inline hardware_id fields")


def _selected_benchmarks(override: Sequence[str] | None, trace_cfg: Mapping[str, Any]) -> list[str | None]:
    raw: Any
    if override:
        raw = list(override)
    elif "benchmark" in trace_cfg:
        raw = trace_cfg["benchmark"]
    else:
        raw = trace_cfg.get("benchmarks", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    else:
        if not isinstance(raw, Sequence):
            raise ValueError("trace.benchmarks must be a sequence of names")
        values = list(raw)
    out: list[str | None] = []
    for value in values:
        if value is None:
            out.append(None)
        else:
            text = str(value)
            if not text:
                raise ValueError("benchmark names must be non-empty")
            if text not in out:
                out.append(text)
    return out


def _resolve_run_id(base_run_id: Any, benchmark: str | None, run_count: int) -> str:
    slug = _slug(benchmark or "default")
    if base_run_id:
        text = str(base_run_id)
        if "{benchmark}" in text:
            return text.format(benchmark=slug)
        return f"{text}_{slug}" if run_count > 1 else text
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"sim_mvp0_{stamp}_{slug}"


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} config must be a mapping")
    return value


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    raise ValueError(f"trace config requires one of: {', '.join(keys)}")


def _first_available(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _trace_pack_path_for_benchmark(
    trace_cfg: Mapping[str, Any],
    *,
    benchmark: str | None,
    benchmark_count: int,
    base_dir: Path,
) -> Path:
    path_mapping = _first_available(trace_cfg, ("trace_pack_paths", "pack_paths", "output_paths"))
    if path_mapping is not None:
        if not isinstance(path_mapping, Mapping):
            raise ValueError("trace.trace_pack_paths must map benchmark name to TracePack path")
        if benchmark is None:
            if len(path_mapping) != 1:
                raise ValueError("trace.trace_pack_paths with multiple entries requires trace.benchmark or trace.benchmarks")
            return _resolve_path(next(iter(path_mapping.values())), base_dir)
        try:
            return _resolve_path(path_mapping[benchmark], base_dir)
        except KeyError as exc:
            raise KeyError(f"trace.trace_pack_paths missing benchmark {benchmark!r}") from exc

    template = _first_available(
        trace_cfg,
        ("trace_pack_path_template", "pack_path_template", "output_path_template"),
    )
    if template is not None:
        if benchmark is None:
            raise ValueError("trace path templates require trace.benchmark or trace.benchmarks")
        return _resolve_path(_format_benchmark_path(template, benchmark), base_dir)

    pack_dir = _first_available(trace_cfg, ("trace_pack_dir", "pack_dir", "output_dir"))
    if pack_dir is not None:
        if benchmark is None:
            raise ValueError("trace trace_pack_dir requires trace.benchmark or trace.benchmarks")
        filename_template = str(trace_cfg.get("filename_template", "{benchmark}"))
        return _resolve_path(Path(str(pack_dir)).expanduser() / _format_benchmark_path(filename_template, benchmark), base_dir)

    path_value = _first_present(trace_cfg, ("trace_pack_path", "pack_path", "output_path"))
    path_text = str(path_value)
    if benchmark is not None and "{" in path_text and "}" in path_text:
        return _resolve_path(_format_benchmark_path(path_text, benchmark), base_dir)
    if benchmark_count > 1:
        raise ValueError(
            "multi-benchmark simulation requires one TracePack per benchmark; "
            "use trace.trace_pack_paths, trace.trace_pack_path_template, or trace.trace_pack_dir"
        )
    return _resolve_path(path_value, base_dir)


def _format_benchmark_path(template: Any, benchmark: str) -> str:
    slug = _slug(benchmark)
    return str(template).format(benchmark=benchmark, benchmark_slug=slug)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    if value is None:
        raise ValueError("path value is required")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (base_dir / path).resolve()


def _project_root_for_config(config_path: str | Path | None) -> Path:
    """Return the project root used for resolving run output directories.

    Config files are commonly stored under ``configs/``.  Relative
    ``run.output_dir`` values should still land in the repository-level
    ``runs/`` directory, independent of the shell's current working directory.
    """

    start = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "pim_hot_cold_moe").is_dir():
            return candidate
    return start


def _resolve_output_path(value: Any, project_root: Path) -> Path:
    if value is None:
        raise ValueError("output path value is required")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _positive_int(value: Any, field_name: str) -> int:
    out = _non_negative_int(value, field_name)
    if out <= 0:
        raise ValueError(f"{field_name} must be positive")
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


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            text_key = str(key)
            if text_key not in seen:
                fields.append(text_key)
                seen.add(text_key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_yaml(value: Mapping[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - PyYAML is a project dependency
        path.write_text(_json_dumps(value), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(_jsonable(value), sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP-0 decode-only simulations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one config, expanding to one run per benchmark")
    run.add_argument("--config", required=True, help="YAML/JSON experiment config")
    run.add_argument("--benchmark", action="append", default=None, help="benchmark to simulate; repeat for multiple")
    run.add_argument("--run-id", default=None, help="override run_id; multi-benchmark runs append the benchmark slug")
    run.add_argument("--output-dir", default=None, help="override run.output_dir")
    run.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    run.set_defaults(func=_cmd_run)
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    results = run_simulations_from_config(
        args.config,
        benchmarks=args.benchmark,
        run_id=args.run_id,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    payload: Any = results[0].to_dict() if len(results) == 1 else [result.to_dict() for result in results]
    print(_json_dumps(payload), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
