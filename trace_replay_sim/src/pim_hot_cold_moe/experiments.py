"""Project-level experiment workflow helpers.

This module intentionally stays thin: it reuses the already implemented
TracePack builder and simulation engine, then adds the run-management glue needed
for reproducible MVP experiments, simple sweeps, and compact reports.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_hardware import load_mapping_config
from .simulation import (
    SimulationResult,
    _project_root_for_config,
    _resolve_output_path,
    run_simulations_from_config,
)
from .trace_pack import build_trace_packs


def build_trace_from_config(
    config_path: str | Path,
    *,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Build per-benchmark TracePack directories from a project YAML/JSON config.

    Multi-benchmark configs produce one independent array-backed TracePack per
    ``(model, benchmark)``. Existing packs raise unless ``overwrite`` is true;
    simulations assume required TracePacks have already been built and fail
    fast if one is missing.
    """

    source_path = Path(config_path).expanduser().resolve()
    source = load_mapping_config(source_path)
    base_dir = source_path.parent
    trace_cfg = _mapping(source.get("trace", source), "trace")
    raw_root = _resolve_path(_required(trace_cfg, "raw_root"), base_dir, must_exist=True)
    model = trace_cfg.get("model", trace_cfg.get("model_id"))
    if not model:
        model_cfg = source.get("model", {})
        if isinstance(model_cfg, Mapping):
            model = model_cfg.get("model_id")
    if not model:
        raise ValueError("trace build config requires trace.model/model_id or inline model.model_id")

    effective_overwrite = bool(trace_cfg.get("overwrite", False)) if overwrite is None else bool(overwrite)
    return build_trace_packs(
        raw_root=raw_root,
        model=str(model),
        benchmarks=_benchmark_list(trace_cfg),
        output_path=_optional_resolved_path(
            _first_available(trace_cfg, ("output_path", "trace_pack_path", "pack_path")),
            base_dir,
        ),
        output_dir=_optional_resolved_path(trace_cfg.get("output_dir"), base_dir),
        output_path_template=_optional_resolved_template(
            _first_available(
                trace_cfg,
                ("output_path_template", "trace_pack_path_template", "pack_path_template"),
            ),
            base_dir,
        ),
        output_paths=_optional_resolved_path_mapping(
            _first_available(trace_cfg, ("output_paths", "trace_pack_paths", "pack_paths")),
            base_dir,
        ),
        filename_template=str(trace_cfg.get("filename_template", "{benchmark}")),
        max_requests_per_benchmark=trace_cfg.get("max_requests_per_benchmark"),
        overwrite=effective_overwrite,
    )


def run_sweep_from_config(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Expand and optionally execute a small Cartesian-product sweep.

    Minimal sweep config shape::

        base_config: configs/experiments/mvp0_100req_debug.yaml
        sweep:
          name: scheduler
          parameters:
            scheduler.policy: [placement_greedy, gpu_baseline]

    The generated per-combination configs are written under
    ``<output_dir>/_sweeps/<sweep_id>/configs``.  The JSON index links each
    settings row to all run IDs produced by that settings row.
    """

    source_path = Path(config_path).expanduser().resolve()
    source = load_mapping_config(source_path)
    base_dir = source_path.parent
    project_root = _project_root_for_config(source_path)
    sweep_cfg = _mapping(source.get("sweep", {}), "sweep")
    base_source = _load_sweep_base(source, base_dir)
    parameters = _parameter_grid(sweep_cfg)
    name = _slug(str(sweep_cfg.get("name", source_path.stem)))
    sweep_id = str(sweep_cfg.get("sweep_id") or f"sweep_{name}_{_utc_stamp()}")
    output_dir = _sweep_output_dir(sweep_cfg, base_source, project_root)
    sweep_dir = output_dir / "_sweeps" / sweep_id
    config_dir = sweep_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, settings in enumerate(parameters):
        combo = deepcopy(base_source)
        for dotted_path, value in settings.items():
            _set_dotted(combo, dotted_path, value)
        run_cfg = combo.setdefault("run", {})
        if not isinstance(run_cfg, dict):
            raise ValueError("base run config must be a mapping")
        run_cfg.setdefault("output_dir", str(output_dir))
        run_cfg["run_id"] = _render_run_id_template(sweep_cfg, sweep_id, index, settings)

        combo_path = config_dir / f"{index:03d}_{_params_slug(settings)}.yaml"
        _write_yaml(combo, combo_path)

        result_dicts: list[dict[str, Any]] = []
        if not dry_run:
            results = run_simulations_from_config(combo_path, overwrite=overwrite)
            result_dicts = [result.to_dict() for result in results]
        rows.append(
            {
                "combo_index": index,
                "settings": settings,
                "config_path": str(combo_path),
                "run_ids": [result["run_id"] for result in result_dicts],
                "results": result_dicts,
            }
        )

    index_json_path = sweep_dir / "index.json"
    payload = {
        "sweep_id": sweep_id,
        "dry_run": dry_run,
        "config_path": str(source_path),
        "index_json_path": str(index_json_path),
        "rows": rows,
    }
    index_json_path.write_text(_json_dumps(payload), encoding="utf-8")
    return payload


def summarize_runs(run_paths: Sequence[str | Path], *, output_format: str = "markdown") -> str:
    """Return a compact comparison report for one or more run directories."""

    metrics = [_load_run_metrics(path) for path in run_paths]
    fmt = output_format.lower()
    if fmt == "json":
        return _json_dumps({"runs": metrics})
    rows = [_report_row(item) for item in metrics]
    if fmt != "markdown":
        raise ValueError("report format must be one of: markdown, json")
    return _markdown_table(rows)


def _load_sweep_base(source: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    if "base_config" in source:
        return load_mapping_config(_resolve_path(source["base_config"], base_dir, must_exist=True))
    if "base" in source:
        base = source["base"]
        if not isinstance(base, Mapping):
            raise ValueError("sweep base must be a mapping")
        return dict(base)
    reserved = {"sweep", "base_config"}
    inline = {key: value for key, value in source.items() if key not in reserved}
    if not inline:
        raise ValueError("sweep config requires base_config, base, or inline experiment config")
    return inline


def _parameter_grid(sweep_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = sweep_cfg.get("parameters", sweep_cfg.get("matrix", {}))
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("sweep.parameters must be a mapping of dotted paths to value lists")
    items = [(str(key), value) for key, value in raw.items()]
    keys = [key for key, _value in items]
    values: list[list[Any]] = []
    for key, raw_values in items:
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise ValueError(f"sweep parameter {key!r} must be a list of values")
        if not raw_values:
            raise ValueError(f"sweep parameter {key!r} must have at least one value")
        values.append(list(raw_values))
    if not keys:
        return [{}]
    max_runs = sweep_cfg.get("max_runs")
    rows = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]
    if max_runs is not None:
        rows = rows[: _positive_int(max_runs, "sweep.max_runs")]
    return rows


def _set_dotted(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("sweep parameter path must be non-empty")
    cursor: dict[str, Any] = target
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set {dotted_path!r}: {part!r} is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def _render_run_id_template(
    sweep_cfg: Mapping[str, Any],
    sweep_id: str,
    index: int,
    settings: Mapping[str, Any],
) -> str:
    params_slug = _params_slug(settings)
    template = str(sweep_cfg.get("run_id_template") or "{sweep_id}_{index:03d}_{params_slug}")
    return template.format(
        sweep_id=sweep_id,
        index=index,
        params_slug=params_slug,
        params_hash=_short_hash(settings),
    )


def _sweep_output_dir(sweep_cfg: Mapping[str, Any], base_source: Mapping[str, Any], project_root: Path) -> Path:
    if "output_dir" in sweep_cfg:
        return _resolve_output_path(sweep_cfg["output_dir"], project_root)
    run_cfg = base_source.get("run", {})
    if isinstance(run_cfg, Mapping) and run_cfg.get("output_dir"):
        return _resolve_output_path(run_cfg["output_dir"], project_root)
    return _resolve_output_path("runs", project_root)


def _load_run_metrics(path: str | Path) -> dict[str, Any]:
    run_path = Path(path).expanduser().resolve()
    metrics_path = run_path if run_path.name == "metrics.json" else run_path / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found for run path: {run_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metrics root must be a JSON object: {metrics_path}")
    return data


def _report_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    latency = _mapping(metrics.get("latency_ms", {}), "latency_ms")
    pred = _mapping(metrics.get("histogram_prediction_error", {}), "histogram_prediction_error")
    hot_miss = _mapping(metrics.get("hot_miss_load_rate", {}), "hot_miss_load_rate")
    oracle_gap = _mapping(metrics.get("oracle_gap_ratio", {}), "oracle_gap_ratio")
    offload = _mapping(metrics.get("offload", {}), "offload")
    memory = _mapping(metrics.get("memory", {}), "memory")
    cost = _mapping(metrics.get("cost", {}), "cost")
    return {
        "run_id": metrics.get("run_id"),
        "benchmark": metrics.get("benchmark"),
        "eval_steps": metrics.get("evaluation", {}).get("step_count")
        if isinstance(metrics.get("evaluation"), Mapping)
        else None,
        "latency_mean_ms": latency.get("mean"),
        "latency_p90_ms": latency.get("p90"),
        "pred_error_mean": pred.get("mean"),
        "hot_miss_mean": hot_miss.get("mean"),
        "oracle_gap_ratio_mean": oracle_gap.get("mean"),
        "offload_backend": offload.get("backend"),
        "hbm_saved_gb": memory.get("hbm_saved_gb"),
        "hbm_saved_fraction": memory.get("hbm_saved_fraction"),
        "remote_expert_gb": memory.get("remote_expert_gb"),
        "offload_expert_gb": memory.get("offload_expert_gb"),
        "hbm_expert_gb_per_gpu": memory.get("hbm_expert_gb_per_gpu"),
        "cost_saved_fraction": cost.get("cost_saved_fraction"),
        "warnings": "; ".join(str(w) for w in metrics.get("warnings", []) or []),
    }


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "run_id",
        "benchmark",
        "eval_steps",
        "latency_mean_ms",
        "latency_p90_ms",
        "pred_error_mean",
        "hot_miss_mean",
        "oracle_gap_ratio_mean",
        "offload_backend",
        "hbm_saved_gb",
        "hbm_saved_fraction",
        "remote_expert_gb",
        "offload_expert_gb",
        "hbm_expert_gb_per_gpu",
        "cost_saved_fraction",
        "warnings",
    ]
    lines = ["# Run Report", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _benchmark_list(trace_cfg: Mapping[str, Any]) -> list[str]:
    if "benchmarks" in trace_cfg:
        raw = trace_cfg["benchmarks"]
    elif "benchmark" in trace_cfg:
        raw = [trace_cfg["benchmark"]]
    else:
        raise ValueError("trace config requires benchmark or benchmarks")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = [str(value) for value in raw]
    else:
        raise ValueError("trace.benchmarks must be a sequence")
    values = [value for value in values if value]
    if not values:
        raise ValueError("at least one benchmark is required")
    return values


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    raise ValueError(f"config requires one of: {', '.join(keys)}")


def _first_available(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"config requires {key!r}")
    return value


def _optional_resolved_path(value: Any, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value, base_dir, must_exist=False)


def _optional_resolved_template(value: Any, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _optional_resolved_path_mapping(value: Any, base_dir: Path) -> dict[str, Path] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, Mapping):
        raise ValueError("benchmark path mapping must be a mapping")
    return {str(key): _resolve_path(path, base_dir, must_exist=False) for key, path in value.items()}


def _resolve_path(value: Any, base_dir: Path, *, must_exist: bool) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        cwd_path = path.resolve()
        resolved = cwd_path if cwd_path.exists() else (base_dir / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def _positive_int(value: Any, field_name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return out


def _params_slug(settings: Mapping[str, Any]) -> str:
    if not settings:
        return "base"
    parts = [f"{_slug(path.split('.')[-1])}_{_slug(value)}" for path, value in sorted(settings.items())]
    slug = "__".join(parts)
    if len(slug) <= 96:
        return slug
    return f"{slug[:80]}_{_short_hash(settings)}"


def _slug(value: Any) -> str:
    text = str(value).lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "value"


def _short_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _utc_stamp() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimulationResult):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_yaml(value: Mapping[str, Any], path: Path) -> None:
    import yaml  # type: ignore

    path.write_text(yaml.safe_dump(_jsonable(value), sort_keys=True), encoding="utf-8")
