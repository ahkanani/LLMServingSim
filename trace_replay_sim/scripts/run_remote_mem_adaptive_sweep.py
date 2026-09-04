#!/usr/bin/env python3
"""Generate and run the remote-memory adaptive placement sweep.

This runner expands the compact manifest in
``configs/experiments/remote_mem/adaptive_extra_budget_sweep.yaml`` into
concrete simulation configs, then executes them with a parallel subprocess
queue. It is intended to run inside tmux and can resume completed jobs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MANIFEST = "configs/experiments/remote_mem/adaptive_extra_budget_sweep.yaml"


@dataclass(frozen=True)
class Experiment:
    name: str
    config_path: Path
    run_dir: Path
    log_path: Path
    kind: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Sweep manifest YAML")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel workers; default uses all detected CPUs")
    parser.add_argument("--generate-only", action="store_true", help="Only generate configs/index, do not run simulations")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite and rerun even if metrics.json exists")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed run directories")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N generated configs, useful for testing")
    parser.add_argument("--python", default=sys.executable, help="Python executable for simulation subprocesses")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_yaml(manifest_path)
    project_root = Path(manifest["project_root"]).expanduser().resolve()

    experiments = generate_configs(manifest, manifest_path)
    if args.limit is not None:
        experiments = experiments[: args.limit]

    _write_index(manifest, experiments)
    counts = _counts_by_kind(experiments)
    print(
        json.dumps(
            {
                "generated": len(experiments),
                "counts_by_kind": counts,
                "config_dir": str(Path(manifest["generated_config_dir"]).expanduser().resolve()),
                "output_dir": str(Path(manifest["output_dir"]).expanduser().resolve()),
                "logs_dir": str(Path(manifest["logs_dir"]).expanduser().resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.generate_only:
        return 0

    jobs = max(1, int(args.jobs))
    resume = not args.no_resume
    start = time.time()
    failures: list[dict[str, Any]] = []
    completed = 0
    skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_exp = {
            executor.submit(
                run_experiment,
                exp,
                project_root=project_root,
                python_exe=args.python,
                overwrite=args.overwrite,
                resume=resume,
            ): exp
            for exp in experiments
        }
        total = len(future_to_exp)
        for future in concurrent.futures.as_completed(future_to_exp):
            exp = future_to_exp[future]
            result = future.result()
            completed += 1
            if result["status"] == "skipped":
                skipped += 1
            if result["status"] == "failed":
                failures.append(result)
            print(
                json.dumps(
                    {
                        "progress": f"{completed}/{total}",
                        "status": result["status"],
                        "name": exp.name,
                        "returncode": result.get("returncode"),
                        "elapsed_s": round(result.get("elapsed_s", 0.0), 3),
                        "log": str(exp.log_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.time() - start
    summary = {
        "total": len(experiments),
        "skipped": skipped,
        "failed": len(failures),
        "succeeded_or_skipped": len(experiments) - len(failures),
        "elapsed_s": round(elapsed, 3),
        "failures": failures,
    }
    summary_path = Path(manifest["output_dir"]).expanduser().resolve() / "runner_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, sort_keys=True), flush=True)
    return 1 if failures else 0


def generate_configs(manifest: dict[str, Any], manifest_path: Path) -> list[Experiment]:
    project_root = Path(manifest["project_root"]).expanduser().resolve()
    source_glob = str(manifest["source_glob"])
    source_paths = sorted(project_root.glob(source_glob))
    if len(source_paths) != 20:
        raise ValueError(f"expected 20 source configs from {source_glob!r}, found {len(source_paths)}")
    exclude_benchmarks = {str(value) for value in manifest.get("exclude_benchmarks", [])}

    generated_dir = Path(manifest["generated_config_dir"]).expanduser().resolve()
    output_dir = Path(manifest["output_dir"]).expanduser().resolve()
    logs_dir = Path(manifest["logs_dir"]).expanduser().resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for old_config in generated_dir.glob("*.yaml"):
        old_config.unlink()

    hardware_paths = {
        key: _resolve_project_path(value, project_root)
        for key, value in _mapping(manifest["hardware"], "hardware").items()
    }
    batch_sizes = [int(v) for v in manifest["replay"]["batch_sizes"]]
    warmup_steps = int(manifest["replay"]["warmup_steps"])
    eval_steps = int(manifest["replay"]["eval_steps"])
    latency_combine_mode = str(manifest.get("scheduler", {}).get("latency_combine_mode", "max"))
    base_fraction = float(manifest["adaptive"]["base_extra_copy_fraction"])
    sweep_model_slug = str(manifest["adaptive"]["fraction_sweep_model_slug"])
    fraction_values = [float(v) for v in manifest["adaptive"]["fraction_sweep_values"]]
    extra_fractions = [v for v in fraction_values if not _same_float(v, base_fraction)]
    static_cfg = dict(manifest.get("static_hot_cold", {}))
    static_enabled = bool(static_cfg.get("enabled", False))
    static_hot_fraction = float(static_cfg.get("hot_fraction", 0.5))
    static_base_cold_fraction = float(static_cfg.get("base_cold_fraction", 0.25))
    static_sweep_model_slug = str(static_cfg.get("cold_fraction_sweep_model_slug", sweep_model_slug))
    static_cold_values = [float(v) for v in static_cfg.get("cold_fraction_sweep_values", [static_base_cold_fraction])]
    static_extra_cold_values = [v for v in static_cold_values if not _same_float(v, static_base_cold_fraction)]
    static_oracle_comparison = bool(static_cfg.get("oracle_comparison", True))

    experiments: list[Experiment] = []
    for source_path in source_paths:
        source = _load_yaml(source_path)
        benchmark = str(source.get("trace", {}).get("benchmark", ""))
        if benchmark in exclude_benchmarks:
            continue
        case_id = _case_id(source_path, source)
        model_slug = _model_slug_from_case(case_id)

        for hw_key, hardware_path in hardware_paths.items():
            for batch_size in batch_sizes:
                run_id = f"adaptive_remote_{case_id}_{hw_key}_b{batch_size}_x{_fraction_tag(base_fraction)}"
                cfg = _base_config(source, source_path, hardware_path, output_dir, run_id, warmup_steps, eval_steps, batch_size)
                cfg["placement"] = {"mode": "adaptive_extra_budget", "extra_copy_fraction": base_fraction}
                cfg["scheduler"] = {
                    "policy": "placement_greedy",
                    "oracle_comparison": True,
                    "latency_combine_mode": latency_combine_mode,
                }
                experiments.append(_write_experiment(generated_dir, logs_dir, output_dir, run_id, cfg, "adaptive_base"))

                if model_slug == sweep_model_slug:
                    for fraction in extra_fractions:
                        run_id = f"adaptive_remote_{case_id}_{hw_key}_b{batch_size}_x{_fraction_tag(fraction)}"
                        cfg = _base_config(source, source_path, hardware_path, output_dir, run_id, warmup_steps, eval_steps, batch_size)
                        cfg["placement"] = {"mode": "adaptive_extra_budget", "extra_copy_fraction": fraction}
                        cfg["scheduler"] = {
                            "policy": "placement_greedy",
                            "oracle_comparison": True,
                            "latency_combine_mode": latency_combine_mode,
                        }
                        experiments.append(_write_experiment(generated_dir, logs_dir, output_dir, run_id, cfg, "adaptive_fraction"))

                if static_enabled:
                    run_id = (
                        f"static_remote_{case_id}_{hw_key}_b{batch_size}"
                        f"_h{_fraction_tag(static_hot_fraction)}_c{_fraction_tag(static_base_cold_fraction)}"
                    )
                    cfg = _base_config(source, source_path, hardware_path, output_dir, run_id, warmup_steps, eval_steps, batch_size)
                    cfg["placement"] = {
                        "mode": "static_hot_cold",
                        "hot_fraction": static_hot_fraction,
                        "cold_fraction": static_base_cold_fraction,
                        "warmup_weight": 1.0,
                        "prediction_weight": 1.0,
                    }
                    cfg["scheduler"] = {
                        "policy": "placement_greedy",
                        "oracle_comparison": static_oracle_comparison,
                        "latency_combine_mode": latency_combine_mode,
                    }
                    experiments.append(_write_experiment(generated_dir, logs_dir, output_dir, run_id, cfg, "static_base"))

                    if model_slug == static_sweep_model_slug:
                        for cold_fraction in static_extra_cold_values:
                            run_id = (
                                f"static_remote_{case_id}_{hw_key}_b{batch_size}"
                                f"_h{_fraction_tag(static_hot_fraction)}_c{_fraction_tag(cold_fraction)}"
                            )
                            cfg = _base_config(source, source_path, hardware_path, output_dir, run_id, warmup_steps, eval_steps, batch_size)
                            cfg["placement"] = {
                                "mode": "static_hot_cold",
                                "hot_fraction": static_hot_fraction,
                                "cold_fraction": cold_fraction,
                                "warmup_weight": 1.0,
                                "prediction_weight": 1.0,
                            }
                            cfg["scheduler"] = {
                                "policy": "placement_greedy",
                                "oracle_comparison": static_oracle_comparison,
                                "latency_combine_mode": latency_combine_mode,
                            }
                            experiments.append(_write_experiment(generated_dir, logs_dir, output_dir, run_id, cfg, "static_cold_fraction"))

        if bool(manifest.get("gpu_baseline", {}).get("enabled", True)):
            baseline_hw_key = str(manifest["gpu_baseline"].get("hardware_key", "bw09"))
            baseline_hw = hardware_paths[baseline_hw_key]
            oracle_comparison = bool(manifest["gpu_baseline"].get("oracle_comparison", False))
            for batch_size in batch_sizes:
                run_id = f"gpu_baseline_{case_id}_b{batch_size}"
                cfg = _base_config(source, source_path, baseline_hw, output_dir, run_id, warmup_steps, eval_steps, batch_size)
                cfg["placement"] = {"mode": "all_gpu"}
                cfg["scheduler"] = {
                    "policy": "gpu_baseline",
                    "oracle_comparison": oracle_comparison,
                    "latency_combine_mode": latency_combine_mode,
                }
                experiments.append(_write_experiment(generated_dir, logs_dir, output_dir, run_id, cfg, "gpu_baseline"))

    source_count = len([path for path in source_paths if str(_load_yaml(path).get("trace", {}).get("benchmark", "")) not in exclude_benchmarks])
    base_count = source_count * len(hardware_paths) * len(batch_sizes)
    sweep_source_count = len(
        [
            path
            for path in source_paths
            if str(_load_yaml(path).get("trace", {}).get("benchmark", "")) not in exclude_benchmarks
            and _model_slug_from_case(_case_id(path, _load_yaml(path))) == sweep_model_slug
        ]
    )
    expected = base_count + (sweep_source_count * len(hardware_paths) * len(batch_sizes) * len(extra_fractions))
    if static_enabled:
        expected += base_count
        static_sweep_source_count = len(
            [
                path
                for path in source_paths
                if str(_load_yaml(path).get("trace", {}).get("benchmark", "")) not in exclude_benchmarks
                and _model_slug_from_case(_case_id(path, _load_yaml(path))) == static_sweep_model_slug
            ]
        )
        expected += static_sweep_source_count * len(hardware_paths) * len(batch_sizes) * len(static_extra_cold_values)
    if bool(manifest.get("gpu_baseline", {}).get("enabled", True)):
        expected += source_count * len(batch_sizes)
    if len(experiments) != expected:
        raise ValueError(f"expected {expected} generated configs, got {len(experiments)}")
    return sorted(experiments, key=lambda exp: exp.name)


def run_experiment(
    exp: Experiment,
    *,
    project_root: Path,
    python_exe: str,
    overwrite: bool,
    resume: bool,
) -> dict[str, Any]:
    metrics_path = exp.run_dir / "metrics.json"
    if resume and not overwrite and metrics_path.is_file():
        return {"status": "skipped", "elapsed_s": 0.0}

    exp.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, "-m", "pim_hot_cold_moe", "simulate", "--config", str(exp.config_path)]
    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    start = time.time()
    with exp.log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    status = "succeeded" if proc.returncode == 0 else "failed"
    return {"status": status, "returncode": proc.returncode, "elapsed_s": elapsed, "log": str(exp.log_path)}


def _base_config(
    source: dict[str, Any],
    source_path: Path,
    hardware_path: Path,
    output_dir: Path,
    run_id: str,
    warmup_steps: int,
    eval_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(source)
    cfg["model"]["config_path"] = str(_resolve_config_path(cfg["model"]["config_path"], source_path.parent))
    cfg["hardware"]["config_path"] = str(hardware_path)
    cfg["replay"] = dict(cfg.get("replay", {}))
    cfg["replay"].update(
        {
            "seed": int(cfg["replay"].get("seed", 123)),
            "max_batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "eval_steps": eval_steps,
        }
    )
    cfg["run"] = dict(cfg.get("run", {}))
    cfg["run"].update({"seed": int(cfg["run"].get("seed", 123)), "output_dir": str(output_dir), "run_id": run_id, "overwrite": False})
    cfg["metrics"] = {"level": "summary"}
    return cfg


def _write_experiment(
    generated_dir: Path,
    logs_dir: Path,
    output_dir: Path,
    run_id: str,
    cfg: dict[str, Any],
    kind: str,
) -> Experiment:
    config_path = generated_dir / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Experiment(
        name=run_id,
        config_path=config_path,
        run_dir=output_dir / run_id,
        log_path=logs_dir / f"{run_id}.log",
        kind=kind,
    )


def _write_index(manifest: dict[str, Any], experiments: list[Experiment]) -> None:
    output_dir = Path(manifest["output_dir"]).expanduser().resolve()
    generated_dir = Path(manifest["generated_config_dir"]).expanduser().resolve()
    payload = [
        {
            "name": exp.name,
            "kind": exp.kind,
            "config_path": str(exp.config_path),
            "run_dir": str(exp.run_dir),
            "log_path": str(exp.log_path),
        }
        for exp in experiments
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (generated_dir / "experiment_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _case_id(source_path: Path, source: dict[str, Any]) -> str:
    run_id = str(source.get("run", {}).get("run_id") or source_path.stem)
    run_id = re.sub(r"^remote_mem_", "", run_id)
    run_id = re.sub(r"_static_hot_cold_bw09$", "", run_id)
    run_id = re.sub(r"_remote_mem_bw09$", "", run_id)
    return _slug(run_id)


def _model_slug_from_case(case_id: str) -> str:
    known = (
        "qwen3_235b_a22b_fp8",
        "deepseek_r1_awq",
        "kimi_k2_thinking",
        "llama_4_maverick_17b_128e_instruct",
    )
    for slug in known:
        if case_id.startswith(slug + "_"):
            return slug
    return case_id


def _fraction_tag(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def _same_float(left: float, right: float) -> bool:
    return abs(left - right) < 1e-12


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")


def _counts_by_kind(experiments: list[Experiment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for exp in experiments:
        out[exp.kind] = out.get(exp.kind, 0) + 1
    return out


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_config_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
