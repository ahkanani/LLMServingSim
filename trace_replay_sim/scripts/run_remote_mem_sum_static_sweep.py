#!/usr/bin/env python3
"""Generate and run the serialized remote-memory static hot/cold sweep.

The matrix is:

* 16 model-workload pairs from remote-memory full runs, excluding HuggingFaceH4.
* static_hot_cold remote-memory runs:
  3 remote bandwidths x 3 batch sizes x 4 cold fractions.
* all-GPU baselines:
  3 batch sizes, deduplicated across remote bandwidth.

Total: 576 static remote-memory runs + 48 GPU baselines = 624 experiments.
Results, generated configs, logs, and the experiment index are written under
``runs/remote_mem_sum`` by default.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOB = "configs/experiments/remote_mem/full_runs/*_remote_mem_bw09.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "remote_mem_sum"
GENERATED_CONFIG_DIR = OUTPUT_ROOT / "generated_configs"
RUNS_DIR = OUTPUT_ROOT / "runs"
LOGS_DIR = OUTPUT_ROOT / "logs"

EXCLUDE_BENCHMARKS = {"HuggingFaceH4"}
HARDWARE_CONFIGS = {
    "bw045": "configs/hardware/8gpu_h200_remote_mem_bw045.yaml",
    "bw09": "configs/hardware/8gpu_h200_remote_mem_bw09.yaml",
    "bw18": "configs/hardware/8gpu_h200_remote_mem_bw18.yaml",
}
BATCH_SIZES = (32, 64, 128)
COLD_FRACTIONS = (0.1, 0.2, 0.25, 0.5)
HOT_FRACTION = 0.5
WARMUP_STEPS = 128
EVAL_STEPS = 1000
LATENCY_COMBINE_MODE = "sum"
GPU_BASELINE_HARDWARE_KEY = "bw09"


@dataclass(frozen=True)
class Experiment:
    name: str
    config_path: Path
    run_dir: Path
    log_path: Path
    kind: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel workers; default uses all detected CPUs")
    parser.add_argument("--generate-only", action="store_true", help="Only generate configs/index, do not run simulations")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite and rerun even if metrics.json exists")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed run directories")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N generated configs, useful for testing")
    parser.add_argument("--python", default=sys.executable, help="Python executable for simulation subprocesses")
    args = parser.parse_args(argv)

    experiments = generate_configs()
    if args.limit is not None:
        experiments = experiments[: args.limit]

    _write_index(experiments)
    counts = _counts_by_kind(experiments)
    print(
        json.dumps(
            {
                "generated": len(experiments),
                "counts_by_kind": counts,
                "config_dir": str(GENERATED_CONFIG_DIR),
                "output_dir": str(RUNS_DIR),
                "logs_dir": str(LOGS_DIR),
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
                        "kind": exp.kind,
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
        "counts_by_kind": counts,
        "skipped": skipped,
        "failed": len(failures),
        "succeeded_or_skipped": len(experiments) - len(failures),
        "elapsed_s": round(elapsed, 3),
        "failures": failures,
    }
    summary_path = OUTPUT_ROOT / "runner_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, sort_keys=True), flush=True)
    return 1 if failures else 0


def generate_configs() -> list[Experiment]:
    source_paths = sorted(PROJECT_ROOT.glob(SOURCE_GLOB))
    if len(source_paths) != 20:
        raise ValueError(f"expected 20 remote-memory bw09 source configs, found {len(source_paths)}")

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _clear_yaml_files(GENERATED_CONFIG_DIR)

    hardware_paths = {key: _resolve_project_path(path) for key, path in HARDWARE_CONFIGS.items()}
    experiments: list[Experiment] = []
    selected_sources: list[tuple[Path, dict[str, Any], str]] = []

    for source_path in source_paths:
        source = _load_yaml(source_path)
        benchmark = str(source.get("trace", {}).get("benchmark", ""))
        if benchmark in EXCLUDE_BENCHMARKS:
            continue
        selected_sources.append((source_path, source, _case_id(source_path, source)))

    for source_path, source, case_id in selected_sources:
        for hw_key, hardware_path in hardware_paths.items():
            for batch_size in BATCH_SIZES:
                for cold_fraction in COLD_FRACTIONS:
                    run_id = (
                        f"static_sum_remote_{case_id}_{hw_key}_b{batch_size}"
                        f"_h{_fraction_tag(HOT_FRACTION)}_c{_fraction_tag(cold_fraction)}"
                    )
                    cfg = _base_config(source, source_path, hardware_path, run_id, batch_size)
                    cfg["placement"] = {
                        "mode": "static_hot_cold",
                        "hot_fraction": HOT_FRACTION,
                        "cold_fraction": cold_fraction,
                        "warmup_weight": 1.0,
                        "prediction_weight": 1.0,
                    }
                    cfg["scheduler"] = {
                        "policy": "placement_greedy",
                        "oracle_comparison": True,
                        "latency_combine_mode": LATENCY_COMBINE_MODE,
                    }
                    experiments.append(_write_experiment(run_id, cfg, "static_hot_cold"))

        baseline_hw = hardware_paths[GPU_BASELINE_HARDWARE_KEY]
        for batch_size in BATCH_SIZES:
            run_id = f"gpu_baseline_sum_{case_id}_b{batch_size}"
            cfg = _base_config(source, source_path, baseline_hw, run_id, batch_size)
            cfg["placement"] = {"mode": "all_gpu"}
            cfg["scheduler"] = {
                "policy": "gpu_baseline",
                "oracle_comparison": False,
                "latency_combine_mode": LATENCY_COMBINE_MODE,
            }
            experiments.append(_write_experiment(run_id, cfg, "gpu_baseline"))

    expected = (
        len(selected_sources) * len(HARDWARE_CONFIGS) * len(BATCH_SIZES) * len(COLD_FRACTIONS)
        + len(selected_sources) * len(BATCH_SIZES)
    )
    if len(selected_sources) != 16:
        raise ValueError(f"expected 16 sources after excluding {sorted(EXCLUDE_BENCHMARKS)}, got {len(selected_sources)}")
    if expected != 624:
        raise ValueError(f"default matrix should produce 624 configs, computed {expected}")
    if len(experiments) != expected:
        raise ValueError(f"expected {expected} generated configs, got {len(experiments)}")
    return sorted(experiments, key=lambda exp: exp.name)


def run_experiment(exp: Experiment, *, python_exe: str, overwrite: bool, resume: bool) -> dict[str, Any]:
    metrics_path = exp.run_dir / "metrics.json"
    if resume and not overwrite and metrics_path.is_file():
        return {"status": "skipped", "elapsed_s": 0.0}

    exp.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, "-m", "pim_hot_cold_moe", "simulate", "--config", str(exp.config_path)]
    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    start = time.time()
    with exp.log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    status = "succeeded" if proc.returncode == 0 else "failed"
    return {"status": status, "returncode": proc.returncode, "elapsed_s": elapsed, "log": str(exp.log_path)}


def _base_config(source: dict[str, Any], source_path: Path, hardware_path: Path, run_id: str, batch_size: int) -> dict[str, Any]:
    cfg = copy.deepcopy(source)
    cfg["model"]["config_path"] = str(_resolve_config_path(cfg["model"]["config_path"], source_path.parent))
    cfg["hardware"]["config_path"] = str(hardware_path)
    cfg["replay"] = dict(cfg.get("replay", {}))
    cfg["replay"].update(
        {
            "seed": int(cfg["replay"].get("seed", 123)),
            "max_batch_size": batch_size,
            "warmup_steps": WARMUP_STEPS,
            "eval_steps": EVAL_STEPS,
        }
    )
    cfg["run"] = dict(cfg.get("run", {}))
    cfg["run"].update({"seed": int(cfg["run"].get("seed", 123)), "output_dir": str(RUNS_DIR), "run_id": run_id, "overwrite": False})
    cfg["metrics"] = {"level": "summary"}
    return cfg


def _write_experiment(run_id: str, cfg: dict[str, Any], kind: str) -> Experiment:
    config_path = GENERATED_CONFIG_DIR / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Experiment(
        name=run_id,
        config_path=config_path,
        run_dir=RUNS_DIR / run_id,
        log_path=LOGS_DIR / f"{run_id}.log",
        kind=kind,
    )


def _write_index(experiments: list[Experiment]) -> None:
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
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "experiment_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (GENERATED_CONFIG_DIR / "experiment_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _case_id(source_path: Path, source: dict[str, Any]) -> str:
    run_id = str(source.get("run", {}).get("run_id") or source_path.stem)
    run_id = re.sub(r"^remote_mem_", "", run_id)
    run_id = re.sub(r"_static_hot_cold_bw09$", "", run_id)
    run_id = re.sub(r"_remote_mem_bw09$", "", run_id)
    return _slug(run_id)


def _fraction_tag(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")


def _counts_by_kind(experiments: list[Experiment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for exp in experiments:
        out[exp.kind] = out.get(exp.kind, 0) + 1
    return out


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
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


def _clear_yaml_files(path: Path) -> None:
    for item in path.glob("*.yaml"):
        item.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
