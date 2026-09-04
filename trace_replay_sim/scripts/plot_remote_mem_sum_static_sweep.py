#!/usr/bin/env python3
"""Plot the serialized remote-memory static hot/cold sweep.

This script is intentionally static-only: adaptive placement is not part of the
sum-mode story because latency-only adaptive placement tends to avoid serialized
remote-memory fetches. Outputs are written under ``runs/remote_mem_sum``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import yaml


MODEL_LABELS = {
    "qwen3_235b_a22b_fp8": "Qwen3",
    "deepseek_r1_awq": "DeepSeek",
    "kimi_k2_thinking": "Kimi",
    "llama_4_maverick_17b_128e_instruct": "Llama4",
}
BENCH_LABELS = {
    "Chinese-SimpleQA": "ChineseQA",
    "hellaswag": "HellaSwag",
    "livecodebench": "LiveCode",
    "mmlu": "MMLU",
    "mmlu_ZH_CN": "MMLU-ZH",
    "mmlu_zh_cn": "MMLU-ZH",
}
BW_ORDER = ["bw045", "bw09", "bw18"]
BW_LABELS = {"bw045": "0.45", "bw09": "0.90", "bw18": "1.80"}
BATCH_ORDER = [32, 64, 128]
COLD_ORDER = [0.10, 0.20, 0.25, 0.50]
COLD_LABELS = {0.10: "c=0.10", 0.20: "c=0.20", 0.25: "c=0.25", 0.50: "c=0.50"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs/remote_mem_sum/runs")
    parser.add_argument("--plots-dir", default="runs/remote_mem_sum/plots")
    parser.add_argument("--cost-dir", default="runs/remote_mem_sum/cost_efficiency_demo")
    parser.add_argument("--hbm-cost-unit", type=float, default=1.0)
    parser.add_argument("--remote-cost-unit", type=float, default=0.2)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    cost_dir = Path(args.cost_dir).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    cost_dir.mkdir(parents=True, exist_ok=True)
    clear_plot_outputs(plots_dir)
    clear_plot_outputs(cost_dir)

    rows = load_rows(runs_dir, hbm_cost_unit=args.hbm_cost_unit, remote_cost_unit=args.remote_cost_unit)
    rows = add_baseline_deltas(rows)
    validate_coverage(rows)
    write_table(plots_dir / "remote_mem_sum_static_metrics.csv", rows)
    write_json(plots_dir / "remote_mem_sum_static_metrics.json", rows)
    write_table(cost_dir / "remote_mem_sum_cost_metrics.csv", rows)
    write_json(cost_dir / "remote_mem_sum_cost_metrics.json", rows)

    plot_run_coverage(rows, plots_dir / "01_run_coverage.png")
    plot_latency_heatmap(rows, plots_dir / "02_latency_delta_heatmap.png")
    plot_latency_case_facets(rows, plots_dir / "03_latency_case_facets.png")
    plot_fixed_mmlu_b128_latency(rows, plots_dir / "04_fixed_mmlu_b128_latency_slices.png")
    plot_pareto_latency_hbm(rows, plots_dir / "05_pareto_latency_vs_hbm_saved.png")
    plot_oracle_gap_heatmap(rows, plots_dir / "06_oracle_gap_heatmap.png")
    plot_fixed_mmlu_b128_lane_decomposition(rows, plots_dir / "07_fixed_mmlu_b128_lane_decomposition.png")

    plot_cost_pareto(rows, cost_dir / "01_pareto_tokens_per_cost_vs_cost_saved.png")
    plot_cost_heatmap(rows, cost_dir / "02_tokens_per_cost_gain_heatmap.png")
    plot_cost_case_facets(rows, cost_dir / "03_cost_efficiency_case_facets.png")
    plot_fixed_mmlu_b128_cost_tradeoff(rows, cost_dir / "04_fixed_mmlu_b128_cost_tradeoff.png")

    summary = summarize(rows, hbm_cost_unit=args.hbm_cost_unit, remote_cost_unit=args.remote_cost_unit)
    plot_stories = build_plot_stories(summary)
    (plots_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (plots_dir / "plot_stories.md").write_text(plot_stories, encoding="utf-8")
    (cost_dir / "cost_efficiency_summary.json").write_text(json.dumps(summary["cost_efficiency"], indent=2, sort_keys=True), encoding="utf-8")
    (cost_dir / "plot_stories.md").write_text(plot_stories, encoding="utf-8")
    print(json.dumps({"plots_dir": str(plots_dir), "cost_dir": str(cost_dir), **summary["coverage"]}, sort_keys=True))
    return 0


def load_rows(runs_dir: Path, *, hbm_cost_unit: float, remote_cost_unit: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = Path(metrics["paths"]["config_path"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        placement = config.get("placement", {})
        replay = config.get("replay", {})
        scheduler = config.get("scheduler", {})
        memory = metrics["memory"]

        model = model_slug_from_id(str(metrics["model_id"]))
        benchmark = str(metrics["benchmark"])
        batch = int(replay["max_batch_size"])
        latency_ms = float(metrics["latency_ms"]["mean"])
        tokens_per_s = batch / (latency_ms / 1000.0)
        placement_mode = str(placement.get("mode"))
        cold_fraction = placement.get("cold_fraction")
        hot_fraction = placement.get("hot_fraction")
        hbm_gb, remote_gb = costed_memory_gb(placement_mode, memory)
        relative_cost = hbm_gb * hbm_cost_unit + remote_gb * remote_cost_unit
        tokens_per_s_per_cost = tokens_per_s / relative_cost
        oracle_gap_ms = float(metrics.get("oracle_gap_ms", {}).get("mean", 0.0) or 0.0)
        oracle_ms = latency_ms - oracle_gap_ms
        oracle_gap_pct = (oracle_gap_ms / oracle_ms * 100.0) if oracle_ms > 0 else 0.0

        rows.append(
            {
                "run_id": metrics["run_id"],
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "benchmark": benchmark,
                "benchmark_label": BENCH_LABELS.get(benchmark, benchmark),
                "case": f"{MODEL_LABELS.get(model, model)} / {BENCH_LABELS.get(benchmark, benchmark)}",
                "hardware_id": metrics["hardware_id"],
                "bw_key": bandwidth_key(str(metrics["hardware_id"])),
                "bw_tbps_each": bandwidth_value(str(metrics["hardware_id"])),
                "batch": batch,
                "warmup_steps": int(replay["warmup_steps"]),
                "eval_steps_configured": int(replay["eval_steps"]),
                "eval_steps_observed": int(metrics["evaluation"]["step_count"]),
                "placement": placement_mode,
                "hot_fraction": none_to_empty(hot_fraction),
                "cold_fraction": none_to_empty(cold_fraction),
                "cold_label": "" if cold_fraction is None else COLD_LABELS.get(float(cold_fraction), str(cold_fraction)),
                "scheduler_policy": str(scheduler.get("policy", "")),
                "latency_combine_mode": str(scheduler.get("latency_combine_mode", metrics.get("scheduler", {}).get("latency_combine_mode", ""))),
                "oracle_comparison": bool(scheduler.get("oracle_comparison", False)),
                "latency_ms": latency_ms,
                "gpu_time_ms": float(metrics["gpu_time_ms"]["mean"]),
                "offload_time_ms": float(metrics["offload_time_ms"]["mean"]),
                "remote_share_of_sum_pct": (
                    float(metrics["offload_time_ms"]["mean"]) / latency_ms * 100.0 if latency_ms > 0 else 0.0
                ),
                "oracle_ms": oracle_ms,
                "oracle_gap_ms": oracle_gap_ms,
                "oracle_gap_pct": oracle_gap_pct,
                "hbm_saved_pct": float(memory.get("hbm_saved_fraction", 0.0) or 0.0) * 100.0,
                "hbm_expert_gb_per_gpu": float(memory.get("hbm_expert_gb_per_gpu", 0.0) or 0.0),
                "hbm_expert_gb": float(memory.get("hbm_expert_gb", 0.0) or 0.0),
                "remote_expert_gb": float(memory.get("remote_expert_gb", memory.get("offload_expert_gb", 0.0)) or 0.0),
                "sim_cost_saved_pct": float(metrics.get("cost", {}).get("cost_saved_fraction", 0.0) or 0.0) * 100.0,
                "hbm_gb_costed": hbm_gb,
                "remote_gb_costed": remote_gb,
                "relative_cost": relative_cost,
                "tokens_per_s": tokens_per_s,
                "tokens_per_s_per_cost": tokens_per_s_per_cost,
            }
        )
    if not rows:
        raise FileNotFoundError(f"no metrics.json files under {runs_dir}")
    return rows


def add_baseline_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (row["model"], row["benchmark"], row["batch"]): row
        for row in rows
        if row["placement"] == "all_gpu"
    }
    missing: list[str] = []
    for row in rows:
        base = baselines.get((row["model"], row["benchmark"], row["batch"]))
        if base is None:
            missing.append(f"{row['model']}/{row['benchmark']}/b{row['batch']}")
            continue
        row["gpu_baseline_latency_ms"] = float(base["latency_ms"])
        row["latency_delta_pct"] = (float(row["latency_ms"]) / float(base["latency_ms"]) - 1.0) * 100.0
        row["speedup_vs_gpu_pct"] = (float(base["latency_ms"]) / float(row["latency_ms"]) - 1.0) * 100.0
        row["gpu_tokens_per_s"] = float(base["tokens_per_s"])
        row["gpu_relative_cost"] = float(base["relative_cost"])
        row["gpu_tokens_per_s_per_cost"] = float(base["tokens_per_s_per_cost"])
        row["tokens_per_s_gain_pct"] = (float(row["tokens_per_s"]) / float(base["tokens_per_s"]) - 1.0) * 100.0
        row["relative_cost_delta_pct"] = (float(row["relative_cost"]) / float(base["relative_cost"]) - 1.0) * 100.0
        row["cost_saved_pct"] = -float(row["relative_cost_delta_pct"])
        row["tokens_per_s_per_cost_gain_pct"] = (
            float(row["tokens_per_s_per_cost"]) / float(base["tokens_per_s_per_cost"]) - 1.0
        ) * 100.0
    if missing:
        raise ValueError("missing GPU baselines for: " + ", ".join(sorted(set(missing))[:10]))
    return rows


def validate_coverage(rows: list[dict[str, Any]]) -> None:
    metrics_count = len(rows)
    if metrics_count != 624:
        raise ValueError(f"expected 624 metrics rows, found {metrics_count}")
    static_rows = [row for row in rows if row["placement"] == "static_hot_cold"]
    baseline_rows = [row for row in rows if row["placement"] == "all_gpu"]
    if len(static_rows) != 576 or len(baseline_rows) != 48:
        raise ValueError(f"expected 576 static + 48 baseline rows, got {len(static_rows)} + {len(baseline_rows)}")


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def plot_run_coverage(rows: list[dict[str, Any]], path: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        key = "GPU baseline" if row["placement"] == "all_gpu" else "Static remote"
        counts[key] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Static remote", "GPU baseline"]
    values = [counts[label] for label in labels]
    bars = ax.bar(labels, values, color=["#4c78a8", "#9ecae9"], edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("completed runs")
    ax.set_title("Remote-memory sum-mode sweep coverage")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, path)


def plot_latency_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    cases = sorted({(row["model"], row["benchmark"]) for row in data}, key=lambda x: (model_sort_key(x[0]), benchmark_sort_key(x[1])))
    columns = [(cold, bw, batch) for cold in COLD_ORDER for bw in BW_ORDER for batch in BATCH_ORDER]
    matrix = np.full((len(cases), len(columns)), np.nan)
    lookup = {(r["model"], r["benchmark"], round(float(r["cold_fraction"]), 2), r["bw_key"], r["batch"]): r for r in data}
    for ri, (model, bench) in enumerate(cases):
        for ci, (cold, bw, batch) in enumerate(columns):
            row = lookup.get((model, bench, cold, bw, batch))
            if row:
                matrix[ri, ci] = float(row["latency_delta_pct"])
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanpercentile(finite, 98))
    vmin = min(0.0, float(np.nanpercentile(finite, 2)))
    fig, ax = plt.subplots(figsize=(24, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"{MODEL_LABELS.get(m, m)} / {BENCH_LABELS.get(b, b)}" for m, b in cases], fontsize=9)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"c={cold:g}\n{BW_LABELS[bw]}TB/s\nb{batch}" for cold, bw, batch in columns], rotation=70, ha="right", fontsize=7)
    for split in [9, 18, 27]:
        ax.axvline(split - 0.5, color="white", linewidth=2.0)
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            value = matrix[ri, ci]
            if np.isfinite(value):
                ax.text(ci, ri, f"{value:.0f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.set_title("Serialized remote memory: latency delta vs matched all-GPU baseline (%)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("latency delta (%)")
    fig.tight_layout()
    save_fig(fig, path)


def plot_cold_fraction_lines(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    agg = aggregate(data, ["model", "cold_fraction"], "latency_delta_pct")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for model in sorted({row["model"] for row in data}, key=model_sort_key):
        series = sorted([row for row in agg if row["model"] == model], key=lambda r: float(r["cold_fraction"]))
        ax.plot(
            [float(row["cold_fraction"]) for row in series],
            [float(row["mean"]) for row in series],
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model, model),
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("cold_fraction")
    ax.set_ylabel("mean latency delta vs GPU (%)")
    ax.set_title("Latency penalty grows with forced cold offload under sum-mode")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)


def plot_bandwidth_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    agg = aggregate(data, ["cold_fraction", "bw_key"], "latency_delta_pct")
    colors = {0.10: "#4c78a8", 0.20: "#72b7b2", 0.25: "#f58518", 0.50: "#e45756"}
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xs = [0, 1, 2]
    for cold in COLD_ORDER:
        series = [next(row for row in agg if round(float(row["cold_fraction"]), 2) == cold and row["bw_key"] == bw) for bw in BW_ORDER]
        ax.plot(xs, [float(row["mean"]) for row in series], marker="o", linewidth=2, color=colors[cold], label=COLD_LABELS[cold])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{BW_LABELS[bw]} TB/s" for bw in BW_ORDER])
    ax.set_xlabel("remote bandwidth per device")
    ax.set_ylabel("mean latency delta vs GPU (%)")
    ax.set_title("Bandwidth reduces serialized remote penalty, especially at high cold fractions")
    ax.grid(True, alpha=0.25)
    ax.legend(title="cold fraction")
    fig.tight_layout()
    save_fig(fig, path)


def plot_pareto_latency_hbm(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    colors = {"bw045": "#d62728", "bw09": "#1f77b4", "bw18": "#2ca02c"}
    markers = {32: "o", 64: "s", 128: "^"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in data:
        ax.scatter(
            row["hbm_saved_pct"],
            row["latency_delta_pct"],
            s=22 + 55 * float(row["cold_fraction"]),
            marker=markers[row["batch"]],
            color=colors[row["bw_key"]],
            alpha=0.42,
            edgecolor="black",
            linewidth=0.2,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("HBM expert memory saved vs all-GPU (%)")
    ax.set_ylabel("latency delta vs all-GPU (%)")
    ax.set_title("Pareto cloud: serialized remote memory trades HBM savings for latency")
    ax.grid(True, alpha=0.23)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[bw], markeredgecolor="black", label=f"{BW_LABELS[bw]} TB/s") for bw in BW_ORDER]
    handles += [Line2D([0], [0], marker=markers[b], color="black", linestyle="", label=f"batch {b}") for b in BATCH_ORDER]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, path)


def _unused_old_plot_oracle_gap_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    cases = sorted({(row["model"], row["benchmark"]) for row in data}, key=lambda x: (model_sort_key(x[0]), benchmark_sort_key(x[1])))
    columns = [(cold, bw) for cold in COLD_ORDER for bw in BW_ORDER]
    matrix = np.full((len(cases), len(columns)), np.nan)
    agg = aggregate(data, ["model", "benchmark", "cold_fraction", "bw_key"], "oracle_gap_pct")
    lookup = {(r["model"], r["benchmark"], round(float(r["cold_fraction"]), 2), r["bw_key"]): r for r in agg}
    for ri, (model, bench) in enumerate(cases):
        for ci, (cold, bw) in enumerate(columns):
            row = lookup.get((model, bench, cold, bw))
            if row:
                matrix[ri, ci] = float(row["mean"])
    finite = matrix[np.isfinite(matrix)]
    vmax = max(1.0, float(np.nanpercentile(finite, 95)))
    fig, ax = plt.subplots(figsize=(14, 8.5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma_r", vmin=0.0, vmax=vmax)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"{MODEL_LABELS.get(m, m)} / {BENCH_LABELS.get(b, b)}" for m, b in cases], fontsize=9)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"c={cold:g}\n{BW_LABELS[bw]}TB/s" for cold, bw in columns], rotation=55, ha="right", fontsize=8)
    for split in [3, 6, 9]:
        ax.axvline(split - 0.5, color="white", linewidth=2.0)
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            value = matrix[ri, ci]
            if np.isfinite(value):
                ax.text(ci, ri, f"{value:.1f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_title("Mean oracle gap by case/cold fraction/bandwidth (%)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("oracle gap (%)")
    fig.tight_layout()
    save_fig(fig, path)


def plot_lane_decomposition(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    agg_gpu = aggregate(data, ["cold_fraction"], "gpu_time_ms")
    agg_off = aggregate(data, ["cold_fraction"], "offload_time_ms")
    x = np.arange(len(COLD_ORDER))
    gpu_vals = [next(row for row in agg_gpu if round(float(row["cold_fraction"]), 2) == cold)["mean"] for cold in COLD_ORDER]
    off_vals = [next(row for row in agg_off if round(float(row["cold_fraction"]), 2) == cold)["mean"] for cold in COLD_ORDER]
    fig, ax = plt.subplots(figsize=(8.3, 5.1))
    ax.bar(x, gpu_vals, color="#4c78a8", label="GPU lane")
    ax.bar(x, off_vals, bottom=gpu_vals, color="#f58518", label="remote lane")
    ax.set_xticks(x)
    ax.set_xticklabels([COLD_LABELS[cold] for cold in COLD_ORDER])
    ax.set_ylabel("mean per-step latency contribution (ms)")
    ax.set_title("Sum-mode lane decomposition: remote time is additive")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)


def plot_cost_pareto(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    colors = {0.10: "#4c78a8", 0.20: "#72b7b2", 0.25: "#f58518", 0.50: "#e45756"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in data:
        cold = round(float(row["cold_fraction"]), 2)
        ax.scatter(
            row["cost_saved_pct"],
            row["tokens_per_s_per_cost_gain_pct"],
            s=30,
            color=colors[cold],
            alpha=0.45,
            edgecolor="black",
            linewidth=0.2,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("relative cost saved vs all-GPU (%)")
    ax.set_ylabel("tokens/s/cost gain vs all-GPU (%)")
    ax.set_title("Cost efficiency Pareto: serialized remote memory")
    ax.grid(True, alpha=0.23)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], markeredgecolor="black", label=COLD_LABELS[c]) for c in COLD_ORDER]
    ax.legend(handles=handles, title="cold fraction", fontsize=8)
    fig.tight_layout()
    save_fig(fig, path)


def _unused_old_plot_cost_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    cases = sorted({(row["model"], row["benchmark"]) for row in data}, key=lambda x: (model_sort_key(x[0]), benchmark_sort_key(x[1])))
    columns = [(cold, bw) for cold in COLD_ORDER for bw in BW_ORDER]
    matrix = np.full((len(cases), len(columns)), np.nan)
    agg = aggregate(data, ["model", "benchmark", "cold_fraction", "bw_key"], "tokens_per_s_per_cost_gain_pct")
    lookup = {(r["model"], r["benchmark"], round(float(r["cold_fraction"]), 2), r["bw_key"]): r for r in agg}
    for ri, (model, bench) in enumerate(cases):
        for ci, (cold, bw) in enumerate(columns):
            row = lookup.get((model, bench, cold, bw))
            if row:
                matrix[ri, ci] = float(row["mean"])
    finite = matrix[np.isfinite(matrix)]
    vmax = max(abs(float(np.nanpercentile(finite, 5))), abs(float(np.nanpercentile(finite, 95))))
    fig, ax = plt.subplots(figsize=(14, 8.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"{MODEL_LABELS.get(m, m)} / {BENCH_LABELS.get(b, b)}" for m, b in cases], fontsize=9)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"c={cold:g}\n{BW_LABELS[bw]}TB/s" for cold, bw in columns], rotation=55, ha="right", fontsize=8)
    for split in [3, 6, 9]:
        ax.axvline(split - 0.5, color="white", linewidth=2.0)
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            value = matrix[ri, ci]
            if np.isfinite(value):
                ax.text(ci, ri, f"{value:.0f}", ha="center", va="center", fontsize=6, color="black")
    ax.set_title("Tokens/s/cost gain vs all-GPU, averaged over batches (%)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label("tokens/s/cost gain (%)")
    fig.tight_layout()
    save_fig(fig, path)


def plot_cost_by_cold_fraction(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    agg = aggregate(data, ["cold_fraction"], "tokens_per_s_per_cost_gain_pct")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    series = sorted(agg, key=lambda r: float(r["cold_fraction"]))
    ax.plot([float(row["cold_fraction"]) for row in series], [float(row["mean"]) for row in series], marker="o", linewidth=2.4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("cold_fraction")
    ax.set_ylabel("mean tokens/s/cost gain vs GPU (%)")
    ax.set_title("Cost efficiency falls as serialized cold offload increases")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, path)


def plot_cost_latency_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    agg = aggregate_many(data, ["cold_fraction"], ["latency_delta_pct", "cost_saved_pct", "tokens_per_s_per_cost_gain_pct"])
    x = np.arange(len(COLD_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    latency = [next(row for row in agg if round(float(row["cold_fraction"]), 2) == cold)["latency_delta_pct_mean"] for cold in COLD_ORDER]
    cost = [next(row for row in agg if round(float(row["cold_fraction"]), 2) == cold)["cost_saved_pct_mean"] for cold in COLD_ORDER]
    tpc = [next(row for row in agg if round(float(row["cold_fraction"]), 2) == cold)["tokens_per_s_per_cost_gain_pct_mean"] for cold in COLD_ORDER]
    ax.bar(x - width, latency, width=width, label="latency delta", color="#e45756")
    ax.bar(x, cost, width=width, label="cost saved", color="#4c78a8")
    ax.bar(x + width, tpc, width=width, label="tokens/s/cost gain", color="#72b7b2")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COLD_LABELS[cold] for cold in COLD_ORDER])
    ax.set_ylabel("mean percent vs all-GPU")
    ax.set_title("System tradeoff by cold fraction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path)


def _unused_old_summarize(rows: list[dict[str, Any]], *, hbm_cost_unit: float, remote_cost_unit: float) -> dict[str, Any]:
    static = [row for row in rows if row["placement"] == "static_hot_cold"]
    baselines = [row for row in rows if row["placement"] == "all_gpu"]
    by_cold = aggregate_many(static, ["cold_fraction"], ["latency_delta_pct", "hbm_saved_pct", "cost_saved_pct", "tokens_per_s_per_cost_gain_pct", "oracle_gap_pct"])
    by_bw = aggregate_many(static, ["bw_key"], ["latency_delta_pct", "tokens_per_s_per_cost_gain_pct", "oracle_gap_pct"])
    by_batch = aggregate_many(static, ["batch"], ["latency_delta_pct", "tokens_per_s_per_cost_gain_pct"])
    best_cost = max(static, key=lambda row: float(row["tokens_per_s_per_cost_gain_pct"]))
    best_latency = min(static, key=lambda row: float(row["latency_delta_pct"]))
    worst_latency = max(static, key=lambda row: float(row["latency_delta_pct"]))
    low_penalty = [row for row in static if float(row["latency_delta_pct"]) <= 10.0]
    return {
        "coverage": {
            "total_metrics": len(rows),
            "static_runs": len(static),
            "gpu_baselines": len(baselines),
            "models": sorted({row["model"] for row in rows}, key=model_sort_key),
            "benchmarks": sorted({row["benchmark"] for row in rows}, key=benchmark_sort_key),
            "batches": BATCH_ORDER,
            "bandwidths": BW_ORDER,
            "cold_fractions": COLD_ORDER,
        },
        "cost_model": {
            "hbm_cost_unit": hbm_cost_unit,
            "remote_cost_unit": remote_cost_unit,
            "hbm_to_remote_ratio": hbm_cost_unit / remote_cost_unit,
        },
        "overall": {
            "latency_delta_pct_mean": mean(row["latency_delta_pct"] for row in static),
            "latency_delta_pct_p50": percentile([row["latency_delta_pct"] for row in static], 50),
            "latency_delta_pct_p90": percentile([row["latency_delta_pct"] for row in static], 90),
            "hbm_saved_pct_mean": mean(row["hbm_saved_pct"] for row in static),
            "cost_saved_pct_mean": mean(row["cost_saved_pct"] for row in static),
            "tokens_per_s_per_cost_gain_pct_mean": mean(row["tokens_per_s_per_cost_gain_pct"] for row in static),
            "oracle_gap_pct_mean": mean(row["oracle_gap_pct"] for row in static),
            "low_penalty_run_count_le_10pct": len(low_penalty),
            "low_penalty_fraction_le_10pct": len(low_penalty) / len(static),
        },
        "by_cold_fraction": by_cold,
        "by_bandwidth": by_bw,
        "by_batch": by_batch,
        "best_tokens_per_cost": pick_summary_row(best_cost),
        "best_latency": pick_summary_row(best_latency),
        "worst_latency": pick_summary_row(worst_latency),
        "cost_efficiency": {
            "by_cold_fraction": by_cold,
            "best_tokens_per_cost": pick_summary_row(best_cost),
            "mean_tokens_per_s_per_cost_gain_pct": mean(row["tokens_per_s_per_cost_gain_pct"] for row in static),
        },
    }


def _unused_old_build_plot_stories(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    by_cold = summary["by_cold_fraction"]
    best_cost = summary["best_tokens_per_cost"]
    worst = summary["worst_latency"]
    lines = [
        "# Remote Memory Sum-Mode Plot Stories",
        "",
        "These plots focus on fixed static hot/cold placement because sum-mode serialized remote fetches make latency-only adaptive placement collapse toward all-GPU.",
        "",
        "## Main Plots",
        "",
        "- `01_run_coverage.png`: verifies the study has the intended 576 static remote-memory runs and 48 matched GPU baselines.",
        "- `02_latency_delta_heatmap.png`: shows which model/workload, cold fraction, bandwidth, and batch combinations pay the largest serialized latency penalty.",
        "- `03_latency_by_cold_fraction.png`: isolates the main systems knob; increasing cold fraction saves more HBM but raises serialized latency.",
        "- `04_bandwidth_sensitivity.png`: shows remote bandwidth can reduce the sum-mode penalty, but it cannot hide remote time the way max-mode could.",
        "- `05_pareto_latency_vs_hbm_saved.png`: is the primary capacity/performance Pareto plot for the final story.",
        "- `06_oracle_gap_heatmap.png`: sanity-checks the gap to the placement-independent sum-mode lower bound; it largely mirrors serialized offload penalty because the oracle can avoid remote fetches.",
        "- `07_gpu_remote_lane_decomposition.png`: makes the new assumption visually explicit: GPU and remote lanes stack in sum-mode.",
        "",
        "## Cost-Efficiency Demo",
        "",
        "- `01_pareto_tokens_per_cost_vs_cost_saved.png`: shows whether cheaper remote memory compensates for lower throughput.",
        "- `02_tokens_per_cost_gain_heatmap.png`: identifies cases where cost efficiency is preserved or lost.",
        "- `03_tokens_per_cost_by_cold_fraction.png`: summarizes the cost-efficiency slope as cold fraction increases.",
        "- `04_latency_cost_tradeoff_by_cold_fraction.png`: gives the headline tradeoff between latency, cost saving, and tokens/s/cost.",
        "",
        "## Initial Read",
        "",
        f"- Mean latency delta across static remote runs is `{overall['latency_delta_pct_mean']:.2f}%`; p90 is `{overall['latency_delta_pct_p90']:.2f}%`.",
        f"- Mean HBM saving is `{overall['hbm_saved_pct_mean']:.2f}%`, while mean cost saving is `{overall['cost_saved_pct_mean']:.2f}%` under the plotted cost model.",
        f"- Only `{overall['low_penalty_run_count_le_10pct']}` of 576 static runs stay within 10% latency penalty.",
        f"- Best tokens/s/cost case is `{best_cost['run_id']}` with `{best_cost['tokens_per_s_per_cost_gain_pct']:.2f}%` gain.",
        f"- Worst latency case is `{worst['run_id']}` with `{worst['latency_delta_pct']:.2f}%` latency delta.",
        "",
        "## Cold-Fraction Means",
        "",
    ]
    for row in sorted(by_cold, key=lambda item: float(item["cold_fraction"])):
        lines.append(
            f"- c={float(row['cold_fraction']):.2f}: latency delta `{row['latency_delta_pct_mean']:.2f}%`, "
            f"HBM saved `{row['hbm_saved_pct_mean']:.2f}%`, tokens/s/cost gain `{row['tokens_per_s_per_cost_gain_pct_mean']:.2f}%`."
        )
    lines.append("")
    return "\n".join(lines)


def aggregate(rows: list[dict[str, Any]], keys: list[str], metric: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(float(row[metric]))
    out: list[dict[str, Any]] = []
    for key_values, values in buckets.items():
        item = {key: value for key, value in zip(keys, key_values)}
        item.update({"mean": mean(values), "count": len(values)})
        out.append(item)
    return out


def aggregate_many(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, bucket in buckets.items():
        item = {key: value for key, value in zip(keys, key_values)}
        item["count"] = len(bucket)
        for metric in metrics:
            values = [float(row[metric]) for row in bucket]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_p50"] = percentile(values, 50)
        out.append(item)
    return out


def pick_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "run_id",
        "model",
        "benchmark",
        "bw_key",
        "batch",
        "cold_fraction",
        "latency_delta_pct",
        "hbm_saved_pct",
        "cost_saved_pct",
        "tokens_per_s_per_cost_gain_pct",
        "oracle_gap_pct",
    ]
    return {key: row[key] for key in keys}


def costed_memory_gb(placement_mode: str, memory: dict[str, Any]) -> tuple[float, float]:
    if placement_mode == "all_gpu":
        return float(memory["baseline_all_gpu_hbm_gb"]), 0.0
    return (
        float(memory.get("hbm_expert_gb", 0.0) or 0.0),
        float(memory.get("remote_expert_gb", memory.get("offload_expert_gb", 0.0)) or 0.0),
    )


def save_fig(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def model_slug_from_id(model_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    for slug in MODEL_LABELS:
        if slug in normalized:
            return slug
    aliases = {
        "qwen3_235b_a22b_fp8": "qwen3_235b_a22b_fp8",
        "deepseek_r1_awq": "deepseek_r1_awq",
        "kimi_k2_thinking": "kimi_k2_thinking",
        "llama_4_maverick_17b_128e_instruct": "llama_4_maverick_17b_128e_instruct",
    }
    for needle, slug in aliases.items():
        if needle in normalized:
            return slug
    return model_id.split("/")[-1]


def bandwidth_key(hardware_id: str) -> str:
    if "bw045" in hardware_id:
        return "bw045"
    if "bw18" in hardware_id:
        return "bw18"
    return "bw09"


def bandwidth_value(hardware_id: str) -> float:
    return {"bw045": 0.45, "bw09": 0.90, "bw18": 1.80}[bandwidth_key(hardware_id)]


def model_sort_key(model: str) -> int:
    return list(MODEL_LABELS).index(model) if model in MODEL_LABELS else len(MODEL_LABELS)


def benchmark_sort_key(benchmark: str) -> int:
    order = ["Chinese-SimpleQA", "hellaswag", "livecodebench", "mmlu", "mmlu_ZH_CN"]
    return order.index(benchmark) if benchmark in order else len(order)


def none_to_empty(value: Any) -> Any:
    return "" if value is None else value


def mean(values: Any) -> float:
    seq = [float(v) for v in values]
    return sum(seq) / len(seq) if seq else 0.0


def percentile(values: Any, q: float) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    idx = (len(seq) - 1) * q / 100.0
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return seq[lo]
    return seq[lo] * (hi - idx) + seq[hi] * (idx - lo)


def clear_plot_outputs(path: Path) -> None:
    for pattern in ("*.png", "*.md", "*.json", "*.csv"):
        for item in path.glob(pattern):
            item.unlink()


def matrix_for_metric(
    rows: list[dict[str, Any]],
    metric: str,
) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[float, str, int]]]:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    cases = sorted({(row["model"], row["benchmark"]) for row in data}, key=lambda x: (model_sort_key(x[0]), benchmark_sort_key(x[1])))
    columns = [(cold, bw, batch) for cold in COLD_ORDER for bw in BW_ORDER for batch in BATCH_ORDER]
    matrix = np.full((len(cases), len(columns)), np.nan)
    lookup = {
        (row["model"], row["benchmark"], round(float(row["cold_fraction"]), 2), row["bw_key"], row["batch"]): row
        for row in data
    }
    for ri, (model, bench) in enumerate(cases):
        for ci, (cold, bw, batch) in enumerate(columns):
            row = lookup.get((model, bench, cold, bw, batch))
            if row:
                matrix[ri, ci] = float(row[metric])
    return matrix, cases, columns


def plot_metric_full_heatmap(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    metric: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    diverging: bool = False,
) -> None:
    matrix, cases, columns = matrix_for_metric(rows, metric)
    finite = matrix[np.isfinite(matrix)]
    if diverging:
        limit = max(abs(float(np.nanpercentile(finite, 2))), abs(float(np.nanpercentile(finite, 98))), 1.0)
        vmin, vmax = -limit, limit
    else:
        vmin = min(0.0, float(np.nanpercentile(finite, 2)))
        vmax = max(1.0, float(np.nanpercentile(finite, 98)))
    fig, ax = plt.subplots(figsize=(24, 9))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([f"{MODEL_LABELS.get(m, m)} / {BENCH_LABELS.get(b, b)}" for m, b in cases], fontsize=9)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"c={cold:g}\n{BW_LABELS[bw]}TB/s\nb{batch}" for cold, bw, batch in columns], rotation=70, ha="right", fontsize=7)
    for split in [9, 18, 27]:
        ax.axvline(split - 0.5, color="white", linewidth=2.0)
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            value = matrix[ri, ci]
            if np.isfinite(value):
                ax.text(ci, ri, f"{value:.0f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.84)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    save_fig(fig, path)


def plot_latency_case_facets(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    cases = sorted({(row["model"], row["benchmark"]) for row in data}, key=lambda x: (model_sort_key(x[0]), benchmark_sort_key(x[1])))
    colors = {"bw045": "#d62728", "bw09": "#1f77b4", "bw18": "#2ca02c"}
    markers = {32: "o", 64: "s", 128: "^"}
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for ax, (model, bench) in zip(axes.ravel(), cases):
        subset = [row for row in data if row["model"] == model and row["benchmark"] == bench]
        for bw in BW_ORDER:
            for batch in BATCH_ORDER:
                series = sorted(
                    [row for row in subset if row["bw_key"] == bw and row["batch"] == batch],
                    key=lambda row: float(row["cold_fraction"]),
                )
                ax.plot(
                    [float(row["cold_fraction"]) for row in series],
                    [float(row["latency_delta_pct"]) for row in series],
                    color=colors[bw],
                    marker=markers[batch],
                    linewidth=0.9,
                    markersize=3.5,
                    alpha=0.75,
                )
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(f"{MODEL_LABELS.get(model, model)} / {BENCH_LABELS.get(bench, bench)}", fontsize=9)
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("cold_fraction")
    for ax in axes[:, 0]:
        ax.set_ylabel("latency delta (%)")
    handles = [Line2D([0], [0], color=colors[bw], marker="o", label=f"{BW_LABELS[bw]} TB/s") for bw in BW_ORDER]
    handles += [Line2D([0], [0], color="black", marker=markers[b], linestyle="", label=f"batch {b}") for b in BATCH_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9)
    fig.suptitle("Per-case latency curves; every marker is one run", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_fig(fig, path)


def fixed_mmlu_b128_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["placement"] == "static_hot_cold" and row["benchmark"] == "mmlu" and row["batch"] == 128
    ]


def plot_fixed_mmlu_b128_latency(rows: list[dict[str, Any]], path: Path) -> None:
    data = fixed_mmlu_b128_rows(rows)
    colors = {"bw045": "#d62728", "bw09": "#1f77b4", "bw18": "#2ca02c"}
    models = sorted({row["model"] for row in data}, key=model_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, model in zip(axes.ravel(), models):
        subset = [row for row in data if row["model"] == model]
        for bw in BW_ORDER:
            series = sorted([row for row in subset if row["bw_key"] == bw], key=lambda row: float(row["cold_fraction"]))
            ax.plot(
                [float(row["cold_fraction"]) for row in series],
                [float(row["latency_delta_pct"]) for row in series],
                marker="o",
                color=colors[bw],
                label=f"{BW_LABELS[bw]} TB/s",
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} / MMLU / batch128")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("cold_fraction")
    for ax in axes[:, 0]:
        ax.set_ylabel("latency delta (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-slice bandwidth sensitivity; no cross-workload averaging")
    fig.tight_layout()
    save_fig(fig, path)


def plot_oracle_gap_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    plot_metric_full_heatmap(
        rows,
        path,
        metric="oracle_gap_pct",
        title="Oracle gap for every static run (%)",
        colorbar_label="oracle gap (%)",
        cmap="magma_r",
    )


def plot_fixed_mmlu_b128_lane_decomposition(rows: list[dict[str, Any]], path: Path) -> None:
    data = fixed_mmlu_b128_rows(rows)
    models = sorted({row["model"] for row in data}, key=model_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    width = 0.22
    offsets = {"bw045": -width, "bw09": 0.0, "bw18": width}
    colors = {"bw045": "#d62728", "bw09": "#1f77b4", "bw18": "#2ca02c"}
    x = np.arange(len(COLD_ORDER))
    for ax, model in zip(axes.ravel(), models):
        subset = [row for row in data if row["model"] == model]
        for bw in BW_ORDER:
            series = sorted([row for row in subset if row["bw_key"] == bw], key=lambda row: float(row["cold_fraction"]))
            gpu = [float(row["gpu_time_ms"]) for row in series]
            off = [float(row["offload_time_ms"]) for row in series]
            xpos = x + offsets[bw]
            ax.bar(xpos, gpu, width=width, color=colors[bw], alpha=0.45)
            ax.bar(xpos, off, bottom=gpu, width=width, color=colors[bw], alpha=0.9, hatch="//")
        ax.set_title(f"{MODEL_LABELS.get(model, model)} / MMLU / batch128")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{cold:g}" for cold in COLD_ORDER])
    for ax in axes[-1, :]:
        ax.set_xlabel("cold_fraction")
    for ax in axes[:, 0]:
        ax.set_ylabel("latency contribution (ms)")
    handles = [Line2D([0], [0], color=colors[bw], linewidth=8, label=f"{BW_LABELS[bw]} TB/s") for bw in BW_ORDER]
    handles += [Line2D([0], [0], color="gray", linewidth=8, label="solid=GPU, hatched=remote")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Fixed-slice sum-mode lane decomposition; every bar is one run", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, path)


def plot_cost_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    plot_metric_full_heatmap(
        rows,
        path,
        metric="tokens_per_s_per_cost_gain_pct",
        title="Tokens/s/cost gain for every static run (%)",
        colorbar_label="tokens/s/cost gain (%)",
        cmap="RdYlGn",
        diverging=True,
    )


def plot_cost_case_facets(rows: list[dict[str, Any]], path: Path) -> None:
    data = [row for row in rows if row["placement"] == "static_hot_cold"]
    cases = sorted({(row["model"], row["benchmark"]) for row in data}, key=lambda x: (model_sort_key(x[0]), benchmark_sort_key(x[1])))
    colors = {"bw045": "#d62728", "bw09": "#1f77b4", "bw18": "#2ca02c"}
    markers = {32: "o", 64: "s", 128: "^"}
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    for ax, (model, bench) in zip(axes.ravel(), cases):
        subset = [row for row in data if row["model"] == model and row["benchmark"] == bench]
        for row in subset:
            ax.scatter(
                row["cost_saved_pct"],
                row["tokens_per_s_per_cost_gain_pct"],
                color=colors[row["bw_key"]],
                marker=markers[row["batch"]],
                s=22 + 45 * float(row["cold_fraction"]),
                alpha=0.65,
                edgecolor="black",
                linewidth=0.2,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} / {BENCH_LABELS.get(bench, bench)}", fontsize=9)
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("cost saved (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("tokens/s/cost gain (%)")
    handles = [Line2D([0], [0], color=colors[bw], marker="o", linestyle="", label=f"{BW_LABELS[bw]} TB/s") for bw in BW_ORDER]
    handles += [Line2D([0], [0], color="black", marker=markers[b], linestyle="", label=f"batch {b}") for b in BATCH_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=9)
    fig.suptitle("Per-case cost-efficiency cloud; marker size increases with cold_fraction", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_fig(fig, path)


def plot_fixed_mmlu_b128_cost_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    data = fixed_mmlu_b128_rows(rows)
    models = sorted({row["model"] for row in data}, key=model_sort_key)
    colors = {"latency_delta_pct": "#e45756", "cost_saved_pct": "#4c78a8", "tokens_per_s_per_cost_gain_pct": "#72b7b2"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    width = 0.22
    x = np.arange(len(COLD_ORDER))
    for ax, model in zip(axes.ravel(), models):
        subset = [
            row
            for row in data
            if row["model"] == model and row["bw_key"] == "bw18"
        ]
        series = sorted(subset, key=lambda row: float(row["cold_fraction"]))
        for i, metric in enumerate(["latency_delta_pct", "cost_saved_pct", "tokens_per_s_per_cost_gain_pct"]):
            ax.bar(
                x + (i - 1) * width,
                [float(row[metric]) for row in series],
                width=width,
                color=colors[metric],
                label=metric.replace("_pct", "").replace("_", " "),
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(f"{MODEL_LABELS.get(model, model)} / MMLU / batch128 / 1.80 TB/s")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{cold:g}" for cold in COLD_ORDER])
    for ax in axes[-1, :]:
        ax.set_xlabel("cold_fraction")
    for ax in axes[:, 0]:
        ax.set_ylabel("percent vs all-GPU")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-slice cost/latency tradeoff; no averaging")
    fig.tight_layout()
    save_fig(fig, path)


def summarize(rows: list[dict[str, Any]], *, hbm_cost_unit: float, remote_cost_unit: float) -> dict[str, Any]:
    static = [row for row in rows if row["placement"] == "static_hot_cold"]
    baselines = [row for row in rows if row["placement"] == "all_gpu"]
    threshold_counts = {
        "latency_delta_le_5pct": sum(1 for row in static if float(row["latency_delta_pct"]) <= 5.0),
        "latency_delta_le_10pct": sum(1 for row in static if float(row["latency_delta_pct"]) <= 10.0),
        "latency_delta_le_25pct": sum(1 for row in static if float(row["latency_delta_pct"]) <= 25.0),
        "tokens_per_cost_gain_gt_0pct": sum(1 for row in static if float(row["tokens_per_s_per_cost_gain_pct"]) > 0.0),
    }
    threshold_counts_by_cold: list[dict[str, Any]] = []
    for cold in COLD_ORDER:
        subset = [row for row in static if round(float(row["cold_fraction"]), 2) == cold]
        threshold_counts_by_cold.append(
            {
                "cold_fraction": cold,
                "run_count": len(subset),
                "latency_delta_le_10pct": sum(1 for row in subset if float(row["latency_delta_pct"]) <= 10.0),
                "tokens_per_cost_gain_gt_0pct": sum(1 for row in subset if float(row["tokens_per_s_per_cost_gain_pct"]) > 0.0),
            }
        )
    best_cost = max(static, key=lambda row: float(row["tokens_per_s_per_cost_gain_pct"]))
    best_latency = min(static, key=lambda row: float(row["latency_delta_pct"]))
    worst_latency = max(static, key=lambda row: float(row["latency_delta_pct"]))
    return {
        "coverage": {
            "total_metrics": len(rows),
            "static_runs": len(static),
            "gpu_baselines": len(baselines),
            "models": sorted({row["model"] for row in rows}, key=model_sort_key),
            "benchmarks": sorted({row["benchmark"] for row in rows}, key=benchmark_sort_key),
            "batches": BATCH_ORDER,
            "bandwidths": BW_ORDER,
            "cold_fractions": COLD_ORDER,
        },
        "cost_model": {
            "hbm_cost_unit": hbm_cost_unit,
            "remote_cost_unit": remote_cost_unit,
            "hbm_to_remote_ratio": hbm_cost_unit / remote_cost_unit,
        },
        "threshold_counts": threshold_counts,
        "threshold_counts_by_cold": threshold_counts_by_cold,
        "best_tokens_per_cost": pick_summary_row(best_cost),
        "best_latency": pick_summary_row(best_latency),
        "worst_latency": pick_summary_row(worst_latency),
        "cost_efficiency": {
            "threshold_counts": threshold_counts,
            "threshold_counts_by_cold": threshold_counts_by_cold,
            "best_tokens_per_cost": pick_summary_row(best_cost),
        },
    }


def build_plot_stories(summary: dict[str, Any]) -> str:
    counts = summary["threshold_counts"]
    best_cost = summary["best_tokens_per_cost"]
    best_latency = summary["best_latency"]
    worst = summary["worst_latency"]
    lines = [
        "# Remote Memory Sum-Mode Plot Stories",
        "",
        "No plot in this folder averages metrics across different workloads, models, bandwidths, or batch sizes. Heatmap cells, scatter points, and fixed-slice bars are individual simulation runs; threshold summaries are counts.",
        "",
        "## Main Plots",
        "",
        "- `01_run_coverage.png`: verifies 576 static remote runs and 48 matched GPU baselines.",
        "- `02_latency_delta_heatmap.png`: one cell per static run, with columns carrying cold_fraction, bandwidth, and batch.",
        "- `03_latency_case_facets.png`: one subplot per model/workload; every marker is one run, with color for bandwidth and marker shape for batch.",
        "- `04_fixed_mmlu_b128_latency_slices.png`: fixed benchmark/batch slice showing per-model bandwidth sensitivity without mixing workloads.",
        "- `05_pareto_latency_vs_hbm_saved.png`: one point per static run for the capacity/performance frontier.",
        "- `06_oracle_gap_heatmap.png`: one cell per static run for gap to the sum-mode oracle.",
        "- `07_fixed_mmlu_b128_lane_decomposition.png`: fixed benchmark/batch slice showing additive GPU and remote lanes.",
        "",
        "## Cost-Efficiency Demo",
        "",
        "- `01_pareto_tokens_per_cost_vs_cost_saved.png`: one point per static run.",
        "- `02_tokens_per_cost_gain_heatmap.png`: one cell per static run.",
        "- `03_cost_efficiency_case_facets.png`: one subplot per model/workload; marker size reflects cold_fraction.",
        "- `04_fixed_mmlu_b128_cost_tradeoff.png`: fixed MMLU/batch128/bw18 slice; every bar is one run.",
        "",
        "## Count-Based Read",
        "",
        f"- Runs within 5% latency penalty: `{counts['latency_delta_le_5pct']}` / 576.",
        f"- Runs within 10% latency penalty: `{counts['latency_delta_le_10pct']}` / 576.",
        f"- Runs within 25% latency penalty: `{counts['latency_delta_le_25pct']}` / 576.",
        f"- Runs with positive tokens/s/cost gain: `{counts['tokens_per_cost_gain_gt_0pct']}` / 576.",
        f"- Best latency run: `{best_latency['run_id']}` at `{best_latency['latency_delta_pct']:.2f}%` latency delta.",
        f"- Best tokens/s/cost run: `{best_cost['run_id']}` at `{best_cost['tokens_per_s_per_cost_gain_pct']:.2f}%` gain.",
        f"- Worst latency run: `{worst['run_id']}` at `{worst['latency_delta_pct']:.2f}%` latency delta.",
        "",
        "## Threshold Counts By Cold Fraction",
        "",
    ]
    for row in summary["threshold_counts_by_cold"]:
        lines.append(
            f"- c={row['cold_fraction']:.2f}: `{row['latency_delta_le_10pct']}` / {row['run_count']} runs within 10% latency, "
            f"`{row['tokens_per_cost_gain_gt_0pct']}` / {row['run_count']} runs with positive tokens/s/cost gain."
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
