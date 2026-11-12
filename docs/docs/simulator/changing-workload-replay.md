---
title: Changing-workload expert replay
sidebar_position: 12
---

# Changing-workload expert replay

Recorded expert replay keeps serving queues and GPU/PIM residency alive while
the request mix changes. The supported group migration policy is anchored
frequency hysteresis. Fixed placement remains available with migration disabled;
both modes retain the same profile-based scheduling choice for BOTH experts.

## Inputs and execution

Run `python -m serving` with `--expert-routing-policy TRACE`, a cluster config,
`--expert-policy-config`, `--expert-trace-pack`, and `--dataset`. For example:

```bash
python -m serving \
  --cluster-config configs/cluster/pim_experiments/expert_replay/qwen3_235b_h200_4p4d_8x16.json \
  --expert-routing-policy TRACE \
  --expert-trace-pack workloads/moe_replay/collection.json \
  --expert-policy-config configs/expert_replay/migration.json \
  --dataset workloads/moe_replay/requests.jsonl \
  --max-num-seqs 128 \
  --output outputs/moe_replay/sim.csv
```

Per-instance sequence limits in the cluster config take precedence over CLI
limits. Set the decode instance's `max_num_seqs` to the desired total cap in
the single-decode layout. The supplied layouts demonstrate different capacities;
inspect their instance settings before running a comparison.

The migration preset reads separate initialization request IDs from
`workloads/moe_replay/calibration.json`. Supply these through `calibration_file`
or `calibration_request_ids`, and set `calibration_batch_size` to the desired
initialization batch cap. Evaluation and calibration requests must be disjoint.

Every request specifies `input_toks`, `output_toks`, `arrival_time_ns`, and
`expert_trace_request_id`. Replay uses recorded routes for every requested decode
position. The first output token is produced by prefill: D recorded decode
forwards correspond to D+1 output tokens. To replay a complete recording, use
that request's full recorded decode length. Domain changes do not reset queues,
placement, anchors, or history.

## Several packs without copying routes

`--expert-trace-pack` accepts a TracePack directory or a collection manifest:

```json
{
  "schema_version": "trace_pack_collection_v1",
  "packs": {
    "english": "../../trace_packs/mmlu",
    "chinese": "../../trace_packs/Chinese-SimpleQA"
  }
}
```

Paths are relative to the collection manifest. Request IDs become
`english::original_request_id`. Sources must agree on model ID, layers, expert
counts, and top-k. Route arrays remain read-only memory maps. Optional
`workload_phase` labels are reporting metadata; online placement receives only
completed routing observations, without labels or future routes.

## Anchored frequency hysteresis

`configs/expert_replay/migration.json` is the runnable policy preset.
`migration.mode: "none"` disables replacement. To enable hysteresis, use:

```json
{
  "migration": {
    "mode": "frequency_hysteresis",
    "interval": 64,
    "fast_window": 8,
    "cooldown": 4
  },
  "hysteresis_options": {
    "relative_margin": 0.25,
    "persistence": 1,
    "gate": "anchored"
  }
}
```

At each scheduled check, average normalized expert frequencies from the latest
completed forwards. A batch means one serving forward, not one new request.
Compare this distribution with an anchor initialized from calibration routes.
The gate opens only when per-layer total-variation distance exceeds the larger
of `gate_floor` and the `noise_quantile` of calibration-window distances.
The implementation owns these defaults in `serving/core/expert_frequency.py`.

An incoming expert must exceed the incumbent frequency by the relative margin
for the configured number of consecutive checks. Rank GPU-only, BOTH, and
PIM-only residency by frequency while retaining incumbents that fail this test.
Cooldown prevents recently moved experts from immediately moving again.
The gate, margin, and persistence serve different purposes: detect distribution
change, require a meaningful advantage, and filter short-lived advantages.

Each changed layer proposes one complete transaction. GPU/PIM/BOTH counts,
duplicate budgets, and endpoint slot counts remain fixed. New copies inherit
vacated slots; unrelated experts retain their physical owners. Anchors,
cooldowns, replacement counts, and reversal counts change only on actual commits.
Stable noisy demand can still cross a finite threshold; zero replacements are
not guaranteed for every workload.

## Local SSD admission

Every decode GPU and PIM group has its own SSD containing immutable expert weights.
Set `ssd_read_budget_gib` to a positive per-device allowance, or leave it null for
unlimited admission. All layers on an endpoint share the allowance. Budgets
reset at integer simulated seconds; unused allowance expires. Initial placement
loading is excluded.

The default `ssd_queue_policy` is `newest_first`. An entire layer update commits
only when every affected endpoint can admit all incoming copy bytes. Otherwise
serving continues with existing residency. A new target replaces obsolete pending
work; identical targets keep their original queue age. Newer targets are tried
first, ties use layer ID, and temporarily infeasible targets are skipped.
Targets are revalidated at policy checks. Budget retries do not advance persistence.

Admitted SSD reads are fully hidden: simulated loading delay and network weight
traffic are zero. Activation communication is charged normally. This is a
fixed-window admission model, not a paced transfer or PCIe-contention model.

## Outputs

The request CSV and neighboring `.replay.json` contain throughput, latency,
queueing, per-forward timing, physical residency, GPU/PIM work, gate checks,
committed replacements, reversals, and local SSD read totals. Capped runs also
report device windows, pending/cancelled bytes, and commit delays. Controller CPU
and wall time are reported separately from simulated serving time.

If the backend exits before replay completes, the input directory retains
`backend_failure.json` with its exit code, terminating signal (if any), completed
request/forward counts, and simulated time. A signal alone does not establish
whether memory pressure caused the termination.

Group replay also reports `pim_weight_read_quanta_by_group` and
`pim_weight_read_bytes_by_group`. An expert with N tokens contributes
`ceil(N / granularity)` expert-weight reads per forward. These modeled PIM
reads are separate from SSD reads used to replace resident copies. Granularity
changes weight-read sharing; it does not add an explicit per-bank compute model.
The optional compiled scheduler supports batches through 256 tokens and retains
the Python scheduler as its reference.

Enable `record_workload_phases` to include executed domain composition and arrival
boundaries. Enable `record_route_batches` with phase reporting to save completed
request/position/source-rank observations in `.routes.jsonl`. The placement CSV
records final physical owners. Use `--save-trace-text` and retained input graphs
when inspecting activation traffic. No experiment-specific plotting or report
generator is required to obtain these outputs.

See [MoE expert routing](./moe-expert-routing.md) for measured GPU-profile
interpretation, analytical PIM equations, communication, and timing limitations.


## Mixed dense and MoE models

A model configuration can provide `moe_layer_ids`, an ordered list of physical
transformer-layer indices. Replay validates the full TracePack shape, but only
these layers contribute expert routes, calibration, placement, and movement.
Dense-layer placeholder routes do not represent expert executions. Placement
CSV rows report physical layer indices; graph audits also include `model_layer`.
`expert_trace_model_id` explicitly identifies a trace source when a measured
profile uses a different publisher or quantization suffix.

`expert_weight_bits` specifies packed expert storage independently of activation
precision; `linear_weight_bits` does the same for linear projections. Packed
storage excludes quantization metadata. Models using MLA can opt into
`compressed_kv_cache` to account for the replicated latent KV and rotary cache.

The added H200 DeepSeek R1 AWQ, Kimi K2 Thinking FP8, and Llama 4 Maverick FP8
configurations use measured MoE and attention profiles. The MLA architecture declares `attention_output_allreduce` because its measured
wrapper contains the output projection compute but excludes multi-GPU
communication; the simulator emits that TP reduction once after attention.
Their bundles currently lack dense FFN measurements. An explicit `dense_ffn_roofline` model setting fills
only missing `gate_up_proj`, `act_fn`, and `down_proj` tables, at each TP degree,
using the maximum of ideal compute time and HBM traffic time. Existing measured
rows always take precedence. The setting specifies hardware, stored weight and
activation widths, non-sparse compute peaks, HBM bandwidth, and token grid.
Runtime warnings and `.replay.json` identify the estimates. These ideal estimates
exclude launch and quantization overheads; they are not measured dense FFN
profiles. Other model configurations retain their existing lookup behavior.
