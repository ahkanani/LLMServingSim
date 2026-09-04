# configs/cluster

This directory contains cluster configuration files that define the hardware topology,
instance layout, memory hierarchy, and interconnect parameters for LLMServingSim.

Pass a config file to `python -m serving` via `--cluster-config configs/cluster/{name}.json`.

## Configuration format

```json
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 0,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {
        "mem_size": 512,
        "mem_bw": 256,
        "mem_latency": 0
      },
      "instances": [
        {
          "model_name": "Qwen/Qwen3-32B",
          "hardware": "RTXPRO6000",
          "npu_mem": {
            "mem_size": 96,
            "mem_bw": 1597,
            "mem_latency": 0
          },
          "num_npus": 2,
          "tp_size": 2,
          "pd_type": null
        }
      ]
    }
  ]
}
```

### Top-level fields

| Field | Type | Description |
| --- | --- | --- |
| `num_nodes` | Integer | Number of nodes in the cluster |
| `link_bw` | Float or Array<Float> | ASTRA-Sim topology link bandwidth in GB/s. A scalar is broadcast to all topology dimensions; an array must match the final `npus_count` rank |
| `link_latency` | Float or Array<Float> | ASTRA-Sim topology link latency in ns. A scalar is broadcast to all topology dimensions; an array must match the final `npus_count` rank |
| `network_topology` | String or Array<String> | ASTRA-Sim analytical topology. Supported values: `Ring`, `Switch`, `FullyConnected`. A scalar is broadcast to all topology dimensions; an array must match the final `npus_count` rank. Defaults to `FullyConnected` for backward compatibility |

### Per-node fields

| Field | Type | Description |
| --- | --- | --- |
| `num_instances` | Integer | Number of instances on this node |
| `cpu_mem.mem_size` | Float | CPU memory capacity in GB |
| `cpu_mem.mem_bw` | Float | CPU memory bandwidth in GB/s |
| `cpu_mem.mem_latency` | Float | CPU memory latency in ns |

### Per-instance fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model_name` | String | Yes | HuggingFace model identifier (must match `configs/model/`) |
| `hardware` | String | Yes | Hardware name matching `profiler/perf_models/{hardware}/` |
| `npu_mem` | Object | Yes | NPU memory config (`mem_size` in GB, `mem_bw` in GB/s, `mem_latency` in ns) |
| `pd_type` | String/null | Yes | `"prefill"`, `"decode"`, or `null` for combined |
| `num_npus` | Integer | * | Total GPUs for this instance (inferred from `tp_size * pp_size` if omitted) |
| `tp_size` | Integer | * | Tensor parallel degree (inferred from `num_npus // pp_size` if omitted) |
| `pp_size` | Integer | No | Pipeline parallel degree (default: 1) |
| `ep_size` | Integer | No | Expert parallel degree (default: `tp_size` for MoE, 1 for dense) |
| `dp_group` | String/null | No | DP group ID. Instances with the same string share experts via cross-instance ALLTOALL |
| `max_num_seqs` | Integer | No | Per-instance override for `--max-num-seqs` (`0` = unlimited) |
| `max_num_batched_tokens` | Integer | No | Per-instance override for `--max-num-batched-tokens` (`0` = unlimited) |
| `long_prefill_token_threshold` | Integer | No | Per-instance override for `--long-prefill-token-threshold` |
| `block_size` | Integer | No | Per-instance override for `--block-size` |
| `dtype` | String | No | Per-instance override for `--dtype` |
| `kv_cache_dtype` | String | No | Per-instance override for `--kv-cache-dtype` |
| `enable_chunked_prefill` | Boolean | No | Per-instance override for `--enable-chunked-prefill` |
| `enable_prefix_caching` | Boolean | No | Per-instance override for `--enable-prefix-caching` |
| `prioritize_prefill` | Boolean | No | Per-instance override for `--prioritize-prefill` |
| `enable_local_offloading` | Boolean | No | Per-instance override for `--enable-local-offloading` |
| `enable_attn_offloading` | Boolean | No | Per-instance override for `--enable-attn-offloading` |
| `enable_expert_offloading` | Boolean | No | Per-instance override for `--enable-expert-offloading` |
| `expert_offload` | Object | No | Runtime options for expert PIM offload support, including custom routing distribution paths |
| `expert_routing_distribution_csv` | String | No | Optional shorthand for `expert_offload.routing_distribution_csv`; path to a custom expert-routing distribution CSV |
| `enable_sub_batch_interleaving` | Boolean | No | Per-instance override for `--enable-sub-batch-interleaving` |
| `enable_block_copy` | Boolean | No | Per-instance override for `--enable-block-copy` |

\* At least one of `num_npus` or `tp_size` must be provided. The other is inferred.

### Per-instance runtime overrides

The runtime fields listed above (`max_num_seqs`, `max_num_batched_tokens`, etc.) support **per-instance overrides** in the cluster config. This enables heterogeneous deployments where different instances in the same cluster use different scheduler limits.

**Precedence rule:**
```
per-instance value (from cluster config) > global CLI value (from --flag)
```

For each field, the runtime reads `instance.get("<field>", args.<field>)` — if the field is present in the cluster config, it takes precedence; otherwise the global CLI value is used.

**Unlimited semantics:**
Setting a numeric field to `0` means "unlimited" (via the `_runtime_limit` helper). For example:
- `max_num_seqs: 0` → no limit on concurrent sequences
- `max_num_batched_tokens: 0` → no limit on batched tokens

**Validation gates:**
- `enable_sub_batch_interleaving: true` requires `enable_attn_offloading: true` (enforced at config load time)
- `expert_routing_distribution_csv` is passed to trace generation when `--expert-routing-policy CUSTOM` is selected.

**Example: heterogeneous P/D instances**

See `single_node_pd_per_instance_config.json` for a concrete example where the prefill instance uses `max_num_seqs: 32` (tight concurrency) and the decode instance uses `max_num_seqs: 256` (high throughput):

```json
{
  "instances": [
    {
      "pd_type": "prefill",
      "max_num_seqs": 32,
      "max_num_batched_tokens": 8192,
      "enable_chunked_prefill": true
    },
    {
      "pd_type": "decode",
      "max_num_seqs": 256,
      "max_num_batched_tokens": 0,
      "enable_chunked_prefill": false
    }
  ]
}
```

### Expert-offload runtime fields

These fields configure expert-offload plumbing. `routing_distribution_csv` is active for CSV-backed custom expert routing when `--expert-routing-policy CUSTOM` is selected; later phases use the remaining manifest/profile fields for PIM expert trace generation. `cold_experts_csv` uses a count-based schema: `experts_in_pim` is the number of lowest-probability experts to place in PIM for a layer, not an explicit expert id. Layer-specific count rows override the wildcard count row.

Example decode-instance shape:

```json
{
  "pd_type": "decode",
  "enable_expert_offloading": true,
  "expert_offload": {
    "apply_to": ["decode"],
    "cold_experts_csv": "configs/expert_offload/qwen3_235b_fp8_cold_experts.csv",
    "routing_distribution_csv": "configs/expert_offload/qwen3_235b_fp8_expert_distribution.csv",
    "pim_expert_profile_csv": "profiler/perf/H200/Qwen/Qwen3-235B-A22B-FP8/fp8-kvfp8/tp1/LPDDR5_2GB_6400/pim_moe.csv",
    "channel_policy": "greedy_load",
    "pim_profile_time_semantics": "end_to_end_excludes_transfer",
    "activation_transfer": {
      "npu_to_pim_bw_GBps": 64,
      "pim_to_npu_bw_GBps": 64,
      "npu_to_pim_latency_ns": 1000,
      "pim_to_npu_latency_ns": 1000
    }
  }
}
```

For active expert offload, `pim_moe.csv` is interpreted as CPU/PIM-visible expert runtime only: routed hidden vectors are already visible to the CPU/PIM side at profile entry and output hidden vectors are CPU/PIM-visible at profile exit. The simulator therefore always requires `expert_offload.activation_transfer` and emits explicit local NPU<->CPU/PIM activation-transfer rows outside `PIM ... PIM END`. Transfer bytes are aggregated per MoE layer from cold expert-token pairs: input bytes = `sum(tokens_for_cold_expert * hidden_size * dtype_bytes)`, and output bytes use the same formula. Transfer latency is `fixed_latency_ns + ceil(bytes / bandwidth_GBps)` for each direction, using decimal GB/s (equivalent to bytes/ns). Expert-PIM rows inside `PIM ... PIM END` carry the `pim_moe.csv` runtime but intentionally emit zero input/output tensor sizes for the ASTRA PIM backend, because that backend adds memory service time from `tensor_size`; this prevents double-counting activation movement already represented by the explicit transfer rows. Existing EP dispatch/combine collectives remain in place for hot NPU expert exchange.

### Parallelism rules:
- `num_npus = tp_size * pp_size`
- TP and EP share the same GPUs: non-MoE layers use TP (ALLREDUCE), MoE layers use EP (ALLTOALL)
- DP is achieved via multiple instances with the same `dp_group`
- Without `dp_group`: `ep_size <= tp_size`
- For MoE models: `ep_size` must divide `num_local_experts`

### Network topology:
By default, `config_builder.py` emits `FullyConnected` for every ASTRA-Sim topology dimension to preserve legacy behavior. Set `network_topology` at the top level to select any ASTRA-Sim analytical topology supported by the backend: `Ring`, `Switch`, or `FullyConnected`.

For a single-dimensional TP=4 instance, use a scalar:

```json
{
  "network_topology": "Switch",
  "link_bw": 128,
  "link_latency": 500
}
```

For multi-dimensional layouts, use either a scalar to broadcast the same topology to all dimensions or an array matching the generated topology rank, for example `"network_topology": ["Switch", "Ring"]`.

### DP+EP topology:
When `dp_group` is set, `config_builder.py` generates a 2D ASTRA-Sim topology
`[tp_size, dp_group_size]` with per-dimension collective routing via `involved_dim`.
ALLREDUCE (TP) runs on dim 0 only, ALLTOALL (EP) runs on dim 1. All instances in a
DP group share one ASTRA-Sim process with wave-synchronized scheduling. MoE expert
weights are sharded by `ep_size` (each instance holds `num_local_experts // ep_size` experts).

### Optional fields

| Field | Scope | Type | Description |
| --- | --- | --- | --- |
| `placement` | instance | Object | Per-layer placement rules for weights and KV cache location |
| `power` | node | Object | Power model config (NPU idle/standby/active, CPU, DRAM, link, NIC, storage) |
| `cxl_mem` | top-level | Object | CXL memory expansion parameters (`mem_size`, `mem_bw`, `mem_latency`, `num_devices`) |
| `pim_config` | node cpu_mem | String | Name of a PIM device config in `configs/pim/` |

## Provided configurations

| File | Description |
| --- | --- |
| `single_node_single_instance.json` | Single node, Qwen3-32B with TP=2 (default) |
| `single_node_single_instance_H100.json` | Single node on H100 with TP=4 |
| `single_node_multi_instance.json` | Single node, two instances |
| `single_node_pd_instance.json` | Single node with prefill/decode disaggregation |
| `single_node_pd_per_instance_config.json` | P/D disaggregation with prefill/decode-specific runtime limits |
| `single_node_moe_single_instance.json` | Single node, Qwen3-MoE with TP=2 EP=2 |
| `single_node_moe_multi_instance.json` | Single node, two MoE instances |
| `single_node_moe_pd_instance.json` | Single node, MoE with P/D disaggregation |
| `single_node_cxl_instance.json` | Single node with CXL memory expansion |
| `single_node_memory_instance.json` | Single node with weight/KV placement control |
| `single_node_pim_instance.json` | Single node with PIM-enabled memory + power model |
| `single_node_power_instance.json` | Single node with power modeling enabled |
| `dual_node_multi_instance.json` | Two nodes, two instances each |
| `dual_node_moe_dp_ep_intra_inter_instance.json` | Two-node MoE DP+EP example with per-dimension intra/inter link settings |
