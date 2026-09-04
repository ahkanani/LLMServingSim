# Real Qwen/DeepSeek MMLU/LiveCodeBench Sweep

Generated configs for the requested real-experiment sweep.

## Fixed Settings

- Models: `qwen3_235b_a22b_fp8`, `deepseek_r1_awq`
- Benchmarks: `mmlu`, `livecodebench`
- Scheduler policies: `gpu_baseline`, `placement_greedy`
- Placement pairs: `(hot=0.25, cold=0.50)`, `(hot=0.50, cold=0.25)`
- Predictor: `heuristic`, `window_size=128`, `decay=0.95`, `use_prefill_context=false`
- Replay: `warmup_steps=256`, `eval_steps=5000`, `max_batch_size in {32,64,128}`
- Hardware: `8gpu_16pim_mvp0`
- Oracle comparison: `true` for every run

This sweep is a PIM-offload experiment. Remote-memory MoE experiments are
available separately under `configs/experiments/remote_mem/` and use hardware
configs with `offload_backend: remote_memory`.

## Experiment Count

- Total configs: `48`
- Debug configs: all `qwen3_235b_a22b_fp8` + `mmlu` configs (`12` total)
- Summary configs: all other configs (`36` total)

## Run Command Template

```bash
PYTHONPATH=src .venv/bin/python -m pim_hot_cold_moe simulate --config <config_path>
```

## Experiment List

| # | config | model | benchmark | scheduler | hot | cold | both | batch | metrics |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `001_qwen3_mmlu_gpu_baseline_h25_c50_b32.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 32 | `debug` |
| 2 | `002_qwen3_mmlu_gpu_baseline_h25_c50_b64.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 64 | `debug` |
| 3 | `003_qwen3_mmlu_gpu_baseline_h25_c50_b128.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 128 | `debug` |
| 4 | `004_qwen3_mmlu_gpu_baseline_h50_c25_b32.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 32 | `debug` |
| 5 | `005_qwen3_mmlu_gpu_baseline_h50_c25_b64.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 64 | `debug` |
| 6 | `006_qwen3_mmlu_gpu_baseline_h50_c25_b128.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 128 | `debug` |
| 7 | `007_qwen3_mmlu_placement_greedy_h25_c50_b32.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 32 | `debug` |
| 8 | `008_qwen3_mmlu_placement_greedy_h25_c50_b64.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 64 | `debug` |
| 9 | `009_qwen3_mmlu_placement_greedy_h25_c50_b128.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 128 | `debug` |
| 10 | `010_qwen3_mmlu_placement_greedy_h50_c25_b32.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 32 | `debug` |
| 11 | `011_qwen3_mmlu_placement_greedy_h50_c25_b64.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 64 | `debug` |
| 12 | `012_qwen3_mmlu_placement_greedy_h50_c25_b128.yaml` | `qwen3_235b_a22b_fp8` | `mmlu` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 128 | `debug` |
| 13 | `013_qwen3_livecodebench_gpu_baseline_h25_c50_b32.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 32 | `summary` |
| 14 | `014_qwen3_livecodebench_gpu_baseline_h25_c50_b64.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 64 | `summary` |
| 15 | `015_qwen3_livecodebench_gpu_baseline_h25_c50_b128.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 128 | `summary` |
| 16 | `016_qwen3_livecodebench_gpu_baseline_h50_c25_b32.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 32 | `summary` |
| 17 | `017_qwen3_livecodebench_gpu_baseline_h50_c25_b64.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 64 | `summary` |
| 18 | `018_qwen3_livecodebench_gpu_baseline_h50_c25_b128.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 128 | `summary` |
| 19 | `019_qwen3_livecodebench_placement_greedy_h25_c50_b32.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 32 | `summary` |
| 20 | `020_qwen3_livecodebench_placement_greedy_h25_c50_b64.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 64 | `summary` |
| 21 | `021_qwen3_livecodebench_placement_greedy_h25_c50_b128.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 128 | `summary` |
| 22 | `022_qwen3_livecodebench_placement_greedy_h50_c25_b32.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 32 | `summary` |
| 23 | `023_qwen3_livecodebench_placement_greedy_h50_c25_b64.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 64 | `summary` |
| 24 | `024_qwen3_livecodebench_placement_greedy_h50_c25_b128.yaml` | `qwen3_235b_a22b_fp8` | `livecodebench` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 128 | `summary` |
| 25 | `025_deepseekr1_mmlu_gpu_baseline_h25_c50_b32.yaml` | `deepseek_r1_awq` | `mmlu` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 32 | `summary` |
| 26 | `026_deepseekr1_mmlu_gpu_baseline_h25_c50_b64.yaml` | `deepseek_r1_awq` | `mmlu` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 64 | `summary` |
| 27 | `027_deepseekr1_mmlu_gpu_baseline_h25_c50_b128.yaml` | `deepseek_r1_awq` | `mmlu` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 128 | `summary` |
| 28 | `028_deepseekr1_mmlu_gpu_baseline_h50_c25_b32.yaml` | `deepseek_r1_awq` | `mmlu` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 32 | `summary` |
| 29 | `029_deepseekr1_mmlu_gpu_baseline_h50_c25_b64.yaml` | `deepseek_r1_awq` | `mmlu` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 64 | `summary` |
| 30 | `030_deepseekr1_mmlu_gpu_baseline_h50_c25_b128.yaml` | `deepseek_r1_awq` | `mmlu` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 128 | `summary` |
| 31 | `031_deepseekr1_mmlu_placement_greedy_h25_c50_b32.yaml` | `deepseek_r1_awq` | `mmlu` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 32 | `summary` |
| 32 | `032_deepseekr1_mmlu_placement_greedy_h25_c50_b64.yaml` | `deepseek_r1_awq` | `mmlu` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 64 | `summary` |
| 33 | `033_deepseekr1_mmlu_placement_greedy_h25_c50_b128.yaml` | `deepseek_r1_awq` | `mmlu` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 128 | `summary` |
| 34 | `034_deepseekr1_mmlu_placement_greedy_h50_c25_b32.yaml` | `deepseek_r1_awq` | `mmlu` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 32 | `summary` |
| 35 | `035_deepseekr1_mmlu_placement_greedy_h50_c25_b64.yaml` | `deepseek_r1_awq` | `mmlu` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 64 | `summary` |
| 36 | `036_deepseekr1_mmlu_placement_greedy_h50_c25_b128.yaml` | `deepseek_r1_awq` | `mmlu` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 128 | `summary` |
| 37 | `037_deepseekr1_livecodebench_gpu_baseline_h25_c50_b32.yaml` | `deepseek_r1_awq` | `livecodebench` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 32 | `summary` |
| 38 | `038_deepseekr1_livecodebench_gpu_baseline_h25_c50_b64.yaml` | `deepseek_r1_awq` | `livecodebench` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 64 | `summary` |
| 39 | `039_deepseekr1_livecodebench_gpu_baseline_h25_c50_b128.yaml` | `deepseek_r1_awq` | `livecodebench` | `gpu_baseline` | 0.25 | 0.50 | 0.25 | 128 | `summary` |
| 40 | `040_deepseekr1_livecodebench_gpu_baseline_h50_c25_b32.yaml` | `deepseek_r1_awq` | `livecodebench` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 32 | `summary` |
| 41 | `041_deepseekr1_livecodebench_gpu_baseline_h50_c25_b64.yaml` | `deepseek_r1_awq` | `livecodebench` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 64 | `summary` |
| 42 | `042_deepseekr1_livecodebench_gpu_baseline_h50_c25_b128.yaml` | `deepseek_r1_awq` | `livecodebench` | `gpu_baseline` | 0.50 | 0.25 | 0.25 | 128 | `summary` |
| 43 | `043_deepseekr1_livecodebench_placement_greedy_h25_c50_b32.yaml` | `deepseek_r1_awq` | `livecodebench` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 32 | `summary` |
| 44 | `044_deepseekr1_livecodebench_placement_greedy_h25_c50_b64.yaml` | `deepseek_r1_awq` | `livecodebench` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 64 | `summary` |
| 45 | `045_deepseekr1_livecodebench_placement_greedy_h25_c50_b128.yaml` | `deepseek_r1_awq` | `livecodebench` | `placement_greedy` | 0.25 | 0.50 | 0.25 | 128 | `summary` |
| 46 | `046_deepseekr1_livecodebench_placement_greedy_h50_c25_b32.yaml` | `deepseek_r1_awq` | `livecodebench` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 32 | `summary` |
| 47 | `047_deepseekr1_livecodebench_placement_greedy_h50_c25_b64.yaml` | `deepseek_r1_awq` | `livecodebench` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 64 | `summary` |
| 48 | `048_deepseekr1_livecodebench_placement_greedy_h50_c25_b128.yaml` | `deepseek_r1_awq` | `livecodebench` | `placement_greedy` | 0.50 | 0.25 | 0.25 | 128 | `summary` |
