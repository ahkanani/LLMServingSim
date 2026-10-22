"""Reproducible Qwen MoE replay comparisons and memory-pressure experiments."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from serving.core.expert_offload import repo_root
from serving.core.expert_replay import ExpertTracePack
from workloads.generators.moe_trace import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/moe_replay'))
    parser.add_argument('--benchmarks', nargs='+', default=['mmlu', 'livecodebench'])
    parser.add_argument('--caps', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--variants', nargs='+', choices=['gpu', 'static', 'adaptive', 'migration'],
                        default=['gpu', 'static', 'adaptive', 'migration'])
    parser.add_argument('--budgets', nargs='+', choices=['normal', 'pressure'], default=['normal', 'pressure'])
    parser.add_argument('--loads', nargs='+', type=float, default=[0, 0.25, 0.5, 0.75])
    parser.add_argument('--num-reqs', type=int, default=256)
    parser.add_argument('--decode-steps', type=int, help='Optional smoke limit on recorded decode forwards')
    parser.add_argument('--calibration-requests', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    root, output = repo_root(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(cap < 2 or cap % 2 for cap in args.caps):
        parser.error('caps must be positive even group-wide sequence counts')
    if any(load < 0 or load > 1 for load in args.loads):
        parser.error('loads must be between zero (saturated) and one')
    if args.decode_steps is not None and args.decode_steps < 0:
        parser.error('decode-steps must be nonnegative')
    base_cluster = json.loads((root / 'configs/cluster/pim_experiments/expert_replay/qwen3_235b_h200.json').read_text())
    summaries = []
    for benchmark in args.benchmarks:
        pack_path = (args.trace_root / benchmark).resolve()
        pack = ExpertTracePack(pack_path)
        for budget in args.budgets:
            for cap in args.caps:
                directory = output / benchmark / budget / f'cap{cap}'
                directory.mkdir(parents=True, exist_ok=True)
                calibration, _ = prepare(pack, args.calibration_requests, args.num_reqs, args.seed)
                cal_file = directory / 'calibration.json'
                cal_file.write_text(json.dumps(dict(request_ids=calibration, seed=args.seed)))
                cluster = json.loads(json.dumps(base_cluster))
                for instance in cluster['nodes'][0]['instances']:
                    if instance['pd_type'] == 'decode':
                        instance['max_num_seqs'] = cap // 2
                        instance['max_num_batched_tokens'] = max(2048, cap // 2)
                cluster_file = directory / 'cluster.json'
                cluster_file.write_text(json.dumps(cluster, indent=2))
                def execute(variant, load, rate=0):
                    name = variant + ('_saturated' if not load else f'_load{load:g}')
                    dataset, result = directory / (name + '.jsonl'), directory / (name + '.csv')
                    _, rows = prepare(pack, args.calibration_requests, args.num_reqs, args.seed, rate)
                    if args.decode_steps is not None:
                        for row in rows:
                            row['output_toks'] = min(row['output_toks'], args.decode_steps + 1)
                    dataset.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                    config = json.loads((root / f'configs/expert_replay/{variant}.json').read_text())
                    config.update(calibration_file=str(cal_file), memory_pressure=budget == 'pressure')
                    config_file = directory / (name + '.policy.json')
                    config_file.write_text(json.dumps(config, indent=2))
                    command = [sys.executable, '-m', 'serving', '--expert-routing-policy', 'TRACE',
                               '--expert-trace-pack', str(pack_path), '--expert-policy-config', str(config_file),
                               '--cluster-config', str(cluster_file), '--dataset', str(dataset), '--output', str(result)]
                    print(f'{benchmark} {budget} cap={cap} {name}', flush=True)
                    with (directory / (name + '.log')).open('w') as log:
                        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=args.timeout)
                    summary = json.loads(result.with_suffix('.replay.json').read_text())
                    summaries.append(dict(benchmark=benchmark, budget=budget, cap=cap, variant=variant,
                        load=load, rate_rps=rate, request_throughput_rps=summary['request_throughput_rps'],
                        latency_mean_ms=summary['latency']['mean_ms'], ttft_mean_ms=summary['ttft']['mean_ms'],
                        tpot_mean_ms=summary['tpot']['mean_ms'], preemptions=summary['preemptions'],
                        migration_bytes=summary['policy'].get('migration_bytes', 0), result=str(result)))
                    with (output / 'summary.csv').open('w') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                        writer.writeheader()
                        writer.writerows(summaries)
                    return summary['request_throughput_rps']
                pilot = execute('gpu', 0)
                for variant in args.variants:
                    if variant != 'gpu' and 0 in args.loads:
                        execute(variant, 0)
                for load in args.loads:
                    if load:
                        for variant in args.variants:
                            execute(variant, load, pilot * load)
    print(output / 'summary.csv')


if __name__ == '__main__':
    main()
