"""Prepare disjoint calibration and serving workloads from recorded MoE routes."""

import json
import random
from pathlib import Path

from serving.core.expert_replay import ExpertTracePack


def register_args(parser):
    parser.add_argument('--trace-pack', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration-output', required=True)
    parser.add_argument('--calibration-requests', type=int, default=32)
    parser.add_argument('--num-reqs', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--sps', type=float, default=0, help='0: saturated; otherwise Poisson requests/s')


def prepare(pack, calibration_count=32, count=256, seed=123, sps=0):
    if calibration_count < 1 or count < 1 or sps < 0:
        raise ValueError('calibration/count must be positive and sps nonnegative')
    ids = sorted(pack.records)
    random.Random(seed).shuffle(ids)
    if calibration_count + count > len(ids):
        raise ValueError('TracePack has insufficient disjoint calibration/evaluation requests')
    calibration = ids[:calibration_count]
    selected = ids[calibration_count:calibration_count + count]
    arrival, rng, rows = 1, random.Random(seed), []
    for index, trace_id in enumerate(selected):
        if index and sps:
            arrival += max(1, round(rng.expovariate(sps) * 1e9))
        record = pack.records[trace_id]
        rows.append(dict(input_toks=int(record['prefill_length']),
                         output_toks=int(record['decode_length']) + 1,
                         arrival_time_ns=arrival, expert_trace_request_id=trace_id))
    return calibration, rows


def run(args):
    pack = ExpertTracePack(args.trace_pack)
    calibration, rows = prepare(pack, args.calibration_requests, args.num_reqs, args.seed, args.sps)
    output, cal_output = Path(args.output), Path(args.calibration_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cal_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    cal_output.write_text(json.dumps(dict(request_ids=calibration, seed=args.seed), indent=2) + '\n')
    return 0
