"""Project-level CLI for PIM hot/cold MoE workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .experiments import build_trace_from_config, run_sweep_from_config, summarize_runs
from .simulation import run_simulations_from_config
from .trace_pack import TracePack


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIM hot/cold MoE MVP workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="TracePack workflows")
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)
    trace_build = trace_sub.add_parser("build", help="build a TracePack from a config")
    trace_build.add_argument("--config", required=True, help="YAML/JSON trace build config")
    trace_build.add_argument("--overwrite", action="store_true", help="override config and rebuild an existing TracePack")
    trace_build.set_defaults(func=_cmd_trace_build)

    trace_inspect = trace_sub.add_parser("inspect", help="print TracePack metadata")
    trace_inspect.add_argument("--trace-pack", "--pack", required=True, dest="trace_pack")
    trace_inspect.set_defaults(func=_cmd_trace_inspect)

    simulate = subparsers.add_parser("simulate", help="run one config, expanding to one run per benchmark")
    simulate.add_argument("--config", required=True, help="YAML/JSON experiment config")
    simulate.add_argument("--benchmark", action="append", default=None, help="benchmark to simulate; repeat for multiple")
    simulate.add_argument("--run-id", default=None, help="override run_id; multi-benchmark runs append benchmark slug")
    simulate.add_argument("--output-dir", default=None, help="override run.output_dir")
    simulate.add_argument("--overwrite", action="store_true", help="replace existing run directories")
    simulate.set_defaults(func=_cmd_simulate)

    sweep = subparsers.add_parser("sweep", help="expand and run a small config sweep")
    sweep.add_argument("--config", required=True, help="YAML/JSON sweep config")
    sweep.add_argument("--overwrite", action="store_true", help="replace existing run directories")
    sweep.add_argument("--dry-run", action="store_true", help="write expanded configs and index without running")
    sweep.set_defaults(func=_cmd_sweep)

    report = subparsers.add_parser("report", help="summarize one or more run directories")
    report.add_argument("--run", action="append", required=True, help="run directory or metrics.json; repeat to compare")
    report.add_argument("--format", choices=["markdown", "json"], default="markdown")
    report.set_defaults(func=_cmd_report)
    return parser


def _cmd_trace_build(args: argparse.Namespace) -> int:
    manifest = build_trace_from_config(
        args.config,
        overwrite=True if args.overwrite else None,
    )
    print(_json_dumps(manifest), end="")
    return 0


def _cmd_trace_inspect(args: argparse.Namespace) -> int:
    with TracePack.open(args.trace_pack) as pack:
        payload = {
            "path": str(Path(args.trace_pack).expanduser().resolve()),
            "benchmarks": pack.benchmarks(),
            "request_count": len(pack.request_ids()),
            "model_metadata": pack.model_metadata(),
        }
    print(_json_dumps(payload), end="")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
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


def _cmd_sweep(args: argparse.Namespace) -> int:
    result = run_sweep_from_config(args.config, overwrite=args.overwrite, dry_run=args.dry_run)
    print(_json_dumps(result), end="")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    print(summarize_runs(args.run, output_format=args.format), end="")
    return 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
