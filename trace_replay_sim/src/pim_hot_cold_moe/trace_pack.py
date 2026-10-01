"""TracePack conversion and continuous-batch replay.

TracePacks are one directory per ``(model, benchmark)``. They keep hot replay
data in a memory-mapped NumPy array and small metadata tables in Parquet/JSON:

``manifest.json``
    TracePack metadata and model compatibility information.
``requests.parquet``
    One row per request in deterministic replay order.
``decode_experts.npy``
    Expert ids with shape ``[request, decode_pos, layer, top_k]``.
``decode_lengths.npy``
    Per-request valid decode lengths.
``prefill_summary.parquet``
    Aggregated prefill counts as ``request_index, layer_id, expert_id, count``.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import posixpath
import random
import shutil
import tarfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .interfaces import ReplayStep

SCHEMA_VERSION = "trace_pack_v1"


@dataclass(frozen=True)
class RawTraceRef:
    """Reference to one raw per-request trace file."""

    benchmark: str
    subject: str
    relative_id: str
    source_path: str
    source_member: str | None = None

    @property
    def request_id(self) -> str:
        return f"{self.benchmark}/{self.relative_id}"

    @property
    def source_label(self) -> str:
        if self.source_member:
            return f"{self.source_path}!{self.source_member}"
        return self.source_path


@dataclass(frozen=True)
class _RequestSummary:
    request_index: int
    request_id: str
    benchmark: str
    subject: str
    relative_id: str
    source_path: str
    source_member: str | None
    prefill_length: int
    decode_length: int
    num_layers: int
    top_k: int
    max_expert_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _strip_json_suffix(path: str) -> str:
    return path[:-5] if path.endswith(".json") else path


def _safe_member_name(name: str) -> str | None:
    norm = posixpath.normpath(name)
    if norm.startswith("../") or norm == ".." or norm.startswith("/"):
        return None
    if not norm.endswith(".json"):
        return None
    return norm


def _subject_from_relative(relative_id: str) -> str:
    parts = [p for p in relative_id.split("/") if p]
    if len(parts) >= 2:
        return parts[0]
    if parts:
        return "__root__"
    return "__unknown__"


def _iter_json_files(benchmark_dir: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(benchmark_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".json"):
                yield Path(root) / name


def _iter_tar_files(benchmark_dir: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(benchmark_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".tar.gz") or name.endswith(".tgz"):
                yield Path(root) / name


def discover_raw_traces(
    raw_root: str | Path,
    model: str,
    benchmarks: Sequence[str],
    max_requests_per_benchmark: int | None = None,
) -> list[RawTraceRef]:
    """Discover raw trace files without mutating the dataset."""

    if max_requests_per_benchmark is not None and max_requests_per_benchmark <= 0:
        raise ValueError("max_requests_per_benchmark must be positive or None")

    root = Path(raw_root).expanduser().resolve()
    refs: list[RawTraceRef] = []

    for benchmark in benchmarks:
        benchmark_dir = root / model / benchmark
        if not benchmark_dir.is_dir():
            raise FileNotFoundError(f"benchmark directory not found: {benchmark_dir}")

        count = 0
        seen_relative_json: set[str] = set()

        def reached_limit() -> bool:
            return max_requests_per_benchmark is not None and count >= max_requests_per_benchmark

        for json_path in _iter_json_files(benchmark_dir):
            rel_json = json_path.relative_to(benchmark_dir).as_posix()
            seen_relative_json.add(rel_json)
            relative_id = _strip_json_suffix(rel_json)
            refs.append(
                RawTraceRef(
                    benchmark=benchmark,
                    subject=_subject_from_relative(relative_id),
                    relative_id=relative_id,
                    source_path=str(json_path),
                )
            )
            count += 1
            if reached_limit():
                break

        if reached_limit():
            continue

        for tar_path in _iter_tar_files(benchmark_dir):
            with tarfile.open(tar_path, "r:*") as tar:
                member_names = sorted(
                    name
                    for name in (_safe_member_name(m.name) for m in tar.getmembers())
                    if name is not None
                )
            for member_name in member_names:
                if member_name in seen_relative_json:
                    continue
                relative_id = _strip_json_suffix(member_name)
                refs.append(
                    RawTraceRef(
                        benchmark=benchmark,
                        subject=_subject_from_relative(relative_id),
                        relative_id=relative_id,
                        source_path=str(tar_path),
                        source_member=member_name,
                    )
                )
                count += 1
                if reached_limit():
                    break
            if reached_limit():
                break

    return refs


def _normalize_expert_rows(value: Any, *, context: str) -> list[list[int]]:
    """Normalize raw selected experts into ``list[token][top_k]`` rows."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list/null expert payload, got {type(value).__name__}")
    if not value:
        return []
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return [list(int(item) for item in value)]

    rows: list[list[int]] = []
    for row_index, row in enumerate(value):
        row_context = f"{context} row {row_index}"
        if row is None:
            continue
        if not isinstance(row, list):
            raise ValueError(f"{row_context}: expected list of expert ids, got {type(row).__name__}")
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in row):
            raise ValueError(f"{row_context}: expert ids must be integers")
        if row:
            rows.append(list(int(item) for item in row))
    return rows


def _rows_to_counts(rows: Iterable[Iterable[int]]) -> collections.Counter[int]:
    counts: collections.Counter[int] = collections.Counter()
    for row in rows:
        counts.update(row)
    return counts


def _read_raw_payload(ref: RawTraceRef) -> Any:
    if ref.source_member:
        with tarfile.open(ref.source_path, "r:*") as tar:
            extracted = tar.extractfile(ref.source_member)
            if extracted is None:
                raise FileNotFoundError(f"member not found in archive: {ref.source_label}")
            return json.load(extracted)
    with open(ref.source_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_trace_payload(payload: Any, *, ref: RawTraceRef) -> list[Mapping[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{ref.source_label}: trace payload must be a list of output-token records")
    if not payload:
        raise ValueError(f"{ref.source_label}: trace payload is empty")
    for token_index, token in enumerate(payload):
        if not isinstance(token, dict):
            raise ValueError(
                f"{ref.source_label} token {token_index}: expected layer dictionary, got {type(token).__name__}"
            )
    return payload  # type: ignore[return-value]


def _summarize_request_payload(
    ref: RawTraceRef,
    *,
    request_index: int,
    request_id: str,
    payload: Sequence[Mapping[str, Any]],
) -> _RequestSummary:
    observed_layers: set[int] = set()
    top_k_values: set[int] = set()
    max_expert_id = -1
    prefill_length = 0

    for token_index, token in enumerate(payload):
        is_prefill = token_index == 0
        for raw_layer_id, raw_rows in token.items():
            try:
                layer_id = int(raw_layer_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{ref.source_label} token {token_index}: invalid layer id {raw_layer_id!r}") from exc

            rows = _normalize_expert_rows(
                raw_rows,
                context=f"{ref.source_label} token {token_index} layer {layer_id}",
            )
            if not rows:
                continue
            observed_layers.add(layer_id)
            if is_prefill:
                prefill_length = max(prefill_length, len(rows))
            else:
                if len(rows) != 1:
                    raise ValueError(
                        f"{ref.source_label} token {token_index} layer {layer_id}: "
                        "TracePack decode storage expects exactly one top-k row per layer"
                    )
            for row in rows:
                top_k_values.add(len(row))
                if row:
                    max_expert_id = max(max_expert_id, max(row))

    if not observed_layers:
        raise ValueError(f"{ref.source_label}: no expert routing rows found")
    if len(top_k_values) != 1:
        raise ValueError(f"{ref.source_label}: mixed top-k values are not supported: {sorted(top_k_values)}")

    return _RequestSummary(
        request_index=request_index,
        request_id=request_id,
        benchmark=ref.benchmark,
        subject=ref.subject,
        relative_id=ref.relative_id,
        source_path=ref.source_path,
        source_member=ref.source_member,
        prefill_length=prefill_length,
        decode_length=max(0, len(payload) - 1),
        num_layers=max(observed_layers) + 1,
        top_k=next(iter(top_k_values)),
        max_expert_id=max_expert_id,
    )


def build_trace_pack(
    *,
    raw_root: str | Path,
    output_path: str | Path,
    model: str,
    benchmarks: Sequence[str],
    max_requests_per_benchmark: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one model-benchmark raw trace slice into a TracePack directory."""

    benchmark = _single_benchmark(benchmarks)
    out_path = Path(output_path).expanduser().resolve()
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"output TracePack already exists: {out_path}")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.mkdir(parents=True, exist_ok=False)

    refs = discover_raw_traces(raw_root, model, [benchmark], max_requests_per_benchmark)
    if not refs:
        raise ValueError("no raw trace files discovered for requested model/benchmarks")

    request_summaries: list[_RequestSummary] = []
    used_request_ids: set[str] = set()
    resolved_ids: list[str] = []
    for request_index, ref in enumerate(refs):
        request_id = _dedupe_request_id(ref, used_request_ids)
        resolved_ids.append(request_id)
        payload = _validate_trace_payload(_read_raw_payload(ref), ref=ref)
        request_summaries.append(
            _summarize_request_payload(ref, request_index=request_index, request_id=request_id, payload=payload)
        )

    num_layers_values = {item.num_layers for item in request_summaries}
    top_k_values = {item.top_k for item in request_summaries}
    if len(num_layers_values) != 1:
        raise ValueError(f"TracePack requires one num_layers value; got {sorted(num_layers_values)}")
    if len(top_k_values) != 1:
        raise ValueError(f"TracePack requires one top_k value; got {sorted(top_k_values)}")

    num_requests = len(request_summaries)
    num_layers = next(iter(num_layers_values))
    top_k = next(iter(top_k_values))
    max_decode_length = max(item.decode_length for item in request_summaries)
    max_expert_id = max(item.max_expert_id for item in request_summaries)
    dtype = _expert_dtype(max_expert_id)

    decode_path = out_path / "decode_experts.npy"
    decode = np.lib.format.open_memmap(
        decode_path,
        mode="w+",
        dtype=dtype,
        shape=(num_requests, max_decode_length, num_layers, top_k),
    )
    decode[:] = 0
    decode_lengths = np.zeros((num_requests,), dtype=np.uint32)
    prefill_rows: list[dict[str, Any]] = []

    for summary, ref, request_id in zip(request_summaries, refs, resolved_ids):
        payload = _validate_trace_payload(_read_raw_payload(ref), ref=ref)
        decode_lengths[summary.request_index] = summary.decode_length
        _fill_request_arrays(
            payload,
            ref=ref,
            request_index=summary.request_index,
            request_id=request_id,
            decode=decode,
            prefill_rows=prefill_rows,
            num_layers=num_layers,
            top_k=top_k,
        )
    decode.flush()
    np.save(out_path / "decode_lengths.npy", decode_lengths)

    request_rows = [item.to_dict() for item in request_summaries]
    _write_parquet(request_rows, out_path / "requests.parquet")
    _write_parquet(prefill_rows, out_path / "prefill_summary.parquet")

    model_metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model,
        "benchmarks": [benchmark],
        "request_count": num_requests,
        "num_layers": num_layers,
        "moe_layer_ids": list(range(num_layers)),
        "experts_per_layer": {str(layer_id): max_expert_id + 1 for layer_id in range(num_layers)},
        "top_k": top_k,
        "decode_token_count": int(sum(item.decode_length for item in request_summaries)),
        "prefill_token_count": int(sum(item.prefill_length for item in request_summaries)),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now_iso(),
        "trace_pack_path": str(out_path),
        "conversion": {
            "raw_root": str(Path(raw_root).expanduser().resolve()),
            "model": model,
            "benchmarks": [benchmark],
            "benchmark": benchmark,
            "max_requests_per_benchmark": max_requests_per_benchmark,
        },
        "model_metadata": model_metadata,
        "layout": {
            "decode_experts": "decode_experts.npy",
            "decode_lengths": "decode_lengths.npy",
            "requests": "requests.parquet",
            "prefill_summary": "prefill_summary.parquet",
            "decode_shape": [num_requests, max_decode_length, num_layers, top_k],
            "decode_dtype": str(np.dtype(dtype)),
        },
        "requests": request_rows,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _dedupe_request_id(ref: RawTraceRef, used_request_ids: set[str]) -> str:
    request_id = ref.request_id
    if request_id in used_request_ids:
        digest = hashlib.sha1(ref.source_label.encode("utf-8")).hexdigest()[:8]
        request_id = f"{request_id}-{digest}"
    used_request_ids.add(request_id)
    return request_id


def _expert_dtype(max_expert_id: int) -> np.dtype[Any]:
    if max_expert_id < 0:
        raise ValueError("no expert ids discovered")
    if max_expert_id <= np.iinfo(np.uint8).max:
        return np.dtype("uint8")
    if max_expert_id <= np.iinfo(np.uint16).max:
        return np.dtype("uint16")
    return np.dtype("uint32")


def _fill_request_arrays(
    payload: Sequence[Mapping[str, Any]],
    *,
    ref: RawTraceRef,
    request_index: int,
    request_id: str,
    decode: np.memmap,
    prefill_rows: list[dict[str, Any]],
    num_layers: int,
    top_k: int,
) -> None:
    prefill_counts_by_layer: dict[int, collections.Counter[int]] = {}
    for token_index, token in enumerate(payload):
        is_prefill = token_index == 0
        decode_pos = token_index - 1
        for raw_layer_id, raw_rows in token.items():
            layer_id = int(raw_layer_id)
            rows = _normalize_expert_rows(
                raw_rows,
                context=f"{ref.source_label} token {token_index} layer {layer_id}",
            )
            if not rows:
                continue
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(f"{ref.source_label} token {token_index}: layer {layer_id} outside 0..{num_layers - 1}")
            if is_prefill:
                prefill_counts_by_layer.setdefault(layer_id, collections.Counter()).update(_rows_to_counts(rows))
                continue
            if len(rows) != 1 or len(rows[0]) != top_k:
                raise ValueError(
                    f"{ref.source_label} token {token_index} layer {layer_id}: expected exactly one top-{top_k} row"
                )
            decode[request_index, decode_pos, layer_id, :] = rows[0]

    for layer_id, counts in sorted(prefill_counts_by_layer.items()):
        for expert_id, count in sorted(counts.items()):
            prefill_rows.append(
                {
                    "request_index": int(request_index),
                    "request_id": request_id,
                    "layer_id": int(layer_id),
                    "expert_id": int(expert_id),
                    "count": int(count),
                }
            )


def build_trace_packs(
    *,
    raw_root: str | Path,
    model: str,
    benchmarks: Sequence[str],
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_path_template: str | Path | None = None,
    output_paths: Mapping[str, str | Path] | None = None,
    filename_template: str = "{benchmark}",
    max_requests_per_benchmark: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one TracePack directory for each requested ``(model, benchmark)``."""

    benchmark_list = _benchmark_list(benchmarks)
    manifests: list[dict[str, Any]] = []
    trace_pack_paths: dict[str, str] = {}
    for benchmark in benchmark_list:
        pack_path = _resolve_benchmark_output_path(
            benchmark=benchmark,
            model=model,
            benchmark_count=len(benchmark_list),
            single_path=output_path,
            output_dir=output_dir,
            path_template=output_path_template,
            paths=output_paths,
            filename_template=filename_template,
            field_name="output",
        )
        manifest = build_trace_pack(
            raw_root=raw_root,
            output_path=pack_path,
            model=model,
            benchmarks=[benchmark],
            max_requests_per_benchmark=max_requests_per_benchmark,
            overwrite=overwrite,
        )
        manifests.append(manifest)
        trace_pack_paths[benchmark] = manifest["trace_pack_path"]
    return {
        "schema_version": SCHEMA_VERSION,
        "layout": "per_model_benchmark_trace_pack",
        "model": model,
        "benchmarks": benchmark_list,
        "trace_pack_paths": trace_pack_paths,
        "trace_packs": manifests,
    }


def _benchmark_list(benchmarks: Sequence[str]) -> list[str]:
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    out: list[str] = []
    for value in benchmarks:
        text = str(value)
        if not text:
            raise ValueError("benchmark names must be non-empty")
        if text not in out:
            out.append(text)
    return out


def _single_benchmark(benchmarks: Sequence[str]) -> str:
    values = _benchmark_list(benchmarks)
    if len(values) != 1:
        raise ValueError(
            "TracePacks are scoped to exactly one model-benchmark; "
            f"got {values}. Build separate TracePacks per benchmark."
        )
    return values[0]


def _resolve_benchmark_output_path(
    *,
    benchmark: str,
    model: str,
    benchmark_count: int,
    single_path: str | Path | None,
    output_dir: str | Path | None,
    path_template: str | Path | None,
    paths: Mapping[str, str | Path] | None,
    filename_template: str,
    field_name: str,
) -> Path:
    if paths is not None:
        try:
            return Path(paths[benchmark]).expanduser().resolve()
        except KeyError as exc:
            raise KeyError(f"{field_name}_paths missing benchmark {benchmark!r}") from exc
    if path_template is not None:
        return Path(_format_benchmark_path(path_template, model, benchmark)).expanduser().resolve()
    if single_path is not None:
        if benchmark_count > 1:
            raise ValueError(f"{field_name}_path is only valid for single-benchmark TracePack builds")
        return Path(single_path).expanduser().resolve()
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve() / _format_benchmark_path(filename_template, model, benchmark)
    raise ValueError(f"trace build requires {field_name}_dir, {field_name}_path_template, or {field_name}_paths")


def _format_benchmark_path(template: str | Path, model: str, benchmark: str) -> str:
    return str(template).format(model=model, model_slug=_slug(model), benchmark=benchmark, benchmark_slug=_slug(benchmark))


def _slug(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "value"


def _write_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("TracePack metadata requires pyarrow to write Parquet") from exc

    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("TracePack metadata requires pyarrow to read Parquet") from exc

    return [dict(row) for row in pq.read_table(path).to_pylist()]


class TracePack:
    """Read-only wrapper around an array-backed TracePack directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"TracePack directory not found: {self.path}")
        self._manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        layout = self._manifest.get("layout", {})
        self.decode = np.load(self.path / layout.get("decode_experts", "decode_experts.npy"), mmap_mode="r")
        self._decode_lengths_array = np.load(self.path / layout.get("decode_lengths", "decode_lengths.npy"), mmap_mode="r")
        self._requests = _read_parquet(self.path / layout.get("requests", "requests.parquet"))
        self._prefill_rows = _read_parquet(self.path / layout.get("prefill_summary", "prefill_summary.parquet"))
        self._request_by_id = {str(row["request_id"]): dict(row) for row in self._requests}
        self._index_by_request_id = {request_id: int(row["request_index"]) for request_id, row in self._request_by_id.items()}
        self._expert_count = int(max(self.model_metadata().get("experts_per_layer", {"0": 0}).values() or [0]))

    @classmethod
    def open(cls, path: str | Path) -> "TracePack":
        return cls(path)

    def close(self) -> None:
        return None

    def __enter__(self) -> "TracePack":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def model_metadata(self) -> dict[str, Any]:
        return dict(self._manifest["model_metadata"])

    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    def benchmarks(self) -> list[str]:
        return sorted({str(row["benchmark"]) for row in self._requests})

    def request_ids(self, split: str | None = None, benchmarks: Sequence[str] | None = None) -> list[str]:
        requested_benchmarks: list[str] = []
        if split is not None:
            requested_benchmarks.append(split)
        if benchmarks is not None:
            requested_benchmarks.extend(benchmarks)
        requested = set(requested_benchmarks)
        rows = self._requests
        if requested:
            rows = [row for row in rows if str(row["benchmark"]) in requested]
        return [str(row["request_id"]) for row in sorted(rows, key=lambda row: (str(row["benchmark"]), str(row["request_id"])))]

    def request_records(self, benchmarks: Sequence[str] | None = None) -> list[dict[str, Any]]:
        requested = set(benchmarks or [])
        rows = self._requests
        if requested:
            rows = [row for row in rows if str(row["benchmark"]) in requested]
        return [dict(row) for row in sorted(rows, key=lambda row: (str(row["benchmark"]), str(row["request_id"])))]

    def request_benchmark_map(self, request_ids: Sequence[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        missing: list[str] = []
        for request_id in request_ids:
            row = self._request_by_id.get(request_id)
            if row is None:
                missing.append(request_id)
            else:
                mapping[request_id] = str(row["benchmark"])
        if missing:
            raise KeyError(f"request ids not found in TracePack: {missing}")
        return mapping

    def prefill_context(self, request_ids: Sequence[str]) -> dict[str, Any]:
        if not request_ids:
            return {}
        selected = set(request_ids)
        context: dict[str, Any] = {}
        for request_id in request_ids:
            row = self._request_by_id[request_id]
            context[request_id] = {"prefill_length": int(row["prefill_length"]), "layer_expert_counts": {}}
        for row in self._prefill_rows:
            request_id = str(row["request_id"])
            if request_id not in selected:
                continue
            layer_counts = context[request_id]["layer_expert_counts"].setdefault(int(row["layer_id"]), {})
            layer_counts[int(row["expert_id"])] = int(row["count"])
        return context

    def decode_tokens(self, request_ids: Sequence[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for request_id in request_ids:
            request_index = self._index_by_request_id[request_id]
            decode_length = int(self._decode_lengths_array[request_index])
            for decode_pos in range(decode_length):
                for layer_id, experts in enumerate(self.decode[request_index, decode_pos]):
                    out.append(
                        {
                            "request_id": request_id,
                            "decode_pos": decode_pos,
                            "layer_id": layer_id,
                            "experts": [int(value) for value in experts.tolist()],
                        }
                    )
        return out

    def decode_lengths(self, request_ids: Sequence[str]) -> dict[str, int]:
        return {request_id: int(self._decode_lengths_array[self._index_by_request_id[request_id]]) for request_id in request_ids}

    def aggregate_decode_counts(self, positions_by_request: Mapping[str, int]) -> dict[int, dict[int, int]]:
        if not positions_by_request:
            return {}
        request_ids = list(positions_by_request)
        indices = np.array([self._index_by_request_id[request_id] for request_id in request_ids], dtype=np.int64)
        positions = np.array([int(positions_by_request[request_id]) for request_id in request_ids], dtype=np.int64)
        batch = np.asarray(self.decode[indices, positions, :, :], dtype=np.int64)
        aggregate: dict[int, dict[int, int]] = {}
        for layer_id in range(batch.shape[1]):
            counts = np.bincount(batch[:, layer_id, :].reshape(-1), minlength=self._expert_count)
            nonzero = np.nonzero(counts)[0]
            if len(nonzero):
                aggregate[layer_id] = {int(expert_id): int(counts[expert_id]) for expert_id in nonzero}
        return aggregate


class ReplayStream:
    """Continuous-batch decode replay over one TracePack benchmark."""

    def __init__(
        self,
        trace_pack: TracePack | str | Path,
        *,
        seed: int = 0,
        max_batch_size: int = 1,
        warmup_steps: int = 0,
        eval_steps: int | None = None,
        benchmarks: Sequence[str] | None = None,
        request_ids: Sequence[str] | None = None,
        limit_per_benchmark: int | None = None,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if eval_steps is not None and eval_steps < 0:
            raise ValueError("eval_steps must be non-negative or None")
        if limit_per_benchmark is not None and limit_per_benchmark <= 0:
            raise ValueError("limit_per_benchmark must be positive or None")

        self.trace_pack = TracePack.open(trace_pack) if isinstance(trace_pack, (str, Path)) else trace_pack
        self.seed = int(seed)
        self.max_batch_size = int(max_batch_size)
        self.warmup_step_count = int(warmup_steps)
        self.eval_step_count = eval_steps
        self.explicit_request_ids = list(request_ids) if request_ids is not None else None
        self.limit_per_benchmark = limit_per_benchmark
        requested_benchmarks = list(dict.fromkeys(benchmarks or []))

        if self.explicit_request_ids is not None:
            benchmark_map = self.trace_pack.request_benchmark_map(self.explicit_request_ids)
            selected_benchmarks = sorted(set(benchmark_map.values()))
            if len(selected_benchmarks) != 1:
                raise ValueError(
                    "ReplayStream request_ids must all come from exactly one benchmark; "
                    f"got {selected_benchmarks}. Create separate ReplayStream runs per benchmark."
                )
            if requested_benchmarks and requested_benchmarks != selected_benchmarks:
                raise ValueError(
                    "ReplayStream benchmark filter does not match explicit request_ids: "
                    f"filter={requested_benchmarks}, request_ids={selected_benchmarks}"
                )
            self.benchmark = selected_benchmarks[0]
        elif requested_benchmarks:
            if len(requested_benchmarks) != 1:
                raise ValueError(
                    "ReplayStream requires exactly one benchmark per run; "
                    f"got {requested_benchmarks}. Create separate ReplayStream runs per benchmark."
                )
            self.benchmark = requested_benchmarks[0]
            available_benchmarks = self.trace_pack.benchmarks()
            if self.benchmark not in available_benchmarks:
                raise ValueError(
                    f"benchmark {self.benchmark!r} not found in TracePack; available benchmarks: {available_benchmarks}"
                )
        else:
            available_benchmarks = self.trace_pack.benchmarks()
            if len(available_benchmarks) != 1:
                raise ValueError(
                    "ReplayStream requires exactly one benchmark per run. "
                    f"The TracePack contains {available_benchmarks}; pass one benchmark explicitly."
                )
            self.benchmark = available_benchmarks[0]

        self.benchmarks = [self.benchmark]

    def selected_request_ids(self) -> list[str]:
        return self._ordered_request_ids()

    def _ordered_request_ids(self) -> list[str]:
        if self.explicit_request_ids is not None:
            return list(self.explicit_request_ids)

        records = self.trace_pack.request_records([self.benchmark])
        ids = [str(record["request_id"]) for record in records]
        rng = random.Random(_stable_seed(self.seed, self.benchmark))
        rng.shuffle(ids)
        if self.limit_per_benchmark is not None:
            ids = ids[: self.limit_per_benchmark]
        return ids

    def __iter__(self) -> Iterator[ReplayStep]:
        ordered_ids = self._ordered_request_ids()
        lengths = self.trace_pack.decode_lengths(ordered_ids)
        queue: collections.deque[str] = collections.deque(
            request_id for request_id in ordered_ids if lengths.get(request_id, 0) > 0
        )
        active: list[str] = []
        positions: dict[str, int] = {}

        def admit() -> None:
            while len(active) < self.max_batch_size and queue:
                request_id = queue.popleft()
                active.append(request_id)
                positions[request_id] = 0

        admit()
        step_id = 0
        while active:
            step_positions = {request_id: positions[request_id] for request_id in active}
            yield ReplayStep(
                step_id=step_id,
                active_request_ids=list(active),
                layer_expert_counts=self.trace_pack.aggregate_decode_counts(step_positions),
                request_positions=step_positions,
                metadata={
                    "seed": self.seed,
                    "max_batch_size": self.max_batch_size,
                    "active_count": len(active),
                    "queued_count": len(queue),
                    "benchmark": self.benchmark,
                    "benchmarks": self.benchmarks,
                },
            )

            survivors: list[str] = []
            for request_id in active:
                next_pos = positions[request_id] + 1
                if next_pos < lengths[request_id]:
                    positions[request_id] = next_pos
                    survivors.append(request_id)
                else:
                    del positions[request_id]
            active = survivors
            admit()
            step_id += 1

    def warmup_steps(self) -> Iterator[ReplayStep]:
        for index, step in enumerate(self):
            if index >= self.warmup_step_count:
                break
            metadata = dict(step.metadata)
            metadata["phase"] = "warmup"
            metadata["warmup_steps"] = self.warmup_step_count
            yield replace(step, metadata=metadata)

    def eval_steps(self) -> Iterator[ReplayStep]:
        emitted = 0
        for index, step in enumerate(self):
            if index < self.warmup_step_count:
                continue
            if self.eval_step_count is not None and emitted >= self.eval_step_count:
                break
            metadata = dict(step.metadata)
            metadata["phase"] = "eval"
            metadata["warmup_steps"] = self.warmup_step_count
            metadata["eval_steps"] = self.eval_step_count
            yield replace(step, metadata=metadata)
            emitted += 1


def replay_step_to_dict(step: ReplayStep) -> dict[str, Any]:
    return asdict(step)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("YAML config support requires PyYAML") from exc
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"config must contain a mapping at top level: {config_path}")
        return data
    raise ValueError(f"unsupported config extension for {config_path}; use .json/.yaml/.yml")


def _cmd_convert_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    trace_cfg = cfg.get("trace", cfg)
    manifest = build_trace_packs(
        raw_root=trace_cfg["raw_root"],
        model=trace_cfg.get("model", trace_cfg.get("model_id")),
        benchmarks=trace_cfg["benchmarks"],
        output_path=trace_cfg.get("output_path", trace_cfg.get("pack_path")),
        output_dir=trace_cfg.get("output_dir"),
        output_path_template=trace_cfg.get("output_path_template", trace_cfg.get("pack_path_template")),
        output_paths=trace_cfg.get("output_paths", trace_cfg.get("pack_paths")),
        filename_template=str(trace_cfg.get("filename_template", "{benchmark}")),
        max_requests_per_benchmark=trace_cfg.get("max_requests_per_benchmark"),
        overwrite=bool(args.overwrite or trace_cfg.get("overwrite", False)),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), end="\n")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    with TracePack.open(args.trace_pack) as pack:
        stream = ReplayStream(
            pack,
            seed=args.seed,
            max_batch_size=args.max_batch_size,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            benchmarks=[args.benchmark] if args.benchmark else None,
            limit_per_benchmark=args.limit_per_benchmark,
        )
        steps = stream.eval_steps() if args.phase == "eval" else stream.warmup_steps()
        for index, step in enumerate(steps):
            if index >= args.steps:
                break
            print(json.dumps(replay_step_to_dict(step), sort_keys=True))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    with TracePack.open(args.trace_pack) as pack:
        payload = {
            "path": str(Path(args.trace_pack).expanduser().resolve()),
            "benchmarks": pack.benchmarks(),
            "request_count": len(pack.request_ids()),
            "model_metadata": pack.model_metadata(),
        }
    print(json.dumps(payload, indent=2, sort_keys=True), end="\n")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TracePack conversion and replay tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_config = subparsers.add_parser("convert-config", help="convert raw JSON traces into TracePack directories")
    convert_config.add_argument("--config", required=True, help="YAML/JSON config")
    convert_config.add_argument("--overwrite", action="store_true")
    convert_config.set_defaults(func=_cmd_convert_config)

    replay = subparsers.add_parser("replay", help="print replay steps from a TracePack")
    replay.add_argument("--trace-pack", required=True, help="TracePack directory")
    replay.add_argument("--benchmark", default=None)
    replay.add_argument("--seed", type=int, default=0)
    replay.add_argument("--max-batch-size", type=int, default=1)
    replay.add_argument("--warmup-steps", type=int, default=0)
    replay.add_argument("--eval-steps", type=int, default=None)
    replay.add_argument("--limit-per-benchmark", type=int, default=None)
    replay.add_argument("--phase", choices=["warmup", "eval"], default="eval")
    replay.add_argument("--steps", type=int, default=5)
    replay.set_defaults(func=_cmd_replay)

    inspect = subparsers.add_parser("inspect", help="print TracePack metadata")
    inspect.add_argument("--trace-pack", required=True, help="TracePack directory")
    inspect.set_defaults(func=_cmd_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
