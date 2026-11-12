"""Read recorded MoE selections without a replay cursor or standalone dependency."""

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .expert_offload import resolve_repo_path


@dataclass(frozen=True)
class TokenSpan:
    request_id: int
    trace_request_id: str
    prompt_length: int
    start: int
    count: int
    source_instance: int


class ExpertTracePack:
    def __init__(self, path, model_config=None, model=None):
        self.path = resolve_repo_path(path)
        manifest_path = self.path if self.path.is_file() else self.path / 'manifest.json'
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text())
        self.moe_layer_ids = None
        self.packs = {}
        if self.manifest.get('schema_version') == 'trace_pack_collection_v1':
            self._load_collection(model_config, model)
            return
        if self.manifest.get("schema_version") != "trace_pack_v1":
            raise ValueError("expert replay requires a trace_pack_v1 TracePack")
        metadata = self.manifest["model_metadata"]
        self.num_layers = int(metadata["num_layers"])
        self.top_k = int(metadata["top_k"])
        self.num_experts = max(map(int, metadata["experts_per_layer"].values()))
        if (set(metadata['experts_per_layer']) != {str(i) for i in range(self.num_layers)}
                or len(set(metadata['experts_per_layer'].values())) != 1):
            raise ValueError('expert replay requires complete, uniform per-layer expert metadata')
        self.moe_layer_ids = tuple((model_config or {}).get('moe_layer_ids', range(self.num_layers)))
        if (not self.moe_layer_ids or tuple(sorted(set(self.moe_layer_ids))) != self.moe_layer_ids
                or any(layer < 0 or layer >= self.num_layers for layer in self.moe_layer_ids)):
            raise ValueError('model MoE layer IDs must be unique, ordered, and in range')
        layout = self.manifest["layout"]
        self.decode = np.load(self.path / layout["decode_experts"], mmap_mode="r")
        self.lengths = np.load(self.path / layout["decode_lengths"], mmap_mode="r")
        records = self.manifest.get("requests")
        if not records:
            raise ValueError("TracePack manifest must contain its request metadata index")
        self.records = {str(row["request_id"]): row for row in records}
        if len(self.records) != len(records):
            raise ValueError("duplicate request IDs in TracePack")
        if {int(row['request_index']) for row in records} != set(range(len(records))):
            raise ValueError('TracePack request indices must be unique and contiguous')
        if self.decode.shape != (len(records), max(self.lengths, default=0), self.num_layers, self.top_k):
            raise ValueError("TracePack decode array shape disagrees with metadata")
        for row in records:
            index = int(row["request_index"])
            if not 0 <= index < len(records) or int(self.lengths[index]) != int(row["decode_length"]):
                raise ValueError("TracePack request index/length disagrees with decode_lengths")
        if model_config is not None:
            expected = (int(model_config["num_hidden_layers"]),
                        int(model_config.get("num_experts", model_config.get("num_local_experts", 0))),
                        int(model_config["num_experts_per_tok"]))
            if (self.num_layers, self.num_experts, self.top_k) != expected:
                raise ValueError(f"TracePack/model dimensions disagree: expected {expected}")
            if model is not None and metadata["model_id"] != model:
                raise ValueError(f"TracePack model {metadata['model_id']} does not match {model}")

    def _load_collection(self, model_config, model):
        sources = self.manifest.get('packs')
        if not isinstance(sources, dict) or not sources:
            raise ValueError('TracePack collection requires a nonempty packs mapping')
        self.records = {}
        metadata = None
        for name, location in sorted(sources.items()):
            if not isinstance(name, str) or not name or '::' in name:
                raise ValueError('invalid TracePack source namespace')
            path = Path(location)
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            pack = ExpertTracePack(path, model_config, model)
            if pack.packs:
                raise ValueError('nested TracePack collections are unsupported')
            current = pack.manifest['model_metadata']
            dimensions = {key: current[key] for key in ('model_id', 'num_layers', 'top_k', 'experts_per_layer')}
            if metadata is not None and metadata != dimensions:
                raise ValueError('TracePack collection model dimensions disagree')
            metadata = dimensions
            self.packs[name] = pack
            for trace_id, record in pack.records.items():
                key = name + '::' + trace_id
                self.records[key] = dict(record, request_id=key, source_pack=name, source_request_id=trace_id)
        self.manifest = dict(self.manifest, model_metadata=metadata)
        self.num_layers, self.num_experts, self.top_k = pack.num_layers, pack.num_experts, pack.top_k
        self.moe_layer_ids = pack.moe_layer_ids

    def decode_rows(self, trace_id, start=0, count=None, layer=None):
        """Read immutable recorded routes through either a pack or collection."""
        record = self.records[str(trace_id)]
        length = int(record['decode_length'])
        count = length - start if count is None else count
        if start < 0 or count < 0 or start + count > length:
            raise ValueError(f'expert trace exhausted for {trace_id}')
        if layer is not None and not 0 <= layer < self.num_layers:
            raise ValueError(f'invalid expert layer {layer}')
        if self.packs:
            return self.packs[record['source_pack']].decode_rows(record['source_request_id'], start, count, layer)
        rows = self.decode[int(record['request_index']), start:start + count]
        return rows if layer is None else rows[:, layer, :]

    def decode_batch(self, trace_ids, position):
        if not trace_ids:
            return np.empty((0, self.num_layers, self.top_k), dtype=np.uint8)
        rows = np.stack([self.decode_rows(trace_id, position, 1)[0] for trace_id in trace_ids])
        rows.setflags(write=False)
        return rows

    def manifest_sha256(self):
        digest = hashlib.sha256(self.manifest_path.read_bytes())
        for name, pack in sorted(self.packs.items()):
            digest.update(name.encode())
            digest.update(pack.manifest_path.read_bytes())
        return digest.hexdigest()

    def validate_request(self, trace_id, prompt, outputs):
        if trace_id not in self.records:
            raise ValueError(f"unknown expert_trace_request_id {trace_id!r}")
        record = self.records[trace_id]
        if int(prompt) != int(record["prefill_length"]):
            raise ValueError(f"prompt length disagrees with expert trace {trace_id}")
        if not 1 <= int(outputs) <= int(record["decode_length"]) + 1:
            raise ValueError(f"output length exhausts expert trace {trace_id}")

    def selections(self, spans, layer):
        """Return [real token, top-k] IDs, marking synthetic prompt tokens."""
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"invalid expert layer {layer}")
        chunks, prompt_tokens = [], 0
        for span in spans:
            if span.trace_request_id not in self.records:
                raise ValueError(f"missing expert trace binding for request {span.request_id}")
            record = self.records[span.trace_request_id]
            if span.prompt_length != int(record["prefill_length"]) or span.start < 0 or span.count < 0:
                raise ValueError("invalid immutable expert token span")
            prompt_count = min(span.count, max(0, span.prompt_length - span.start))
            if prompt_count:
                # Deterministic balanced top-k, with identity stable across chunks.
                positions = np.arange(span.start, span.start + prompt_count, dtype=np.int64)
                chunks.append((positions[:, None] * self.top_k + np.arange(self.top_k)) % self.num_experts)
                prompt_tokens += prompt_count
            decode_count = span.count - prompt_count
            if decode_count:
                begin = max(span.start, span.prompt_length) - span.prompt_length
                if begin + decode_count > int(record["decode_length"]):
                    raise ValueError(f"expert trace exhausted for {span.trace_request_id}")
                rows = self.decode_rows(span.trace_request_id, begin, decode_count, layer)
                if np.any(rows < 0) or np.any(rows >= self.num_experts):
                    raise ValueError(f"out-of-range expert ID in {span.trace_request_id}")
                if np.any(np.diff(np.sort(rows, axis=1), axis=1) == 0):
                    raise ValueError(f"duplicate top-k expert in {span.trace_request_id}")
                chunks.append(rows)
        ids = np.concatenate(chunks).astype(np.int64, copy=False) if chunks else np.empty((0, self.top_k), dtype=np.int64)
        ids.setflags(write=False)
        return ids, prompt_tokens


def routing_from_ids(ids, num_experts, ep_size, source_tokens=None, devices=None, owners=None):
    """Count tokens once per destination rank, and pairs once per expert."""
    from .gate_function import RoutingResult
    counts = np.bincount(ids.reshape(-1), minlength=num_experts)
    owners = (np.arange(num_experts) * ep_size // num_experts if owners is None
              else np.asarray(owners))
    selected = np.ones(num_experts, dtype=bool) if devices is None else np.asarray(devices) == "GPU"
    local, active, by_rank = [], [], []
    for rank in range(ep_size):
        mask = (owners == rank) & selected
        local.append(int(np.count_nonzero(np.any(mask[ids], axis=1))))
        active.append(int(np.count_nonzero(counts * mask)))
        by_rank.append((counts * mask).tolist())
    return RoutingResult(local, active, list(source_tokens or [0] * ep_size), counts.tolist(), by_rank,
                         token_expert_ids=ids)


def source_buffer_tokens(batches, tp_size):
    return math.ceil(max((sum(s.count for s in b.token_spans) for b in batches), default=0) / tp_size)
