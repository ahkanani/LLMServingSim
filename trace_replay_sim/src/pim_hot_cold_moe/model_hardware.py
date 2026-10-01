"""Model-shape and aggregate GPU/PIM hardware definitions for MVP-0.

This module is intentionally independent of predictor, placement, scheduler, and
metrics code. It owns config loading, unit normalization, hardware-ratio helpers,
and validation that TracePack metadata is compatible with the selected model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ROOT_KEY = "model"
HARDWARE_ROOT_KEY = "hardware"
OFFLOAD_BACKENDS = {"pim", "remote_memory"}


@dataclass(frozen=True)
class ModelSpec:
    """Trace-compatible model-shape contract shared by the simulator.

    ``num_layers`` matches the TracePack layer-slot dimension, including
    dense/non-MoE slots that may have empty routing. Units are explicit:
    ``expert_weight_bytes`` is the byte cost of one routed expert MLP at a
    layer. MVP-0 supports either uniform values or per-layer dictionaries for
    future heterogeneous models.
    """

    model_id: str
    num_layers: int
    experts_per_layer: int | dict[int, int]
    top_k: int
    expert_weight_bytes: int | dict[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        _validate_uniform_or_layer_ints(
            self.experts_per_layer,
            field_name="experts_per_layer",
            num_layers=self.num_layers,
        )
        _validate_uniform_or_layer_ints(
            self.expert_weight_bytes,
            field_name="expert_weight_bytes",
            num_layers=self.num_layers,
        )

    def experts_for_layer(self, layer_id: int) -> int:
        """Return configured expert count for ``layer_id``."""

        return _value_for_layer(self.experts_per_layer, layer_id, self.num_layers, "experts_per_layer")

    def expert_weight_bytes_for_layer(self, layer_id: int | None = None) -> int:
        """Return the per-expert byte cost.

        ``layer_id`` may be omitted for models with a uniform byte cost. It is
        required when ``expert_weight_bytes`` is per-layer.
        """

        if isinstance(self.expert_weight_bytes, int):
            return self.expert_weight_bytes
        if layer_id is None:
            raise ValueError("layer_id is required when expert_weight_bytes is per-layer")
        return _value_for_layer(self.expert_weight_bytes, layer_id, self.num_layers, "expert_weight_bytes")

    def total_expert_count(self) -> int:
        """Return the model-wide routed expert count."""

        if isinstance(self.experts_per_layer, int):
            return self.num_layers * self.experts_per_layer
        return sum(self.experts_for_layer(layer_id) for layer_id in range(self.num_layers))

    def total_expert_weight_bytes(self) -> int:
        """Return bytes for one copy of all routed expert MLP weights."""

        total = 0
        for layer_id in range(self.num_layers):
            total += self.experts_for_layer(layer_id) * self.expert_weight_bytes_for_layer(layer_id)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "experts_per_layer": _serialize_int_or_layer_map(self.experts_per_layer),
            "top_k": self.top_k,
            "expert_weight_bytes": _serialize_int_or_layer_map(self.expert_weight_bytes),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class HardwareModel:
    """Aggregate GPU/offload hardware-latency model for MVP-0.

    Bandwidth fields are decimal terabytes/second (TB/s) per device. Latency
    helpers return seconds and implement the report equations:
    ``active_experts * W / (N_gpu * B_gpu)`` for GPU and
    ``pim_quanta * W / (N_pim * B_pim)`` for PIM. Remote-memory offload
    uses active expert fetches over aggregate remote bandwidth.
    """

    hardware_id: str
    gpu_count: int
    gpu_bw_tbps_each: float
    pim_count: int
    pim_bw_tbps_each: float
    pim_granularity: int
    offload_backend: str = "pim"
    remote_count: int | None = None
    remote_bw_tbps_each: float | None = None
    remote_capacity_gb_each: float | None = None
    hbm_capacity_gb_each: float | None = None
    hbm_cost_per_gb: float | None = None
    remote_cost_per_gb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hardware_id:
            raise ValueError("hardware_id must be non-empty")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if self.pim_count <= 0:
            raise ValueError("pim_count must be positive")
        if self.gpu_bw_tbps_each <= 0:
            raise ValueError("gpu_bw_tbps_each must be positive")
        if self.pim_bw_tbps_each <= 0:
            raise ValueError("pim_bw_tbps_each must be positive")
        if self.pim_granularity <= 0:
            raise ValueError("pim_granularity must be positive")
        backend = str(self.offload_backend).lower()
        if backend not in OFFLOAD_BACKENDS:
            raise ValueError(f"offload_backend must be one of {sorted(OFFLOAD_BACKENDS)}")
        object.__setattr__(self, "offload_backend", backend)
        if backend == "remote_memory":
            if self.remote_count is None:
                raise ValueError("remote_count is required when offload_backend is remote_memory")
            if self.remote_bw_tbps_each is None:
                raise ValueError("remote_bw_tbps_each is required when offload_backend is remote_memory")
        _validate_optional_positive_int(self.remote_count, "remote_count")
        _validate_optional_positive_float(self.remote_bw_tbps_each, "remote_bw_tbps_each")
        _validate_optional_positive_float(self.remote_capacity_gb_each, "remote_capacity_gb_each")
        _validate_optional_positive_float(self.hbm_capacity_gb_each, "hbm_capacity_gb_each")
        _validate_optional_positive_float(self.hbm_cost_per_gb, "hbm_cost_per_gb")
        _validate_optional_positive_float(self.remote_cost_per_gb, "remote_cost_per_gb")

    def total_gpu_bw_tbps(self) -> float:
        return self.gpu_count * self.gpu_bw_tbps_each

    def total_pim_bw_tbps(self) -> float:
        return self.pim_count * self.pim_bw_tbps_each

    def is_remote_memory_backend(self) -> bool:
        return self.offload_backend == "remote_memory"

    def total_remote_bw_tbps(self) -> float:
        if self.remote_count is None or self.remote_bw_tbps_each is None:
            raise ValueError("remote_count and remote_bw_tbps_each are required for remote memory bandwidth")
        return self.remote_count * self.remote_bw_tbps_each

    def offload_label(self) -> str:
        return "remote" if self.is_remote_memory_backend() else "pim"

    def hardware_ratio(self) -> float:
        """Return ``(N_pim * B_pim) / (N_gpu * B_gpu)``."""

        return self.total_pim_bw_tbps() / self.total_gpu_bw_tbps()

    def gpu_expert_time(self, active_experts: int, model: ModelSpec, layer_id: int | None = None) -> float:
        """Return GPU-side expert latency in seconds for a layer split."""

        if active_experts < 0:
            raise ValueError("active_experts must be non-negative")
        weight_bytes = model.expert_weight_bytes_for_layer(layer_id)
        return (active_experts * weight_bytes) / _tbps_to_bytes_per_second(self.total_gpu_bw_tbps())

    def pim_expert_time(self, pim_quanta: int, model: ModelSpec, layer_id: int | None = None) -> float:
        """Return PIM-side expert latency in seconds for ``pim_quanta`` reads."""

        if pim_quanta < 0:
            raise ValueError("pim_quanta must be non-negative")
        weight_bytes = model.expert_weight_bytes_for_layer(layer_id)
        return (pim_quanta * weight_bytes) / _tbps_to_bytes_per_second(self.total_pim_bw_tbps())

    def offload_expert_time(self, active_experts: int, model: ModelSpec, layer_id: int | None = None) -> float:
        """Return offload-lane latency for PIM quanta or remote active experts."""

        if active_experts < 0:
            raise ValueError("active_experts must be non-negative")
        if not self.is_remote_memory_backend():
            return self.pim_expert_time(active_experts, model, layer_id=layer_id)
        weight_bytes = model.expert_weight_bytes_for_layer(layer_id)
        return (active_experts * weight_bytes) / _tbps_to_bytes_per_second(self.total_remote_bw_tbps())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_id": self.hardware_id,
            "gpu_count": self.gpu_count,
            "gpu_bw_tbps_each": self.gpu_bw_tbps_each,
            "pim_count": self.pim_count,
            "pim_bw_tbps_each": self.pim_bw_tbps_each,
            "pim_granularity": self.pim_granularity,
            "offload_backend": self.offload_backend,
            "remote_count": self.remote_count,
            "remote_bw_tbps_each": self.remote_bw_tbps_each,
            "remote_capacity_gb_each": self.remote_capacity_gb_each,
            "hbm_capacity_gb_each": self.hbm_capacity_gb_each,
            "hbm_cost_per_gb": self.hbm_cost_per_gb,
            "remote_cost_per_gb": self.remote_cost_per_gb,
            "metadata": self.metadata,
        }


class TraceMetadataMismatch(ValueError):
    """Raised when trace metadata conflicts with a selected ``ModelSpec``."""


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON mapping config.

    YAML support uses PyYAML when available; JSON remains dependency-free.
    """

    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("YAML config support requires PyYAML; use JSON config or install pyyaml") from exc
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    return data


def model_spec_from_dict(data: Mapping[str, Any]) -> ModelSpec:
    """Create a ``ModelSpec`` from a config mapping."""

    cfg = _unwrap_root(data, MODEL_ROOT_KEY)
    required = {"model_id", "num_layers", "experts_per_layer", "top_k", "expert_weight_bytes"}
    missing = sorted(key for key in required if key not in cfg)
    if missing:
        raise ValueError(f"model config missing required fields: {missing}")

    metadata = dict(cfg.get("metadata") or {})
    for key, value in cfg.items():
        if key not in required and key != "metadata":
            metadata.setdefault(key, value)

    return ModelSpec(
        model_id=str(cfg["model_id"]),
        num_layers=_positive_int(cfg["num_layers"], "num_layers"),
        experts_per_layer=_coerce_uniform_or_layer_map(cfg["experts_per_layer"], "experts_per_layer"),
        top_k=_positive_int(cfg["top_k"], "top_k"),
        expert_weight_bytes=_coerce_uniform_or_layer_map(cfg["expert_weight_bytes"], "expert_weight_bytes"),
        metadata=metadata,
    )


def hardware_model_from_dict(data: Mapping[str, Any]) -> HardwareModel:
    """Create a ``HardwareModel`` from a config mapping."""

    cfg = _unwrap_root(data, HARDWARE_ROOT_KEY)
    required = {"hardware_id", "gpu_count", "gpu_bw_tbps_each", "pim_count", "pim_bw_tbps_each", "pim_granularity"}
    optional = {
        "offload_backend",
        "remote_count",
        "remote_bw_tbps_each",
        "remote_capacity_gb_each",
        "hbm_capacity_gb_each",
        "hbm_cost_per_gb",
        "remote_cost_per_gb",
    }
    missing = sorted(key for key in required if key not in cfg)
    if missing:
        raise ValueError(f"hardware config missing required fields: {missing}")

    metadata = dict(cfg.get("metadata") or {})
    for key, value in cfg.items():
        if key not in required and key not in optional and key != "metadata":
            metadata.setdefault(key, value)

    return HardwareModel(
        hardware_id=str(cfg["hardware_id"]),
        gpu_count=_positive_int(cfg["gpu_count"], "gpu_count"),
        gpu_bw_tbps_each=_positive_float(cfg["gpu_bw_tbps_each"], "gpu_bw_tbps_each"),
        pim_count=_positive_int(cfg["pim_count"], "pim_count"),
        pim_bw_tbps_each=_positive_float(cfg["pim_bw_tbps_each"], "pim_bw_tbps_each"),
        pim_granularity=_positive_int(cfg["pim_granularity"], "pim_granularity"),
        offload_backend=str(cfg.get("offload_backend", "pim")),
        remote_count=_optional_positive_int(cfg.get("remote_count"), "remote_count"),
        remote_bw_tbps_each=_optional_positive_float(cfg.get("remote_bw_tbps_each"), "remote_bw_tbps_each"),
        remote_capacity_gb_each=_optional_positive_float(
            cfg.get("remote_capacity_gb_each"),
            "remote_capacity_gb_each",
        ),
        hbm_capacity_gb_each=_optional_positive_float(cfg.get("hbm_capacity_gb_each"), "hbm_capacity_gb_each"),
        hbm_cost_per_gb=_optional_positive_float(cfg.get("hbm_cost_per_gb"), "hbm_cost_per_gb"),
        remote_cost_per_gb=_optional_positive_float(cfg.get("remote_cost_per_gb"), "remote_cost_per_gb"),
        metadata=metadata,
    )


def load_model_spec(path: str | Path) -> ModelSpec:
    """Load a ``ModelSpec`` from a YAML/JSON model config file."""

    return model_spec_from_dict(load_mapping_config(path))


def load_hardware_model(path: str | Path) -> HardwareModel:
    """Load a ``HardwareModel`` from a YAML/JSON hardware config file."""

    return hardware_model_from_dict(load_mapping_config(path))


def validate_trace_metadata(model: ModelSpec, trace_metadata_or_manifest: Mapping[str, Any]) -> None:
    """Validate that trace metadata is compatible with ``model``.

    ``trace_metadata_or_manifest`` may be either the manifest root or its
    ``model_metadata`` sub-mapping. Observed expert maxima from a tiny trace
    slice are allowed to be below the configured model expert count, but they may not
    exceed it.
    """

    metadata = _extract_model_metadata(trace_metadata_or_manifest)
    required = {"model_id", "num_layers", "top_k"}
    missing = sorted(key for key in required if key not in metadata)
    if missing:
        raise TraceMetadataMismatch(f"trace model_metadata missing required fields: {missing}")

    trace_model_id = str(metadata["model_id"])
    accepted_ids = {model.model_id, *(str(alias) for alias in model.metadata.get("aliases", []) or [])}
    if trace_model_id not in accepted_ids:
        raise TraceMetadataMismatch(
            f"trace model_id {trace_model_id!r} does not match selected model {model.model_id!r}"
        )

    trace_layers = _positive_int(metadata["num_layers"], "trace num_layers")
    if trace_layers != model.num_layers:
        raise TraceMetadataMismatch(
            f"trace num_layers {trace_layers} does not match selected model num_layers {model.num_layers}"
        )

    trace_top_k = _positive_int(metadata["top_k"], "trace top_k")
    if trace_top_k != model.top_k:
        raise TraceMetadataMismatch(f"trace top_k {trace_top_k} does not match selected model top_k {model.top_k}")

    for layer_id in _metadata_layer_ids(metadata):
        if layer_id < 0 or layer_id >= model.num_layers:
            raise TraceMetadataMismatch(
                f"trace contains layer {layer_id}, outside selected model layer range [0, {model.num_layers})"
            )

    observed_experts = metadata.get("experts_per_layer")
    if observed_experts is not None:
        if isinstance(observed_experts, Mapping):
            observed_map = _coerce_layer_map(observed_experts, "trace experts_per_layer")
            for layer_id, observed_count in observed_map.items():
                if layer_id < 0 or layer_id >= model.num_layers:
                    raise TraceMetadataMismatch(
                        f"trace experts_per_layer has layer {layer_id}, outside selected model layer range"
                    )
                configured = model.experts_for_layer(layer_id)
                if observed_count > configured:
                    raise TraceMetadataMismatch(
                        "trace observed more experts than the selected model allows: "
                        f"layer {layer_id} observed {observed_count}, configured {configured}"
                    )
        else:
            observed_count = _positive_int(observed_experts, "trace experts_per_layer")
            for layer_id in range(model.num_layers):
                configured = model.experts_for_layer(layer_id)
                if observed_count > configured:
                    raise TraceMetadataMismatch(
                        "trace observed more experts than the selected model allows: "
                        f"uniform observed {observed_count}, layer {layer_id} configured {configured}"
                    )


def trace_metadata_validation_report(model: ModelSpec, trace_metadata_or_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a small JSON-friendly validation report, raising on mismatch."""

    metadata = _extract_model_metadata(trace_metadata_or_manifest)
    validate_trace_metadata(model, metadata)
    return {
        "status": "ok",
        "model_id": model.model_id,
        "trace_model_id": metadata.get("model_id"),
        "num_layers": model.num_layers,
        "top_k": model.top_k,
    }


def pim_quanta_for_token_counts(token_counts: Sequence[int], pim_granularity: int) -> int:
    """Return ``sum(ceil(x_i / G))`` for PIM-routed token counts.

    This is the ``S_N^(G)`` term from the candidate-score report.
    """

    if pim_granularity <= 0:
        raise ValueError("pim_granularity must be positive")
    total = 0
    for count in token_counts:
        count_int = _non_negative_int(count, "token count")
        if count_int:
            total += math.ceil(count_int / pim_granularity)
    return total


def normalized_split_latency(
    *,
    active_experts: int,
    cold_experts: int,
    pim_quanta: int,
    hardware: HardwareModel,
) -> float:
    """Return report-normalized split latency.

    Implements ``max(S_N^(G) / R, N_active - N_cold)`` where ``R`` is the
    aggregate PIM/GPU bandwidth ratio.
    """

    active = _non_negative_int(active_experts, "active_experts")
    cold = _non_negative_int(cold_experts, "cold_experts")
    quanta = _non_negative_int(pim_quanta, "pim_quanta")
    if cold > active:
        raise ValueError("cold_experts cannot exceed active_experts")
    return max(quanta / hardware.hardware_ratio(), active - cold)


def split_latency_seconds(
    *,
    active_experts: int,
    cold_experts: int,
    pim_quanta: int,
    model: ModelSpec,
    hardware: HardwareModel,
    layer_id: int | None = None,
    latency_combine_mode: str = "max",
) -> dict[str, float]:
    """Return GPU/PIM/total latency seconds for one layer split."""

    active = _non_negative_int(active_experts, "active_experts")
    cold = _non_negative_int(cold_experts, "cold_experts")
    if cold > active:
        raise ValueError("cold_experts cannot exceed active_experts")
    gpu_time = hardware.gpu_expert_time(active - cold, model, layer_id=layer_id)
    pim_time = hardware.pim_expert_time(pim_quanta, model, layer_id=layer_id)
    if latency_combine_mode == "sum":
        total_latency = gpu_time + pim_time
    elif latency_combine_mode == "max":
        total_latency = max(gpu_time, pim_time)
    else:
        raise ValueError("latency_combine_mode must be one of ['max', 'sum']")
    return {"gpu_time": gpu_time, "pim_time": pim_time, "total_latency": total_latency}


def _unwrap_root(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key in data:
        nested = data[key]
        if not isinstance(nested, Mapping):
            raise ValueError(f"{key} config must be a mapping")
        return nested
    return data


def _extract_model_metadata(metadata_or_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = metadata_or_manifest.get("model_metadata", metadata_or_manifest)
    if not isinstance(metadata, Mapping):
        raise TraceMetadataMismatch("trace model_metadata must be a mapping")
    return metadata


def _metadata_layer_ids(metadata: Mapping[str, Any]) -> list[int]:
    layer_ids: set[int] = set()
    if "moe_layer_ids" in metadata:
        raw_layer_ids = metadata["moe_layer_ids"]
        if not isinstance(raw_layer_ids, Sequence) or isinstance(raw_layer_ids, (str, bytes)):
            raise TraceMetadataMismatch("trace moe_layer_ids must be a sequence")
        for raw_layer in raw_layer_ids:
            layer_ids.add(_non_negative_int(raw_layer, "trace moe_layer_ids item"))
    observed_experts = metadata.get("experts_per_layer")
    if isinstance(observed_experts, Mapping):
        layer_ids.update(_coerce_layer_map(observed_experts, "trace experts_per_layer").keys())
    return sorted(layer_ids)


def _coerce_uniform_or_layer_map(value: Any, field_name: str) -> int | dict[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return _positive_int(value, field_name)
    if isinstance(value, Mapping):
        return _coerce_layer_map(value, field_name)
    raise ValueError(f"{field_name} must be a positive int or layer->positive-int mapping")


def _coerce_layer_map(value: Mapping[Any, Any], field_name: str) -> dict[int, int]:
    if not value:
        raise ValueError(f"{field_name} mapping must be non-empty")
    result: dict[int, int] = {}
    for raw_layer, raw_count in value.items():
        try:
            layer_id = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} has non-integer layer id {raw_layer!r}") from exc
        if isinstance(raw_layer, bool) or layer_id < 0:
            raise ValueError(f"{field_name} layer ids must be non-negative integers")
        if layer_id in result:
            raise ValueError(f"{field_name} has duplicate layer id after integer coercion: {layer_id}")
        result[layer_id] = _positive_int(raw_count, f"{field_name}[{layer_id}]")
    return dict(sorted(result.items()))


def _tbps_to_bytes_per_second(tbps: float) -> float:
    """Convert decimal TB/s to bytes/s."""

    return tbps * 1_000_000_000_000.0


def _validate_uniform_or_layer_ints(value: int | Mapping[int, int], *, field_name: str, num_layers: int) -> None:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an int or layer->int mapping")
    if not value:
        raise ValueError(f"{field_name} mapping must be non-empty")
    for raw_layer, raw_count in value.items():
        if not isinstance(raw_layer, int) or isinstance(raw_layer, bool):
            raise ValueError(f"{field_name} layer ids must be integers")
        if raw_layer < 0 or raw_layer >= num_layers:
            raise ValueError(f"{field_name} layer id out of range: {raw_layer}")
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count <= 0:
            raise ValueError(f"{field_name}[{raw_layer}] must be a positive integer")


def _value_for_layer(value: int | Mapping[int, int], layer_id: int, num_layers: int, field_name: str) -> int:
    if layer_id < 0 or layer_id >= num_layers:
        raise ValueError(f"layer_id out of range for {field_name}: {layer_id}")
    if isinstance(value, int):
        return value
    if layer_id not in value:
        raise KeyError(f"{field_name} does not define layer {layer_id}")
    return value[layer_id]


def _serialize_int_or_layer_map(value: int | Mapping[int, int]) -> int | dict[str, int]:
    if isinstance(value, int):
        return value
    return {str(layer_id): count for layer_id, count in sorted(value.items())}


def _positive_int(value: Any, field_name: str) -> int:
    int_value = _strict_int(value, field_name, adjective="positive")
    if int_value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int_value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _validate_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _non_negative_int(value: Any, field_name: str) -> int:
    int_value = _strict_int(value, field_name, adjective="non-negative")
    if int_value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int_value


def _strict_int(value: Any, field_name: str, *, adjective: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a {adjective} integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a {adjective} integer")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a {adjective} integer") from exc
    if str(value).strip() != str(int_value) and not (isinstance(value, float) and value.is_integer()):
        raise ValueError(f"{field_name} must be a {adjective} integer")
    return int_value


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive float")
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive float") from exc
    if float_value <= 0:
        raise ValueError(f"{field_name} must be a positive float")
    return float_value


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, field_name)


def _validate_optional_positive_float(value: float | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _cmd_inspect(args: argparse.Namespace) -> int:
    model = load_model_spec(args.model_config)
    hardware = load_hardware_model(args.hardware_config)
    payload = {
        "model": model.to_dict(),
        "hardware": hardware.to_dict(),
        "derived": {
            "hardware_ratio": hardware.hardware_ratio(),
            "total_gpu_bw_tbps": hardware.total_gpu_bw_tbps(),
            "total_pim_bw_tbps": hardware.total_pim_bw_tbps(),
            "expert_weight_bytes": model.expert_weight_bytes_for_layer(0),
            "total_expert_count": model.total_expert_count(),
            "total_expert_weight_bytes": model.total_expert_weight_bytes(),
            "one_gpu_expert_time_seconds": hardware.gpu_expert_time(1, model, layer_id=0),
            "one_pim_quantum_time_seconds": hardware.pim_expert_time(1, model, layer_id=0),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_validate_trace(args: argparse.Namespace) -> int:
    model = load_model_spec(args.model_config)
    trace_payload = load_mapping_config(args.trace_metadata)
    report = trace_metadata_validation_report(model, trace_payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model/hardware config utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="load configs and print derived quantities")
    inspect.add_argument("--model-config", required=True)
    inspect.add_argument("--hardware-config", required=True)
    inspect.set_defaults(func=_cmd_inspect)

    validate = subparsers.add_parser("validate-trace", help="validate a TracePack manifest/model_metadata file")
    validate.add_argument("--model-config", required=True)
    validate.add_argument("--trace-metadata", required=True, help="TracePack manifest JSON/YAML or model_metadata file")
    validate.set_defaults(func=_cmd_validate_trace)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
