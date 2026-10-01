"""MVP-0 predictors for near-future MoE expert demand.

The module keeps the prediction contract independent of placement, migration,
and scheduling. Predictors consume ``ReplayStep`` objects from the
TracePack/ReplayStream path; the JSON import is only for the optional smoke CLI
debug print. Non-oracle predictors only consume warmup steps at initialization
and explicitly updated past decode steps thereafter. Oracle access is opt-in
and requires caller-provided future ``ReplayStep`` objects.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, Sequence, cast, runtime_checkable

from .interfaces import PerLayerExpertHistograms, PredictionFrame, ReplayStep

if TYPE_CHECKING:  # pragma: no cover - import-only typing helper
    from .model_hardware import ModelSpec

Histogram = PerLayerExpertHistograms

PREDICTOR_TYPES = {"heuristic", "oracle", "learned"}


@dataclass(frozen=True)
class PredictorConfig:
    """Validated configuration shared by MVP predictor implementations.

    ``decay`` is used by the heuristic predictor.  A value of ``1.0`` gives an
    unweighted sliding-window mean; smaller values give exponentially larger
    weight to more recent observations.  ``per_layer_histograms`` are reported
    as the mean predicted demand per decode step across the requested horizon,
    while metadata records this aggregation choice explicitly.
    """

    predictor_type: str = "heuristic"
    horizon: int = 1
    window_size: int = 32
    decay: float = 1.0
    oracle_permissions: bool = False
    use_prefill_context: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        predictor_type = self.predictor_type.lower()
        if predictor_type not in PREDICTOR_TYPES:
            raise ValueError(f"unsupported predictor type {self.predictor_type!r}; choose from {sorted(PREDICTOR_TYPES)}")
        if self.horizon <= 0:
            raise ValueError("predictor horizon must be positive")
        if self.window_size <= 0:
            raise ValueError("predictor window_size must be positive")
        if not 0.0 <= float(self.decay) <= 1.0:
            raise ValueError("predictor decay must be in [0.0, 1.0]")
        object.__setattr__(self, "predictor_type", predictor_type)
        object.__setattr__(self, "decay", float(self.decay))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.predictor_type,
            "horizon": self.horizon,
            "window_size": self.window_size,
            "decay": self.decay,
            "oracle_permissions": self.oracle_permissions,
            "use_prefill_context": self.use_prefill_context,
            "metadata": self.metadata,
        }


@runtime_checkable
class Predictor(Protocol):
    """Common predictor interface used by the simulation control loop."""

    def initialize(self, warmup_steps: Iterable[ReplayStep], prefill_context: Mapping[str, Any] | None = None) -> None:
        ...

    def update(self, replay_step: ReplayStep) -> None:
        ...

    def predict(self, step_id: int, horizon: int) -> PredictionFrame:
        ...


def predictor_config_from_dict(data: Mapping[str, Any] | PredictorConfig | None = None) -> PredictorConfig:
    """Create ``PredictorConfig`` from a root or ``predictor:`` config mapping."""

    if data is None:
        return PredictorConfig()
    if isinstance(data, PredictorConfig):
        return data
    cfg = data.get("predictor", data)
    if not isinstance(cfg, Mapping):
        raise ValueError("predictor config must be a mapping")

    known = {
        "type",
        "predictor_type",
        "horizon",
        "window_size",
        "window",
        "decay",
        "oracle_permissions",
        "allow_oracle",
        "use_prefill_context",
        "metadata",
    }
    metadata = dict(cfg.get("metadata") or {})
    extra = {str(k): v for k, v in cfg.items() if k not in known}
    if extra:
        metadata.setdefault("extra_config", extra)

    return PredictorConfig(
        predictor_type=str(cfg.get("type", cfg.get("predictor_type", "heuristic"))),
        horizon=_positive_int(cfg.get("horizon", 1), "predictor.horizon"),
        window_size=_positive_int(cfg.get("window_size", cfg.get("window", 32)), "predictor.window_size"),
        decay=float(cfg.get("decay", 1.0)),
        oracle_permissions=bool(cfg.get("oracle_permissions", cfg.get("allow_oracle", False))),
        use_prefill_context=bool(cfg.get("use_prefill_context", False)),
        metadata=metadata,
    )


def make_predictor(
    config: Mapping[str, Any] | PredictorConfig | None = None,
    *,
    model: "ModelSpec | None" = None,
    oracle_steps: Iterable[ReplayStep] | Mapping[int, ReplayStep] | None = None,
) -> Predictor:
    """Instantiate an MVP predictor from config."""

    cfg = predictor_config_from_dict(config)
    if cfg.predictor_type == "heuristic":
        return HeuristicPredictor(cfg, model=model)
    if cfg.predictor_type == "oracle":
        return OraclePredictor(cfg, oracle_steps=oracle_steps, model=model)
    raise NotImplementedError("learned predictor is intentionally deferred beyond MVP-0")


class BasePredictor:
    """Shared state and ordering checks for non-learned MVP predictors."""

    predictor_name = "base"
    is_oracle = False

    def __init__(self, config: PredictorConfig | Mapping[str, Any] | None = None, *, model: "ModelSpec | None" = None):
        self.config = predictor_config_from_dict(config)
        self.model = model
        self._history: collections.deque[Histogram] = collections.deque(maxlen=self.config.window_size)
        self._observed_universe: dict[int, set[int]] = {}
        self._predicted_step_ids: set[int] = set()
        self._updated_step_ids: set[int] = set()
        self._last_observed_step_id: int | None = None
        self._warmup_step_count = 0
        self._prefill_context_count = 0
        self._initialized = False

    def initialize(self, warmup_steps: Iterable[ReplayStep], prefill_context: Mapping[str, Any] | None = None) -> None:
        """Initialize from warmup history only.

        Passing steps explicitly marked as ``phase: eval`` is rejected so callers
        do not accidentally preload future evaluation ground truth into a
        non-oracle predictor.
        """

        self._reset_state()
        if prefill_context:
            self._prefill_context_count = len(prefill_context)
            if self.config.use_prefill_context:
                for histogram in _prefill_context_histograms(prefill_context):
                    self._append_history(histogram)

        for step in warmup_steps:
            if step.metadata.get("phase") == "eval":
                raise ValueError("predictor.initialize accepts warmup/history steps only, not eval steps")
            self._append_history(_histogram_from_replay_step(step))
            self._updated_step_ids.add(int(step.step_id))
            self._last_observed_step_id = int(step.step_id)
            self._warmup_step_count += 1
        self._initialized = True
        self._after_initialize()

    def _reset_state(self) -> None:
        self._history = collections.deque(maxlen=self.config.window_size)
        self._observed_universe = {}
        self._predicted_step_ids = set()
        self._updated_step_ids = set()
        self._last_observed_step_id = None
        self._warmup_step_count = 0
        self._prefill_context_count = 0
        self._initialized = False

    def _after_initialize(self) -> None:
        """Hook for subclasses."""

    def update(self, replay_step: ReplayStep) -> None:
        self._validate_update(replay_step)
        self._append_history(_histogram_from_replay_step(replay_step))
        self._updated_step_ids.add(int(replay_step.step_id))
        self._last_observed_step_id = int(replay_step.step_id)

    def predict(self, step_id: int, horizon: int) -> PredictionFrame:
        raise NotImplementedError

    def _validate_predict(self, step_id: int, horizon: int) -> int:
        if not self._initialized:
            raise RuntimeError("predictor must be initialized before predict")
        step_id = int(step_id)
        horizon = _positive_int(horizon, "horizon")
        if not self.is_oracle and self._last_observed_step_id is not None and step_id <= self._last_observed_step_id:
            raise RuntimeError(
                f"cannot predict step {step_id}: step has already been observed through {self._last_observed_step_id}"
            )
        self._predicted_step_ids.add(step_id)
        return horizon

    def _validate_update(self, replay_step: ReplayStep) -> None:
        if not self._initialized:
            raise RuntimeError("predictor must be initialized before update")
        step_id = int(replay_step.step_id)
        if step_id in self._updated_step_ids:
            raise ValueError(f"step {step_id} has already been used to update predictor state")
        if self._last_observed_step_id is not None and step_id <= self._last_observed_step_id:
            raise ValueError(
                f"predictor updates must be strictly increasing; last observed {self._last_observed_step_id}, got {step_id}"
            )
        if not self.is_oracle and replay_step.metadata.get("phase") != "warmup" and step_id not in self._predicted_step_ids:
            raise RuntimeError(
                f"non-oracle predictor update for step {step_id} is only allowed after predict({step_id}, ...)"
            )

    def _append_history(self, histogram: Histogram) -> None:
        copied = _normalize_histogram_by_layer(_copy_histogram(histogram), model=self.model)
        self._history.append(copied)
        self._observe_universe(copied)

    def _observe_universe(self, histogram: Histogram) -> None:
        for layer_id, layer_counts in histogram.items():
            experts = self._observed_universe.setdefault(int(layer_id), set())
            experts.update(int(expert_id) for expert_id in layer_counts)

    def _base_metadata(self, *, horizon: int) -> dict[str, Any]:
        return {
            "predictor_type": self.predictor_name,
            "config": self.config.to_dict(),
            "history_observations": len(self._history),
            "warmup_steps": self._warmup_step_count,
            "prefill_context_requests": self._prefill_context_count,
            "prefill_context_used": self.config.use_prefill_context,
            "horizon_aggregation": "mean_predicted_demand_per_step",
            "horizon": horizon,
        }


class HeuristicPredictor(BasePredictor):
    """MVP-0 sliding/decayed history-average predictor.

    This follows the high-level sliding-window idea from ``../qwen3_prdt`` while
    normalizing every layer to expert proportions.  For a multi-step horizon it
    recursively appends each lead prediction and returns the mean predicted
    expert-proportion histogram over the horizon.
    """

    predictor_name = "heuristic"

    def __init__(self, config: PredictorConfig | Mapping[str, Any] | None = None, *, model: "ModelSpec | None" = None):
        super().__init__(config, model=model)
        self._horizon_one_prediction_cache: Histogram | None = None
        self._horizon_one_weighted_sum: Histogram = {}

    def _reset_state(self) -> None:
        super()._reset_state()
        self._horizon_one_prediction_cache = None
        self._horizon_one_weighted_sum = {}

    def _append_history(self, histogram: Histogram) -> None:
        evicted = self._history[0] if len(self._history) == self.config.window_size else None
        super()._append_history(histogram)
        self._update_horizon_one_accumulator(self._history[-1], evicted=evicted)
        self._horizon_one_prediction_cache = None

    def _update_horizon_one_accumulator(
        self,
        appended: Mapping[int, Mapping[int, float]],
        *,
        evicted: Mapping[int, Mapping[int, float]] | None,
    ) -> None:
        decay = self.config.decay
        if decay <= 0.0:
            self._horizon_one_weighted_sum = _copy_histogram(appended)
            return

        if decay >= 1.0:
            if evicted is not None:
                _add_scaled_normalized(self._horizon_one_weighted_sum, evicted, -1.0)
            _add_scaled_normalized(self._horizon_one_weighted_sum, appended, 1.0)
            return

        evicted_scale = decay ** self.config.window_size
        _scale_histogram_in_place(self._horizon_one_weighted_sum, decay)
        if evicted is not None:
            _add_scaled_normalized(self._horizon_one_weighted_sum, evicted, -evicted_scale)
        _add_scaled_normalized(self._horizon_one_weighted_sum, appended, 1.0)

    def predict(self, step_id: int, horizon: int) -> PredictionFrame:
        horizon = self._validate_predict(step_id, horizon)
        if horizon == 1:
            if self._horizon_one_prediction_cache is None:
                self._horizon_one_prediction_cache = _normalize_accumulated_normalized(self._horizon_one_weighted_sum)
            cached_histogram = self._horizon_one_prediction_cache
            histogram = _copy_histogram(cached_histogram)
        else:
            histogram = _recursive_history_prediction(
                list(self._history),
                horizon=horizon,
                window_size=self.config.window_size,
                decay=self.config.decay,
            )
        return PredictionFrame(
            step_id=int(step_id),
            horizon=horizon,
            per_layer_histograms=histogram,
            metadata=self._base_metadata(horizon=horizon),
        )


class OraclePredictor(BasePredictor):
    """Explicit future-ground-truth predictor for upper-bound studies."""

    predictor_name = "oracle"
    is_oracle = True

    def __init__(
        self,
        config: PredictorConfig | Mapping[str, Any] | None = None,
        *,
        oracle_steps: Iterable[ReplayStep] | Mapping[int, ReplayStep] | None = None,
        model: "ModelSpec | None" = None,
    ):
        super().__init__(config, model=model)
        if not self.config.oracle_permissions:
            raise ValueError("oracle predictor requires predictor config oracle_permissions: true")
        self._oracle_histograms: dict[int, Histogram] = {}
        if oracle_steps is not None:
            self.set_oracle_steps(oracle_steps)

    def set_oracle_steps(self, oracle_steps: Iterable[ReplayStep] | Mapping[int, ReplayStep]) -> None:
        histograms: dict[int, Histogram] = {}
        if isinstance(oracle_steps, Mapping):
            mapping_steps = cast(Mapping[int, ReplayStep], oracle_steps)
            for step_id, step in mapping_steps.items():
                histograms[int(step_id)] = _normalize_histogram_by_layer(
                    _histogram_from_replay_step(step),
                    model=self.model,
                )
        else:
            for step in oracle_steps:
                histograms[int(step.step_id)] = _normalize_histogram_by_layer(
                    _histogram_from_replay_step(step),
                    model=self.model,
                )
        self._oracle_histograms = histograms

    def update(self, replay_step: ReplayStep) -> None:
        self._validate_update(replay_step)
        self._updated_step_ids.add(int(replay_step.step_id))
        self._last_observed_step_id = int(replay_step.step_id)
        self._observe_universe(_normalize_histogram_by_layer(_histogram_from_replay_step(replay_step), model=self.model))

    def predict(self, step_id: int, horizon: int) -> PredictionFrame:
        horizon = self._validate_predict(step_id, horizon)
        needed = list(range(int(step_id), int(step_id) + horizon))
        missing = [future_step_id for future_step_id in needed if future_step_id not in self._oracle_histograms]
        if missing:
            raise KeyError(
                f"oracle predictor is missing future ground-truth steps for horizon {horizon}: {missing[:10]}"
            )
        histogram = _mean_histograms([self._oracle_histograms[future_step_id] for future_step_id in needed])
        metadata = self._base_metadata(horizon=horizon)
        metadata["oracle_source"] = "explicit_future_replay_steps"
        metadata["future_step_ids"] = needed
        return PredictionFrame(
            step_id=int(step_id),
            horizon=horizon,
            per_layer_histograms=histogram,
            metadata=metadata,
        )


def prediction_frame_to_dict(frame: PredictionFrame) -> dict[str, Any]:
    """Return a JSON-serializable dictionary for a prediction frame."""

    return asdict(frame)


def rank_layer_experts(frame: PredictionFrame, layer_id: int, *, descending: bool = True) -> list[tuple[int, float]]:
    """Return one layer's ``(expert_id, score)`` pairs ranked by predicted demand."""

    layer_counts = frame.per_layer_histograms.get(int(layer_id), {})
    if descending:
        return sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))
    return sorted(layer_counts.items(), key=lambda item: (item[1], item[0]))


def rank_prediction(frame: PredictionFrame, layer_id: int, *, descending: bool = True) -> list[tuple[int, float]]:
    """Backward-compatible alias for :func:`rank_layer_experts`."""

    return rank_layer_experts(frame, layer_id, descending=descending)


def _positive_int(value: Any, field_name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return out


def _histogram_from_replay_step(step: ReplayStep) -> Histogram:
    return {
        int(layer_id): {
            int(expert_id): float(count)
            for expert_id, count in sorted(layer_counts.items())
            if float(count) != 0.0
        }
        for layer_id, layer_counts in sorted(step.layer_expert_counts.items())
    }


def _prefill_context_histograms(prefill_context: Mapping[str, Any]) -> list[Histogram]:
    histograms: list[Histogram] = []
    for request_id in sorted(prefill_context):
        value = prefill_context[request_id]
        if not isinstance(value, Mapping):
            continue
        raw_counts = value.get("layer_expert_counts", {})
        if isinstance(raw_counts, Mapping):
            histograms.append(
                {
                    int(layer_id): {
                        int(expert_id): float(count)
                        for expert_id, count in sorted(cast_mapping(layer_counts).items())
                        if float(count) != 0.0
                    }
                    for layer_id, layer_counts in sorted(cast_mapping(raw_counts).items())
                }
            )
    return histograms


def cast_mapping(value: Any) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return value


def _copy_histogram(histogram: Mapping[int, Mapping[int, float]]) -> Histogram:
    return {
        int(layer_id): {
            int(expert_id): float(value)
            for expert_id, value in sorted(layer_counts.items())
            if float(value) != 0.0
        }
        for layer_id, layer_counts in sorted(histogram.items())
        if layer_counts
    }


def _add_scaled(target: Histogram, source: Mapping[int, Mapping[int, float]], scale: float) -> None:
    for layer_id, layer_counts in source.items():
        out_layer = target.setdefault(int(layer_id), {})
        for expert_id, value in layer_counts.items():
            out_layer[int(expert_id)] = out_layer.get(int(expert_id), 0.0) + float(value) * scale


def _add_scaled_normalized(target: Histogram, source: Mapping[int, Mapping[int, float]], scale: float) -> None:
    for layer_id, layer_counts in source.items():
        out_layer = target.get(layer_id)
        if out_layer is None:
            out_layer = {}
            target[layer_id] = out_layer
        for expert_id, value in layer_counts.items():
            if value != 0.0:
                out_layer[expert_id] = out_layer.get(expert_id, 0.0) + value * scale


def _scale_histogram_in_place(histogram: Histogram, scale: float) -> None:
    for layer_counts in histogram.values():
        for expert_id in list(layer_counts):
            layer_counts[expert_id] *= scale


def _normalize_accumulated_normalized(histogram: Mapping[int, Mapping[int, float]]) -> Histogram:
    out: Histogram = {}
    for layer_id, layer_counts in sorted(histogram.items()):
        total = sum(value for value in layer_counts.values() if value > 0.0)
        if total <= 0.0:
            continue
        out[layer_id] = {
            expert_id: value / total
            for expert_id, value in sorted(layer_counts.items())
            if value > 0.0
        }
    return out


def _mean_histograms(histograms: Sequence[Mapping[int, Mapping[int, float]]]) -> Histogram:
    if not histograms:
        return {}
    out: Histogram = {}
    scale = 1.0 / len(histograms)
    for histogram in histograms:
        _add_scaled(out, histogram, scale)
    return _normalize_histogram_by_layer(_drop_zeros(out))


def _mean_normalized_histograms(histograms: Sequence[Mapping[int, Mapping[int, float]]]) -> Histogram:
    if not histograms:
        return {}
    out: Histogram = {}
    scale = 1.0 / len(histograms)
    for histogram in histograms:
        _add_scaled_normalized(out, histogram, scale)
    return _normalize_accumulated_normalized(out)


def _drop_zeros(histogram: Histogram, *, eps: float = 0.0) -> Histogram:
    return {
        layer_id: {
            expert_id: value
            for expert_id, value in sorted(layer_counts.items())
            if abs(value) > eps
        }
        for layer_id, layer_counts in sorted(histogram.items())
        if any(abs(value) > eps for value in layer_counts.values())
    }


def _weighted_normalized_history_average(
    histories: Sequence[Mapping[int, Mapping[int, float]]],
    *,
    window_size: int,
    decay: float,
) -> Histogram:
    if not histories:
        return {}
    window = list(histories[-window_size:])
    return _weighted_normalized_history_average_from_window(window, decay=decay)


def _weighted_normalized_history_average_from_window(
    window: Sequence[Mapping[int, Mapping[int, float]]],
    *,
    decay: float,
) -> Histogram:
    if not window:
        return {}
    if decay >= 1.0:
        weights = [1.0] * len(window)
    elif decay <= 0.0:
        weights = [0.0] * (len(window) - 1) + [1.0]
    else:
        newest_index = len(window) - 1
        weights = [decay ** (newest_index - index) for index in range(len(window))]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        return {}
    out: Histogram = {}
    for histogram, weight in zip(window, weights, strict=True):
        if weight:
            _add_scaled_normalized(out, histogram, weight / weight_sum)
    return _normalize_accumulated_normalized(out)


def _recursive_history_prediction(histories: Sequence[Mapping[int, Mapping[int, float]]], *, horizon: int, window_size: int, decay: float) -> Histogram:
    if not histories:
        return {}
    rolling = list(histories[-window_size:])
    leads: list[Histogram] = []
    for _ in range(horizon):
        predicted = _weighted_normalized_history_average(rolling, window_size=window_size, decay=decay)
        leads.append(predicted)
        rolling.append(predicted)
        if len(rolling) > window_size:
            rolling = rolling[-window_size:]
    return _mean_normalized_histograms(leads)


def _normalize_histogram_by_layer(
    histogram: Mapping[int, Mapping[int, float]],
    *,
    model: "ModelSpec | None" = None,
) -> Histogram:
    """Normalize each layer to an expert-proportion simplex.

    This follows the qwen3_prdt predictor contract: every predicted layer is a
    distribution over experts, so forecasts are batch-size invariant and
    histogram error can be computed in probability space.
    """

    layers = range(model.num_layers) if model is not None else sorted(int(layer) for layer in histogram)
    out: Histogram = {}
    for layer_id in layers:
        layer_counts = histogram.get(layer_id, {})
        total = sum(max(0.0, float(value)) for value in layer_counts.values())
        if total <= 0.0:
            if model is not None:
                expert_count = model.experts_for_layer(layer_id)
                out[layer_id] = {expert_id: 1.0 / expert_count for expert_id in range(expert_count)}
            continue
        out[layer_id] = {
            int(expert_id): max(0.0, float(value)) / total
            for expert_id, value in sorted(layer_counts.items())
            if float(value) > 0.0
        }
    return out


def _cmd_smoke(args: argparse.Namespace) -> int:
    from .trace_pack import ReplayStream, TracePack

    with TracePack.open(args.trace_pack) as pack:
        warmup_stream = ReplayStream(
            pack,
            seed=args.seed,
            max_batch_size=args.max_batch_size,
            warmup_steps=args.warmup_steps,
            eval_steps=args.eval_steps,
            benchmarks=[args.benchmark] if args.benchmark else None,
            limit_per_benchmark=args.limit_per_benchmark,
        )
        warmup_steps = list(warmup_stream.warmup_steps())
        eval_steps = list(warmup_stream.eval_steps())
        if not eval_steps:
            raise ValueError("no eval steps available after warmup")

        config = PredictorConfig(
            predictor_type=args.predictor_type,
            horizon=args.horizon,
            window_size=args.window_size,
            decay=args.decay,
            oracle_permissions=args.predictor_type == "oracle",
        )
        oracle_steps = eval_steps if args.predictor_type == "oracle" else None
        predictor = make_predictor(config, oracle_steps=oracle_steps)
        predictor.initialize(warmup_steps)
        first_eval = eval_steps[0]
        frame = predictor.predict(first_eval.step_id, args.horizon)
    print(json.dumps(prediction_frame_to_dict(frame), sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prediction utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="initialize a predictor from a TracePack warmup window and print one frame")
    smoke.add_argument("--trace-pack", "--pack", required=True, dest="trace_pack", help="TracePack directory")
    smoke.add_argument("--benchmark", default=None, help="single benchmark to replay when pack contains multiple")
    smoke.add_argument("--predictor-type", choices=sorted(PREDICTOR_TYPES - {"learned"}), default="heuristic")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--max-batch-size", type=int, default=1)
    smoke.add_argument("--warmup-steps", type=int, default=1)
    smoke.add_argument("--eval-steps", type=int, default=None)
    smoke.add_argument("--limit-per-benchmark", type=int, default=None)
    smoke.add_argument("--horizon", type=int, default=1)
    smoke.add_argument("--window-size", type=int, default=32)
    smoke.add_argument("--decay", type=float, default=1.0)
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
