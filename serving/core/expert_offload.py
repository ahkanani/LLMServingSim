"""Schema loaders for static MoE expert offloading.

This module is intentionally side-effect free: it validates and stores external
expert-offload inputs, but B2 does not wire the data into trace generation.
Paths are resolved relative to the repository root so callers keep working even
when ``serving/__main__.py`` changes the current working directory to
``astra-sim/``.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


_VALID_STAGES = {"prefill", "decode"}
_VALID_CHANNEL_POLICIES = {"greedy_load"}
_VALID_PROFILE_TIME_SEMANTICS = {"end_to_end_excludes_transfer"}
_VALID_PLACEMENTS = {"PIM"}
_REQUIRED_MANIFEST_COLUMNS = {"layer", "experts_in_pim", "placement", "channel"}
_REQUIRED_DISTRIBUTION_COLUMNS = {"layer", "expert_id", "prob"}
_REQUIRED_PROFILE_COLUMNS = {"tokens", "number_of_expert_executed_in_pim", "time_us"}
_REQUIRED_ACTIVATION_TRANSFER_FIELDS = {
    "npu_to_pim_bw_GBps",
    "pim_to_npu_bw_GBps",
    "npu_to_pim_latency_ns",
    "pim_to_npu_latency_ns",
}


def repo_root() -> Path:
    """Return the LLMServingSim repository root for this module."""

    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path, *, base_dir: Optional[str | Path] = None) -> Path:
    """Resolve a user path against the repository root.

    ``serving/__main__.py`` changes cwd to ``astra-sim/`` before building runtime
    state. Using the module location as the default anchor makes config paths
    such as ``configs/expert_offload/foo.csv`` resolve the same way from repo
    root, from ``astra-sim/``, and from unit-level scripts.
    """

    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    anchor = Path(base_dir).resolve() if base_dir is not None else repo_root()
    return (anchor / raw).resolve()


def _require_columns(path: Path, fieldnames: Optional[Sequence[str]], required: Set[str]) -> None:
    present = set(fieldnames or [])
    missing = sorted(required - present)
    if missing:
        raise KeyError(f"{path}: missing required CSV column(s): {', '.join(missing)}")


def _read_csv(path: Path, required: Set[str]) -> List[Mapping[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path}: CSV file does not exist")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _require_columns(path, reader.fieldnames, required)
        return list(reader)


def _parse_layer(value: str, *, path: Path, row_num: int, num_layers: Optional[int]) -> Optional[int]:
    value = (value or "").strip()
    if value == "*":
        return None
    if value == "":
        raise ValueError(f"{path}: row {row_num}: layer must be '*' or a non-negative integer")
    try:
        layer = int(value)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_num}: invalid layer '{value}'") from exc
    if layer < 0:
        raise ValueError(f"{path}: row {row_num}: layer must be non-negative, got {layer}")
    if num_layers is not None and layer >= num_layers:
        raise ValueError(
            f"{path}: row {row_num}: layer {layer} out of range for num_layers={num_layers}")
    return layer


def _parse_expert_id(
    value: str,
    *,
    path: Path,
    row_num: int,
    num_experts: Optional[int],
    allow_wildcard: bool = False,
) -> Optional[int]:
    value = (value or "").strip()
    if allow_wildcard and value == "*":
        return None
    if value == "":
        raise ValueError(f"{path}: row {row_num}: expert id must be a non-negative integer")
    try:
        expert_id = int(value)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_num}: invalid expert id {value!r}") from exc
    if expert_id < 0:
        raise ValueError(f"{path}: row {row_num}: expert id must be non-negative, got {expert_id}")
    if num_experts is not None and expert_id >= num_experts:
        raise ValueError(
            f"{path}: row {row_num}: expert id {expert_id} out of range for num_experts={num_experts}")
    return expert_id


def _parse_expert_count(
    value: str,
    *,
    path: Path,
    row_num: int,
    num_experts: Optional[int],
) -> int:
    value = (value or "").strip()
    if value == "":
        raise ValueError(f"{path}: row {row_num}: experts_in_pim must be a non-negative integer count")
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_num}: invalid experts_in_pim count {value!r}") from exc
    if count < 0:
        raise ValueError(f"{path}: row {row_num}: experts_in_pim count must be non-negative, got {count}")
    if num_experts is not None and count > num_experts:
        raise ValueError(
            f"{path}: row {row_num}: experts_in_pim count {count} out of range for num_experts={num_experts}")
    return count


def _parse_optional_channel(value: str, *, path: Path, row_num: int) -> Optional[int]:
    value = (value or "").strip()
    if value == "":
        return None
    try:
        channel = int(value)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_num}: invalid channel '{value}'") from exc
    if channel < 0:
        raise ValueError(f"{path}: row {row_num}: channel must be non-negative, got {channel}")
    return channel


@dataclass(frozen=True)
class ColdExpertPlacement:
    layer: Optional[int]
    expert_id: int
    placement: str = "PIM"
    channel: Optional[int] = None


@dataclass(frozen=True)
class ColdExpertCountRule:
    """Count-based PIM placement rule from ``cold_experts.csv``.

    ``experts_in_pim`` is intentionally a count, not an explicit expert id.
    Concrete cold expert ids are derived from the routing distribution by taking
    the lowest-probability experts for the queried layer.
    """

    layer: Optional[int]
    experts_in_pim: int
    placement: str = "PIM"
    channel: Optional[int] = None


@dataclass
class ColdExpertManifest:
    """Static externally supplied count of experts assigned to PIM.

    The manifest declares how many cold experts should be placed in PIM for a
    layer.  It does not list explicit expert ids.  Callers derive ids with
    ``experts_for_layer(layer, distribution=...)`` so the routing distribution
    remains the single source of expert hot/cold ordering.
    """

    path: Path
    num_experts: Optional[int] = None
    num_layers: Optional[int] = None
    wildcard: Optional[ColdExpertCountRule] = None
    per_layer: Dict[int, ColdExpertCountRule] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        num_experts: Optional[int] = None,
        num_layers: Optional[int] = None,
        base_dir: Optional[str | Path] = None,
    ) -> "ColdExpertManifest":
        resolved = resolve_repo_path(path, base_dir=base_dir)
        rows = _read_csv(resolved, _REQUIRED_MANIFEST_COLUMNS)
        manifest = cls(path=resolved, num_experts=num_experts, num_layers=num_layers)
        seen_layers: Set[Optional[int]] = set()
        for row_num, row in enumerate(rows, start=2):
            layer = _parse_layer(row["layer"], path=resolved, row_num=row_num, num_layers=num_layers)
            if layer in seen_layers:
                label = "*" if layer is None else str(layer)
                raise ValueError(f"{resolved}: row {row_num}: duplicate cold expert count for layer {label}")
            seen_layers.add(layer)
            experts_in_pim = _parse_expert_count(
                row["experts_in_pim"], path=resolved, row_num=row_num, num_experts=num_experts)
            placement = (row.get("placement") or "").strip().upper()
            if placement not in _VALID_PLACEMENTS:
                raise ValueError(
                    f"{resolved}: row {row_num}: unsupported placement '{row.get('placement')}'. "
                    f"Supported: {sorted(_VALID_PLACEMENTS)}")
            channel = _parse_optional_channel(row.get("channel", ""), path=resolved, row_num=row_num)
            rule = ColdExpertCountRule(
                layer=layer, experts_in_pim=experts_in_pim, placement=placement, channel=channel)
            if layer is None:
                manifest.wildcard = rule
            else:
                manifest.per_layer[layer] = rule
        if manifest.wildcard is None and not manifest.per_layer:
            raise ValueError(f"{resolved}: cold expert manifest has no rows")
        return manifest

    def count_rule_for_layer(self, layer: int) -> ColdExpertCountRule:
        if layer < 0:
            raise ValueError(f"layer must be non-negative, got {layer}")
        rule = self.per_layer.get(layer)
        if rule is not None:
            return rule
        if self.wildcard is not None:
            return self.wildcard
        raise ValueError(f"{self.path}: layer {layer} has no experts_in_pim count and no layer='*' fallback")

    def count_for_layer(self, layer: int) -> int:
        return self.count_rule_for_layer(layer).experts_in_pim

    def cold_expert_ids_for_layer(
        self,
        layer: int,
        distribution: "ExpertDistribution",
        *,
        num_experts: Optional[int] = None,
    ) -> List[int]:
        n = num_experts if num_experts is not None else self.num_experts
        if n is None:
            n = distribution.num_experts
        if n is None:
            raise ValueError("num_experts is required to derive cold expert ids")
        rule = self.count_rule_for_layer(layer)
        if rule.experts_in_pim > n:
            raise ValueError(
                f"{self.path}: layer {layer}: experts_in_pim count {rule.experts_in_pim} "
                f"out of range for num_experts={n}")
        if rule.experts_in_pim == 0:
            return []
        probs = distribution.probabilities_for_layer(layer, num_experts=n)

        # For equal-probability ties, do not bias PIM placement toward low
        # expert ids.  Low expert ids often map to the same EP shard/rank, so
        # using expert_id as the tie-breaker can offload only one rank and hide
        # the expected MoE speedup.  Keep probability as the primary key (lower
        # probability still goes to PIM first), but break exact ties with a
        # deterministic pseudo-random value.  Include the layer in the hash so
        # equal-probability wildcard distributions spread PIM experts across
        # layers while remaining reproducible across runs.
        def tie_break(expert_id: int) -> int:
            key = f"cold_expert_random_tie_v1:{layer}:{expert_id}".encode("utf-8")
            return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

        order = sorted(range(n), key=lambda expert_id: (probs[expert_id], tie_break(expert_id)))
        return order[:rule.experts_in_pim]

    def placements_for_layer(
        self,
        layer: int,
        distribution: Optional["ExpertDistribution"] = None,
        *,
        num_experts: Optional[int] = None,
    ) -> List[ColdExpertPlacement]:
        if distribution is None:
            raise ValueError(
                "ColdExpertManifest.placements_for_layer requires an ExpertDistribution; "
                "experts_in_pim is a count, not explicit expert ids")
        rule = self.count_rule_for_layer(layer)
        return [
            ColdExpertPlacement(layer=layer, expert_id=expert_id, placement=rule.placement, channel=rule.channel)
            for expert_id in self.cold_expert_ids_for_layer(layer, distribution, num_experts=num_experts)
        ]

    def experts_for_layer(
        self,
        layer: int,
        distribution: Optional["ExpertDistribution"] = None,
        *,
        num_experts: Optional[int] = None,
    ) -> Set[int]:
        if distribution is None:
            raise ValueError(
                "ColdExpertManifest.experts_for_layer requires an ExpertDistribution; "
                "experts_in_pim is a count, not explicit expert ids")
        return set(self.cold_expert_ids_for_layer(layer, distribution, num_experts=num_experts))

    def has_layer_specific_rows(self) -> bool:
        """Return True when PIM expert counts can vary by transformer layer."""
        return bool(self.per_layer)


@dataclass
class ExpertDistribution:
    """CSV-backed probability model for synthetic expert routing."""

    path: Path
    num_experts: Optional[int] = None
    num_layers: Optional[int] = None
    wildcard_probs: Dict[Optional[int], float] = field(default_factory=dict)
    per_layer_probs: Dict[int, Dict[Optional[int], float]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        num_experts: Optional[int] = None,
        num_layers: Optional[int] = None,
        base_dir: Optional[str | Path] = None,
    ) -> "ExpertDistribution":
        resolved = resolve_repo_path(path, base_dir=base_dir)
        rows = _read_csv(resolved, _REQUIRED_DISTRIBUTION_COLUMNS)
        dist = cls(path=resolved, num_experts=num_experts, num_layers=num_layers)
        for row_num, row in enumerate(rows, start=2):
            layer = _parse_layer(row["layer"], path=resolved, row_num=row_num, num_layers=num_layers)
            expert_id = _parse_expert_id(
                row["expert_id"], path=resolved, row_num=row_num, num_experts=num_experts,
                allow_wildcard=True)
            prob_s = (row.get("prob") or "").strip()
            try:
                prob = float(prob_s)
            except ValueError as exc:
                raise ValueError(f"{resolved}: row {row_num}: invalid prob '{prob_s}'") from exc
            if not math.isfinite(prob) or prob < 0:
                raise ValueError(f"{resolved}: row {row_num}: prob must be finite and non-negative, got {prob_s}")
            target = dist.wildcard_probs if layer is None else dist.per_layer_probs.setdefault(layer, {})
            target[expert_id] = prob
        if not dist.wildcard_probs and not dist.per_layer_probs:
            raise ValueError(f"{resolved}: distribution CSV has no rows")
        return dist

    def has_layer_specific_rows(self) -> bool:
        """Return True when probabilities can vary by transformer layer."""
        return bool(self.per_layer_probs)

    def probabilities_for_layer(self, layer: int, *, num_experts: Optional[int] = None) -> List[float]:
        n = num_experts if num_experts is not None else self.num_experts
        if n is None:
            raise ValueError("num_experts is required to expand an expert distribution")
        if n <= 0:
            raise ValueError(f"num_experts must be positive, got {n}")
        if layer < 0:
            raise ValueError(f"layer must be non-negative, got {layer}")

        entries: Dict[Optional[int], float] = dict(self.wildcard_probs)
        layer_entries = self.per_layer_probs.get(layer)
        if layer_entries is None and not self.wildcard_probs:
            raise ValueError(
                f"{self.path}: layer {layer} has no distribution and no layer='*' fallback")
        if layer_entries:
            entries.update(layer_entries)
        default = entries.get(None)
        probs: List[Optional[float]] = [default for _ in range(n)]
        for expert_id, prob in entries.items():
            if expert_id is None:
                continue
            if expert_id >= n:
                raise ValueError(
                    f"{self.path}: expert id {expert_id} out of range for num_experts={n}")
            probs[expert_id] = prob
        missing = [idx for idx, prob in enumerate(probs) if prob is None]
        if missing:
            raise ValueError(
                f"{self.path}: layer {layer} missing probability for expert(s) {missing}; "
                "provide explicit rows or an expert_id='*' fallback")
        numeric = [float(prob) for prob in probs]
        total = sum(numeric)
        if total <= 0:
            raise ValueError(f"{self.path}: layer {layer} distribution sum must be positive")
        return [prob / total for prob in numeric]

    def allocate_counts_for_layer(
        self,
        layer: int,
        total_pairs: int,
        *,
        num_experts: Optional[int] = None,
    ) -> List[int]:
        """Allocate expert-token pairs for ``layer`` with largest-remainder rounding.

        The CSV stores probabilities, while the simulator needs an integer
        expert-token histogram. Use largest-remainder rounding, but break ties
        randomly so lower expert ids do not systematically receive extra tokens
        when fractional remainders are equal.
        """

        if total_pairs < 0:
            raise ValueError(f"total_pairs must be non-negative, got {total_pairs}")
        probs = self.probabilities_for_layer(layer, num_experts=num_experts)
        raw = [prob * total_pairs for prob in probs]
        counts = [int(math.floor(value)) for value in raw]
        remainder = total_pairs - sum(counts)
        if remainder > 0:
            order = list(range(len(raw)))
            random.shuffle(order)
            order.sort(key=lambda idx: raw[idx] - counts[idx], reverse=True)
            for idx in order[:remainder]:
                counts[idx] += 1
        return counts


@dataclass(frozen=True)
class PIMExpertProfileRow:
    tokens: int
    number_of_expert_executed_in_pim: int
    time_us: float

    @property
    def time_ns(self) -> float:
        return self.time_us * 1000.0


@dataclass
class PIMExpertProfileDB:
    """Validated PIM expert runtime table with deterministic 2D lookup.

    The profile CSV maps ``(tokens, number_of_expert_executed_in_pim)`` to
    end-to-end PIM expert runtime in microseconds.  Runtime lookups return
    nanoseconds and use the same linear interpolation/extrapolation policy as
    the NPU MoE table: interpolate over tokens within each expert-count row,
    then interpolate over the expert-count axis.
    """

    path: Path
    rows: Dict[Tuple[int, int], PIMExpertProfileRow]
    expert_count_values: List[int] = field(default_factory=list)
    rows_by_expert_count: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        base_dir: Optional[str | Path] = None,
    ) -> "PIMExpertProfileDB":
        resolved = resolve_repo_path(path, base_dir=base_dir)
        rows = _read_csv(resolved, _REQUIRED_PROFILE_COLUMNS)
        table: Dict[Tuple[int, int], PIMExpertProfileRow] = {}
        for row_num, row in enumerate(rows, start=2):
            try:
                tokens = int((row.get("tokens") or "").strip())
            except ValueError as exc:
                raise ValueError(f"{resolved}: row {row_num}: invalid tokens '{row.get('tokens')}'") from exc
            try:
                experts = int((row.get("number_of_expert_executed_in_pim") or "").strip())
            except ValueError as exc:
                raise ValueError(
                    f"{resolved}: row {row_num}: invalid number_of_expert_executed_in_pim "
                    f"'{row.get('number_of_expert_executed_in_pim')}'") from exc
            try:
                time_us = float((row.get("time_us") or "").strip())
            except ValueError as exc:
                raise ValueError(f"{resolved}: row {row_num}: invalid time_us '{row.get('time_us')}'") from exc
            if tokens <= 0:
                raise ValueError(f"{resolved}: row {row_num}: tokens must be positive, got {tokens}")
            if experts <= 0:
                raise ValueError(
                    f"{resolved}: row {row_num}: number_of_expert_executed_in_pim must be positive, got {experts}")
            if not math.isfinite(time_us) or time_us < 0:
                raise ValueError(f"{resolved}: row {row_num}: time_us must be finite and non-negative, got {time_us}")
            key = (tokens, experts)
            if key in table:
                raise ValueError(f"{resolved}: duplicate PIM expert profile key tokens={tokens}, experts={experts}")
            table[key] = PIMExpertProfileRow(tokens=tokens, number_of_expert_executed_in_pim=experts, time_us=time_us)
        if not table:
            raise ValueError(f"{resolved}: profile CSV has no rows")

        rows_by_expert_count: Dict[int, Dict[str, List[float]]] = {}
        for experts in sorted({key[1] for key in table}):
            expert_rows = [row for (tokens, row_experts), row in table.items() if row_experts == experts]
            expert_rows.sort(key=lambda item: item.tokens)
            rows_by_expert_count[experts] = {
                "tokens": [row.tokens for row in expert_rows],
                "time_ns": [row.time_ns for row in expert_rows],
            }
        return cls(
            path=resolved,
            rows=table,
            expert_count_values=sorted(rows_by_expert_count),
            rows_by_expert_count=rows_by_expert_count,
        )

    def get_exact(self, tokens: int, number_of_expert_executed_in_pim: int) -> PIMExpertProfileRow:
        key = (tokens, number_of_expert_executed_in_pim)
        try:
            return self.rows[key]
        except KeyError as exc:
            raise KeyError(
                f"{self.path}: missing PIM expert profile row for tokens={tokens}, "
                f"number_of_expert_executed_in_pim={number_of_expert_executed_in_pim}") from exc

    @staticmethod
    def _linear_interpolate(x0: float, y0: float, x1: float, y1: float, query: float) -> float:
        if x1 == x0:
            return y0
        t = (query - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @classmethod
    def _lookup_bounds(cls, values: Sequence[int], query: int) -> Tuple[int, int]:
        if not values:
            raise KeyError("empty PIM expert profile axis")
        if len(values) == 1:
            return 0, 0
        if query <= values[0]:
            return 0, 0
        for idx, value in enumerate(values):
            if query == value:
                return idx, idx
            if query < value:
                return idx - 1, idx
        return len(values) - 2, len(values) - 1

    @classmethod
    def _lookup_1d(cls, keys: Sequence[int], values: Sequence[float], query: int) -> float:
        if not keys:
            raise KeyError("empty PIM expert token table")
        if len(keys) != len(values):
            raise ValueError("PIM expert profile token/value table length mismatch")
        lo, hi = cls._lookup_bounds(keys, query)
        if lo == hi:
            return float(values[lo])
        return cls._linear_interpolate(keys[lo], values[lo], keys[hi], values[hi], query)

    def _lookup_for_expert_count(self, expert_count: int, tokens: int) -> float:
        table = self.rows_by_expert_count.get(expert_count)
        if table is None or not table.get("tokens"):
            raise KeyError(
                f"{self.path}: missing PIM expert profile token table for "
                f"number_of_expert_executed_in_pim={expert_count}; available expert counts: "
                f"{sorted(self.rows_by_expert_count)}")
        return self._lookup_1d(table["tokens"], table["time_ns"], tokens)

    def lookup_time_ns(self, tokens: int, number_of_expert_executed_in_pim: int) -> int:
        """Return interpolated/extrapolated PIM expert runtime in ns.

        Exact CSV keys return the exact CSV value converted to ns. Other
        lookups follow ``trace_generator._lookup_moe`` semantics: first do 1D
        token interpolation/extrapolation within each expert-count table, then
        linearly interpolate/extrapolate over the
        ``number_of_expert_executed_in_pim`` axis. Queries below the minimum
        profiled axis value clamp to that minimum, matching the existing MoE
        lookup helper.
        """
        tokens = int(tokens)
        experts = int(number_of_expert_executed_in_pim)
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        if experts <= 0:
            raise ValueError(f"number_of_expert_executed_in_pim must be positive, got {experts}")

        row = self.rows.get((tokens, experts))
        if row is not None:
            return max(0, int(round(row.time_ns)))
        if not self.expert_count_values:
            raise KeyError(f"{self.path}: empty PIM expert profile")
        lo, hi = self._lookup_bounds(self.expert_count_values, experts)
        lo_experts = self.expert_count_values[lo]
        val_lo = self._lookup_for_expert_count(lo_experts, tokens)
        if lo == hi:
            return max(0, int(round(val_lo)))
        hi_experts = self.expert_count_values[hi]
        val_hi = self._lookup_for_expert_count(hi_experts, tokens)
        value_ns = self._linear_interpolate(lo_experts, val_lo, hi_experts, val_hi, experts)
        return max(0, int(round(value_ns)))

    def lookup_time_us(self, tokens: int, number_of_expert_executed_in_pim: int) -> float:
        return self.lookup_time_ns(tokens, number_of_expert_executed_in_pim) / 1000.0

    # Short aliases for future trace-generation call sites.
    def lookup_ns(self, tokens: int, number_of_expert_executed_in_pim: int) -> int:
        return self.lookup_time_ns(tokens, number_of_expert_executed_in_pim)

    def lookup_us(self, tokens: int, number_of_expert_executed_in_pim: int) -> float:
        return self.lookup_time_us(tokens, number_of_expert_executed_in_pim)



@dataclass(frozen=True)
class ActivationTransferConfig:
    """NPU<->CPU/PIM activation movement parameters for cold experts.

    Bandwidths are expressed in decimal GB/s, which is numerically equal to
    bytes/ns. Fixed link/setup latencies are charged once per aggregate
    transfer row.
    """

    npu_to_pim_bw_GBps: float
    pim_to_npu_bw_GBps: float
    npu_to_pim_latency_ns: int
    pim_to_npu_latency_ns: int

    @staticmethod
    def _parse_positive_float(raw: object, *, key: str) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expert_offload.activation_transfer.{key} must be a positive number") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"expert_offload.activation_transfer.{key} must be positive and finite, got {raw!r}")
        return value

    @staticmethod
    def _parse_nonnegative_int(raw: object, *, key: str) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expert_offload.activation_transfer.{key} must be a non-negative integer") from exc
        if value < 0:
            raise ValueError(
                f"expert_offload.activation_transfer.{key} must be non-negative, got {value}")
        return value

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, object]]) -> Optional["ActivationTransferConfig"]:
        if raw in (None, ""):
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("expert_offload.activation_transfer must be an object")
        missing = sorted(_REQUIRED_ACTIVATION_TRANSFER_FIELDS - set(raw.keys()))
        if missing:
            raise ValueError(
                "expert_offload.activation_transfer missing required field(s): "
                + ", ".join(missing))
        return cls(
            npu_to_pim_bw_GBps=cls._parse_positive_float(
                raw.get("npu_to_pim_bw_GBps"), key="npu_to_pim_bw_GBps"),
            pim_to_npu_bw_GBps=cls._parse_positive_float(
                raw.get("pim_to_npu_bw_GBps"), key="pim_to_npu_bw_GBps"),
            npu_to_pim_latency_ns=cls._parse_nonnegative_int(
                raw.get("npu_to_pim_latency_ns"), key="npu_to_pim_latency_ns"),
            pim_to_npu_latency_ns=cls._parse_nonnegative_int(
                raw.get("pim_to_npu_latency_ns"), key="pim_to_npu_latency_ns"),
        )

    @staticmethod
    def _latency_ns(num_bytes: int, bw_GBps: float, fixed_latency_ns: int) -> int:
        num_bytes = int(num_bytes)
        if num_bytes <= 0:
            return 0
        # 1 decimal GB/s == 1 byte/ns. Use ceil so nonzero bytes always
        # pay at least one bandwidth cycle in addition to fixed latency.
        return int(fixed_latency_ns) + int(math.ceil(num_bytes / float(bw_GBps)))

    def npu_to_pim_time_ns(self, num_bytes: int) -> int:
        return self._latency_ns(num_bytes, self.npu_to_pim_bw_GBps, self.npu_to_pim_latency_ns)

    def pim_to_npu_time_ns(self, num_bytes: int) -> int:
        return self._latency_ns(num_bytes, self.pim_to_npu_bw_GBps, self.pim_to_npu_latency_ns)

@dataclass
class ExpertOffloadConfig:
    """Runtime schema object for expert offloading configuration."""

    enabled: bool = False
    apply_to: Tuple[str, ...] = ("decode",)
    cold_experts_csv: Optional[Path] = None
    routing_distribution_csv: Optional[Path] = None
    pim_expert_profile_csv: Optional[Path] = None
    channel_policy: str = "greedy_load"
    pim_profile_time_semantics: Optional[str] = None
    activation_transfer: Optional[ActivationTransferConfig] = None

    @classmethod
    def from_dict(
        cls,
        raw: Optional[Mapping[str, object]],
        *,
        enabled: bool = False,
        base_dir: Optional[str | Path] = None,
    ) -> "ExpertOffloadConfig":
        data = dict(raw or {})
        apply_to_raw = data.get("apply_to", ["decode"])
        if isinstance(apply_to_raw, str):
            apply_to = (apply_to_raw,)
        else:
            try:
                apply_to = tuple(str(v) for v in apply_to_raw)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError("expert_offload.apply_to must be a string or list of strings") from exc
        invalid = sorted(set(apply_to) - _VALID_STAGES)
        if invalid:
            raise ValueError(
                f"expert_offload.apply_to contains unsupported stage(s): {invalid}. "
                f"Supported: {sorted(_VALID_STAGES)}")
        channel_policy = str(data.get("channel_policy", "greedy_load"))
        if channel_policy not in _VALID_CHANNEL_POLICIES:
            raise ValueError(
                f"expert_offload.channel_policy '{channel_policy}' is unsupported. "
                f"Supported: {sorted(_VALID_CHANNEL_POLICIES)}")
        raw_time_semantics = data.get("pim_profile_time_semantics")
        profile_time_semantics = (
            None if raw_time_semantics in (None, "") else str(raw_time_semantics)
        )
        if (profile_time_semantics is not None
                and profile_time_semantics not in _VALID_PROFILE_TIME_SEMANTICS):
            raise ValueError(
                f"expert_offload.pim_profile_time_semantics '{profile_time_semantics}' is unsupported. "
                f"Supported: {sorted(_VALID_PROFILE_TIME_SEMANTICS)}")

        activation_transfer = ActivationTransferConfig.from_dict(data.get("activation_transfer"))

        def optional_path(key: str) -> Optional[Path]:
            value = data.get(key)
            if value in (None, ""):
                return None
            return resolve_repo_path(str(value), base_dir=base_dir)

        return cls(
            enabled=enabled,
            apply_to=apply_to,
            cold_experts_csv=optional_path("cold_experts_csv"),
            routing_distribution_csv=optional_path("routing_distribution_csv"),
            pim_expert_profile_csv=optional_path("pim_expert_profile_csv"),
            channel_policy=channel_policy,
            pim_profile_time_semantics=profile_time_semantics,
            activation_transfer=activation_transfer,
        )

    def load_manifest(self, *, num_experts: int, num_layers: Optional[int] = None) -> ColdExpertManifest:
        if self.cold_experts_csv is None:
            raise ValueError("expert_offload.cold_experts_csv is required to load a cold expert manifest")
        return ColdExpertManifest.load(self.cold_experts_csv, num_experts=num_experts, num_layers=num_layers)

    def load_distribution(self, *, num_experts: int, num_layers: Optional[int] = None) -> ExpertDistribution:
        if self.routing_distribution_csv is None:
            raise ValueError("expert_offload.routing_distribution_csv is required to load an expert distribution")
        return ExpertDistribution.load(self.routing_distribution_csv, num_experts=num_experts, num_layers=num_layers)

    def load_profile_db(self) -> PIMExpertProfileDB:
        if self.pim_expert_profile_csv is None:
            raise ValueError("expert_offload.pim_expert_profile_csv is required to load a PIM expert profile")
        return PIMExpertProfileDB.load(self.pim_expert_profile_csv)
