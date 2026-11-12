"""Anchored frequency hysteresis with atomic, endpoint-local expert replacement."""

import collections
from dataclasses import asdict, dataclass
import math
from time import perf_counter, process_time

import numpy as np


@dataclass(frozen=True)
class FrequencyMigrationConfig:
    mode: str = 'frequency_hysteresis'
    interval: int = 64
    fast_window: int = 8
    cooldown: int = 4

    def __post_init__(self):
        if self.mode != 'frequency_hysteresis':
            raise ValueError('group migration supports only frequency_hysteresis or none')
        for name in ('interval', 'fast_window'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.cooldown) is not int or self.cooldown < 0:
            raise ValueError('cooldown must be a nonnegative integer')


def reassign_slots(values, gpu, pim, target):
    """Refill only vacated physical slots for a complete frequency assignment."""
    if collections.Counter(values) != collections.Counter(target):
        raise ValueError('target residency changed per-layer budgets')
    owners, loads = [], []
    for device, old in (('GPU', gpu), ('PIM', pim)):
        new = [-1] * len(values)
        free, added = [], []
        for expert, (before, after) in enumerate(zip(values, target)):
            had, has = before in (device, 'BOTH'), after in (device, 'BOTH')
            if had and has:
                new[expert] = old[expert]
            elif had:
                free.append(old[expert])
            elif has:
                added.append(expert)
        for endpoint, expert in zip(free, added):
            new[expert] = endpoint
            loads.append((device, endpoint, expert))
        owners.append(tuple(new))
    return tuple(target), *owners, tuple(loads)


@dataclass(frozen=True)
class FrequencyHysteresisConfig:
    relative_margin: float = 0.25
    persistence: int = 1
    gate: str = 'anchored'
    gate_floor: float = 0.03
    noise_quantile: float = 0.99

    def __post_init__(self):
        for name in ('relative_margin', 'gate_floor', 'noise_quantile'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.relative_margin < 0 or not 0 <= self.gate_floor <= 1:
            raise ValueError('invalid margin or gate threshold')
        if not 0 < self.noise_quantile < 1:
            raise ValueError('noise_quantile must be between zero and one')
        if type(self.persistence) is not int or not 1 <= self.persistence <= 32767:
            raise ValueError('persistence must be a positive integer no greater than 32767')
        if self.gate != 'anchored':
            raise ValueError('frequency hysteresis supports only the anchored gate')


class HysteresisExpertMigration:
    """Rank exchanges retain incumbents until observed advantages persist.

    GPU/BOTH/PIM have rank priorities 2/1/0. This
    priority is a ranking objective, not a prediction of device service time.
    Owners change only on committed replacement; BOTH dispatch stays adaptive.
    """

    def __init__(self, config, model, cost, calibration_routes, **options):
        self.config = FrequencyMigrationConfig(**config)
        self.model, self.cost = model, cost
        self.hysteresis = FrequencyHysteresisConfig(**options)
        history = np.stack([self.histogram(ids) for ids in calibration_routes])
        self.histories = collections.deque(history[-self.config.fast_window:], maxlen=self.config.fast_window)
        self.anchor = history.mean(axis=0)
        window = self.config.fast_window
        if len(history) >= window:
            cumulative = np.concatenate((np.zeros_like(history[:1]), history.cumsum(axis=0)))
            recent = (cumulative[window:] - cumulative[:-window]) / window
            noise = np.abs(recent - self.anchor).sum(axis=2) / 2
            threshold = np.quantile(noise, self.hysteresis.noise_quantile, axis=0)
        else:
            threshold = np.zeros(model.num_layers)
        self.gate_thresholds = np.maximum(self.hysteresis.gate_floor, threshold)
        self.streaks = np.zeros((model.num_layers, model.experts_per_layer, model.experts_per_layer),
                               dtype=np.int16)
        self.proposal_anchors = {}
        self.completed = 0
        self.last_moves, self.previous_moves = {}, {}
        self.events, self.checks = [], []
        self.stats = collections.Counter()
        self.ssd_bytes_gpu = [0] * cost.gpu_count
        self.ssd_bytes_pim = [0] * cost.groups.count
        self._last_observation = self._last_decision = -1
        self.wall_seconds = 0
        self.decision_seconds, self.decision_cpu_seconds = [], []

    def histogram(self, ids):
        counts = np.stack([np.bincount(ids[:, layer].reshape(-1), minlength=self.model.experts_per_layer)
                           for layer in range(self.model.num_layers)]).astype(float)
        total = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, total, out=np.zeros_like(counts), where=total != 0)

    def observe(self, step_id, works):
        if step_id <= self._last_observation:
            raise ValueError('migration observations must be completed in increasing order')
        self._last_observation = step_id
        counts = np.asarray([work.counts for work in works], dtype=float)
        total = counts.sum(axis=1, keepdims=True)
        self.histories.append(np.divide(counts, total, out=np.zeros_like(counts), where=total != 0))
        self.completed += 1

    def decide(self, state, gpu_owners, pim_owners):
        if (not self.completed or self.completed % self.config.interval
                or self.completed == self._last_decision):
            return None
        self._last_decision = self.completed
        started, cpu_started = perf_counter(), process_time()
        history = np.asarray(self.histories)
        demand = history.mean(axis=0)
        reference = self.anchor
        distances = np.abs(demand - reference).sum(axis=1) / 2
        residency = dict(state.residency)
        gpu, pim, events = list(gpu_owners), list(pim_owners), []
        self.proposal_anchors = {}
        opened, changed_layers = [], []
        for layer in range(self.model.num_layers):
            frequencies = demand[layer]
            eligible = frequencies[:, None] > ((1 + self.hysteresis.relative_margin) * frequencies[None, :] + 1e-15)
            self.streaks[layer] = np.where(eligible,
                np.minimum(self.streaks[layer].astype(np.int32) + 1, self.hysteresis.persistence), 0)
            if distances[layer] <= self.gate_thresholds[layer]:
                self.stats['stable_layer_checks'] += 1
                continue
            opened.append(layer)
            values = tuple(residency[layer, expert] for expert in range(self.model.experts_per_layer))
            priority = np.array([{'PIM': 0, 'BOTH': 1, 'GPU': 2}[v] for v in values])
            target = priority.copy()
            ready = eligible & (self.streaks[layer] >= self.hysteresis.persistence)
            available = np.array([self.completed - self.last_moves.get((layer, e), -self.config.cooldown)
                                  >= self.config.cooldown for e in range(len(values))])
            ready &= available[:, None] & available[None, :]
            swaps = 0
            while True:
                priority_gap = target[None, :] - target[:, None]
                gain = (frequencies[:, None] - frequencies[None, :]) * priority_gap
                gain = np.where(ready & (priority_gap > 0), gain, -np.inf)
                incoming, outgoing = np.unravel_index(np.argmax(gain), gain.shape)
                if not np.isfinite(gain[incoming, outgoing]):
                    break
                target[incoming], target[outgoing] = target[outgoing], target[incoming]
                swaps += 1
            if not swaps:
                self.stats['no_qualified_exchange_layer_checks'] += 1
                continue
            classes = ('PIM', 'BOTH', 'GPU')
            wanted = tuple(classes[level] for level in target)
            values_after, gpu[layer], pim[layer], loads = reassign_slots(values, gpu[layer], pim[layer], wanted)
            changed = [e for e, (before, after) in enumerate(zip(values, values_after)) if before != after]
            events.append(dict(round=self.completed, layer=layer, experts=changed,
                residency_before=[values[e] for e in changed], residency_after=[values_after[e] for e in changed],
                local_loads=[list(load) for load in loads], predicted_gain_ns=None,
                trigger='persistent_frequency_advantage', gate_distance=float(distances[layer]),
                gate_threshold=float(self.gate_thresholds[layer]), rank_exchanges=swaps))
            residency.update(((layer, e), value) for e, value in enumerate(values_after))
            self.proposal_anchors[layer] = frequencies.copy()
            changed_layers.append(layer)
        self.checks.append(dict(round=self.completed, fired_layers=opened, active_layers=changed_layers,
            max_distance=float(max(distances, default=0)), exchanges=len(events),
            stable_layers=self.model.num_layers - len(opened)))
        elapsed = perf_counter() - started
        self.wall_seconds += elapsed
        self.decision_seconds.append(elapsed)
        self.decision_cpu_seconds.append(process_time() - cpu_started)
        return (residency, gpu, pim, events) if events else None

    def committed(self, events, anchors=None):
        anchors = self.proposal_anchors if anchors is None else anchors
        for event in events:
            layer = event['layer']
            self.anchor[layer] = anchors[layer]
            self.events.append(event)
            self.stats['placement_exchanges'] += 1
            self.stats['expert_replacements'] += len(event['local_loads'])
            for expert, before, value in zip(event['experts'], event['residency_before'], event['residency_after']):
                previous = self.previous_moves.get((layer, expert))
                if previous is not None and previous[1] == value:
                    self.stats['replacement_reversals'] += 1
                self.previous_moves[layer, expert] = value, before
                self.last_moves[layer, expert] = self.completed
            for device, endpoint, _ in event['local_loads']:
                target = self.ssd_bytes_gpu if device == 'GPU' else self.ssd_bytes_pim
                target[endpoint] += self.cost.weight_bytes
                self.stats['local_ssd_load_bytes'] += self.cost.weight_bytes
        self.proposal_anchors = {}

    def summary(self):
        return dict(self.stats, mode=self.config.mode, config=asdict(self.config),
            hysteresis_options=asdict(self.hysteresis), gate_thresholds=self.gate_thresholds.tolist(),
            network_weight_bytes=0, ssd_load_time_ns=0, ssd_load_model='fully_hidden_endpoint_local_reads',
            local_ssd_load_bytes_by_gpu=self.ssd_bytes_gpu, local_ssd_load_bytes_by_pim_group=self.ssd_bytes_pim,
            checks=self.checks, events=self.events, policy_wall_seconds=self.wall_seconds,
            decision_seconds=self.decision_seconds, decision_cpu_seconds=self.decision_cpu_seconds)
