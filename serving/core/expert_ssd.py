"""Atomic layer admission under independent, fixed-window local SSD budgets."""

import collections
import copy
import math


class ExpertSSDBudget:
    window_ns = 1_000_000_000

    def __init__(self, gib_per_second, queue_policy='newest_first', weight_bytes=0):
        if (isinstance(gib_per_second, bool) or not isinstance(gib_per_second, (int, float))
                or not math.isfinite(gib_per_second) or gib_per_second <= 0):
            raise ValueError('ssd_read_budget_gib must be finite and positive, or null for unlimited')
        if queue_policy not in ('newest_first', 'fifo'):
            raise ValueError('ssd_queue_policy must be newest_first or fifo')
        self.capacity = int(gib_per_second * 2**30)
        if self.capacity < weight_bytes or weight_bytes <= 0:
            raise ValueError('SSD budget must fit at least one complete expert copy')
        self.gib_per_second, self.queue_policy = gib_per_second, queue_policy
        self.weight_bytes = weight_bytes
        self.pending, self.windows, self.events = {}, {}, []
        self.stats = collections.Counter()
        for name in ('requested_bytes', 'admitted_bytes', 'cancelled_bytes', 'requested_layer_updates',
                     'admitted_layer_updates', 'cancelled_layer_updates', 'deferred_layer_updates',
                     'coalesced_proposals', 'blocked_admission_attempts', 'peak_pending_bytes'):
            self.stats[name] = 0
        self.blocked_windows = set()
        self.last_check = -1
        self.last_time_ns = -1

    def synchronize(self, proposal, anchors, now_ns, completed):
        """Replace stale unstarted intent; identical targets keep queue age."""
        if completed == self.last_check:
            raise ValueError('SSD intent may be refreshed only once per policy check')
        if proposal is not None:
            layers = [event['layer'] for event in proposal[3]]
            if len(set(layers)) != len(layers):
                raise ValueError('SSD admission requires one complete transaction per layer')
        self.last_check = completed
        wanted = {}
        if proposal is not None:
            residency, gpu, pim, events = proposal
            for event in events:
                layer = event['layer']
                values = tuple(residency[layer, e] for e in range(len(gpu[layer])))
                signature = (values, tuple(gpu[layer]), tuple(pim[layer]))
                loads = collections.Counter((device, endpoint) for device, endpoint, _ in event['local_loads'])
                costs = {key: count * self.weight_bytes for key, count in loads.items()}
                if max(costs.values(), default=0) > self.capacity:
                    raise ValueError('atomic layer update exceeds a full per-device SSD window')
                item = dict(layer=layer, values=values, gpu=tuple(gpu[layer]), pim=tuple(pim[layer]),
                    signature=signature, costs=costs, event=copy.deepcopy(event),
                    anchor=copy.deepcopy(anchors.get(layer)), proposed_ns=now_ns,
                    proposed_round=completed, validated_round=completed)
                old = self.pending.get(layer)
                if old is not None and old['signature'] == signature:
                    item['proposed_ns'], item['proposed_round'] = old['proposed_ns'], old['proposed_round']
                    self.stats['coalesced_proposals'] += 1
                else:
                    if old is not None:
                        self._cancel(old, now_ns, 'superseded')
                    self.stats['requested_bytes'] += sum(costs.values())
                    self.stats['requested_layer_updates'] += 1
                    self.events.append(dict(kind='enqueue', time_ns=now_ns, round=completed,
                                            layer=layer, bytes=sum(costs.values())))
                wanted[layer] = item
        for layer in self.pending.keys() - wanted.keys():
            self._cancel(self.pending[layer], now_ns, 'no_longer_qualified')
        self.pending = wanted
        self.stats['peak_pending_bytes'] = max(self.stats['peak_pending_bytes'], self.pending_bytes)

    def _cancel(self, item, now_ns, reason):
        size = sum(item['costs'].values())
        self.stats['cancelled_bytes'] += size
        self.stats['cancelled_layer_updates'] += 1
        self.events.append(dict(kind='cancel', time_ns=now_ns, layer=item['layer'], bytes=size, reason=reason))

    @property
    def pending_bytes(self):
        return sum(sum(item['costs'].values()) for item in self.pending.values())

    def select(self, now_ns):
        if now_ns < self.last_time_ns:
            raise ValueError('SSD admission time must be monotonic')
        self.last_time_ns = now_ns
        window = now_ns // self.window_ns
        used = collections.Counter(self.windows.get(window, {}))
        sign = -1 if self.queue_policy == 'newest_first' else 1
        ordered = sorted(self.pending.values(), key=lambda item: (sign * item['proposed_ns'], item['layer']))
        selected = []
        for item in ordered:
            if any(used[key] + size > self.capacity for key, size in item['costs'].items()):
                self.stats['blocked_admission_attempts'] += 1
                self.blocked_windows.update((window, *key) for key, size in item['costs'].items()
                                            if used[key] + size > self.capacity)
                continue
            used.update(item['costs'])
            selected.append(item)
        return selected

    def committed(self, selected, now_ns, completed):
        window = now_ns // self.window_ns
        used = self.windows.setdefault(window, collections.Counter())
        for item in selected:
            if self.pending[item['layer']] is not item:
                raise RuntimeError('SSD admission intent changed before commit')
            used.update(item['costs'])
            if max(used.values(), default=0) > self.capacity:
                raise RuntimeError('per-device SSD window capacity exceeded')
            size = sum(item['costs'].values())
            self.stats['admitted_bytes'] += size
            self.stats['admitted_layer_updates'] += 1
            delay = now_ns - item['proposed_ns']
            self.stats['deferred_layer_updates'] += delay > 0
            self.events.append(dict(kind='admit', time_ns=now_ns, round=completed, layer=item['layer'],
                bytes=size, proposed_ns=item['proposed_ns'], proposed_round=item['proposed_round'],
                validated_round=item['validated_round'], delay_ns=delay,
                device_bytes=[[device, endpoint, value] for (device, endpoint), value in sorted(item['costs'].items())]))
            del self.pending[item['layer']]

    def snapshot(self):
        return dict(ssd_requested_bytes=self.stats['requested_bytes'], ssd_admitted_bytes=self.stats['admitted_bytes'],
                    ssd_cancelled_bytes=self.stats['cancelled_bytes'], ssd_pending_bytes=self.pending_bytes,
                    ssd_pending_layers=len(self.pending))

    def summary(self):
        assert self.stats['requested_bytes'] == self.stats['admitted_bytes'] + self.stats['cancelled_bytes'] + self.pending_bytes
        return dict(self.stats, gib_per_second=self.gib_per_second, capacity_bytes=self.capacity,
            window_ns=self.window_ns, queue_policy=self.queue_policy,
            model='fixed_window_admission_with_hidden_admitted_reads', initial_loading_excluded=True,
            pending_bytes=self.pending_bytes, pending_layers=len(self.pending),
            blocked_device_windows=[list(key) for key in sorted(self.blocked_windows)],
            pending=[dict(layer=item['layer'], proposed_ns=item['proposed_ns'],
                          proposed_round=item['proposed_round'], bytes=sum(item['costs'].values()))
                     for item in sorted(self.pending.values(), key=lambda item: item['layer'])],
            windows=[dict(window=window, device_bytes=[[d, e, value] for (d, e), value in sorted(used.items())])
                     for window, used in sorted(self.windows.items())], events=self.events)
