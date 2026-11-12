"""Optional exact accelerator for BOTH scheduling and placement calibration.

The Python ProfileAdaptive implementation remains the reference. Batches above
256 tokens use that reference; Numba is optional for ordinary serving.
"""

import math

import numpy as np
from numba import njit
from functools import lru_cache


@njit(cache=True)
def _popcount(value):
    result = 0
    while value:
        value &= value - np.uint64(1)
        result += 1
    return result


@njit(cache=True)
def _evaluate(selected, masks, counts, sources, gpu, pim, table, gpu_count, pim_count,
              tp_size, granularity, weight_bytes, bandwidth, hidden, direct):
    words = masks.shape[1]
    gpu_masks = np.zeros((gpu_count, words), dtype=np.uint64)
    gpu_active = np.zeros(gpu_count, dtype=np.int64)
    pim_masks = np.zeros((pim_count, words), dtype=np.uint64)
    pim_quanta = np.zeros(pim_count, dtype=np.int64)
    for expert in range(len(counts)):
        if not counts[expert]:
            continue
        if selected[expert]:
            rank = gpu[expert]
            for word in range(words):
                gpu_masks[rank, word] |= masks[expert, word]
            gpu_active[rank] += 1
        else:
            group = pim[expert]
            for word in range(words):
                pim_masks[group, word] |= masks[expert, word]
            pim_quanta[group] += (counts[expert] + granularity - 1) // granularity
    compute = 0
    for rank in range(gpu_count):
        if gpu_active[rank]:
            n = 0
            for word in range(words):
                n += _popcount(gpu_masks[rank, word])
            compute = max(compute, table[n, gpu_active[rank]])
    for group in range(pim_count):
        compute = max(compute, math.ceil(pim_quanta[group] * weight_bytes / bandwidth))
    if direct:
        vectors = 0
        for word in range(words):
            vectors += _popcount(pim_masks[0, word])
        return compute, 2 * hidden * vectors
    domains = gpu_count // tp_size
    source_domains = np.zeros((domains, words), dtype=np.uint64)
    target_domains = np.zeros((domains, words), dtype=np.uint64)
    vectors = 0
    for rank in range(gpu_count):
        domain = rank // tp_size
        for word in range(words):
            source_domains[domain, word] |= sources[rank, word]
            target_domains[domain, word] |= gpu_masks[rank, word]
            if rank % tp_size:
                vectors += 2 * _popcount(sources[rank, word])
                vectors += (_popcount(gpu_masks[rank, word] & ~sources[rank, word])
                            + _popcount(gpu_masks[rank, word]))
    for domain in range(domains):
        for target in range(domains):
            if domain != target:
                for word in range(words):
                    vectors += 2 * _popcount(source_domains[domain, word] & target_domains[target, word])
        for group in range(pim_count):
            for word in range(words):
                vectors += 2 * _popcount(source_domains[domain, word] & pim_masks[group, word])
    return compute, vectors * hidden


@njit(cache=True)
def score_routes(values, gpu, pim, masks, counts, sources, table, gpu_count, pim_count,
                 tp_size, granularity, weight_bytes, bandwidth, hidden, direct):
    scores = np.empty((len(masks), 2), dtype=np.int64)
    choices = np.zeros((len(masks), len(values)), dtype=np.bool_)
    for row in range(len(masks)):
        experts = len(values)
        selected = np.zeros(experts, dtype=np.bool_)
        both = np.empty(experts, dtype=np.int64)
        n_both = 0
        for expert in range(experts):
            if counts[row, expert]:
                selected[expert] = values[expert] != 1
                if values[expert] == 2:
                    # Ascending (count, expert) order, matching Python exactly.
                    position = n_both
                    while position and counts[row, both[position - 1]] > counts[row, expert]:
                        both[position] = both[position - 1]
                        position -= 1
                    both[position] = expert
                    n_both += 1
        best_time, best_bytes = _evaluate(selected, masks[row], counts[row], sources[row], gpu, pim, table,
            gpu_count, pim_count, tp_size, granularity, weight_bytes, bandwidth, hidden, direct)
        best_selected = selected.copy()
        for index in range(n_both):
            selected[both[index]] = False
            time, size = _evaluate(selected, masks[row], counts[row], sources[row], gpu, pim, table,
                gpu_count, pim_count, tp_size, granularity, weight_bytes, bandwidth, hidden, direct)
            if time < best_time or (time == best_time and size < best_bytes):
                best_time, best_bytes = time, size
                best_selected = selected.copy()
        for _ in range(2):
            changed = False
            for expert in range(experts):
                if values[expert] != 2 or not counts[row, expert]:
                    continue
                best_selected[expert] = not best_selected[expert]
                time, size = _evaluate(best_selected, masks[row], counts[row], sources[row], gpu, pim, table,
                    gpu_count, pim_count, tp_size, granularity, weight_bytes, bandwidth, hidden, direct)
                if time < best_time or (time == best_time and size < best_bytes):
                    best_time, best_bytes, changed = time, size, True
                else:
                    best_selected[expert] = not best_selected[expert]
            if not changed:
                break
        scores[row, 0], scores[row, 1] = best_time, best_bytes
        choices[row] = best_selected
    return scores, choices


@njit(cache=True)
def regret_routes(selected, masks, counts, sources, trial_gpu, trial_pim, table, gpu_count, pim_count,
                  tp_size, granularity, weight_bytes, bandwidth, hidden, direct):
    experts = counts.shape[1]
    loss_gpu, loss_pim, activity = np.zeros(experts), np.zeros(experts), np.zeros(experts, dtype=np.int64)
    for row in range(len(masks)):
        choice = selected[row].copy()
        for expert in range(experts):
            if not counts[row, expert]:
                continue
            activity[expert] += 1
            old = choice[expert]
            choice[expert] = True
            gpu_time, _ = _evaluate(choice, masks[row], counts[row], sources[row],
                trial_gpu[expert, 0], trial_pim[expert, 0], table, gpu_count, pim_count,
                tp_size, granularity, weight_bytes, bandwidth, hidden, direct)
            choice[expert] = False
            pim_time, _ = _evaluate(choice, masks[row], counts[row], sources[row],
                trial_gpu[expert, 1], trial_pim[expert, 1], table, gpu_count, pim_count,
                tp_size, granularity, weight_bytes, bandwidth, hidden, direct)
            choice[expert] = old
            reference = min(gpu_time, pim_time)
            loss_gpu[expert] += gpu_time - reference
            loss_pim[expert] += pim_time - reference
    return loss_gpu, loss_pim, activity


class CompiledProfileCost:
    def __init__(self, cost):
        self.cost = cost
        self.tables = {}
        self.prepared = lru_cache(maxsize=256)(self._prepare)

    @staticmethod
    def _prepare(masks, counts, sources, words):
        low = (1 << 64) - 1
        def pack(rows):
            return np.asarray([[[int(mask >> (word * 64) & low) for word in range(words)]
                                for mask in row] for row in rows], dtype=np.uint64)
        return pack(masks), np.asarray(counts, dtype=np.int64), pack(sources)

    def _inputs(self, values, works):
        tokens = max(sum(mask.bit_count() for mask in work.sources) for work in works)
        if tokens > 256:
            return None
        key = tokens, len(values)
        if key not in self.tables:
            table = np.zeros((tokens + 1, len(values) + 1), dtype=np.int64)
            for token in range(1, tokens + 1):
                for active in range(1, len(values) + 1):
                    table[token, active] = self.cost.lookup(token, active)
            self.tables[key] = table
        masks, counts, sources = self.prepared(tuple(work.masks for work in works),
            tuple(work.counts for work in works), tuple(work.sources for work in works), max(1, (tokens + 63) // 64))
        return masks, counts, sources, self.tables[key]

    def _run(self, values, gpu, pim, works):
        inputs = self._inputs(values, works)
        if inputs is None:
            return None
        masks, counts, sources, table = inputs
        return score_routes(np.asarray([{'GPU': 0, 'PIM': 1, 'BOTH': 2}[value] for value in values], dtype=np.int8),
            np.asarray(gpu, dtype=np.int64), np.asarray(pim, dtype=np.int64),
            masks, counts, sources,
            table, self.cost.gpu_count, self.cost.groups.count, self.cost.tp_size,
            self.cost.groups.granularity, self.cost.weight_bytes, self.cost.groups.bandwidth_bytes_per_ns,
            self.cost.hidden, self.cost.groups.attachment == 'direct')

    def schedule(self, work, values, gpu, pim):
        result = self._run(values, gpu, pim, [work])
        if result is None:
            return None
        _, choices = result
        bits = sum(1 << int(expert) for expert in np.flatnonzero(choices[0]))
        return self.cost.evaluate(work, bits, gpu, pim)

    def regret(self, values, works, decisions, trial_owners):
        inputs = self._inputs(values, works)
        if inputs is None:
            return None
        masks, counts, sources, table = inputs
        selected = np.asarray([[bool(decision.gpu_bits & (1 << e)) for e in range(len(values))]
                               for decision in decisions], dtype=np.bool_)
        gpu = np.asarray([[trial_owners[e, device][0] for device in ('GPU', 'PIM')]
                          for e in range(len(values))], dtype=np.int64)
        pim = np.asarray([[trial_owners[e, device][1] for device in ('GPU', 'PIM')]
                          for e in range(len(values))], dtype=np.int64)
        return regret_routes(selected, masks, counts, sources, gpu, pim, table, self.cost.gpu_count,
            self.cost.groups.count, self.cost.tp_size, self.cost.groups.granularity, self.cost.weight_bytes,
            self.cost.groups.bandwidth_bytes_per_ns, self.cost.hidden, self.cost.groups.attachment == 'direct')
