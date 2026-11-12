"""Profile-based adaptive placement and scheduling for physical replay devices.

The portable policy equations remain in moe_policy. Serving's adaptive policy
uses this evaluator because CSV coordinates require token identities and actual
GPU/PIM ownership, which aggregate expert histograms do not retain.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from .moe_policy.placement import PlacementState


def round_robin_owners(residency, ranked, gpu_count, pim_count):
    gpu, pim = [-1] * len(residency), [-1] * len(residency)
    gpu_position = pim_position = 0
    for expert in ranked:
        if residency[expert] in ('GPU', 'BOTH'):
            gpu[expert] = gpu_position % gpu_count
            gpu_position += 1
        if residency[expert] in ('PIM', 'BOTH'):
            pim[expert] = pim_position % pim_count
            pim_position += 1
    return tuple(gpu), tuple(pim)


class ExpertWork:
    """Token-instance bitsets, immutable across candidate assignments."""

    def __init__(self, selections, experts, source_ranks, gpu_count):
        masks = [0] * experts
        counts = [0] * experts
        sources = [0] * gpu_count
        if len(source_ranks) != len(selections):
            raise ValueError('profile scheduling requires one source per real token')
        for token, (row, source) in enumerate(zip(selections, source_ranks)):
            bit = 1 << token
            sources[source] |= bit
            for expert in row:
                expert = int(expert)
                masks[expert] |= bit
                counts[expert] += 1
        self.masks, self.counts, self.sources = tuple(masks), tuple(counts), tuple(sources)
        self.active = tuple(expert for expert, count in enumerate(counts) if count)
        self.active_bits = sum(1 << expert for expert in self.active)


@dataclass(frozen=True)
class ExpertCost:
    gpu_bits: int
    gpu_ns: tuple
    pim_ns: tuple
    activation_bytes: int

    @property
    def compute_ns(self):
        return max((*self.gpu_ns, *self.pim_ns), default=0)

    @property
    def score(self):
        return self.compute_ns, self.activation_bytes


class ProfileAdaptive:
    def __init__(self, lookup, gpu_count, tp_size, groups, weight_bytes, hidden_size, accelerate=False):
        if lookup is None:
            raise ValueError('adaptive group replay requires a GPU MoE CSV lookup')
        self.lookup = lru_cache(maxsize=32768)(lookup)
        self.gpu_count, self.tp_size = gpu_count, tp_size
        self.groups, self.weight_bytes, self.hidden = groups, weight_bytes, hidden_size
        self.evaluations = self.schedules = 0
        self.accelerator = None
        if accelerate:
            try:
                from .expert_profile_accel import CompiledProfileCost
            except ImportError:
                pass
            else:
                self.accelerator = CompiledProfileCost(self)

    def evaluate(self, work, gpu_bits, gpu_owners, pim_owners):
        self.evaluations += 1
        gpu_masks, gpu_active = [0] * self.gpu_count, [0] * self.gpu_count
        pim_masks, pim_quanta = [0] * self.groups.count, [0] * self.groups.count
        for expert in work.active:
            if gpu_bits & (1 << expert):
                rank = gpu_owners[expert]
                if rank < 0:
                    raise ValueError('profile candidate selects a nonresident GPU copy')
                gpu_masks[rank] |= work.masks[expert]
                gpu_active[rank] += 1
            else:
                group = pim_owners[expert]
                if group < 0:
                    raise ValueError('profile candidate selects a nonresident PIM copy')
                pim_masks[group] |= work.masks[expert]
                pim_quanta[group] += math.ceil(work.counts[expert] / self.groups.granularity)
        gpu_ns = tuple(int(self.lookup(mask.bit_count(), active)) if active else 0
                       for mask, active in zip(gpu_masks, gpu_active))
        pim_ns = tuple(math.ceil(quanta * self.weight_bytes / self.groups.bandwidth_bytes_per_ns)
                       for quanta in pim_quanta)
        # Tie-break using actual logical activation vectors, not expert pairs.
        # This is a payload tie-break, not a network-latency prediction.
        if self.groups.attachment == 'direct':
            vectors = 2 * pim_masks[0].bit_count()
        else:
            domains = self.gpu_count // self.tp_size
            source_domains, target_domains = [0] * domains, [0] * domains
            vectors = 0
            for rank, (source, target) in enumerate(zip(work.sources, gpu_masks)):
                domain = rank // self.tp_size
                source_domains[domain] |= source
                target_domains[domain] |= target
                if rank % self.tp_size:
                    vectors += 2 * source.bit_count()
                    vectors += (target & ~source).bit_count() + target.bit_count()
            targets = target_domains + pim_masks
            for source_domain, source in enumerate(source_domains):
                vectors += 2 * sum((source & target).bit_count() for domain, target in enumerate(targets)
                                   if source_domain != domain)
        return ExpertCost(gpu_bits, gpu_ns, pim_ns, vectors * self.hidden)

    def schedule(self, work, residency, gpu_owners, pim_owners):
        """Best count-sorted prefix followed by two deterministic flip passes."""
        self.schedules += 1
        if self.accelerator is not None:
            result = self.accelerator.schedule(work, residency, gpu_owners, pim_owners)
            if result is not None:
                return result
        gpu_bits = sum(1 << e for e in work.active if residency[e] in ('GPU', 'BOTH'))
        both = sorted((e for e in work.active if residency[e] == 'BOTH'),
                      key=lambda e: (work.counts[e], e))
        cache = {}

        def cost(bits):
            if bits not in cache:
                cache[bits] = self.evaluate(work, bits, gpu_owners, pim_owners)
            return cache[bits]

        best = cost(gpu_bits)
        for expert in both:
            gpu_bits &= ~(1 << expert)
            trial = cost(gpu_bits)
            if trial.score < best.score:
                best = trial
        for _ in range(2):
            changed = False
            for expert in sorted(both):
                trial = cost(best.gpu_bits ^ (1 << expert))
                if trial.score < best.score:
                    best, changed = trial, True
            if not changed:
                break
        return best

    def calibrate(self, routes, hotness, extra_fraction, feasible):
        """Learn fixed residency using measured counterfactual calibration loss.

        Full candidate ownership is rebuilt before scoring. Counterfactuals
        hold other experts' domain choices fixed, as in the original regret
        method. Complete proposals are re-scheduled and accepted only if the
        measured objective improves; the result is a bounded heuristic.
        """
        layers, experts = len(hotness), len(hotness[0])
        extra_fraction = float(extra_fraction)
        if not 0 <= extra_fraction <= 1:
            raise ValueError('extra_copy_fraction must be between zero and one')
        copies = round(experts * extra_fraction)
        ranked = [tuple(sorted(range(experts), key=lambda e: (-hotness[layer][e], e)))
                  for layer in range(layers)]
        state = {(layer, expert): 'GPU' for layer in range(layers) for expert in range(experts)}
        for layer in range(layers):
            for expert in ranked[layer][-copies:] if copies else ():
                state[layer, expert] = 'BOTH'
        if not feasible(state):
            # Offload may be required simply to fit the model. Try smaller GPU
            # seeds before rejecting a configuration with a feasible PIM home.
            for gpu_primary in ((experts - copies + 1) // 2, 0):
                candidate = {(layer, expert): 'PIM' for layer in range(layers) for expert in range(experts)}
                for layer in range(layers):
                    for expert in ranked[layer][:gpu_primary]:
                        candidate[layer, expert] = 'GPU'
                    for expert in ranked[layer][gpu_primary:gpu_primary + copies]:
                        candidate[layer, expert] = 'BOTH'
                if feasible(candidate):
                    state = candidate
                    break
            else:
                raise ValueError('adaptive calibration has no feasible hotness-based residency seed')
        summaries = []
        for layer in range(layers):
            works = routes[layer]
            order = ranked[layer]

            def ownership(values):
                return round_robin_owners(values, order, self.gpu_count, self.groups.count)

            def fits(values):
                candidate = dict(state)
                candidate.update(((layer, e), value) for e, value in enumerate(values))
                return feasible(candidate)

            def score(values):
                owners = ownership(values)
                decisions = [self.schedule(work, values, *owners) for work in works]
                return (sum(d.compute_ns for d in decisions), sum(d.activation_bytes for d in decisions)), decisions

            current = tuple(state[layer, e] for e in range(experts))
            alternatives = [current]
            for gpu_primary in ((experts - copies + 1) // 2, 0):
                candidate = ['PIM'] * experts
                for expert in order[:gpu_primary]:
                    candidate[expert] = 'GPU'
                for expert in order[gpu_primary:gpu_primary + copies]:
                    candidate[expert] = 'BOTH'
                alternatives.append(tuple(candidate))
            best_score, decisions = score(current)
            for candidate in alternatives[1:]:
                if fits(candidate):
                    trial_score, trial_decisions = score(candidate)
                    if trial_score < best_score:
                        current, best_score, decisions = candidate, trial_score, trial_decisions
            initial_score, accepted = best_score, 0
            for _ in range(4):
                loss_gpu, loss_pim, activity = [0] * experts, [0] * experts, [0] * experts
                trial_owners = {}
                for expert in range(experts):
                    for device in ('GPU', 'PIM'):
                        candidate = list(current)
                        candidate[expert] = device
                        trial_owners[expert, device] = ownership(candidate)
                regret = (self.accelerator.regret(current, works, decisions, trial_owners)
                          if self.accelerator is not None else None)
                for work, baseline in (() if regret is not None else zip(works, decisions)):
                    for expert in work.active:
                        activity[expert] += 1
                        bit = 1 << expert
                        gpu = self.evaluate(work, baseline.gpu_bits | bit, *trial_owners[expert, 'GPU'])
                        pim = self.evaluate(work, baseline.gpu_bits & ~bit, *trial_owners[expert, 'PIM'])
                        reference = min(gpu.compute_ns, pim.compute_ns)
                        loss_gpu[expert] += gpu.compute_ns - reference
                        loss_pim[expert] += pim.compute_ns - reference
                if regret is not None:
                    loss_gpu, loss_pim, activity = regret
                proposal, opposite = [], []
                for expert in range(experts):
                    if not activity[expert]:
                        primary = 'PIM' if current[expert] == 'PIM' else 'GPU'
                    else:
                        primary = 'PIM' if loss_pim[expert] < loss_gpu[expert] else 'GPU'
                    proposal.append(primary)
                    opposite.append(loss_gpu[expert] if primary == 'GPU' else loss_pim[expert])
                for expert in sorted(range(experts), key=lambda e: (-opposite[e], -activity[e], e))[:copies]:
                    proposal[expert] = 'BOTH'
                proposal = tuple(proposal)
                if proposal == current or not fits(proposal):
                    break
                trial_score, trial_decisions = score(proposal)
                if trial_score >= best_score:
                    break
                current, best_score, decisions = proposal, trial_score, trial_decisions
                accepted += 1
            state.update(((layer, e), value) for e, value in enumerate(current))
            summaries.append(dict(layer=layer, initial_compute_ns=initial_score[0],
                final_compute_ns=best_score[0], accepted_passes=accepted,
                gpu_only=current.count('GPU'), pim_only=current.count('PIM'), both=current.count('BOTH')))
        return PlacementState(0, state, dict(policy='adaptive_extra_budget',
            latency_model='gpu_csv_per_rank_pim_analytical_per_group',
            calibration_layers=summaries, extra_copy_fraction=extra_fraction,
            search='three hotness seeds; up to four regret refinement passes',
            objective='sum of per-layer max expert compute; activation bytes break ties'))

    def summary(self):
        cache = self.lookup.cache_info()
        result = dict(candidate_evaluations=self.evaluations, schedule_searches=self.schedules,
                      gpu_csv_unique_queries=cache.misses, gpu_csv_cache_hits=cache.hits)
        if self.accelerator is not None:
            result.update(backend='numba_exact_schedule_and_regret',
                evaluation_counter='final-choice Python evaluations; compiled searches counted by schedule_searches')
        return result
