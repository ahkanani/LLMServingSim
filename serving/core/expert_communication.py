"""Immutable placement-aware, hierarchical expert exchange payloads.

Matrices count FP8 hidden vectors, not expert pairs or padded source slots.
The domain leader is the first GPU in each TP domain. CPU/PIM groups have
one network endpoint each; their internal TP is part of the group model.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertExchange:
    gather: tuple
    dispatch: tuple
    scatter: tuple
    reduce: tuple
    combine: tuple
    return_tokens: tuple

    def stages(self):
        return (self.gather, self.dispatch, self.scatter, self.reduce,
                self.combine, self.return_tokens)


STAGES = ('source_gather', 'domain_dispatch', 'gpu_dispatch',
          'gpu_combine', 'domain_combine', 'source_return')


def plan_expert_exchange(selections, devices, gpu_owners, pim_owners,
                        source_ranks, tp_size, ep_size, pim_groups):
    if ep_size % tp_size or len(selections) != len(source_ranks):
        raise ValueError('expert exchange sources must match complete TP domains')
    gpu_domains = ep_size // tp_size
    domains = gpu_domains + pim_groups
    matrices = [[[0] * size for _ in range(size)]
                for size in (ep_size, domains, ep_size, ep_size, domains, ep_size)]
    gather, dispatch, scatter, reduce, combine, returns = matrices
    for source, experts in zip(source_ranks, selections):
        source_domain = source // tp_size
        source_leader = source_domain * tp_size
        if source != source_leader:
            gather[source][source_leader] += 1
            returns[source_leader][source] += 1
        gpu_targets, destinations = set(), set()
        for expert in experts:
            expert = int(expert)
            if devices[expert] == 'GPU':
                target = gpu_owners[expert]
                if not 0 <= target < ep_size:
                    raise ValueError('GPU expert has no physical owner')
                gpu_targets.add(target)
                destinations.add(target // tp_size)
            elif devices[expert] == 'PIM':
                group = pim_owners[expert]
                if not 0 <= group < pim_groups:
                    raise ValueError('PIM expert has no physical group')
                destinations.add(gpu_domains + group)
            else:
                raise ValueError('expert execution must select one resident device')
        for target in gpu_targets:
            leader = target // tp_size * tp_size
            # Retain a token already available at its original source GPU.
            if target != leader and target != source:
                scatter[leader][target] += 1
            if target != leader:
                reduce[target][leader] += 1
        for destination in destinations:
            if destination != source_domain:
                dispatch[source_domain][destination] += 1
                # Each domain reduces its local expert contributions first.
                combine[destination][source_domain] += 1
    return ExpertExchange(*(tuple(tuple(row) for row in matrix) for matrix in matrices))
