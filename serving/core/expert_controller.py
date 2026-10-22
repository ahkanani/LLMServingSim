"""Group-scoped policy decisions, physical residency, and migration reservations."""

import collections
from dataclasses import dataclass

import numpy as np

from .expert_replay import routing_from_ids, source_buffer_tokens
from .moe_policy.interfaces import ReplayStep
from .moe_policy.model_hardware import ModelSpec, HardwareModel
from .moe_policy.prediction import make_predictor
from .moe_policy.placement import make_placement_policy
from .moe_policy.migration import make_migration_policy
from .moe_policy.scheduler import make_scheduler


@dataclass(frozen=True)
class ExpertLayerPlan:
    selections: object
    routing: object
    gpu_routing: object
    devices: tuple
    pim_counts: tuple


@dataclass(frozen=True)
class ExpertRoundPlan:
    step_id: int
    placement_version: int
    layers: tuple
    source_buffer_tokens: int
    group_tokens: int
    prompt_approximation_tokens: int


class ExpertPolicyController:
    def __init__(self, pack, schedulers, config, gpu_bw_gbps, pim_channels, pim_capacity,
                 expert_weight_bytes):
        self.pack, self.schedulers, self.config = pack, schedulers, config
        self.ep_size = sum(s.num_npus for s in schedulers)
        self.tp_size = schedulers[0].tp_size
        self.channels = int(pim_channels)
        self.channel_capacity = int(pim_capacity) // self.channels
        self.weight_bytes = int(expert_weight_bytes)
        self.model = ModelSpec(pack.manifest['model_metadata']['model_id'], pack.num_layers,
                               pack.num_experts, pack.top_k, self.weight_bytes)
        self.hardware = HardwareModel('serving_replay', self.ep_size, gpu_bw_gbps / 1000,
                                      self.channels, float(config.get('pim_bw_tbps_each', 0.6)),
                                      int(config.get('pim_granularity', 1)))
        self.predictor = make_predictor(config.get('predictor',
            dict(type='heuristic', horizon=1, window_size=128, decay=0.95)), model=self.model)
        self.scheduler = make_scheduler(config.get('scheduler', dict(policy='placement_greedy')))
        self.migration = make_migration_policy(config.get('migration', dict(mode='none')))
        warmup = self._calibration(config['calibration_request_ids'], int(config.get('warmup_steps', 128)))
        self.predictor.initialize(warmup)
        self.step_id = len(warmup)
        self.prediction = self.predictor.predict(self.step_id, 1)
        self.state = make_placement_policy(config.get('placement', {})).initialize(
            warmup, self.prediction, self.model, self.hardware,
            scheduler_config=self.scheduler.config)
        self.base_weights = [s.memory.weight - pack.num_layers *
                             (pack.num_experts // self.ep_size) * self.weight_bytes for s in schedulers]
        self.pending_migration = None
        self.pending_prediction = True
        self.stats = collections.Counter()
        self.coverage = collections.Counter()
        self.rounds = []
        if not self._can_reserve(self.state.residency):
            raise ValueError('initial expert residency exceeds GPU/PIM capacity')
        self._reserve(self.state.residency)
        self.initial_residency = dict(self.state.residency)

    def owner(self, expert):
        return expert * self.ep_size // self.pack.num_experts

    def channel(self, layer, expert):
        return (layer * self.pack.num_experts + expert) % self.channels

    def _calibration(self, request_ids, limit):
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError('calibration_request_ids must be nonempty and unique')
        records = [self.pack.records[request_id] for request_id in request_ids]
        result = []
        for position in range(min(limit, max(int(r['decode_length']) for r in records))):
            active = [r for r in records if position < int(r['decode_length'])]
            ids = self.pack.decode[[int(r['request_index']) for r in active], position]
            counts = {}
            for layer in range(self.pack.num_layers):
                values = np.bincount(ids[:, layer].reshape(-1), minlength=self.pack.num_experts)
                counts[layer] = {int(e): int(values[e]) for e in np.flatnonzero(values)}
            result.append(ReplayStep(position, [r['request_id'] for r in active], counts,
                                     {r['request_id']: position for r in active}))
        return result

    def residency_bytes(self, residency):
        gpu, pim = [0] * self.ep_size, [0] * self.channels
        for (layer, expert), device in residency.items():
            if device in ('GPU', 'BOTH'):
                gpu[self.owner(expert)] += self.weight_bytes
            if device in ('PIM', 'BOTH'):
                pim[self.channel(layer, expert)] += self.weight_bytes
        return gpu, pim

    def _weights(self, residency):
        gpu, pim = self.residency_bytes(residency)
        weights, offset = [], 0
        for index, sched in enumerate(self.schedulers):
            weights.append(self.base_weights[index] + max(gpu[offset:offset + sched.num_npus]))
            offset += sched.num_npus
        return weights, pim

    def _can_reserve(self, residency):
        weights, pim = self._weights(residency)
        if max(pim, default=0) > self.channel_capacity:
            return False
        for sched, weight in zip(self.schedulers, weights):
            memory = sched.memory
            blocks = (int(memory.npu_mem * memory.npu_memory_utilization) - weight) // memory._npu_bytes_per_block
            if blocks < max(1, memory.npu_pool.used_blocks):
                return False
        return True

    def _reserve(self, residency):
        weights, _ = self._weights(residency)
        for sched, weight in zip(self.schedulers, weights):
            if not sched.memory.try_set_resident_weight(weight):
                raise RuntimeError('expert residency reservation changed after feasibility check')

    def prepare_migration(self):
        if self.pending_migration is not None:
            raise RuntimeError('migration is already in flight')
        if not self.pending_prediction:
            self.prediction = self.predictor.predict(self.step_id, 1)
        self.pending_prediction = False
        if not self.migration.should_run(self.step_id):
            return ()
        plan = self.migration.plan(self.state, self.prediction, self.model, self.hardware)
        peak, updates, accepted = dict(self.state.residency), {}, []
        for action in plan.actions:
            candidate = dict(peak)
            candidate[(action.layer, action.promote_expert)] = 'BOTH'
            if not self._can_reserve(candidate):
                self.stats['deferred_migration_pairs'] += 1
                continue
            peak = candidate
            updates.update(action.updates())
            accepted.append(action)
        if accepted:
            self._reserve(peak)
            self.pending_migration = self.state.clone_with_updates(updates)
            self.stats['migration_pairs'] += len(accepted)
            self.stats['migration_bytes'] += len(accepted) * self.weight_bytes
        return tuple(accepted)

    def commit_migration(self):
        if self.pending_migration is None:
            raise RuntimeError('no migration to commit')
        self.state = self.pending_migration
        self.pending_migration = None
        self._reserve(self.state.residency)

    def plan_forward(self, batches):
        spans = tuple(span for batch in batches for span in batch.token_spans)
        layer_ids, histograms, approximations = [], {}, 0
        source_tokens = []
        for batch in batches:
            count = sum(span.count for span in batch.token_spans)
            source_tokens.extend([count // self.tp_size + (rank < count % self.tp_size)
                                  for rank in range(self.tp_size)])
        for layer in range(self.pack.num_layers):
            ids, prompt_count = self.pack.selections(spans, layer)
            layer_ids.append(ids)
            if layer == 0:
                approximations = prompt_count
            values = np.bincount(ids.reshape(-1), minlength=self.pack.num_experts)
            histograms[layer] = {int(e): int(values[e]) for e in np.flatnonzero(values)}
        step = ReplayStep(self.step_id, [str(span.request_id) for span in spans], histograms, {})
        decision = self.scheduler.schedule(step, self.state, self.model, self.hardware)
        assignments = {(item.layer, item.expert): item.device for item in decision.assignments}
        layers = []
        for layer, ids in enumerate(layer_ids):
            devices = tuple(assignments.get((layer, expert), 'GPU') for expert in range(self.pack.num_experts))
            for expert in histograms[layer]:
                resident = self.state.get(layer, expert)
                if resident != 'BOTH' and resident != devices[expert]:
                    raise ValueError('expert scheduler selected a nonresident copy; oracle_split is diagnostic only')
            routing = routing_from_ids(ids, self.pack.num_experts, self.ep_size, source_tokens)
            gpu = routing_from_ids(ids, self.pack.num_experts, self.ep_size, source_tokens, devices)
            pim_counts = tuple(count if devices[expert] == 'PIM' else 0
                               for expert, count in enumerate(routing.per_expert_tokens))
            layers.append(ExpertLayerPlan(ids, routing, gpu, devices, pim_counts))
        buffer_tokens = source_buffer_tokens(batches, self.tp_size)
        plan = ExpertRoundPlan(self.step_id, self.state.version, tuple(layers), buffer_tokens,
                               sum(span.count for span in spans), approximations)
        self._inflight_step = step
        self.stats['prompt_approximation_tokens'] += approximations
        self.stats['expert_pairs'] += sum(sum(h.values()) for h in histograms.values())
        self.stats['pim_pairs'] += sum(sum(layer.pim_counts) for layer in layers)
        return plan

    def complete_forward(self, start, finish):
        self.predictor.update(self._inflight_step)
        self.rounds.append(dict(step_id=self.step_id, start_ns=start, finish_ns=finish,
                                placement_version=self.state.version))
        self.step_id += 1

    def summary(self):
        gpu, pim = self.residency_bytes(self.state.residency)
        return dict(self.stats, profile_coverage=dict(self.coverage),
                    gpu_expert_bytes_by_rank=gpu, pim_expert_bytes_by_channel=pim,
                    kv_blocks_by_instance=[s.memory.npu_pool.num_blocks for s in self.schedulers],
                    placement_version=self.state.version)
