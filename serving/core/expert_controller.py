"""Group-scoped policy decisions, physical residency, and migration reservations."""

import collections
import fcntl
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from .expert_replay import routing_from_ids, source_buffer_tokens
from .expert_communication import plan_expert_exchange, STAGES
from .expert_cost import ExpertWork, ProfileAdaptive, round_robin_owners
from .expert_frequency import HysteresisExpertMigration
from .moe_policy.interfaces import ReplayStep
from .moe_policy.model_hardware import ModelSpec, HardwareModel
from .moe_policy.prediction import make_predictor
from .moe_policy.placement import PlacementState, make_placement_policy, placement_scores, warmup_stats_to_histogram
from .moe_policy.migration import make_migration_policy
from .moe_policy.scheduler import make_scheduler


@dataclass(frozen=True)
class ExpertLayerPlan:
    selections: object
    routing: object
    gpu_routing: object
    devices: tuple
    pim_counts: tuple
    pim_group_tokens: tuple = ()
    pim_group_pairs: tuple = ()
    pim_group_runtime_ns: tuple = ()
    network_exchange: object = None
    gpu_runtime_ns: tuple = ()


@dataclass(frozen=True)
class ExpertRoundPlan:
    step_id: int
    placement_version: int
    layers: tuple
    source_buffer_tokens: int
    group_tokens: int
    prompt_approximation_tokens: int
    source_ranks: tuple = ()
    token_keys: tuple = ()


class ExpertPolicyController:
    def __init__(self, pack, schedulers, config, gpu_bw_gbps, pim_channels, pim_capacity,
                 expert_weight_bytes, groups=None, gpu_lookup=None):
        self.pack, self.schedulers, self.config = pack, schedulers, config
        self.layer_ids = pack.moe_layer_ids
        self.layer_index = {layer: index for index, layer in enumerate(self.layer_ids)}
        self.num_layers = len(self.layer_ids)
        self.ep_size = sum(s.num_npus for s in schedulers)
        self.tp_size = schedulers[0].tp_size
        self.channels = int(pim_channels)
        self.groups = groups
        self.group_mode = groups is not None
        self.channel_capacity = int(pim_capacity) // self.channels
        self.weight_bytes = int(expert_weight_bytes)
        self.model = ModelSpec(pack.manifest['model_metadata']['model_id'], self.num_layers,
                               pack.num_experts, pack.top_k, self.weight_bytes)
        self.hardware = HardwareModel('serving_replay', self.ep_size, gpu_bw_gbps / 1000,
            groups.count * groups.packages_per_group if groups else self.channels,
            groups.bandwidth_tbps_per_package if groups else float(config.get('pim_bw_tbps_each', 0.6)),
            groups.granularity if groups else int(config.get('pim_granularity', 1)))
        self.predictor = make_predictor(config.get('predictor',
            dict(type='heuristic', horizon=1, window_size=128, decay=0.95)), model=self.model)
        self.scheduler = make_scheduler(config.get('scheduler', dict(policy='placement_greedy')))
        migration_config = config.get('migration', dict(mode='none'))
        self.local_migration = None
        local_mode = migration_config.get('mode') == 'frequency_hysteresis'
        if groups and migration_config.get('mode', 'none') not in ('none', 'frequency_hysteresis'):
            raise ValueError('group migration supports only frequency_hysteresis or none')
        if not local_mode and 'hysteresis_options' in config:
            raise ValueError('hysteresis_options requires migration mode frequency_hysteresis')
        self.migration = make_migration_policy(dict(mode='none') if local_mode else migration_config)
        self.gpu_owners = self.pim_owners = None
        warmup = self._calibration(config['calibration_request_ids'], int(config.get('warmup_steps', 128)))
        self.predictor.initialize(warmup)
        self.step_id = len(warmup)
        self.prediction = self.predictor.predict(self.step_id, 1)
        if groups and self.migration.config.mode != 'none':
            raise ValueError('CPU/PIM group replay requires migration mode none')
        self.base_weights = [s.memory.weight - self.num_layers *
                             (pack.num_experts // self.ep_size) * self.weight_bytes for s in schedulers]
        self.hotness = (placement_scores(warmup_histogram=warmup_stats_to_histogram(warmup),
            prediction_histogram=self.prediction.per_layer_histograms, model=self.model,
            config=config.get('placement')) if groups else None)
        self.profile_cost = None
        self.gpu_moe_db = None
        self.calibration_cache_hit = False
        if groups and config.get('placement', {}).get('mode') == 'adaptive_extra_budget':
            if self.scheduler.config.latency_combine_mode != 'max':
                raise ValueError('adaptive serving overlaps GPU/PIM compute and requires max combination')
            self.profile_cost = ProfileAdaptive(gpu_lookup, self.ep_size, self.tp_size, groups,
                                               self.weight_bytes, config.get('hidden_size', 4096),
                                               accelerate=config.get('accelerate_profile_schedule', False))
            routes = [[] for _ in range(self.num_layers)]
            for ids in self.calibration_routes:
                sources = self._calibration_sources(len(ids))
                for layer in range(self.num_layers):
                    routes[layer].append(ExpertWork(ids[:, layer], pack.num_experts, sources, self.ep_size))
            self.state = self._calibrate_residency(routes)
        else:
            self.state = make_placement_policy(config.get('placement', {})).initialize(
                warmup, self.prediction, self.model, self.hardware,
                scheduler_config=self.scheduler.config)
        self.gpu_owners, self.pim_owners = self._physical_owners(self.state.residency) if groups else ([], [])
        self.pending_migration = None
        self.pending_prediction = True
        self.prepare_wall_seconds, self.prepare_cpu_seconds = [], []
        self.stats = collections.Counter()
        self.coverage = collections.Counter()
        self.gpu_query_shapes = collections.Counter()
        self.gpu_pairs_by_rank = [0] * self.ep_size
        self.gpu_runtime_by_rank = [0] * self.ep_size
        self.pim_pairs_by_group = [0] * (groups.count if groups else 0)
        self.pim_tokens_by_group = [0] * (groups.count if groups else 0)
        self.pim_runtime_by_group = [0] * (groups.count if groups else 0)
        self.pim_weight_read_quanta_by_group = [0] * (groups.count if groups else 0)
        self.rounds = []
        self.copy_uses = {}
        self.copy_last_device = {}
        if not self._can_reserve(self.state.residency):
            raise ValueError('initial expert residency exceeds GPU/PIM capacity')
        self._reserve(self.state.residency)
        self.initial_residency = dict(self.state.residency)
        self.initial_gpu_owners = tuple(self.gpu_owners) if groups else ()
        self.initial_pim_owners = tuple(self.pim_owners) if groups else ()
        if local_mode:
            if self.profile_cost is None:
                raise ValueError('local SSD migration requires adaptive_extra_budget CPU/PIM group replay')
            self.local_migration = HysteresisExpertMigration(migration_config, self.model, self.profile_cost,
                self.calibration_routes, **config.get('hysteresis_options', {}))
        self.ssd_budget = None
        if config.get('ssd_read_budget_gib') is not None:
            from .expert_ssd import ExpertSSDBudget
            if not groups or (self.local_migration is None and migration_config.get('mode', 'none') != 'none'):
                raise ValueError('SSD read budgets require local group placement or fixed placement')
            self.ssd_budget = ExpertSSDBudget(config['ssd_read_budget_gib'],
                config.get('ssd_queue_policy', 'newest_first'), self.weight_bytes)

    def owner(self, expert, layer=0):
        if self.group_mode:
            return self.gpu_owners[layer][expert]
        return expert * self.ep_size // self.pack.num_experts

    def channel(self, layer, expert):
        return (layer * self.pack.num_experts + expert) % self.channels

    def _calibration(self, request_ids, limit):
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError('calibration_request_ids must be nonempty and unique')
        if limit < 1:
            raise ValueError('warmup_steps must be positive')
        records = [self.pack.records[request_id] for request_id in request_ids]
        batch_size = int(self.config.get('calibration_batch_size', len(records)))
        if batch_size < 1:
            raise ValueError('calibration_batch_size must be positive')
        result = []
        self.calibration_routes = []
        sizes, token_forwards = collections.Counter(), 0
        route_hash = hashlib.sha256()
        for begin in range(0, len(records), batch_size):
            cohort = records[begin:begin + batch_size]
            for position in range(min(limit, max(int(r['decode_length']) for r in cohort))):
                active = [r for r in cohort if position < int(r['decode_length'])]
                ids = self.pack.decode_batch([r['request_id'] for r in active], position)[:, self.layer_ids, :]
                ids.setflags(write=False)
                self.calibration_routes.append(ids)
                route_hash.update(ids.tobytes())
                sizes[len(active)] += 1
                token_forwards += len(active)
                counts = {}
                for layer in range(self.num_layers):
                    values = np.bincount(ids[:, layer].reshape(-1), minlength=self.pack.num_experts)
                    counts[layer] = {int(e): int(values[e]) for e in np.flatnonzero(values)}
                result.append(ReplayStep(len(result), [r['request_id'] for r in active], counts,
                                         {r['request_id']: position for r in active}))
        self.calibration_stats = dict(requests=len(records), batch_cap=batch_size,
            per_request_forward_limit=limit, group_forwards=len(result), token_forwards=token_forwards,
            batch_size_histogram=dict(sizes), routes_sha256=route_hash.hexdigest(),
            batching='sequential disjoint cohorts, capped independently of learning pool size')
        return result

    def _calibrate_residency(self, routes):
        def learn():
            return self.profile_cost.calibrate(routes, self.hotness,
                self.config['placement'].get('extra_copy_fraction', 0.25), self._can_reserve)

        cache_file = self.config.get('calibration_cache_file')
        if cache_file is None:
            return learn()
        # Cached state is valid only for identical learning data, implementation,
        # timing profiles, geometry, memory constraints, and policy settings.
        sources = [Path(__file__), Path(__file__).with_name('expert_cost.py'),
                   Path(__file__).with_name('expert_groups.py'), Path(__file__).with_name('trace_generator.py'),
                   Path(__file__).with_name('expert_profile_accel.py'), Path(__file__).with_name('expert_replay.py')]
        sources.extend(sorted(Path(__file__).with_name('moe_policy').glob('*.py')))
        signature = dict(model=self.model.to_dict(), groups=self.groups.to_dict(),
            gpu_count=self.ep_size, tp_size=self.tp_size, hidden_size=self.config.get('hidden_size', 4096), hotness=self.hotness,
            calibration=self.calibration_stats, ids=self.config['calibration_request_ids'],
            predictor=self.config.get('predictor'), placement=self.config['placement'],
            scheduler=self.config.get('scheduler'), profile=self.config['calibration_profile_sha256'],
            memory=[dict(capacity=s.memory.npu_mem, utilization=s.memory.npu_memory_utilization,
                         base_weight=base, block_bytes=s.memory._npu_bytes_per_block)
                    for s, base in zip(self.schedulers, self.base_weights)],
            source_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in sources])
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        path = Path(cache_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['key'] == key:
                    state = PlacementState(0, {(int(l), int(e)): d for l, e, d in saved['residency']}, saved['metadata'])
                    if not self._can_reserve(state.residency):
                        raise ValueError('cached expert residency no longer fits its memory budget')
                    self.calibration_cache_hit = True
                    return state
            state = learn()
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(dict(key=key,
                residency=[[l, e, d] for (l, e), d in state.residency.items()], metadata=state.metadata)))
            temporary.replace(path)
            return state

    def _calibration_sources(self, count):
        sources = []
        members = len(self.schedulers)
        for member in range(members):
            tokens = count // members + (member < count % members)
            for rank in range(self.tp_size):
                sources.extend([member * self.tp_size + rank] *
                               (tokens // self.tp_size + (rank < tokens % self.tp_size)))
        return tuple(sources)

    def _physical_owners(self, residency):
        gpu, pim = [], []
        for layer in range(self.num_layers):
            values = tuple(residency[layer, expert] for expert in range(self.pack.num_experts))
            ranked = sorted(range(self.pack.num_experts), key=lambda e: (-self.hotness[layer][e], e))
            g, p = round_robin_owners(values, ranked, self.ep_size, self.groups.count)
            gpu.append(g)
            pim.append(p)
        return gpu, pim

    def residency_bytes(self, residency, gpu_owners=None, pim_owners=None):
        gpu, pim = [0] * self.ep_size, [0] * (self.groups.count if self.groups else self.channels)
        if self.groups and gpu_owners is None:
            if self.gpu_owners is not None and residency is self.state.residency:
                gpu_owners, pim_owners = self.gpu_owners, self.pim_owners
            else:
                gpu_owners, pim_owners = self._physical_owners(residency)
        for (layer, expert), device in residency.items():
            if device in ('GPU', 'BOTH'):
                rank = gpu_owners[layer][expert] if self.groups else self.owner(expert, layer)
                gpu[rank] += self.weight_bytes
            if device in ('PIM', 'BOTH'):
                rank = pim_owners[layer][expert] if self.groups else self.channel(layer, expert)
                pim[rank] += self.weight_bytes
        return gpu, pim

    def _weights(self, residency, gpu_owners=None, pim_owners=None):
        gpu, pim = self.residency_bytes(residency, gpu_owners, pim_owners)
        weights, offset = [], 0
        for index, sched in enumerate(self.schedulers):
            weights.append(self.base_weights[index] + max(gpu[offset:offset + sched.num_npus]))
            offset += sched.num_npus
        return weights, pim

    def _can_reserve(self, residency, gpu_owners=None, pim_owners=None):
        weights, pim = self._weights(residency, gpu_owners, pim_owners)
        if max(pim, default=0) > (self.groups.capacity_bytes if self.groups else self.channel_capacity):
            return False
        if self.groups:
            shard = math.ceil(self.weight_bytes / self.groups.packages_per_group)
            if any(size // self.weight_bytes * shard > self.groups.capacity_bytes // self.groups.packages_per_group
                   for size in pim):
                return False
        for sched, weight in zip(self.schedulers, weights):
            memory = sched.memory
            blocks = (int(memory.npu_mem * memory.npu_memory_utilization) - weight) // memory._npu_bytes_per_block
            if blocks < max(1, memory.npu_pool.used_blocks):
                return False
        return True

    def _reserve(self, residency, gpu_owners=None, pim_owners=None):
        weights, _ = self._weights(residency, gpu_owners, pim_owners)
        for sched, weight in zip(self.schedulers, weights):
            if not sched.memory.try_set_resident_weight(weight):
                raise RuntimeError('expert residency reservation changed after feasibility check')

    def prepare_migration(self, now_ns=0):
        wall, cpu = perf_counter(), process_time()
        try:
            return self._prepare_migration(now_ns)
        finally:
            self.prepare_wall_seconds.append(perf_counter() - wall)
            self.prepare_cpu_seconds.append(process_time() - cpu)

    def _prepare_migration(self, now_ns):
        if self.pending_migration is not None:
            raise RuntimeError('migration is already in flight')
        if self.local_migration is not None:
            proposal = self.local_migration.decide(self.state, self.gpu_owners, self.pim_owners)
            if self.ssd_budget is not None:
                local = self.local_migration
                if local.completed and local.completed % local.config.interval == 0 and local.completed != self.ssd_budget.last_check:
                    self.ssd_budget.synchronize(proposal, local.proposal_anchors, now_ns, local.completed)
                selected = self.ssd_budget.select(now_ns)
                if selected:
                    residency = dict(self.state.residency)
                    gpu, pim, events = list(self.gpu_owners), list(self.pim_owners), []
                    anchors = {}
                    for item in selected:
                        layer = item['layer']
                        residency.update(((layer, expert), value) for expert, value in enumerate(item['values']))
                        gpu[layer], pim[layer] = item['gpu'], item['pim']
                        event = dict(item['event'], round=local.completed, decision_round=item['validated_round'],
                                     proposed_round=item['proposed_round'], commit_time_ns=now_ns,
                                     proposal_time_ns=item['proposed_ns'], queue_delay_ns=now_ns-item['proposed_ns'])
                        events.append(event)
                        if item['anchor'] is not None:
                            anchors[layer] = item['anchor']
                    if self._commit_local_placement((residency, gpu, pim, events), anchors):
                        self.ssd_budget.committed(selected, now_ns, local.completed)
            elif proposal is not None:
                self._commit_local_placement(proposal)
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

    def _commit_local_placement(self, proposal, anchors=None):
        residency, gpu, pim, events = proposal
        for layer in range(self.num_layers):
            before = collections.Counter(self.state.get(layer, e) for e in range(self.pack.num_experts))
            after = collections.Counter(residency[layer, e] for e in range(self.pack.num_experts))
            if before != after:
                raise RuntimeError('local migration changed per-layer residency budgets')
            for device, old, new in (('GPU', self.gpu_owners[layer], gpu[layer]),
                                      ('PIM', self.pim_owners[layer], pim[layer])):
                if collections.Counter(rank for rank in old if rank >= 0) != collections.Counter(rank for rank in new if rank >= 0):
                    raise RuntimeError('local migration changed endpoint weight capacity')
                for expert, rank in enumerate(new):
                    resident = residency[layer, expert] in (device, 'BOTH')
                    limit = self.ep_size if device == 'GPU' else self.groups.count
                    if (resident and not 0 <= rank < limit) or (not resident and rank != -1):
                        raise RuntimeError('local migration has an invalid physical owner')
                    if rank >= 0 and old[expert] >= 0 and rank != old[expert]:
                        raise RuntimeError('local migration moved a retained physical copy')
        if not self._can_reserve(residency, gpu, pim):
            self.stats['deferred_local_placement_updates'] += 1
            return
        # No forward is executing here. Evicted slots can be refilled from the
        # endpoint's full-model SSD before the new version becomes visible.
        self._reserve(residency, gpu, pim)
        self.state = self.state.clone_with_updates(residency)
        self.gpu_owners, self.pim_owners = gpu, pim
        self.local_migration.committed(events, anchors)
        return True

    def commit_migration(self):
        if self.pending_migration is None:
            raise RuntimeError('no migration to commit')
        self.state = self.pending_migration
        self.pending_migration = None
        self._reserve(self.state.residency)

    def plan_forward(self, batches):
        spans = tuple(span for batch in batches for span in batch.token_spans)
        layer_ids, histograms, approximations = [], {}, 0
        source_tokens, source_ranks = [], []
        for batch in batches:
            count = sum(span.count for span in batch.token_spans)
            offset = len(source_tokens)
            counts = [count // self.tp_size + (rank < count % self.tp_size) for rank in range(self.tp_size)]
            source_tokens.extend(counts)
            source_ranks.extend(rank + offset for rank, n in enumerate(counts) for _ in range(n))
        for layer, physical_layer in enumerate(self.layer_ids):
            ids, prompt_count = self.pack.selections(spans, physical_layer)
            layer_ids.append(ids)
            if layer == 0:
                approximations = prompt_count
            values = np.bincount(ids.reshape(-1), minlength=self.pack.num_experts)
            histograms[layer] = {int(e): int(values[e]) for e in np.flatnonzero(values)}
        step = ReplayStep(self.step_id, [str(span.request_id) for span in spans], histograms, {})
        profile_decisions = []
        self._inflight_works = []
        if self.profile_cost is not None:
            assignments = {}
            for layer, ids in enumerate(layer_ids):
                work = ExpertWork(ids, self.pack.num_experts, source_ranks, self.ep_size)
                self._inflight_works.append(work)
                residency = tuple(self.state.get(layer, e) for e in range(self.pack.num_experts))
                choice = self.profile_cost.schedule(work, residency, self.gpu_owners[layer], self.pim_owners[layer])
                profile_decisions.append(choice)
                for expert in work.active:
                    assignments[layer, expert] = 'GPU' if choice.gpu_bits & (1 << expert) else 'PIM'
        else:
            decision = self.scheduler.schedule(step, self.state, self.model, self.hardware)
            assignments = {(item.layer, item.expert): item.device for item in decision.assignments}
        layers = []
        for layer, ids in enumerate(layer_ids):
            devices = tuple(assignments.get((layer, expert), 'GPU') for expert in range(self.pack.num_experts))
            for expert in histograms[layer]:
                resident = self.state.get(layer, expert)
                if resident != 'BOTH' and resident != devices[expert]:
                    raise ValueError('expert scheduler selected a nonresident copy; oracle_split is diagnostic only')
            owners = self.gpu_owners[layer] if self.groups else None
            routing = routing_from_ids(ids, self.pack.num_experts, self.ep_size, source_tokens, owners=owners)
            gpu = routing_from_ids(ids, self.pack.num_experts, self.ep_size, source_tokens, devices, owners)
            pim_counts = tuple(count if devices[expert] == 'PIM' else 0
                               for expert, count in enumerate(routing.per_expert_tokens))
            if self.groups:
                for expert, count in histograms[layer].items():
                    key, device = (layer, expert), devices[expert]
                    self.copy_uses[key] = self.copy_uses.get(key, 0) | (1 if device == 'GPU' else 2)
                    if self.state.get(layer, expert) == 'BOTH':
                        self.stats['both_' + device.lower() + '_pairs'] += count
                        self.stats['both_' + device.lower() + '_expert_forwards'] += 1
                        previous = self.copy_last_device.get(key)
                        if previous is not None and previous != device:
                            self.stats['both_execution_switches'] += 1
                        self.copy_last_device[key] = device
            if sum(map(sum, gpu.tokens_by_rank_expert)) + sum(pim_counts) != ids.size:
                raise RuntimeError('GPU/PIM assignment did not conserve recorded expert pairs')
            group_tokens, group_pairs, runtimes = [], [], []
            if self.groups:
                for group in range(self.groups.count):
                    selected = np.array([devices[e] == 'PIM' and self.pim_owners[layer][e] == group
                                         for e in range(self.pack.num_experts)])
                    tokens = tuple(map(int, np.flatnonzero(np.any(selected[ids], axis=1))))
                    pairs = [count for e, count in enumerate(pim_counts) if selected[e]]
                    group_tokens.append(tokens)
                    group_pairs.append(sum(pairs))
                    runtimes.append(self.groups.runtime_ns(pairs, self.weight_bytes))
                    self.pim_pairs_by_group[group] += sum(pairs)
                    self.pim_tokens_by_group[group] += len(tokens)
                    self.pim_runtime_by_group[group] += runtimes[-1]
                    self.pim_weight_read_quanta_by_group[group] += sum(
                        math.ceil(count / self.groups.granularity) for count in pairs)
                self.stats['pim_activation_input_bytes'] += sum(map(len, group_tokens)) * self.config.get('hidden_size', 4096)
                self.stats['gpu_distinct_tokens'] += sum(gpu.local_tokens)
                self.stats['gpu_actual_pairs'] += sum(sum(row) for row in gpu.tokens_by_rank_expert)
                for rank, row in enumerate(gpu.tokens_by_rank_expert):
                    self.gpu_pairs_by_rank[rank] += sum(row)
                    if gpu.local_tokens[rank]:
                        self.gpu_query_shapes[(gpu.local_tokens[rank], gpu.activated_experts[rank], sum(row))] += 1
                    else:
                        self.coverage['gpu_empty_rank_layer_queries'] += 1
                self.stats['gpu_profile_implied_pairs'] += sum(gpu.local_tokens) * self.pack.top_k
                self.coverage['gpu_partial_route_tokens'] += sum(
                    np.count_nonzero((np.sum((np.asarray(owners)[ids] == rank) &
                        (np.asarray(devices)[ids] == 'GPU'), axis=1) > 0) &
                        (np.sum((np.asarray(owners)[ids] == rank) &
                        (np.asarray(devices)[ids] == 'GPU'), axis=1) < self.pack.top_k))
                    for rank in range(self.ep_size))
                self.stats['pim_compute_work_ns'] += sum(runtimes)
                self.stats['pim_compute_critical_ns'] += max(runtimes, default=0)
            exchange = None
            if self.groups and self.groups.attachment == 'network':
                exchange = plan_expert_exchange(ids, devices, owners, self.pim_owners[layer],
                    source_ranks, self.tp_size, self.ep_size, self.groups.count)
                for name, matrix in zip(STAGES, exchange.stages()):
                    self.stats['network_' + name + '_bytes'] += (
                        sum(map(sum, matrix)) * self.config.get('hidden_size', 4096))
            layers.append(ExpertLayerPlan(ids, routing, gpu, devices, pim_counts,
                                          tuple(group_tokens), tuple(group_pairs), tuple(runtimes), exchange,
                                          profile_decisions[layer].gpu_ns if profile_decisions else ()))
            if profile_decisions and tuple(runtimes) != profile_decisions[layer].pim_ns:
                raise RuntimeError("adaptive PIM estimate differs from execution plan")
        buffer_tokens = source_buffer_tokens(batches, self.tp_size)
        plan = ExpertRoundPlan(self.step_id, self.state.version, tuple(layers), buffer_tokens,
                               sum(span.count for span in spans), approximations, tuple(source_ranks),
                               tuple((s.request_id, pos) for s in spans for pos in range(s.start, s.start + s.count)))
        self._inflight_step = step
        self._inflight_plan = plan
        self.stats['prompt_approximation_tokens'] += approximations
        self.stats['expert_pairs'] += sum(sum(h.values()) for h in histograms.values())
        self.stats['pim_pairs'] += sum(sum(layer.pim_counts) for layer in layers)
        return plan

    def complete_forward(self, start, finish):
        self.predictor.update(self._inflight_step)
        if self.local_migration is not None:
            self.local_migration.observe(self.step_id, self._inflight_works)
        record = dict(step_id=self.step_id, start_ns=start, finish_ns=finish,
                      placement_version=self.state.version)
        if self.groups:
            record.update(real_tokens=self._inflight_plan.group_tokens,
                          source_buffer_tokens=self._inflight_plan.source_buffer_tokens,
                          source_rank_tokens=[self._inflight_plan.source_ranks.count(rank) for rank in range(self.ep_size)])
            record['pim_weight_read_quanta'] = sum(self.pim_weight_read_quanta_by_group)
        if self.profile_cost is not None:
            record['expert_compute_estimate'] = dict(
                gpu_max_ns=[max(layer.gpu_runtime_ns, default=0) for layer in self._inflight_plan.layers],
                pim_max_ns=[max(layer.pim_group_runtime_ns, default=0) for layer in self._inflight_plan.layers])
        if self.local_migration is not None:
            record.update(local_ssd_load_bytes=self.local_migration.stats['local_ssd_load_bytes'],
                          expert_replacements=self.local_migration.stats['expert_replacements'],
                          replacement_reversals=self.local_migration.stats['replacement_reversals'])
        if self.ssd_budget is not None:
            record.update(self.ssd_budget.snapshot())
        self.rounds.append(record)
        self.step_id += 1

    def summary(self):
        gpu, pim = self.residency_bytes(self.state.residency)
        result = dict(self.stats, profile_coverage={k: int(v) for k, v in self.coverage.items()},
                    gpu_expert_bytes_by_rank=gpu, pim_expert_bytes_by_channel=pim,
                    kv_blocks_by_instance=[s.memory.npu_pool.num_blocks for s in self.schedulers],
                    placement_version=self.state.version)
        result['calibration'] = dict(self.calibration_stats, cache_hit=self.calibration_cache_hit)
        result['placement_controller_timing'] = dict(
            wall_seconds=self.prepare_wall_seconds, cpu_seconds=self.prepare_cpu_seconds,
            included_in_serving_clock=False)
        if self.ssd_budget is not None:
            result['ssd_budget'] = self.ssd_budget.summary()
        if self.groups:
            initial_gpu, initial_pim = self.residency_bytes(self.initial_residency,
                self.initial_gpu_owners, self.initial_pim_owners)
            result['initial_gpu_expert_bytes_by_rank'] = initial_gpu
            result['initial_pim_expert_bytes_by_group'] = initial_pim
            result['initial_placement_sha256'] = hashlib.sha256(json.dumps(dict(
                residency=sorted((layer, expert, value) for (layer, expert), value in self.initial_residency.items()),
                gpu=self.initial_gpu_owners, pim=self.initial_pim_owners), sort_keys=True).encode()).hexdigest()
        if self.local_migration is not None:
            migration = self.local_migration.summary()
            result['local_migration'] = migration
            for key in ('local_ssd_load_bytes', 'expert_replacements', 'replacement_reversals',
                        'network_weight_bytes', 'ssd_load_time_ns'):
                result[key] = migration.get(key, 0)
        if self.groups:
            result.pop('pim_expert_bytes_by_channel')
            result.update(pim_groups=self.groups.to_dict(), pim_expert_bytes_by_group=pim,
                pim_expert_bytes_per_package=[size // self.weight_bytes * math.ceil(self.weight_bytes / self.groups.packages_per_group) for size in pim],
                gpu_pairs_by_rank=self.gpu_pairs_by_rank, gpu_runtime_ns_by_rank=self.gpu_runtime_by_rank,
                pim_pairs_by_group=self.pim_pairs_by_group, pim_tokens_by_group=self.pim_tokens_by_group,
                pim_runtime_ns_by_group=self.pim_runtime_by_group,
                pim_weight_read_quanta_by_group=self.pim_weight_read_quanta_by_group,
                pim_weight_read_bytes_by_group=[count * self.weight_bytes
                                               for count in self.pim_weight_read_quanta_by_group],
                gpu_query_shapes=[dict(tokens=key[0], active_experts=key[1], actual_pairs=key[2], count=count)
                                  for key, count in sorted(self.gpu_query_shapes.items())],
                activation_transfer_bytes=2 * self.stats['pim_activation_input_bytes'])
            counts = collections.Counter(self.state.residency.values())
            both_usage = collections.Counter(self.copy_uses.get(key, 0) for key, value in self.state.residency.items()
                                              if value == 'BOTH')
            result['copy_usage'] = dict(residency=dict(counts), both_fraction=counts['BOTH'] / len(self.state.residency),
                both_never_used=both_usage[0], both_used_gpu_only=both_usage[1],
                both_used_pim_only=both_usage[2], both_used_on_both_domains=both_usage[3],
                unused_gpu_copies=sum(value in ('GPU', 'BOTH') and not self.copy_uses.get(key, 0) & 1
                                      for key, value in self.state.residency.items()),
                unused_pim_copies=sum(value in ('PIM', 'BOTH') and not self.copy_uses.get(key, 0) & 2
                                      for key, value in self.state.residency.items()),
                switch_definition='domain changes between successive active uses of each BOTH expert')
        if self.profile_cost is not None:
            result['adaptive_profile'] = dict(self.profile_cost.summary(), placement=self.state.metadata,
                objective='max expert compute across physical GPU ranks and PIM groups; activation payload tie-break',
                oracle_comparison='no bandwidth-based oracle; search is a bounded heuristic')
        return result
