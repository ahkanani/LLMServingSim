"""Full serving replay driver using the existing scheduler and ASTRA-Sim.

Forward, handoff, migration, and arrival-wakeup jobs have separate completion
callbacks. Backend iteration counters never stand in for request progress.
"""

import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import platform
import resource
from time import perf_counter

import numpy as np
import yaml

from .config_builder import build_cluster_config
from .controller import Controller
from .expert_controller import ExpertPolicyController
from .expert_groups import PIMGroups
from .expert_offload import resolve_repo_path
from .expert_replay import ExpertTracePack
from .memory_model import calculate_expert_sizes
from .replay_graph import control_graphs, forward_graph, add_scaleout_groups
from .request import Batch, RequestStatus
from .scheduler import Scheduler
from .utils import get_config


class ReplayRuntime:
    def __init__(self, args, astra_sim, paths):
        self.wall_start = perf_counter()
        self.cpu_start = resource.getrusage(resource.RUSAGE_SELF)
        self.child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.graph_wall_seconds = 0
        from serving.__main__ import _build_instance_runtime_configs
        if not args.expert_trace_pack or not args.expert_policy_config or not args.dataset:
            raise ValueError('TRACE requires --expert-trace-pack, --expert-policy-config, and --dataset')
        if args.skip_prefill:
            raise ValueError('TRACE models prefill and decode together; --skip-prefill is unsupported')
        if args.network_backend != 'analytical' or args.enable_attn_offloading or args.enable_local_offloading:
            raise ValueError('TRACE currently supports analytical execution without attention/local offload')
        self.args, self.astra_sim, self.paths = args, astra_sim, paths
        self.config = yaml.safe_load(resolve_repo_path(args.expert_policy_config).read_text())
        unsupported = {'hindsight_phases', 'hindsight_fit_options', 'hindsight_forward_limit',
                       'utility_options', 'gpu_moe_profile'} & self.config.keys()
        if unsupported:
            raise ValueError(f'unsupported experimental replay settings: {sorted(unsupported)}')
        cal = self.config.get('calibration_request_ids')
        if cal is None:
            cal = json.loads(resolve_repo_path(self.config['calibration_file']).read_text())['request_ids']
        self.config['calibration_request_ids'] = cal
        if self.config.get('scheduler', {}).get('policy') == 'oracle_split':
            raise ValueError('oracle_split is diagnostic only; use oracle_comparison with a residency-aware scheduler')
        cluster_path = os.path.relpath(resolve_repo_path(args.cluster_config), Path(astra_sim).parent)
        self.cluster = build_cluster_config(astra_sim, cluster_path, inputs_root=paths.inputs_root,
                                            enable_pim=True, native_pd=True)
        self.cluster['save_trace_text'] = args.save_trace_text
        self.groups = (PIMGroups.from_config(self.cluster['expert_replay'])
                       if self.cluster['expert_replay'] is not None else None)
        self.instances = self.cluster['instances']
        self.prefills = [i for i in self.instances if i['pd_type'] == 'prefill']
        self.decodes = [i for i in self.instances if i['pd_type'] == 'decode']
        if (len(self.prefills) != 1 or len(self.decodes) not in (1, 2) or
                len(self.instances) != 1 + len(self.decodes)):
            raise ValueError('TRACE requires one prefill instance and one or two decode instances')
        first = self.prefills[0]
        self.model = first['model_name']
        if any(i['model_name'] != self.model for i in self.instances):
            raise ValueError('TRACE instances must use the same model')
        if len(self.decodes) > 1 and (self.decodes[0].get('dp_group') is None or
                any(i.get('dp_group') != self.decodes[0].get('dp_group') for i in self.decodes)):
            raise ValueError('TRACE decode instances must share one DP/EP group')
        if any(i['ep_size'] != sum(d['num_npus'] for d in self.decodes) for i in self.decodes):
            raise ValueError('TRACE EP must span all decode instances')
        self.pack = ExpertTracePack(args.expert_trace_pack, get_config(self.model), get_config(self.model).get('expert_trace_model_id', self.model))
        self.runtime = _build_instance_runtime_configs(self.instances, args,
            dict(float16=16, bfloat16=16, float32=32, fp8=8, int8=8))
        if len({(r['dtype'], r['kv_cache_dtype']) for r in self.runtime}) != 1:
            raise ValueError('TRACE instances must use matching weight and KV dtypes')
        if any(i['hardware'] != self.decodes[0]['hardware'] for i in self.decodes):
            raise ValueError('TRACE decode GPUs must use one hardware profile')
        if len({json.dumps(self.runtime[i['instance_id']]['expert_offload'], sort_keys=True)
                for i in self.decodes}) != 1:
            raise ValueError('TRACE decode instances must share expert_offload settings')
        if any(r['enable_sub_batch_interleaving'] or r['enable_attn_offloading'] or r['enable_local_offloading']
               for r in self.runtime):
            raise ValueError('TRACE does not combine sub-batch interleaving or other offload paths')
        if args.prefix_storage != 'None' or args.enable_prefix_sharing:
            raise ValueError('TRACE currently supports GPU prefix caching without a shared lower tier')
        rows = [json.loads(line) for line in resolve_repo_path(args.dataset).read_text().splitlines() if line.strip()]
        if args.num_reqs:
            rows = rows[:args.num_reqs]
        if not rows:
            raise ValueError('TRACE dataset is empty')
        for row in rows:
            self.pack.validate_request(row.get('expert_trace_request_id'), row['input_toks'], row['output_toks'])
            if row['expert_trace_request_id'] in set(cal):
                raise ValueError('calibration and evaluation requests overlap')
        self.rows = rows
        self.report_phases = bool(self.config.get('record_workload_phases', False))
        self.schedulers = []
        for i, instance in enumerate(self.instances):
            r = self.runtime[i]
            self.schedulers.append(Scheduler(self.model, instance['node_id'], i,
                r['max_num_seqs'], r['max_num_batched_tokens'], instance['num_npus'], instance['tp_size'], 1,
                instance['npu_mem']['mem_size'], self.cluster['cpu_mem_size'][instance['node_id']],
                self.cluster['inst2npu_mapping'][i], instance['pd_type'], r['fp'], r['block_size'], len(rows),
                r['enable_prefix_caching'], False, None, None, r['enable_chunked_prefill'],
                r['long_prefill_token_threshold'], ep_size=instance['ep_size'], kv_cache_dtype=r['kv_cache_dtype'],
                npu_memory_utilization=r['npu_memory_utilization'], reserve_full_isl=r['reserve_full_isl'],
                native_pd=True))
        self.prefill = self.schedulers[first['instance_id']]
        self.decode_scheds = [self.schedulers[i['instance_id']] for i in self.decodes]
        if self.config.get('memory_pressure', False):
            mean_length = sum(r['input_toks'] + r['output_toks'] for r in rows) / len(rows)
            for sched in self.decode_scheds:
                blocks = math.ceil(16 * mean_length / sched.memory.block_size)
                util = (sched.memory.weight + blocks * sched.memory._npu_bytes_per_block) / sched.memory.npu_mem
                sched.memory.npu_memory_utilization = util
                if not sched.memory.try_set_resident_weight(sched.memory.weight):
                    raise ValueError('memory pressure budget cannot be initialized')
        pim = self.cluster['pim_models'][self.decodes[0]['node_id']]
        if any(i['node_id'] != self.decodes[0]['node_id'] for i in self.decodes):
            raise ValueError('TRACE decode group must share one PIM memory node')
        pcfg = pim.get_config() if pim is not None else dict(mem_size=256, dimm_size=8)
        expert_bytes = calculate_expert_sizes(self.model, 1, fp=self.runtime[first['instance_id']]['fp'] // 8)[1]
        self.config['hidden_size'] = get_config(self.model)['hidden_size']
        from .trace_generator import resolve_variant, _load_perf_db, _lookup_moe
        decode_runtime = self.runtime[self.decodes[0]['instance_id']]
        variant = resolve_variant(decode_runtime['dtype'], decode_runtime['kv_cache_dtype'], get_config(self.model))
        gpu_db = _load_perf_db(self.decodes[0]['hardware'], self.model, variant, {1},
                               get_config(self.model)['model_type'])
        profile_path = Path(astra_sim).parent / 'profiler/perf' / self.decodes[0]['hardware'] / self.model / variant / 'tp1/moe.csv'
        self.gpu_moe_profile = dict(path=str(profile_path.resolve()),
            sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            override=False,
            applies_to='decode placement, calibration and expert execution; prefill uses original moe.csv')
        self.config['calibration_profile_sha256'] = self.gpu_moe_profile['sha256']
        self.output_path = (resolve_repo_path(args.output) if args.output else
                            resolve_repo_path(f'outputs/{args.run_id}/sim.csv'))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.with_suffix('.progress.json').write_text(json.dumps(dict(
            phase='calibration', pid=os.getpid(), requests=len(rows), calibration_requests=len(cal),
            calibration_batch_size=self.config.get('calibration_batch_size', len(cal)))))
        calibration_start = perf_counter()
        self.policy = ExpertPolicyController(self.pack, self.decode_scheds, self.config,
            self.decodes[0]['npu_mem']['mem_bw'], int(pcfg['mem_size'] // pcfg['dimm_size']),
            int(pcfg['mem_size'] * 2**30), expert_bytes, groups=self.groups,
            gpu_lookup=lambda tokens, active: _lookup_moe(gpu_db, tokens, active))
        self.policy.gpu_moe_db = gpu_db
        self.calibration_wall_seconds = perf_counter() - calibration_start
        cfg = self.runtime[self.decodes[0]['instance_id']]['expert_offload_config']
        if not self.groups:
            self.policy.pim_profile = cfg.load_profile_db()
        self.transfer = cfg.activation_transfer
        if self.transfer is None and (not self.groups or self.groups.attachment == 'direct'):
            raise ValueError('TRACE requires explicit expert_offload.activation_transfer')
        # Native replay uses the same explicit PIM settings on all decode members.
        for inst in self.decodes:
            r = self.runtime[inst['instance_id']]
            r['expert_offload_config'] = cfg
        self.cluster['replay_ep_group'] = len(self.instances) + 1
        groups = {str(i['instance_id'] + 1): self.ranks(i['instance_id']) for i in self.instances}
        groups[str(self.cluster['replay_ep_group'])] = [rank for i in self.decodes for rank in self.ranks(i['instance_id'])]
        self.groups_path = Path(paths.inputs_root) / 'comm_groups.json'
        self.groups_path.write_text(json.dumps(dict(groups=groups)))
        self.group_ranks = []
        self.total_endpoints = self.cluster['total_npu']
        if self.groups:
            if self.groups.attachment == 'network':
                self.group_ranks = list(range(self.total_endpoints, self.total_endpoints + self.groups.count))
                self.total_endpoints += self.groups.count
            self._configure_group_network()
        self.domain_group = None
        if self.group_ranks:
            self.domain_group = self.cluster['replay_ep_group'] + 1
            groups[str(self.domain_group)] = [self.ranks(i['instance_id'])[0]
                for i in self.decodes] + self.group_ranks
            self.groups_path.write_text(json.dumps(dict(groups=groups)))
        for index, row in enumerate(rows):
            self.prefill.add_request([index, self.model, row['input_toks'], row['input_toks'] + row['output_toks'],
                row['arrival_time_ns'], first['instance_id']], expert_trace_request_id=row['expert_trace_request_id'])
        self.pending_handoffs = collections.deque()
        self.handoff_ready = {}
        self.handoff_wait = []
        self.complete = []
        self.busy = {}
        self.assignments = {}
        self.iterations = collections.Counter()
        self.job_id = 0
        self.jobs = []
        self.decode_phase = None
        self.decode_batches = None
        self.decode_cursor = 0
        self.current = 1
        self.retained_workloads = {}
        self.last_decode_artifact = None
        self.pruned_workloads = 0

    def _configure_group_network(self):
        network_path = Path(self.paths.network_config)
        network = yaml.safe_load(network_path.read_text())
        tp = self.decodes[0]['tp_size']
        if self.total_endpoints % tp:
            raise ValueError('network-attached group endpoint count must fit the TP-sized rank container')
        # npus_count is the rank container required by ASTRA. The explicit
        # domain map distinguishes CPU service endpoints from four-GPU domains.
        network['npus_count'] = [tp, self.total_endpoints // tp]
        domains = [rank // tp for rank in range(self.cluster['total_npu'])]
        domains.extend(range(max(domains) + 1, max(domains) + 1 + len(self.group_ranks)))
        network['endpoint_domains'] = domains
        network['endpoint_roles'] = ['GPU'] * self.cluster['total_npu'] + ['CPU_PIM'] * len(self.group_ranks)
        network_path.write_text(yaml.safe_dump(network))
        system_path = Path(self.paths.system_config)
        system = json.loads(system_path.read_text())
        system.update({'replay-event-driven': True, 'replay-domain-map': domains,
                       'replay-scaleout': dict(topology=network['topology'][-1],
                           bandwidth=network['bandwidth'][-1], latency=network['latency'][-1])})
        if self.group_ranks:
            system['replay-local-bandwidth'] = network['bandwidth'][0]
        system_path.write_text(json.dumps(system, indent=2))

    def ranks(self, instance_id):
        start = self.cluster['inst2npu_mapping'][instance_id]
        return list(range(start, start + self.instances[instance_id]['num_npus']))

    def available(self, schedulers):
        return all(rank not in self.busy for sched in schedulers for rank in self.ranks(sched.instance_id))

    def path(self, kind):
        self.job_id += 1
        return f'replay_{kind}_{self.job_id}'

    def launch(self, kind, ranks, workload, callback):
        if any(rank in self.busy for rank in ranks):
            raise RuntimeError('attempt to overlap control jobs on a busy GPU')
        job = dict(kind=kind, remaining=set(ranks), finish=0, start=self.current, callback=callback,
                   workload=workload, decode_round=len(self.policy.rounds) if kind == 'decode' else None)
        for rank in ranks:
            self.iterations[rank] += 1
            self.busy[rank] = (job, self.iterations[rank])
            self.assignments[rank] = workload

    def done_report(self, rank, iteration, finish):
        entry = self.busy.get(rank)
        if entry is None or entry[1] != iteration:
            return
        job = entry[0]
        job['remaining'].discard(rank)
        job['finish'] = max(job['finish'], finish)
        del self.busy[rank]
        if not job['remaining']:
            self.jobs.append(dict(kind=job['kind'], start_ns=job['start'], finish_ns=job['finish']))
            job['callback'](job['finish'])
            self._finish_artifacts(job)
            if job['kind'] == 'decode' and len(self.policy.rounds) % 64 == 0:
                self._write_progress('serving')

    def _write_progress(self, phase):
        row = dict(phase=phase, pid=os.getpid(), requests=len(self.rows), completed_requests=len(self.complete),
            decode_rounds=len(self.policy.rounds), token_forwards=self.policy.stats['expert_pairs'] //
            (self.policy.num_layers * self.pack.top_k), simulated_time_ns=self.current,
            wall_seconds=perf_counter() - self.wall_start, calibration_wall_seconds=self.calibration_wall_seconds,
            pruned_workloads=self.pruned_workloads)
        path = self.output_path.with_suffix('.progress.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(row))
        temporary.replace(path)

    def _finish_artifacts(self, job):
        if not self.config.get('prune_completed_graphs', False):
            return
        folder = Path(job['workload']).parent
        root = Path(self.paths.inputs_root) / 'workload'
        if folder.parent != root:
            raise RuntimeError('refusing to prune a workload outside this replay run')
        samples = set(self.config.get('retain_decode_rounds', []))
        if job['kind'] == 'decode':
            if self.last_decode_artifact is not None:
                previous, index = self.last_decode_artifact
                if index not in samples:
                    shutil.rmtree(previous)
                    self.retained_workloads.pop(previous.name)
                    self.pruned_workloads += 1
            self.last_decode_artifact = folder, job['decode_round']
            keep = True  # Keep the latest completed forward, including the final tail.
        else:
            keep = not any(value['kind'] == job['kind'] for value in self.retained_workloads.values())
        if keep:
            self.retained_workloads[folder.name] = dict(kind=job['kind'], decode_round=job['decode_round'],
                start_ns=job['start'], finish_ns=job['finish'], folder=str(folder.relative_to(self.paths.inputs_root)))
        else:
            shutil.rmtree(folder)
            self.pruned_workloads += 1

    def _complete_batch(self, sched, batch, finish):
        ended = []
        for rank in self.ranks(sched.instance_id):
            _, _, requests = sched.add_done(batch.batch_id + 1, rank, finish)
            ended.extend(requests)
        return ended

    def schedule_prefill(self):
        if self.pending_handoffs or not self.available([self.prefill]):
            return
        batch = self.prefill.schedule(self.current, self.prefill.start_npu)
        if batch is None:
            arrivals = [r.arrival for r in self.prefill.waiting if r.arrival > self.current]
            if arrivals:
                self.wakeup(min(arrivals), self.ranks(self.prefill.instance_id))
            return
        folder = self.path('prefill')
        i = self.prefill.instance_id
        graph_start = perf_counter()
        forward_graph(batch, self.instances[i], self.runtime[i], self.cluster, None, None, folder)
        self.graph_wall_seconds += perf_counter() - graph_start
        def finished(finish):
            for req in self._complete_batch(self.prefill, batch, finish):
                if req.num_tokens_reached >= req.output:
                    self.complete.append(req)
                else:
                    self.handoff_ready[req.id] = finish
                    self.pending_handoffs.append(req)
        self.launch('prefill', self.ranks(i), str(Path(self.paths.inputs_root) / 'workload' / folder / 'llm'), finished)

    def wakeup(self, time_ns, ranks):
        from .graph_generator import _get_llm_converter
        from chakra.src.third_party.utils.protolib import encodeMessage
        folder = Path(self.paths.inputs_root) / 'workload' / self.path('arrival')
        folder.mkdir(parents=True)
        converter = _get_llm_converter()('', str(folder / 'llm'), 1)
        for rank in ranks:
            with (folder / f'llm.{rank}.et').open('wb') as stream:
                encodeMessage(stream, converter.get_global_metadata())
                encodeMessage(stream, converter.get_comp_node('arrival_wakeup', max(1, int(time_ns - self.current))))
        self.launch('arrival', ranks, str(folder / 'llm'), lambda finish: None)

    def schedule_handoff(self):
        if not self.pending_handoffs or not self.available(self.schedulers):
            return
        req = self.pending_handoffs[0]
        target = None
        for delta in range(len(self.decode_scheds)):
            sched = self.decode_scheds[(self.decode_cursor + delta) % len(self.decode_scheds)]
            if len(sched.running) >= sched.max_num_seqs:
                continue
            if not sched.kv.can_fit_full_sequence(req):
                continue
            if sched.kv.allocate_slots(req, 1) is not None:
                target = sched
                self.decode_cursor = (self.decode_scheds.index(sched) + 1) % len(self.decode_scheds)
                break
        if target is None:
            if not any(s.running or s.waiting for s in self.decode_scheds):
                raise ValueError(f'request {req.id} cannot fit the decode KV budget')
            return
        self.pending_handoffs.popleft()
        self.handoff_wait.append(self.current - self.handoff_ready.pop(req.id))
        target.kv.take_traffic()
        size = self.prefill.memory.get_kv(req.num_computed_tokens)
        transfers = [(req.id, src, dst, size) for src, dst in zip(self.ranks(self.prefill.instance_id), self.ranks(target.instance_id))]
        folder = Path(self.paths.inputs_root) / 'workload' / self.path('kv_transfer')
        ranks = list(range(self.cluster['total_npu']))
        workload = control_graphs(folder, ranks, 'kv_transfer', transfers=transfers)
        def finished(finish):
            self.prefill.kv.free(req)
            req.instance_id, req.status = target.instance_id, RequestStatus.RUNNING
            target.running.append(req)
        self.launch('kv_transfer', ranks, workload, finished)

    def schedule_decode(self):
        if not self.available(self.decode_scheds):
            return
        if self.decode_phase == 'forward_ready':
            self.launch_decode_forward()
            return
        if self.decode_phase is not None:
            return
        batches = [s.schedule(self.current, s.start_npu) for s in self.decode_scheds]
        if not any(batches):
            return
        for index, (sched, batch) in enumerate(zip(self.decode_scheds, batches)):
            if batch is None:
                batch = Batch(sched.get_batch_id(), self.model, 0, 0, [], [], 0, 0, [], [], [], self.current, 0)
                batch.scheduled_tokens = {}
                sched.inflight.append(batch)
                batches[index] = batch
        self.decode_batches = batches
        self.decode_phase = 'forward_ready'
        actions = self.policy.prepare_migration(self.current)
        if actions:
            self.decode_phase = 'migration'
            folder = Path(self.paths.inputs_root) / 'workload' / self.path('migration')
            ranks = [rank for s in self.decode_scheds for rank in self.ranks(s.instance_id)]
            workload = control_graphs(folder, ranks, 'migration', migration=actions,
                                      controller=self.policy, transfer_config=self.transfer)
            def finished(finish):
                self.policy.commit_migration()
                self.decode_phase = 'forward_ready'
            self.launch('migration', ranks, workload, finished)
        else:
            self.launch_decode_forward()

    def launch_decode_forward(self):
        graph_start = perf_counter()
        batches = self.decode_batches
        plan = self.policy.plan_forward(batches)
        padded = plan.source_buffer_tokens * self.decode_scheds[0].tp_size
        folder = self.path('decode')
        for index, (sched, batch) in enumerate(zip(self.decode_scheds, batches)):
            batch.total_len = padded
            # lm_head sees padded slots; attention keeps its real token lists.
            batch.num_decode += padded - sum(span.count for span in batch.token_spans)
            forward_graph(batch, self.instances[sched.instance_id], self.runtime[sched.instance_id],
                          self.cluster, plan, self.policy, folder, coordinator=index == 0)
        if self.group_ranks:
            add_scaleout_groups(Path(self.paths.inputs_root) / 'workload' / folder,
                               plan, self.policy, self.group_ranks, self.domain_group)
        self.graph_wall_seconds += perf_counter() - graph_start
        start = self.current
        self.decode_phase = 'forward'
        def finished(finish):
            for sched, batch in zip(self.decode_scheds, batches):
                self.complete.extend(self._complete_batch(sched, batch, finish))
            self.policy.complete_forward(start, finish)
            if self.report_phases:
                self._record_workload_round()
            self.decode_phase, self.decode_batches = None, None
        ranks = [rank for s in self.decode_scheds for rank in self.ranks(s.instance_id)] + self.group_ranks
        self.launch('decode', ranks, str(Path(self.paths.inputs_root) / 'workload' / folder / 'llm'), finished)

    def _record_workload_round(self):
        """Attach experiment labels after completion, outside the online policy."""
        plan = self.policy._inflight_plan
        counts = collections.Counter(self.rows[request]['workload_phase'] for request, _ in plan.token_keys)
        record = self.policy.rounds[-1]
        record['workload_tokens'] = dict(counts)
        record['pim_pairs'] = sum(sum(layer.pim_counts) for layer in plan.layers)
        record['expert_pairs'] = sum(layer.selections.size for layer in plan.layers)
        record['pim_only_pairs'] = sum(
            count for layer, item in enumerate(plan.layers) for expert, count in enumerate(item.pim_counts)
            if self.policy.state.get(layer, expert) == 'PIM')
        if self.config.get('record_route_batches', False):
            path = self.output_path.with_suffix('.routes.jsonl')
            row = dict(round=len(self.policy.rounds) - 1, start_ns=record['start_ns'], finish_ns=record['finish_ns'],
                dominant_phase=min(counts, key=lambda phase: (-counts[phase], phase)), workload_tokens=dict(counts),
                token_position_basis='decode_offset',
                tokens=[[self.rows[request]['expert_trace_request_id'], position - self.rows[request]['input_toks'], source]
                        for (request, position), source in zip(plan.token_keys, plan.source_ranks)])
            with path.open('w' if len(self.policy.rounds) == 1 else 'a') as stream:
                stream.write(json.dumps(row) + '\n')
        if 'expert_compute_estimate' not in record:
            from .trace_generator import _lookup_moe
            record['expert_compute_estimate'] = dict(
                gpu_max_ns=[max((_lookup_moe(self.policy.gpu_moe_db, tokens, active) if active else 0
                    for tokens, active in zip(item.gpu_routing.local_tokens, item.gpu_routing.activated_experts)), default=0)
                    for item in plan.layers],
                pim_max_ns=[max(item.pim_group_runtime_ns, default=0) for item in plan.layers])

    def run(self):
        total = self.total_endpoints
        initial = control_graphs(Path(self.paths.inputs_root) / 'workload' / 'replay_init', range(total), 'init')
        binary = Path(self.astra_sim) / 'build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware'
        command = [str(binary), '--workload-configuration=' + initial,
                   '--system-configuration=' + self.paths.system_config,
                   '--network-configuration=' + self.paths.network_config,
                   '--memory-configuration=' + self.paths.memory_config,
                   '--comm-group-configuration=' + str(self.groups_path),
                   '--start-npu-ids=' + ','.join(map(str, range(total)))]
        parser = Controller(total)
        with (Path(self.paths.inputs_root) / 'backend.log').open('w') as errors:
            # The analytical backend creates tmp__mem files relative to cwd.
            # Keep them private when multiple serving runs start concurrently.
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=errors, text=True, bufsize=1, cwd=self.paths.inputs_root)
            try:
                while True:
                    report = None
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            code = process.wait(timeout=5)
                            failure = dict(returncode=code, signal=-code if code < 0 else None,
                                completed_requests=len(self.complete), expected_requests=len(self.rows),
                                completed_decode_rounds=len(self.policy.rounds), simulated_time_ns=self.current)
                            (Path(self.paths.inputs_root) / 'backend_failure.json').write_text(
                                json.dumps(failure, indent=2) + '\n')
                            raise RuntimeError(f'ASTRA-Sim exited before replay completion '
                                               f'(return code {code}); inspect backend.log and backend_failure.json')
                        if 'Waiting' in line:
                            break
                        parsed = parser.parse_output(line)
                        if parsed:
                            report = parsed
                    if report is None:
                        raise RuntimeError('ASTRA-Sim requested work without a completion report')
                    rank = report['sys']
                    self.current = report['cycle']
                    self.done_report(rank, report['id'], self.current)
                    if len(self.complete) == len(self.rows) and not self.busy:
                        process.stdin.write('exit\n')
                        process.stdin.flush()
                        break
                    self.schedule_handoff()
                    self.schedule_prefill()
                    self.schedule_decode()
                    response = self.assignments.pop(rank, None)
                    if response is None:
                        response = 'pass -1' if self.groups and self.assignments else 'pass'
                    if self.groups and not self.busy and len(self.complete) < len(self.rows):
                        raise RuntimeError('replay has unfinished requests but no schedulable work; check KV capacity')
                    process.stdin.write(response + '\n')
                    process.stdin.flush()
                process.wait(timeout=30)
                if process.returncode:
                    raise RuntimeError(f'ASTRA-Sim exited with code {process.returncode}')
                self.backend_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        return self.write_results()

    def write_results(self):
        output = self.output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        for index, sched in enumerate(self.schedulers):
            sched.save_output(str(output), is_append=index > 0)
        finish = max(req.end_time for req in self.complete)
        start = min(row['arrival_time_ns'] for row in self.rows)
        metric = lambda values: dict(mean_ms=float(np.mean(values)) / 1e6,
                                     p50_ms=float(np.percentile(values, 50)) / 1e6,
                                     p99_ms=float(np.percentile(values, 99)) / 1e6)
        from .trace_generator import resolve_variant
        variant = resolve_variant(self.runtime[self.decodes[0]['instance_id']]['dtype'],
                                  self.runtime[self.decodes[0]['instance_id']]['kv_cache_dtype'], get_config(self.model))
        profile_root = Path(self.astra_sim).parent / 'profiler' / 'perf' / self.decodes[0]['hardware'] / self.model / variant
        profile_hashes = {str(path.relative_to(profile_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in profile_root.glob('tp*/*.csv')}
        job_times = collections.Counter()
        for job in self.jobs:
            job_times[job['kind']] += job['finish_ns'] - job['start_ns']
        summary = dict(total_clocks_ns=finish, requests=len(self.complete),
            request_throughput_rps=len(self.complete) * 1e9 / max(1, finish - start),
            output_throughput_tps=sum(row['output_toks'] for row in self.rows) * 1e9 / max(1, finish - start),
            latency=metric([r.latency for r in self.complete]), ttft=metric([r.ttft for r in self.complete]),
            tpot=metric([r.tpot for r in self.complete]), queueing=metric([r.queuing_delay for r in self.complete]),
            handoff_queueing=metric(self.handoff_wait or [0]),
            itl=metric([t for r in self.complete for t in r.itl] or [0]),
            preemptions=sum(s.num_preemptions for s in self.schedulers),
            recompute_tokens=sum(s.recompute_tokens for s in self.schedulers),
            policy=self.policy.summary(), timing_limitations=[
                'analytical PIM group timing; CPU and internal TP overhead excluded' if self.groups else 'synthetic PIM profile',
                'GPU MoE tables approximate partial top-k and uneven routing'],
            jobs=self.jobs, rounds=self.policy.rounds, job_time_ns=dict(job_times),
            gpu_profile_sha256=profile_hashes,
            gpu_moe_profile=self.gpu_moe_profile,
            network_config=yaml.safe_load(Path(self.paths.network_config).read_text()),
            resolved_instances=self.instances,
            memory_utilization_by_instance=[s.memory.npu_memory_utilization for s in self.schedulers],
            policy_config=self.config, policy_hardware=self.policy.hardware.to_dict(),
            request_trace_ids={str(index): row['expert_trace_request_id'] for index, row in enumerate(self.rows)},
            prefill_routing='BALANCED approximation',
            trace_pack_manifest_sha256=self.pack.manifest_sha256(),
            pim_profile_sha256=(None if self.groups else hashlib.sha256(self.policy.pim_profile.path.read_bytes()).hexdigest()))
        model_config = get_config(self.model)
        omitted = sorted(set(range(self.pack.num_layers)) - set(self.policy.layer_ids))
        summary['model_layer_mapping'] = dict(transformer_layers=self.pack.num_layers,
            routed_layers=list(self.policy.layer_ids), dense_layers=omitted,
            expert_weight_bytes=self.policy.weight_bytes,
            expert_weight_bits=model_config.get('expert_weight_bits', self.runtime[self.decodes[0]['instance_id']]['fp']))
        if omitted and model_config.get('dense_ffn_roofline'):
            summary['dense_ffn_roofline'] = model_config['dense_ffn_roofline']
            summary['timing_limitations'].append(
                f'Dense FFNs on layers {omitted} use ideal roofline estimates; measured dense FFN profiles are missing')
        if model_config.get('expert_weight_bits'):
            summary['timing_limitations'].append('Packed weight bytes exclude quantization scale/zero-point metadata')
        if self.report_phases:
            summary['workload_phases'] = self._workload_summary(metric)
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        child = self.backend_usage
        cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), platform.processor())
        summary['host_timing'] = dict(wall_seconds=perf_counter() - self.wall_start,
            calibration_wall_seconds=self.calibration_wall_seconds, graph_wall_seconds=self.graph_wall_seconds,
            parent_user_seconds=cpu.ru_utime - self.cpu_start.ru_utime,
            parent_system_seconds=cpu.ru_stime - self.cpu_start.ru_stime,
            backend_user_seconds=child.ru_utime - self.child_start.ru_utime,
            backend_system_seconds=child.ru_stime - self.child_start.ru_stime,
            parent_peak_rss_kib=cpu.ru_maxrss, backend_peak_rss_kib=child.ru_maxrss,
            cpu_model=cpu_model, hostname=platform.node(), platform=platform.platform(),
            available_cpus=len(os.sched_getaffinity(0)), save_trace_text=self.args.save_trace_text,
            log_level=self.args.log_level)
        summary['decode_statistics'] = dict(rounds=len(self.policy.rounds),
            token_forwards=int(self.policy.stats['expert_pairs'] // (self.policy.num_layers * self.pack.top_k)),
            generated_tokens=sum(row['output_toks'] - 1 for row in self.rows),
            instance_sequence_caps=[s.max_num_seqs for s in self.decode_scheds],
            busy_ns=job_times.get('decode', 0),
            elapsed_ns=(self.policy.rounds[-1]['finish_ns'] - self.policy.rounds[0]['start_ns']
                        if self.policy.rounds else 0))
        summary['execution_artifacts'] = dict(pruned_workloads=self.pruned_workloads,
            retained_workloads=list(self.retained_workloads.values()),
            retention='sampled forwards plus final forward and first job of each control kind'
                      if self.config.get('prune_completed_graphs', False) else 'all generated graphs until run cleanup')
        if self.groups:
            summary['replay_schema'] = 'cpu_pim_groups_v3' if self.group_ranks else 'cpu_pim_groups_v2'
            if self.group_ranks:
                summary['expert_communication'] = dict(
                    algorithm='hierarchical_placement_aware_direct_alltoallv',
                    domain_members=[self.ranks(i['instance_id'])[0] for i in self.decodes] + self.group_ranks,
                    domain_group=self.domain_group, padding='no communication from padded or dummy tokens',
                    metadata='FP8 hidden vectors only; expert IDs, scales and gate weights excluded',
                    reduction='one combined vector per token per domain; reduction arithmetic unmodeled')
            summary['activation_wire_dtype'] = 'fp8; no scale/metadata/conversion overhead'
            network = summary['network_config']
            summary['network_units'] = dict(
                convention='existing analytical backend interprets configured GB/s as GiB/s',
                effective_bandwidth_bytes_per_ns=[value * 2**30 / 1e9 for value in network['bandwidth']],
                latency='ns per hop; Switch has two hops',
                direct_transfer_bandwidth='decimal GB/s')
            summary['timing_limitations'].extend([
                'GPU MoE profile includes inseparable router/combine costs; empty ranks have no MoE lookup',
                'FP8 wire format excludes quantization, dequantization, scales, and routing metadata costs'])
            root = Path(self.astra_sim).parent
            tracked_sources = list((root / 'serving').rglob('*.py'))
            tracked_sources.extend((root / 'astra-sim/astra-sim').rglob('*.cc'))
            tracked_sources.extend((root / 'astra-sim/astra-sim').rglob('*.hh'))
            tracked_sources.extend((root / 'astra-sim/extern/graph_frontend/chakra/src/converter').rglob('*.py'))
            source_hash = hashlib.sha256()
            for path in sorted(tracked_sources):
                source_hash.update(str(path.relative_to(root)).encode())
                source_hash.update(path.read_bytes())
            summary['simulator_source_sha256'] = source_hash.hexdigest()
            summary['model_config_sha256'] = hashlib.sha256(json.dumps(get_config(self.model), sort_keys=True).encode()).hexdigest()
            summary['cluster_config_sha256'] = hashlib.sha256(resolve_repo_path(self.args.cluster_config).read_bytes()).hexdigest()
            summary['dataset_sha256'] = hashlib.sha256(resolve_repo_path(self.args.dataset).read_bytes()).hexdigest()
            summary['gpu_profile_metadata_sha256'] = hashlib.sha256((profile_root / 'meta.yaml').read_bytes()).hexdigest()
            route_hash = hashlib.sha256()
            for trace_id in sorted(set(self.config['calibration_request_ids']) |
                                   {r['expert_trace_request_id'] for r in self.rows}):
                record = self.pack.records[trace_id]
                route_hash.update(json.dumps(record, sort_keys=True).encode())
                route_hash.update(self.pack.decode_rows(trace_id).tobytes())
            summary['selected_trace_sha256'] = route_hash.hexdigest()
            with output.with_suffix('.placement.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['layer', 'expert', 'hotness', 'residency', 'gpu_ep_rank', 'pim_group'])
                for layer in range(self.policy.num_layers):
                    for expert in range(self.pack.num_experts):
                        writer.writerow([self.policy.layer_ids[layer], expert, self.policy.hotness[layer][expert],
                            self.policy.state.get(layer, expert), self.policy.gpu_owners[layer][expert],
                            self.policy.pim_owners[layer][expert]])
        summary['host_timing']['wall_seconds'] = perf_counter() - self.wall_start
        summary_path = output.with_suffix('.replay.json')
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        self._write_progress('complete')
        print(f'Total clocks (ns): {finish}')
        print(f'Replay results: {output}')
        print(f'Replay summary: {summary_path}')
        if not self.args.keep_inputs and not self.args.save_trace_text:
            shutil.rmtree(self.paths.inputs_root)
        return summary

    def _workload_summary(self, metric):
        phases = {}
        for request in self.complete:
            phase = self.rows[request.id]['workload_phase']
            phases.setdefault(phase, []).append(request)
        result = {}
        for phase, requests in phases.items():
            arrivals = [self.rows[r.id]['arrival_time_ns'] for r in requests]
            rounds = [r for r in self.policy.rounds if r['workload_tokens'].get(phase, 0)]
            majority = [r for r in rounds if r['workload_tokens'][phase] * 2 > r['real_tokens']]
            result[phase] = dict(requests=len(requests), first_arrival_ns=min(arrivals), last_arrival_ns=max(arrivals),
                first_decode_ns=rounds[0]['start_ns'] if rounds else None,
                majority_decode_ns=majority[0]['start_ns'] if majority else None,
                last_completion_ns=max(r.end_time for r in requests),
                latency=metric([r.latency for r in requests]), ttft=metric([r.ttft for r in requests]),
                tpot=metric([r.tpot for r in requests]), queueing=metric([r.queuing_delay for r in requests]),
                steady_itl=metric([t for r in requests for t in r.itl[1:]] or [0]))
        return result


def run_replay(args, astra_sim, paths):
    return ReplayRuntime(args, astra_sim, paths).run()
