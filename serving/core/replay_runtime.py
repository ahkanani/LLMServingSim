"""Full serving replay driver using the existing scheduler and ASTRA-Sim.

Forward, handoff, migration, and arrival-wakeup jobs have separate completion
callbacks. Backend iteration counters never stand in for request progress.
"""

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import yaml

from .config_builder import build_cluster_config
from .controller import Controller
from .expert_controller import ExpertPolicyController
from .expert_offload import resolve_repo_path
from .expert_replay import ExpertTracePack
from .memory_model import calculate_expert_sizes
from .replay_graph import control_graphs, forward_graph
from .request import Batch, RequestStatus
from .scheduler import Scheduler
from .utils import get_config


class ReplayRuntime:
    def __init__(self, args, astra_sim, paths):
        from serving.__main__ import _build_instance_runtime_configs
        if not args.expert_trace_pack or not args.expert_policy_config or not args.dataset:
            raise ValueError('TRACE requires --expert-trace-pack, --expert-policy-config, and --dataset')
        if args.skip_prefill:
            raise ValueError('TRACE models prefill and decode together; --skip-prefill is unsupported')
        if args.network_backend != 'analytical' or args.enable_attn_offloading or args.enable_local_offloading:
            raise ValueError('TRACE currently supports analytical execution without attention/local offload')
        self.args, self.astra_sim, self.paths = args, astra_sim, paths
        self.config = yaml.safe_load(resolve_repo_path(args.expert_policy_config).read_text())
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
        self.instances = self.cluster['instances']
        self.prefills = [i for i in self.instances if i['pd_type'] == 'prefill']
        self.decodes = [i for i in self.instances if i['pd_type'] == 'decode']
        if len(self.prefills) != 1 or len(self.decodes) != 2 or len(self.instances) != 3:
            raise ValueError('TRACE requires one prefill instance and two decode DP instances')
        first = self.prefills[0]
        self.model = first['model_name']
        if any(i['model_name'] != self.model for i in self.instances):
            raise ValueError('TRACE instances must use the same model')
        if self.decodes[0].get('dp_group') is None or self.decodes[0].get('dp_group') != self.decodes[1].get('dp_group'):
            raise ValueError('TRACE decode instances must share one DP/EP group')
        if any(i['ep_size'] != sum(d['num_npus'] for d in self.decodes) for i in self.decodes):
            raise ValueError('TRACE EP must span both decode instances')
        self.pack = ExpertTracePack(args.expert_trace_pack, get_config(self.model), self.model)
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
        pcfg = pim.get_config()
        expert_bytes = calculate_expert_sizes(self.model, 1, fp=self.runtime[first['instance_id']]['fp'] // 8)[1]
        self.policy = ExpertPolicyController(self.pack, self.decode_scheds, self.config,
            self.decodes[0]['npu_mem']['mem_bw'], int(pcfg['mem_size'] // pcfg['dimm_size']),
            int(pcfg['mem_size'] * 2**30), expert_bytes)
        cfg = self.runtime[self.decodes[0]['instance_id']]['expert_offload_config']
        self.policy.pim_profile = cfg.load_profile_db()
        self.transfer = cfg.activation_transfer
        if self.transfer is None:
            raise ValueError('TRACE requires explicit expert_offload.activation_transfer')
        # Native replay uses the same explicit PIM settings on both DP members.
        for inst in self.decodes:
            r = self.runtime[inst['instance_id']]
            r['expert_offload_config'] = cfg
        self.cluster['replay_ep_group'] = len(self.instances) + 1
        groups = {str(i['instance_id'] + 1): self.ranks(i['instance_id']) for i in self.instances}
        groups[str(self.cluster['replay_ep_group'])] = [rank for i in self.decodes for rank in self.ranks(i['instance_id'])]
        self.groups_path = Path(paths.inputs_root) / 'comm_groups.json'
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
        job = dict(kind=kind, remaining=set(ranks), finish=0, start=self.current, callback=callback)
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
        forward_graph(batch, self.instances[i], self.runtime[i], self.cluster, None, None, folder)
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
        actions = self.policy.prepare_migration()
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
        start = self.current
        self.decode_phase = 'forward'
        def finished(finish):
            for sched, batch in zip(self.decode_scheds, batches):
                self.complete.extend(self._complete_batch(sched, batch, finish))
            self.policy.complete_forward(start, finish)
            self.decode_phase, self.decode_batches = None, None
        ranks = [rank for s in self.decode_scheds for rank in self.ranks(s.instance_id)]
        self.launch('decode', ranks, str(Path(self.paths.inputs_root) / 'workload' / folder / 'llm'), finished)

    def run(self):
        total = self.cluster['total_npu']
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
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=errors, text=True, bufsize=1)
            try:
                while True:
                    report = None
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            raise RuntimeError('ASTRA-Sim exited before replay completion; inspect backend.log')
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
                    response = self.assignments.pop(rank, None) or 'pass'
                    process.stdin.write(response + '\n')
                    process.stdin.flush()
                process.wait(timeout=30)
                if process.returncode:
                    raise RuntimeError(f'ASTRA-Sim exited with code {process.returncode}')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        return self.write_results()

    def write_results(self):
        output = resolve_repo_path(self.args.output) if self.args.output else resolve_repo_path(f'outputs/{self.args.run_id}/sim.csv')
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
            policy=self.policy.summary(), timing_limitations=['synthetic PIM profile',
                'GPU MoE tables approximate partial top-k and uneven routing'],
            jobs=self.jobs, rounds=self.policy.rounds, job_time_ns=dict(job_times),
            gpu_profile_sha256=profile_hashes,
            network_config=yaml.safe_load(Path(self.paths.network_config).read_text()),
            resolved_instances=self.instances,
            memory_utilization_by_instance=[s.memory.npu_memory_utilization for s in self.schedulers],
            policy_config=self.config, policy_hardware=self.policy.hardware.to_dict(),
            request_trace_ids={str(index): row['expert_trace_request_id'] for index, row in enumerate(self.rows)},
            prefill_routing='BALANCED approximation',
            trace_pack_manifest_sha256=hashlib.sha256((self.pack.path / 'manifest.json').read_bytes()).hexdigest(),
            pim_profile_sha256=hashlib.sha256(self.policy.pim_profile.path.read_bytes()).hexdigest())
        summary_path = output.with_suffix('.replay.json')
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Total clocks (ns): {finish}')
        print(f'Replay results: {output}')
        print(f'Replay summary: {summary_path}')
        if not self.args.keep_inputs and not self.args.save_trace_text:
            shutil.rmtree(self.paths.inputs_root)
        return summary


def run_replay(args, astra_sim, paths):
    return ReplayRuntime(args, astra_sim, paths).run()
