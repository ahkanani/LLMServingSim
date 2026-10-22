"""Recorded-route, policy parity, memory safety, and native-backend regression tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from serving.core.block_pool import BlockPool, Device
from serving.core.expert_replay import ExpertTracePack, TokenSpan, routing_from_ids
from serving.core.moe_policy.interfaces import ReplayStep
from serving.core.moe_policy.model_hardware import ModelSpec, HardwareModel
from serving.core.moe_policy.prediction import make_predictor
from serving.core.moe_policy.placement import make_placement_policy
from serving.core.moe_policy.migration import make_migration_policy
from serving.core.expert_controller import ExpertPolicyController
from serving.core.moe_policy.migration import MigrationAction, MigrationPlan
from serving.core.moe_policy.scheduler import make_scheduler
from workloads.generators.moe_trace import prepare


ROOT = Path(__file__).resolve().parents[1]


def make_pack(path, layers=2, experts=8, top_k=2, model='fixture'):
    path.mkdir()
    lengths = [3, 0, 2, 1, 2]
    prompts = [4, 1, 5, 3, 4]
    data = np.zeros((5, 3, layers, top_k), dtype=np.uint8)
    for request in range(5):
        for position in range(lengths[request]):
            for layer in range(layers):
                data[request, position, layer] = (np.arange(top_k) + request * top_k + position * 7 + layer) % experts
    np.save(path / 'decode_experts.npy', data)
    np.save(path / 'decode_lengths.npy', np.asarray(lengths, dtype=np.uint32))
    records = [dict(request_id=f'r{i}', request_index=i, prefill_length=prompts[i],
                    decode_length=lengths[i]) for i in range(5)]
    manifest = dict(schema_version='trace_pack_v1', requests=records,
        model_metadata=dict(model_id=model, num_layers=layers, top_k=top_k,
                            experts_per_layer={str(i): experts for i in range(layers)}),
        layout=dict(decode_experts='decode_experts.npy', decode_lengths='decode_lengths.npy'))
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return ExpertTracePack(path)


class ExpertReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pack = make_pack(Path(self.tmp.name) / 'pack')

    def test_absolute_positions_survive_chunks_and_recomputation(self):
        whole = TokenSpan(1, 'r0', 4, 0, 7, 0)
        ids, prompt = self.pack.selections([whole], 1)
        self.assertEqual(prompt, 4)
        pieces = [self.pack.selections([TokenSpan(1, 'r0', 4, start, count, 0)], 1)[0]
                  for start, count in [(0, 2), (2, 3), (5, 2)]]
        np.testing.assert_array_equal(ids, np.concatenate(pieces))
        np.testing.assert_array_equal(ids, self.pack.selections([whole], 1)[0])
        np.testing.assert_array_equal(ids[4:], self.pack.decode[0, :, 1])
        self.assertFalse(ids.flags.writeable)

    def test_bounds_binding_and_zero_decode(self):
        self.pack.validate_request('r1', 1, 1)
        for arguments in [('unknown', 1, 1), ('r1', 2, 1), ('r1', 1, 2)]:
            with self.assertRaises(ValueError):
                self.pack.validate_request(*arguments)
        with self.assertRaises(ValueError):
            self.pack.selections([TokenSpan(0, 'r0', 4, 7, 1, 0)], 0)

    def test_exact_tokens_are_distinct_from_expert_pairs(self):
        ids = np.array([[0, 1], [0, 2], [4, 5]])
        result = routing_from_ids(ids, 8, 2, [2, 1])
        self.assertEqual(result.local_tokens, [2, 1])
        self.assertEqual(result.activated_experts, [3, 2])
        self.assertEqual(sum(result.per_expert_tokens), 6)
        gpu = routing_from_ids(ids, 8, 2, [2, 1], ['GPU', 'PIM', 'GPU', 'GPU', 'PIM', 'PIM', 'GPU', 'GPU'])
        self.assertEqual(gpu.local_tokens, [2, 0])
        self.assertEqual(gpu.activated_experts, [2, 0])

    def test_workload_split_is_disjoint_and_arrivals_repeat(self):
        calibration, rows = prepare(self.pack, 1, 4, 123, 10)
        self.assertTrue(set(calibration).isdisjoint(row['expert_trace_request_id'] for row in rows))
        self.assertEqual((calibration, rows), prepare(self.pack, 1, 4, 123, 10))
        for row in rows:
            self.assertEqual(row['output_toks'], self.pack.records[row['expert_trace_request_id']]['decode_length'] + 1)

    def test_reservation_never_moves_or_preempts_live_kv(self):
        pool = BlockPool(Device.NPU, 8, 16, 1024)
        live = pool.get_new_blocks(4)
        identities = [(block.block_id, id(block), block.ref_cnt) for block in live]
        self.assertFalse(pool.resize_free_capacity(3))
        self.assertEqual(pool.num_blocks, 8)
        self.assertTrue(pool.resize_free_capacity(5))
        self.assertTrue(pool.resize_free_capacity(10))
        self.assertEqual(identities, [(block.block_id, id(block), block.ref_cnt) for block in live])
        self.assertEqual(pool.get_num_free_blocks(), 6)
        pool.free_blocks(live)
        self.assertTrue(pool.is_free())

    def test_policy_equations_match_independent_source_fixtures(self):
        expected = json.loads((ROOT / 'tests/fixtures/expert_policy_golden.json').read_text())['expected']
        warmup = [ReplayStep(i, ['cal'], {0: {0: 8, 1: 4, 2: 2, 3: 1}, 1: {7: 9, 6: 5, 5: 2}}, {})
                  for i in range(3)]
        current = ReplayStep(3, ['eval'], {0: {0: 1, 4: 2, 5: 4, 6: 8}, 1: {0: 5, 1: 2, 7: 1}}, {})
        model = ModelSpec('Qwen/Qwen3-235B-A22B-FP8', 2, 8, 2, 18874368)
        hardware = HardwareModel('fixture', 2, 4.8, 2, .6, 1)
        for mode in expected:
            predictor = make_predictor(dict(type='heuristic', window_size=128, decay=.95, horizon=1), model=model)
            predictor.initialize(warmup)
            prediction = predictor.predict(3, 1)
            scheduler = make_scheduler(dict(policy='gpu_baseline' if mode == 'all_gpu' else 'placement_greedy'))
            placement = make_placement_policy(dict(mode=mode, hot_fraction=.5, cold_fraction=.25,
                extra_copy_fraction=.25)).initialize(warmup, prediction, model, hardware, scheduler_config=scheduler.config)
            migration = make_migration_policy(dict(mode='predictive_budgeted_regret', budget_pairs_per_step=2)).plan(
                placement, prediction, model, hardware)
            decision = scheduler.schedule(current, placement, model, hardware)
            actual = dict(residency=[[l, e, d] for (l, e), d in placement.residency.items()],
                assignments=[[a.layer, a.expert, a.token_count, a.device] for a in decision.assignments],
                gpu_time_s=decision.gpu_time_s, pim_time_s=decision.pim_time_s,
                migrations=[[a.layer, a.promote_expert, a.promote_from, a.demote_expert, a.demote_to] for a in migration.actions])
            self.assertEqual(actual, expected[mode])

    def test_migration_deferral_is_atomic_with_live_kv(self):
        from types import SimpleNamespace
        memories = []
        for _ in range(2):
            memory = SimpleNamespace(weight=80, npu_mem=1000, npu_memory_utilization=1,
                                     _npu_bytes_per_block=1, npu_pool=BlockPool(Device.NPU, 920, 1, 1))
            def reserve(weight, mem=memory):
                if not mem.npu_pool.resize_free_capacity(1000 - weight):
                    return False
                mem.weight = weight
                return True
            memory.try_set_resident_weight = reserve
            memories.append(memory)
        schedulers = [SimpleNamespace(num_npus=1, tp_size=1, memory=memory) for memory in memories]
        controller = ExpertPolicyController(self.pack, schedulers,
            dict(calibration_request_ids=['r0'], placement=dict(mode='static_hot_cold')),
            4800, 2, 1000, 10)
        residency = {(layer, expert): 'GPU' for layer in range(2) for expert in range(8)}
        residency[(0, 0)], residency[(0, 7)] = 'PIM', 'BOTH'
        controller.state = controller.state.clone_with_updates(residency)
        controller._reserve(residency)
        live = memories[0].npu_pool.get_new_blocks(memories[0].npu_pool.get_num_free_blocks())
        identities = [(b.block_id, b.ref_cnt) for b in live]
        action = MigrationAction(0, 0, 'PIM', 7, 'PIM', 1)
        controller.migration = SimpleNamespace(should_run=lambda step: True,
            plan=lambda *args: MigrationPlan(controller.step_id, (action,)))
        state = controller.state
        self.assertEqual(controller.prepare_migration(), ())
        self.assertIs(controller.state, state)
        self.assertIsNone(controller.pending_migration)
        self.assertEqual(identities, [(b.block_id, b.ref_cnt) for b in live])
        self.assertEqual(controller.stats['deferred_migration_pairs'], 1)
        memories[0].npu_pool.free_blocks(live)
        self.assertEqual(controller.prepare_migration(), (action,))
        self.assertIs(controller.state, state)
        controller.commit_migration()
        self.assertEqual(controller.state.get(0, 0), 'BOTH')
        self.assertEqual(controller.state.get(0, 7), 'PIM')


@unittest.skipUnless(os.environ.get('LLMSERVINGSIM_REPLAY_INTEGRATION') == '1', 'requires built ASTRA-Sim and Chakra')
class ReplayBackendTests(unittest.TestCase):
    def test_migrations_share_the_configured_transfer_bandwidth(self):
        from types import SimpleNamespace
        from serving.core.replay_graph import control_graphs
        from serving.core.expert_offload import ActivationTransferConfig
        from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
        from chakra.src.third_party.utils.protolib import decodeMessage
        controller = SimpleNamespace(weight_bytes=1000, schedulers=[SimpleNamespace(start_npu=4, node_id=0)])
        actions = [MigrationAction(0, 0, 'GPU', 1, 'GPU', 1),
                   MigrationAction(1, 3, 'GPU', 4, 'GPU', 1),
                   MigrationAction(2, 6, 'PIM', 7, 'PIM', 1)]
        transfer = ActivationTransferConfig(100, 50, 10, 20)
        with tempfile.TemporaryDirectory() as directory:
            control_graphs(directory, [4, 5], 'migration', migration=actions,
                           controller=controller, transfer_config=transfer)
            nodes = []
            with (Path(directory) / 'llm.4.et').open('rb') as stream:
                decodeMessage(stream, GlobalMetadata())
                node = Node()
                while decodeMessage(stream, node):
                    nodes.append(node)
                    node = Node()
            self.assertEqual([node.duration_micros for node in nodes], [30, 40])
            self.assertEqual(list(nodes[1].data_deps), [nodes[0].id])

    def test_pim_fork_join_dependencies_and_channel_attributes(self):
        from chakra.src.converter.llm_converter import LLMConverter
        from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
        from chakra.src.third_party.utils.protolib import decodeMessage
        rows = [
            ['embedding', '1', 'REMOTE:0', '0', 'LOCAL', '0', 'LOCAL', '0', 'NONE', '0', 'NONE'],
            ['EXPERT', '0', 'NONE', '0'],
            ['expert_hot', '100', 'LOCAL', '0', 'LOCAL', '0', 'LOCAL', '0', 'NONE', '0', 'NONE'],
            ['xfer_in_L0', '10', 'REMOTE:0.0', '128', 'REMOTE:0.0', '0', 'REMOTE:0.0', '128', 'NONE', '0', 'PIM_XFER_IN'],
            ['PIM', '0'],
            ['expert_pim_L0_E1', '200', 'REMOTE:0.1', '0', 'REMOTE:0.1', '18874368', 'REMOTE:0.1', '0', 'NONE', '0', 'EXPERT'],
            ['PIM', 'END'],
            ['xfer_out_L0', '20', 'REMOTE:0.0', '128', 'REMOTE:0.0', '0', 'REMOTE:0.0', '128', 'NONE', '0', 'PIM_XFER_OUT'],
            ['expert_pim_join', '1', 'LOCAL', '0', 'LOCAL', '0', 'LOCAL', '0', 'NONE', '0', 'NONE'],
            ['EXPERT', 'END', 'NONE', '0'],
            ['sampler', '1', 'LOCAL', '0', 'LOCAL', '0', 'REMOTE:0', '0', 'NONE', '0', 'NONE'],
        ]
        with tempfile.TemporaryDirectory() as directory:
            converter = LLMConverter('', str(Path(directory) / 'llm'), 1)
            converter.convert_rows('COLOCATED model_parallel_NPU_group: 1', rows, comm_groups={'tp': 1, 'ep': 1})
            nodes = []
            with (Path(directory) / 'llm.0.et').open('rb') as stream:
                decodeMessage(stream, GlobalMetadata())
                node = Node()
                while decodeMessage(stream, node):
                    nodes.append(node)
                    node = Node()
            find = lambda text: next(node for node in nodes if text in node.name)
            hot, incoming, pim, outgoing, join = [find(text) for text in
                ('COMP_NODE_expert_hot', 'xfer_in', 'PIM_COMP_NODE_expert_pim_L0', 'xfer_out', 'expert_pim_join')]
            self.assertFalse(any('MEM_LOAD_NODE_expert_pim' in node.name for node in nodes))
            self.assertNotIn(hot.id, incoming.data_deps)
            self.assertEqual(set(hot.data_deps), set(incoming.data_deps))
            self.assertIn(incoming.id, pim.data_deps)
            self.assertIn(pim.id, outgoing.data_deps)
            self.assertIn(hot.id, join.data_deps)
            self.assertIn(outgoing.id, join.data_deps)
            attributes = {attr.name: attr for attr in pim.attr}
            self.assertEqual(attributes['tensor_channel'].uint32_val, 1)
            self.assertEqual(attributes['tensor_size'].uint64_val, 0)

    def test_native_handoffs_backpressure_zero_decode_and_pim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = make_pack(root / 'pack', 94, 128, 8, 'Qwen/Qwen3-235B-A22B-FP8')
            dataset = root / 'workload.jsonl'
            dataset.write_text(''.join(json.dumps(dict(input_toks=r['prefill_length'], output_toks=r['decode_length'] + 1,
                arrival_time_ns=1, expert_trace_request_id=r['request_id'])) + '\n' for r in list(pack.records.values())[1:]))
            cluster = json.loads((ROOT / 'configs/cluster/pim_experiments/expert_replay/qwen3_235b_h200.json').read_text())
            for instance in cluster['nodes'][0]['instances']:
                if instance['pd_type'] == 'decode':
                    instance['max_num_seqs'] = 1
            cluster_path = root / 'cluster.json'
            cluster_path.write_text(json.dumps(cluster))
            observed_clocks = {}
            for variant in ['gpu', 'static', 'migration']:
                config = json.loads((ROOT / f'configs/expert_replay/{variant}.json').read_text())
                config.pop('calibration_file')
                config['calibration_request_ids'] = ['r0']
                policy = root / 'policy.json'
                policy.write_text(json.dumps(config))
                output = root / (variant + '.csv')
                with (root / (variant + '.log')).open('w') as log:
                    result = subprocess.run([sys.executable, '-m', 'serving', '--expert-routing-policy', 'TRACE',
                        '--expert-trace-pack', str(pack.path), '--expert-policy-config', str(policy),
                        '--cluster-config', str(cluster_path), '--dataset', str(dataset), '--output', str(output)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(result.returncode, 0, (root / (variant + '.log')).read_text())
                summary = json.loads(output.with_suffix('.replay.json').read_text())
                observed_clocks[variant] = summary['total_clocks_ns']
                self.assertEqual(summary['requests'], 4)
                self.assertEqual(summary['policy']['expert_pairs'], 5 * 94 * 8)
                self.assertEqual(sum(j['kind'] == 'kv_transfer' for j in summary['jobs']), 3)
                self.assertGreater(summary['policy']['profile_coverage']['gpu_moe_lookups'], 0)
                self.assertEqual(len(summary['policy']['gpu_expert_bytes_by_rank']), 8)
                if variant != 'gpu':
                    self.assertGreater(summary['policy']['pim_pairs'], 0)
            fixture = ROOT / 'tests/fixtures/expert_replay_clocks.json'
            if os.environ.get('LLMSERVINGSIM_UPDATE_REPLAY_BASELINES') == '1':
                fixture.write_text(json.dumps(observed_clocks, indent=2) + '\n')
            self.assertEqual(observed_clocks, json.loads(fixture.read_text()))


if __name__ == '__main__':
    unittest.main()
