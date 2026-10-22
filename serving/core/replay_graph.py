"""Typed control workloads and full-model forwards for the replay driver."""

from pathlib import Path

from .graph_generator import _get_llm_converter, generate_graph
from .trace_generator import generate_trace


def forward_graph(batch, instance, runtime, cluster, plan, controller, folder, coordinator=False):
    batch.expert_replay = True
    batch.expert_round_plan = plan
    batch.expert_controller = controller
    batch.expert_pim_coordinator = coordinator
    if controller is not None:
        members = controller.schedulers
        member_index = next(i for i, sched in enumerate(members) if sched.instance_id == instance['instance_id'])
        batch.expert_ep_offset = member_index * instance['tp_size']
    trace = generate_trace(batch, instance['hardware'], instance['tp_size'], 1,
        instance['local_ep'], instance['ep_total'], pd_type=None,
        node_id=instance['node_id'], instance_id=instance['instance_id'],
        max_num_batched_tokens=runtime['max_num_batched_tokens'], max_num_seqs=runtime['max_num_seqs'],
        placement=cluster['placement'][instance['instance_id']],
        enable_block_copy=False, expert_routing_policy='BALANCED', fp=runtime['fp'],
        dtype=runtime['dtype'], kv_cache_dtype=runtime['kv_cache_dtype'],
        pim_model=cluster['pim_models'][instance['node_id']],
        inputs_root=cluster['inputs_root'], expert_offload_config=runtime['expert_offload_config'])
    trace.comm_groups = dict(tp=instance['instance_id'] + 1,
                             ep=cluster['replay_ep_group'] if instance['pd_type'] == 'decode'
                             else instance['instance_id'] + 1)
    generate_graph(batch, instance['hardware'], instance['num_npus'], instance['node_id'],
                   instance['instance_id'], cluster['inst2npu_mapping'][instance['instance_id']],
                   workload_name=folder, inputs_root=cluster['inputs_root'],
                   save_trace_text=cluster['save_trace_text'], trace=trace)


def control_graphs(folder, ranks, kind, transfers=(), migration=(), controller=None, transfer_config=None):
    """Write matching SEND/RECV or serialized PIM copy nodes; no generation."""
    from chakra.src.third_party.utils.protolib import encodeMessage
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    converter = _get_llm_converter()('', str(folder / 'llm'), 1)
    for rank in ranks:
        nodes = []
        for tag, src, dst, size in transfers:
            if rank in (src, dst):
                node = converter.get_comm_node(rank == src, 'kv_handoff', 'NONE', size, src, dst, id=tag)
                # The converter's shared dictionary gives SEND/RECV identical tags.
                nodes.append(node)
        if migration and rank == controller.schedulers[0].start_npu:
            # activation_transfer describes a node-wide link. Aggregate copies
            # before timing them; assigning its full bandwidth to each PIM
            # channel would multiply the available external bandwidth.
            previous = None
            for origin in ('GPU', 'PIM'):
                size = sum(controller.weight_bytes for action in migration if action.promote_from == origin)
                if not size:
                    continue
                latency = (transfer_config.npu_to_pim_time_ns(size) if origin == 'GPU'
                           else transfer_config.pim_to_npu_time_ns(size))
                node = converter.get_pim_compute_node('expert_weight_copy_from_' + origin, 'WEIGHT', latency,
                    f'REMOTE:{controller.schedulers[0].node_id}.0', 0)
                if previous is not None:
                    converter.add_parent(node, previous)
                nodes.append(node)
                previous = node
        if not nodes:
            nodes = [converter.get_comp_node(kind + '_idle', 1)]
        with (folder / f'llm.{rank}.et').open('wb') as stream:
            encodeMessage(stream, converter.get_global_metadata())
            for node in nodes:
                encodeMessage(stream, node)
    return str(folder / 'llm')
