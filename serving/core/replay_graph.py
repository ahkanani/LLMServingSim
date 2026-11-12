"""Typed control workloads and full-model forwards for the replay driver."""

from pathlib import Path
import heapq

from .graph_generator import _get_llm_converter, generate_graph
from .trace_generator import generate_trace


def _read_nodes(path):
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
    from chakra.src.third_party.utils.protolib import decodeMessage
    nodes = []
    with path.open('rb') as stream:
        decodeMessage(stream, GlobalMetadata())
        node = Node()
        while decodeMessage(stream, node):
            nodes.append(node)
            node = Node()
    return nodes


def _write_nodes(path, converter, nodes):
    from chakra.src.third_party.utils.protolib import encodeMessage
    with path.open('wb') as stream:
        encodeMessage(stream, converter.get_global_metadata())
        for node in nodes:
            encodeMessage(stream, node)


def _topological_nodes(nodes):
    """Keep dependencies before children for Chakra's bounded streaming feeder."""
    by_id = {n.id: n for n in nodes}
    children, remaining = {}, {}
    for node in nodes:
        remaining[node.id] = len(node.data_deps)
        for parent in node.data_deps:
            if parent not in by_id:
                raise ValueError(f'missing graph parent {parent}')
            children.setdefault(parent, []).append(node.id)
    ready = [key for key, count in remaining.items() if not count]
    heapq.heapify(ready)
    order = []
    while ready:
        key = heapq.heappop(ready)
        order.append(key)
        for child in children.get(key, ()):
            remaining[child] -= 1
            if not remaining[child]:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        raise ValueError('cycle in CPU/PIM replay graph')
    renumber = {key: i for i, key in enumerate(order)}
    for key in order:
        node = by_id[key]
        node.id = renumber[key]
        node.data_deps[:] = [renumber[parent] for parent in node.data_deps]
    return [by_id[key] for key in order]


def _exchange_node(converter, name, matrix, index, group_id, collective_id, hidden):
    from chakra.schema.protobuf.et_def_pb2 import AttributeProto, Uint64List
    sends = [count * hidden for count in matrix[index]]
    receives = [row[index] * hidden for row in matrix]
    node = converter.get_comm_coll_node(name, 'ALLTOALL', sum(sends))
    node.attr.extend([
        AttributeProto(name='comm_group_id', int32_val=group_id),
        AttributeProto(name='replay_alltoallv', bool_val=True),
        AttributeProto(name='replay_collective_id', uint64_val=collective_id),
        AttributeProto(name='replay_send_counts', uint64_list=Uint64List(values=sends)),
        AttributeProto(name='replay_recv_counts', uint64_list=Uint64List(values=receives)),
    ])
    return node


def add_scaleout_groups(folder, plan, controller, group_ranks, domain_group):
    """Replace logical expert exchanges with a common hierarchical all-to-all-v.

    Local TP-domain gather -> domain dispatch -> local expert dispatch ->
    GPU/PIM computation -> local reduction -> domain combine -> source return.
    Both GPU-domain leaders and CPU/PIM endpoints join the same outer collective.
    """
    import json
    from chakra.schema.protobuf.et_def_pb2 import AttributeProto
    from .expert_communication import STAGES

    folder = Path(folder)
    source_ranks = [s.start_npu + local for s in controller.schedulers for local in range(s.num_npus)]
    tp = controller.tp_size
    gpu_domains = len(controller.schedulers)
    leaders = source_ranks[::tp]
    domain_ranks = leaders + list(group_ranks)
    gpu_nodes = {rank: _read_nodes(folder / f'llm.{rank}.et') for rank in source_ranks}
    converter = _get_llm_converter()('', str(folder / 'llm'), 1)
    converter.next_node_id = max(n.id for nodes in gpu_nodes.values() for n in nodes) + 1
    group_nodes = {rank: [] for rank in group_ranks}
    previous = {rank: [] for rank in group_ranks}
    starts = {rank: [n for n in nodes if n.name == 'COMM_COLL_NODE_expert_start_ALLTOALL']
              for rank, nodes in gpu_nodes.items()}
    ends = {rank: [n for n in nodes if n.name == 'COMM_COLL_NODE_expert_end_ALLTOALL']
            for rank, nodes in gpu_nodes.items()}
    if any(len(nodes) != len(plan.layers) for nodes in (*starts.values(), *ends.values())):
        raise ValueError('network replay needs one logical expert exchange per layer')
    manifest = dict(algorithm='hierarchical_placement_aware_direct_alltoallv',
                    wire_dtype='fp8', vector_bytes=controller.config['hidden_size'],
                    domain_members=domain_ranks, gpu_domains=[source_ranks[i:i + tp]
                        for i in range(0, len(source_ranks), tp)], layers=[])
    hidden = controller.config['hidden_size']
    for layer_id, layer in enumerate(plan.layers):
        matrices = layer.network_exchange.stages()
        base = (plan.step_id * len(plan.layers) + layer_id) * len(STAGES)
        audit = dict(layer=layer_id, model_layer=controller.layer_ids[layer_id], stages=[])
        for stage, matrix in enumerate(matrices):
            audit['stages'].append(dict(name=STAGES[stage], token_counts=matrix,
                bytes=sum(map(sum, matrix)) * hidden,
                communicator='domains' if stage in (1, 4) else 'local_tp'))
        manifest['layers'].append(audit)
        for domain, sched in enumerate(controller.schedulers):
            begin = domain * tp
            local_matrices = [tuple(tuple(row[begin:begin + tp])
                                   for row in matrix[begin:begin + tp])
                              if stage not in (1, 4) else matrix
                              for stage, matrix in enumerate(matrices)]
            for local in range(tp):
                rank = source_ranks[begin + local]
                start, end = starts[rank][layer_id], ends[rank][layer_id]
                group = sched.instance_id + 1
                gather = _exchange_node(converter, f'expert_L{layer_id}_source_gather',
                    local_matrices[0], local, group, base, hidden)
                gather.data_deps.extend(start.data_deps)
                dispatch = _exchange_node(converter, f'expert_L{layer_id}_gpu_dispatch',
                    local_matrices[2], local, group, base + 2, hidden)
                reduce = _exchange_node(converter, f'expert_L{layer_id}_gpu_combine',
                    local_matrices[3], local, group, base + 3, hidden)
                reduce.data_deps.extend(end.data_deps)
                # Empty expert ranks must also wait for their dispatch.
                if start.id not in reduce.data_deps:
                    reduce.data_deps.append(start.id)
                returns = _exchange_node(converter, f'expert_L{layer_id}_source_return',
                    local_matrices[5], local, group, base + 5, hidden)
                if local == 0:
                    outer_in = _exchange_node(converter, f'expert_L{layer_id}_domain_dispatch',
                        matrices[1], domain, domain_group, base + 1, hidden)
                    outer_in.data_deps.append(gather.id)
                    outer_out = _exchange_node(converter, f'expert_L{layer_id}_domain_combine',
                        matrices[4], domain, domain_group, base + 4, hidden)
                    outer_out.data_deps.append(reduce.id)
                    dispatch.data_deps.append(outer_in.id)
                    returns.data_deps.append(outer_out.id)
                    gpu_nodes[rank].extend([outer_in, outer_out])
                else:
                    dispatch.data_deps.append(gather.id)
                    returns.data_deps.append(reduce.id)
                # Preserve anchor IDs, so existing compute and next-layer edges
                # point to actual dispatch completion and combined source data.
                dispatch.id = start.id
                returns.id = end.id
                start.CopyFrom(dispatch)
                end.CopyFrom(returns)
                gpu_nodes[rank].extend([gather, reduce])
        for group, endpoint in enumerate(group_ranks):
            domain = gpu_domains + group
            dispatch = _exchange_node(converter, f'expert_L{layer_id}_domain_dispatch',
                matrices[1], domain, domain_group, base + 1, hidden)
            dispatch.data_deps.extend(previous[endpoint])
            combine = _exchange_node(converter, f'expert_L{layer_id}_domain_combine',
                matrices[4], domain, domain_group, base + 4, hidden)
            nodes = [dispatch]
            if layer.pim_group_runtime_ns[group]:
                compute = converter.get_comp_node(f'cpu_pim_group_L{layer_id}_G{group}',
                                                  layer.pim_group_runtime_ns[group])
                compute.attr.append(AttributeProto(name='is_cpu_op', bool_val=True))
                compute.data_deps.append(dispatch.id)
                combine.data_deps.append(compute.id)
                nodes.append(compute)
            else:
                combine.data_deps.append(dispatch.id)
            nodes.append(combine)
            group_nodes[endpoint].extend(nodes)
            previous[endpoint] = [combine.id]
    for rank, nodes in {**gpu_nodes, **group_nodes}.items():
        _write_nodes(folder / f'llm.{rank}.et', converter, _topological_nodes(nodes))
    # Authoritative readable expansion of the logical ALLTOALL trace markers.
    # The protobufs carry these same per-rank byte vectors for native execution.
    (folder / 'expert_communication.json').write_text(json.dumps(manifest) + '\n')


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
    if plan is not None and controller.groups and controller.groups.attachment == 'network':
        trace.comm_groups['expert_exchange'] = True
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
