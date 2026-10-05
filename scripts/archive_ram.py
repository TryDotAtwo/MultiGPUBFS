"""Startup-only bounded host pipeline admission; stable across graph degrees."""
SLOT_BYTES=8<<20


def plan(config,n,world):
    host=config.get('host_available_bytes',8<<30)
    if type(host) is not int or host<=0 or world not in (1,2,4,8) or not 2<=n<=128:
        raise ValueError('invalid archive host memory/topology')
    # Includes reader payload, reusable packed/shift arenas and metadata.
    compact=config.get('retention_policy')=='last_complete_small_1000'
    slot_bytes=(1<<20) if compact else SLOT_BYTES
    workspace=world*((16 if compact else 64)<<20)+(64<<20)
    inventory=config.get('gpu_inventory')
    if inventory:
        if len(inventory)!=world:raise ValueError('archive GPU inventory/topology')
        sizes=[x.get('total_bytes',x.get('free_bytes')) for x in inventory]
        if any(type(x) is not int or x<=0 for x in sizes):raise ValueError('archive GPU memory inventory')
        target=sum(sizes)
    else:
        per_gpu=config.get('resource_plan',{}).get('free_bytes')
        target=per_gpu*world if type(per_gpu) is int and per_gpu>0 else host//2
    storage=config.get('archive_storage_mode','hybrid')
    if storage not in ('hybrid','ram'):raise ValueError('invalid archive storage mode')
    total_target=target
    if compact:target=world*(32<<20)
    elif storage=='hybrid':target=max(world*(32<<20),target//4)
    budget=config.get('archive_ram_budget_bytes',min(target+workspace,host*3//4))
    if type(budget) is not int or not workspace<budget<=host-(64<<20):
        raise ValueError('insufficient host archive memory')
    slots=config.get('archive_ram_slots')
    if slots is None:
        slots=(budget-workspace)//(world*slot_bytes)
        if slots>=8:slots=slots//8*8
    if type(slots) is not int or not 2<=slots<=65536:
        raise ValueError('invalid archive RAM slot count')
    pinned=world*slots*slot_bytes
    if pinned+workspace>budget:
        raise ValueError('insufficient host archive memory')
    initial=config.get('archive_initial_slots',min(slots,32 if compact else 64))
    if type(initial) is not int or not 2<=initial<=slots:raise ValueError('invalid initial archive RAM slots')
    return dict(policy='compact-prefix-host-ring-v1' if compact else 'hybrid-vram-comparable-ring-v1' if storage=='hybrid' else 'vram-comparable-host-ring-v1',host_available_bytes=host,
        storage_mode='compact' if compact else storage,
        target_total_queue_bytes=target if compact else total_target,
        ssd_queue_target_bytes=0 if compact else max(0,total_target-pinned),
        ssd_target_scope='packed closed parts waiting for publication; target, not a physical reservation',
        target_total_pinned_bytes=target,host_limited=pinned+world*8*slot_bytes<target,
        total_budget_bytes=budget,workspace_reserve_bytes=workspace,
        slot_bytes=slot_bytes,slots_per_rank=slots,initial_slots_per_rank=initial,
        expansion='activate preallocated reserve on exhausted credits; no hot-path allocation',
        pinned_bytes_per_rank=slots*slot_bytes,
        total_pinned_bytes=pinned,rows_per_slot=max(1000,slot_bytes//n),
        geometry_scope='same rounded pinned bytes and slot count across graph degrees')
