"""Startup-only bounded host pipeline admission; stable across graph degrees."""
SLOT_BYTES=8<<20


def plan(config,n,world):
    host=config.get('host_available_bytes',8<<30)
    if type(host) is not int or host<=0 or world not in (1,2,4,8) or not 2<=n<=128:
        raise ValueError('invalid archive host memory/topology')
    # Includes reader payload, reusable packed/shift arenas and metadata.
    workspace=world*(64<<20)+(64<<20)
    inventory=config.get('gpu_inventory')
    if inventory:
        if len(inventory)!=world:raise ValueError('archive GPU inventory/topology')
        sizes=[x.get('total_bytes',x.get('free_bytes')) for x in inventory]
        if any(type(x) is not int or x<=0 for x in sizes):raise ValueError('archive GPU memory inventory')
        target=sum(sizes)
    else:
        per_gpu=config.get('resource_plan',{}).get('free_bytes')
        target=per_gpu*world if type(per_gpu) is int and per_gpu>0 else host//2
    budget=config.get('archive_ram_budget_bytes',min(target+workspace,host*3//4))
    if type(budget) is not int or not workspace<budget<=host-(64<<20):
        raise ValueError('insufficient host archive memory')
    slots=config.get('archive_ram_slots')
    if slots is None:
        slots=(budget-workspace)//(world*SLOT_BYTES)
        if slots>=8:slots=slots//8*8
    if type(slots) is not int or not 2<=slots<=65536:
        raise ValueError('invalid archive RAM slot count')
    pinned=world*slots*SLOT_BYTES
    if pinned+workspace>budget:
        raise ValueError('insufficient host archive memory')
    return dict(policy='vram-comparable-host-ring-v1',host_available_bytes=host,
        target_total_pinned_bytes=target,host_limited=pinned+world*8*SLOT_BYTES<target,
        total_budget_bytes=budget,workspace_reserve_bytes=workspace,
        slot_bytes=SLOT_BYTES,slots_per_rank=slots,pinned_bytes_per_rank=slots*SLOT_BYTES,
        total_pinned_bytes=pinned,rows_per_slot=max(1000,SLOT_BYTES//n),
        geometry_scope='same rounded pinned bytes and slot count across graph degrees')
