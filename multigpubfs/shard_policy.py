"""Plan logical shards from live-state geometry, with independently addressed segments.

This is a sizing heuristic. Exact routing can be skewed, so the runtime still
uses independently addressed segments when any logical shard grows further.
"""
from collections import Counter
from math import comb, gcd

from .memory_admission import bitmap_bytes_by_rank, bitmap_domain, bitmap_frontier_banks, bitmap_frontier_row_bytes

LOCAL_SHARD_TARGET = 1 << 32
MAX_LOGICAL_SHARDS = 4096


def _cyclic_orbit(graph):
    generators=graph.action.get('generators', [])
    n=len(graph.start); identity=tuple(range(n))
    generators=[tuple(g) for g in generators if tuple(g)!=identity]
    if not generators:
        return 1
    base=generators[0]
    if len(base)!=n or sorted(base)!=list(identity):
        return None
    inverse=[0]*n
    for i,v in enumerate(base):
        inverse[v]=i
    inverse=tuple(inverse)
    if any(g!=base and g!=inverse for g in generators):
        return None
    seen=set(); result=1
    for i in range(n):
        if i in seen:
            continue
        values=[]; j=i
        while j not in seen:
            seen.add(j); values.append(graph.start[j]); j=base[j]
        period=len(values)
        for candidate in range(1,len(values)+1):
            if len(values)%candidate==0 and all(values[k]==values[k%candidate] for k in range(len(values))):
                period=candidate; break
        result=result//gcd(result,period)*period
    return result


def _orbit_upper(graph):
    action=graph.action
    if action['kind']=='permutation':
        cyclic=_cyclic_orbit(graph)
        if cyclic is not None:
            return cyclic,'PERMUTATION_CYCLIC_ORBIT'
        result=1; used=0
        for count in Counter(graph.start).values():
            result*=comb(used+count,count); used+=count
        return result,'PERMUTATION_MULTISET_UPPER'
    if action['kind']=='matrix':
        moduli={g['modulo'] for g in action['generators']}
        if len(moduli)==1:
            modulus=next(iter(moduli))
            if modulus>0:
                return modulus**len(graph.start)+(not all(0<=v<modulus for v in graph.start)),'MATRIX_STATE_SPACE_UPPER'
    return None,'UNKNOWN'


def _rank_share(admission, world, bitmap):
    cuts=admission.get('owner_cuts')
    if not bitmap and isinstance(cuts,list) and len(cuts)==world+1 and cuts[0]==0 and cuts[-1]==1<<32 and all(type(x) is int for x in cuts) and all(a<=b for a,b in zip(cuts,cuts[1:])):
        return max(b-a for a,b in zip(cuts,cuts[1:])),1<<32
    return 1,world


def _ceil_div(a,b):
    return (a+b-1)//b


def balanced_pack4_geometry(graph, world, env):
    """Cold starting geometry for large banked Pack4 traversals on two ranks."""
    if (world != 2 or env.get('MGBFS_MEMORY_PRESET') != 'balanced'
            or env.get('MGBFS_EXPERIMENTAL_BITMAP') != '1'
            or env.get('MGBFS_SEGMENT_PACK4') != '1'
            or env.get('MGBFS_SEGMENT_STRATEGY') != 'atomic'
            or env.get('MGBFS_SEGMENT_EXACT_EXCHANGE', '1') != '1'
            or env.get('MGBFS_BITMAP_BANKS', '3') != '3'
            or env.get('MGBFS_SEGMENT_BITMAP_OWNER_PAGE', '0') != '0'
            or any(key in env for key in ('MGBFS_SEGMENT_PACKET_CONTROL_ADDR',
                                         'MGBFS_SEGMENT_OWNER_LANES', 'MGBFS_SEGMENT_QUEUE_CAP',
                                         'MGBFS_SEGMENT_MAX_RANK_STATES', 'MGBFS_SEGMENT_SHARD_TARGET_ROWS',
                                         'MGBFS_SEGMENT_ROWS'))):
        return None
    from .workload_geometry import coxeter_frontier_bound

    bound = coxeter_frontier_bound(graph)
    if bound is None or bound['maximum_frontier'] < 1 << 28:
        return None
    # The S13 comparison used four concurrent owners on eight shards and
    # 98,304 generated candidates per tile. Keep both limits as branching grows.
    return {'kind': 'LARGE_TWO_RANK_PACK4_EXACT', 'minimum_shards': 8,
            'source_batch_cap': max(1, min(8192, 98_304 // graph.generator_count))}


def choose_segmented_shards(graph, admission, *, bitmap=False, target_rows=LOCAL_SHARD_TARGET,
                            rank_limit=None, segment_rows=None, env=None, minimum_shards=1):
    if type(target_rows) is not int or not 1<=target_rows<=LOCAL_SHARD_TARGET:
        raise ValueError('INVALID_SHARD_TARGET_ROWS')
    if rank_limit is not None and (type(rank_limit) is not int or rank_limit<1):
        raise ValueError('INVALID_SEGMENT_RANK_LIMIT')
    if segment_rows is not None and (type(segment_rows) is not int or not 1<=segment_rows<=LOCAL_SHARD_TARGET):
        raise ValueError('INVALID_SEGMENT_ROWS')
    plan=admission['plan']
    world=len(admission.get('devices') or admission.get('inventory') or [0])
    env = env or {}
    if bitmap:
        banks = bitmap_frontier_banks(env, maxmemory=env.get('MGBFS_MEMORY_PRESET') == 'max_capacity')
    else:
        banks = int(plan.get('history_layers',3))
    if not bitmap and banks not in (1,3):
        raise ValueError('INVALID_SHARD_HISTORY_BANKS')

    from .workload_geometry import coxeter_frontier_bound, independent_swap_frontier_bound
    certified=coxeter_frontier_bound(graph) if banks in (2,3) else None
    if bitmap and certified is None:
        certified=independent_swap_frontier_bound(graph)
    orbit,kind=_orbit_upper(graph)
    if certified is not None:
        # Full simultaneously retained consecutive layers, not just the peak.
        global_peak=certified['maximum_frontier']
        orbit=certified['order']; kind=certified['kind']
        global_live=min(global_peak*banks,certified.get('maximum_live_states',orbit))
    else:
        global_live=orbit
        global_peak=orbit

    # An explicit caller estimate can restrict a global orbit; never infer
    # uniformity as a correctness guarantee.
    declared=graph.to_dict().get('expected_max_unique_states')
    if type(declared) is int and declared>0:
        orbit=declared if orbit is None else min(orbit,declared)
        global_live=declared if global_live is None else min(global_live,declared)
        global_peak=declared if global_peak is None else min(global_peak,declared)

    numerator,denominator=_rank_share(admission,world,bitmap)
    estimated=None if global_live is None else _ceil_div(global_live*numerator*5,denominator*4)
    peak_estimate=None if global_peak is None else _ceil_div(global_peak*numerator*5,denominator*4)

    # Estimate one shard's segment from the layer/domain and configured cap.
    # This is used for cost estimation; actual row admission uses live VRAM.
    rows=segment_rows if segment_rows is not None else min(target_rows,max(32,peak_estimate or target_rows))
    count=rows*banks; elements=int(plan.get('elements',len(graph.start)))
    width=int(plan.get('state_bytes',1 if all(0<=v<=255 for v in graph.start) else 8))
    row_bytes=bitmap_frontier_row_bytes(elements,width,env.get('MGBFS_SEGMENT_PACK4')=='1') if bitmap else elements*width
    slots=0 if bitmap else 1<<(2*count-1).bit_length()
    segment_bytes=count*row_bytes + (0 if bitmap or banks==1 else count*8) + slots*8 + 24
    reserve=int(admission.get('additional_reserve_bytes',2<<30))
    budgets=[max(0,item['free_bytes']-item.get('additional_reserve_bytes',reserve))
             for item in admission.get('inventory',[]) if type(item.get('free_bytes')) is int]
    memory_limit=None
    shard_budget=MAX_LOGICAL_SHARDS
    if budgets:
        bitmap_bytes=0
        if bitmap:
            domain=bitmap_domain(graph,position_orbits=env.get('_MGBFS_BITMAP_POSITION_ORBITS_CAPABLE')=='1')
            bitmap_bytes=max(bitmap_bytes_by_rank(domain if domain is not None else 1<<64,world))
        available=[max(0,budget-bitmap_bytes)//segment_bytes for budget in budgets]
        memory_limit=min(available)*count
        # Every active shard owns at least one whole segment. This is a
        # necessary bound, not a complete workspace or production admission.
        # Segment size shrinks when there are more shards. Bound their count
        # by minimum allocation, not the single-shard allocation above.
        minimum_rows=min(32,target_rows,segment_rows or target_rows)
        minimum_slots=0 if bitmap else 1<<(2*minimum_rows*banks-1).bit_length()
        minimum_bytes=minimum_rows*banks*(row_bytes+(0 if bitmap or banks==1 else 8))+minimum_slots*8+24
        shard_budget=min(MAX_LOGICAL_SHARDS,max(1,min(max(0,budget-bitmap_bytes)//minimum_bytes for budget in budgets)))

    memory_limited=memory_limit is not None and (estimated is None or estimated>memory_limit)
    effective=memory_limit if estimated is None else estimated
    if memory_limit is not None:
        effective=min(effective,memory_limit) if effective is not None else memory_limit
    if rank_limit is not None:
        live_limit=rank_limit*banks
        effective=min(effective,live_limit) if effective is not None else live_limit
    # Shards divide each bank separately. Three consecutive layers coexist,
    # but they do not share one bank's index space; sizing by their sum
    # needlessly multiplies shard count without increasing row capacity.
    peak_effective=peak_estimate
    if memory_limit is not None:
        peak_memory_limit=memory_limit//banks
        peak_effective=min(peak_effective,peak_memory_limit) if peak_effective is not None else peak_memory_limit
    if rank_limit is not None:
        peak_effective=min(peak_effective,rank_limit) if peak_effective is not None else rank_limit
    if effective is None:
        effective=target_rows*banks
    if peak_effective is None:
        peak_effective=target_rows
    effective=max(1,effective)
    peak_effective=max(1,peak_effective)
    needed=max(minimum_shards,_ceil_div(peak_effective,target_rows))
    rounded=1<<(needed-1).bit_length()
    shard_ceiling=1<<(shard_budget.bit_length()-1)
    return {'target_shards':min(MAX_LOGICAL_SHARDS,shard_ceiling,rounded),
            'target_rows_per_shard':target_rows,
            'global_live_bound':global_live, 'global_peak_bank_bound':global_peak, 'bound_kind':kind,
            'estimated_peak_bank_rows_per_rank':peak_effective,
            'estimated_live_rows_per_rank':effective,
            'estimated_rank_live_before_memory_cap':estimated,
            'estimated_max_live_rows_in_vram':memory_limit,
            'memory_limited':memory_limited,
            'shard_count_capped':rounded>min(MAX_LOGICAL_SHARDS,shard_ceiling),
            'requested_shards_before_storage_cap':rounded,
            'minimum_shard_segment_budget':shard_budget,
            'segment_rows_used_for_estimate':rows,
            'rank_balance_margin':1.25,
            'guarantees_shard_limit':False,
            'scope':'startup heuristic for peak per bank and retained states; skew uses independent segments and actual VRAM admission'}
