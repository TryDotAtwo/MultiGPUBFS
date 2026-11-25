"""Capacity-aware shard sizing stays independent of 24-bit segment overflow."""
from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.shard_policy import choose_segmented_shards

def coxeter(n):
    generators=[]
    for i in range(n-1):
        x=list(range(n)); x[i],x[i+1]=x[i+1],x[i]; generators.append(x)
    return GraphDefinition.permutation(generators,list(range(n)))

def admission(n=14, world=8, free=180_000_000_000):
    return {'devices':list(range(world)),
            'inventory':[{'free_bytes':free} for _ in range(world)],
            'additional_reserve_bytes':2<<30,
            'rank_plans':[{'capacity':1_000_000_000} for _ in range(world)],
            'plan':{'history_layers':3,'elements':n,'state_bytes':1},
            'owner_cuts':[i*(1<<32)//world for i in range(world+1)]}

def test_coxeter_s14_uses_all_three_live_layers_and_configured_cap():
    p=choose_segmented_shards(coxeter(14),admission())
    assert p['global_live_bound']==11_141_606_753
    assert p['target_shards']==1,p
    assert p['target_rows_per_shard']==1<<32

def test_bitmap_has_same_live_state_geometry():
    p=choose_segmented_shards(coxeter(14),admission(),bitmap=True)
    assert p['target_shards']==1,p

def test_bitmap_geometry_uses_effective_domain_and_frontier_codec():
    n = 60
    graph = GraphDefinition.permutation(
        [[p ^ 1 if p // 2 == i else p for p in range(n)] for i in range(n // 2)],
        [-7 if p & 1 else 511 for p in range(n)],
    )
    hardware = admission(n=n, world=2, free=6 << 30)
    hardware["plan"]["state_bytes"] = 8
    unpacked = choose_segmented_shards(
        graph, hardware, bitmap=True, target_rows=1 << 20,
        env={"MGBFS_BITMAP_BANKS": "2", "_MGBFS_BITMAP_POSITION_ORBITS_CAPABLE": "1"},
    )
    packed = choose_segmented_shards(
        graph, hardware, bitmap=True, target_rows=1 << 20,
        env={"MGBFS_BITMAP_BANKS": "2", "MGBFS_SEGMENT_PACK4": "1",
             "_MGBFS_BITMAP_POSITION_ORBITS_CAPABLE": "1"},
    )
    assert packed["bound_kind"] == "CERTIFIED_INDEPENDENT_SWAPS"
    assert packed["global_live_bound"] == 2 * packed["global_peak_bank_bound"]
    assert unpacked["memory_limited"] and not packed["memory_limited"]
    assert packed["target_shards"] > unpacked["target_shards"]

def test_small_graph_remains_one_shard():
    p=choose_segmented_shards(coxeter(4),admission(n=4,world=2))
    assert p['target_shards']==1,p

def test_large_pack4_balanced_bounds_production_geometry(tmp_path, monkeypatch):
    import subprocess
    import pytest
    from multigpubfs import autotune, host_memory, launch, nccl_preflight

    graph = coxeter(13)
    hardware = admission(n=13, world=2, free=9751 << 20)
    hardware['plan']['batch'] = 87_381
    probed = []

    def admit_shards(graph, devices, capacity, shards, native, env, directory):
        probed.append(shards)
        assert env['MGBFS_SEGMENT_EXACT_EXCHANGE'] == '1'
        return hardware

    def production_probe(command, *, env, **_):
        assert command[1] == 'graph-info' and command[-3:] == ['8', 'auto', '8192']
        assert env['MGBFS_SEGMENT_EXACT_EXCHANGE'] == '1'
        return subprocess.CompletedProcess(command, 1, '', 'CPU_FIXTURE_PRODUCTION_REACHED')

    monkeypatch.setattr(launch, 'native_runtime', lambda _: ('mgbfs', {}))
    monkeypatch.setattr(launch, '_require_memory_backend', lambda *_: {
        'segmented_exact_exchange_v1': True, 'bitmap_pack4_experimental_v1': True,
    })
    monkeypatch.setattr(host_memory, 'admit', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launch, 'native_query', production_probe)
    monkeypatch.setattr(nccl_preflight, 'prepare', lambda native, devices, env, output, graph: (env, None))
    monkeypatch.setattr(autotune, '_admit', admit_shards)
    with pytest.raises(RuntimeError, match='CPU_FIXTURE_PRODUCTION_REACHED'):
        launch.run_graph(graph, tmp_path / 'out', devices=[0, 1], memory_backend='permutation_bitmap',
                         bitmap_pack4=True, _profile_options={'pilot_limit': 3})
    assert probed == [1, 8]


def test_pack4_geometry_keeps_other_workloads_and_explicit_knobs():
    from multigpubfs.shard_policy import balanced_pack4_geometry

    env = {'MGBFS_MEMORY_PRESET': 'balanced', 'MGBFS_EXPERIMENTAL_BITMAP': '1',
           'MGBFS_SEGMENT_PACK4': '1', 'MGBFS_SEGMENT_STRATEGY': 'atomic'}
    assert balanced_pack4_geometry(coxeter(12), 2, env) is None
    assert balanced_pack4_geometry(coxeter(13), 1, env) is None
    assert balanced_pack4_geometry(coxeter(13), 2, dict(env, MGBFS_MEMORY_PRESET='max_capacity')) is None
    assert balanced_pack4_geometry(coxeter(13), 2, dict(env, MGBFS_SEGMENT_QUEUE_CAP='2048')) is None

def test_insufficient_memory_caps_shard_work_without_claiming_fit():
    p=choose_segmented_shards(coxeter(14),admission(free=12<<30))
    assert 1<=p['target_shards']<128,p
    assert p['memory_limited'] is True
    assert p['guarantees_shard_limit'] is False

def test_emulated_small_shard_limit_scales_up():
    p=choose_segmented_shards(coxeter(8),admission(n=8,world=2,free=12<<30),target_rows=16,segment_rows=1024)
    assert p['target_shards']>1,p
    assert p['estimated_peak_bank_rows_per_rank']<=p['target_shards']*16

def test_directed_graph_uses_visited_orbit_bound():
    graph=GraphDefinition.permutation([[1,2,3,4,5,6,7,0]],list(range(8)))
    a=admission(n=8,world=2,free=12<<30)
    a['plan']['history_layers']=1
    p=choose_segmented_shards(graph,a,target_rows=2)
    assert p['bound_kind']!='CERTIFIED_COXETER_GLOBAL_FRONTIER'
    assert p['global_live_bound']>=8
    assert p['target_shards']>=4


def test_tuner_does_not_try_below_sized_target(tmp_path,monkeypatch):
    import subprocess
    from multigpubfs import autotune
    import multigpubfs.hardware_profiles as hardware
    import multigpubfs.launch as launch
    graph=coxeter(14); a=admission(); a['plan'].update(capacity=100000,batch=65536,history_algorithm='HASH',owner_lanes=0)
    native=tmp_path/'native'; native.write_bytes(b'executable')
    seen=[]
    monkeypatch.setattr(autotune,'_admit',lambda *args,**kwargs:a)
    monkeypatch.setattr(autotune,'_system_info',lambda *args:subprocess.CompletedProcess([],1,'',''))
    monkeypatch.setattr(autotune,'_dependency_identity',lambda *args:None)
    monkeypatch.setattr(hardware,'small_permutation_workload',lambda *args:False)
    monkeypatch.setattr(hardware,'startup_policy',lambda *args:{'shards':[1,4,8,16],'probe_capacity_per_rank':50000})
    def pilot(graph,target,**kwargs):
        shards=kwargs['shards']; lanes=int(kwargs['_native_env']['MGBFS_SEGMENT_OWNER_LANES']); seen.append(shards)
        return {'backend':'EXPERIMENTAL_SEGMENTED24','segment_memory_by_rank':[{'owner_lanes':lanes,'exact_exchange':kwargs['_native_env'].get('MGBFS_SEGMENT_EXACT_EXCHANGE','0')=='1'} for _ in range(8)],
                'layer_sizes':[1,20000,20000,20000,1], 'layer_seconds':[.1]*4,
                'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
    monkeypatch.setattr(launch,'run_graph',pilot)
    result=autotune.choose_profile(graph,list(range(8)),None,90,str(native),{'MGBFS_EXPERIMENTAL_SEGMENTED24':'1'},allow_specialized=False)
    assert seen and min(seen)>=1,seen
    assert result['shards']>=1,result
    assert result['shard_geometry']['target_shards']==1,result


def test_many_shards_pay_at_least_one_segment_each():
    a=admission(n=12,world=2,free=12<<30)
    p=choose_segmented_shards(coxeter(12),a,target_rows=1024)
    assert p['target_shards'] <= 4096,p
    assert p['target_shards'] <= p['minimum_shard_segment_budget']
    assert p['segment_rows_used_for_estimate'] <= 1024

def test_tuner_does_not_measure_resource_stopped_candidate(tmp_path,monkeypatch):
    import subprocess
    import pytest
    from multigpubfs import autotune
    import multigpubfs.hardware_profiles as hardware
    import multigpubfs.launch as launch
    graph=coxeter(14); a=admission(); a['plan'].update(capacity=100000,batch=65536,history_algorithm='HASH',owner_lanes=0)
    native=tmp_path/'native'; native.write_bytes(b'executable')
    monkeypatch.setattr(autotune,'_admit',lambda *args,**kwargs:a)
    monkeypatch.setattr(autotune,'_system_info',lambda *args:subprocess.CompletedProcess([],1,'',''))
    monkeypatch.setattr(autotune,'_dependency_identity',lambda *args:None)
    monkeypatch.setattr(hardware,'small_permutation_workload',lambda *args:False)
    monkeypatch.setattr(hardware,'startup_policy',lambda *args:{'shards':[1,4,8,16],'probe_capacity_per_rank':50000})
    def pilot(graph,target,**kwargs):
        lanes=int(kwargs['_native_env']['MGBFS_SEGMENT_OWNER_LANES'])
        return {'backend':'EXPERIMENTAL_SEGMENTED24','segment_memory_by_rank':[{'owner_lanes':lanes,'exact_exchange':kwargs['_native_env'].get('MGBFS_SEGMENT_EXACT_EXCHANGE','0')=='1'} for _ in range(8)],
                'layer_sizes':[1,20000,20000,20000,1], 'layer_seconds':[.01]*4,
                'status':'INCOMPLETE','reason':'RESOURCE_10'}
    monkeypatch.setattr(launch,'run_graph',pilot)
    # Detailed receipt validation has independent tests. This selector fixture
    # supplies its successful outcome without fabricating native receipt files.
    monkeypatch.setattr(autotune,'_validate_segmented_resource_report',
                        lambda *args:{'validated_ranks':8,'fully_retained_layer_sha256':{}})
    result=autotune.choose_profile(graph,list(range(8)),None,90,str(native),
                                   {'MGBFS_EXPERIMENTAL_SEGMENTED24':'1'},allow_specialized=False)
    assert result['status']=='VALIDATED_RESOURCE_LIMIT_UNMEASURED_PROFILE'
    assert not result['measured'] and len(result['resource_rejections'])>=2


def test_public_configuration_accepts_maximum_segment_rows():
    from multigpubfs.cayleypy_backend import NativeOptions
    assert NativeOptions().segment_rows == 1 << 32
    assert NativeOptions(segment_rows=1 << 25).segment_rows == 1 << 25


def test_segment_rows_validation():
    import pytest
    from multigpubfs.cayleypy_backend import NativeOptions
    for value in (0, -1, (1 << 32) + 1, True, 1.5):
        with pytest.raises(ValueError, match="INVALID_SEGMENT_ROWS"):
            NativeOptions(segment_rows=value)

def test_custom_segment_rows_forwarded(tmp_path, monkeypatch):
    import json
    import multigpubfs.cayleypy_backend as api
    from cayleypy import CayleyGraph, PermutationGroups
    graph=CayleyGraph(PermutationGroups.coxeter(3), device="cpu")
    seen={}
    def stub(definition, output, **kwargs):
        seen.update(kwargs)
        output.mkdir(parents=True)
        (output/"states.json").write_text(json.dumps({"current":[],"previous_small":[]}))
        return {"status":"COMPLETE","layer_sizes":[1]}
    monkeypatch.setattr(api,"run_graph",stub)
    api.bfs(graph, output=tmp_path/"result", native_options=api.NativeOptions(segment_rows=1<<25,autotune=False))
    assert seen["segment_rows"] == 1<<25

def test_shard_sizing_uses_peak_bank_and_exposes_live_bound():
    a=admission(n=8,world=2,free=180_000_000_000)
    p=choose_segmented_shards(coxeter(8),a,target_rows=2048)
    assert p['global_live_bound']>p['global_peak_bank_bound']
    assert p['global_peak_bank_bound']==3836
    assert p['target_shards']==2, p
    assert p['estimated_peak_bank_rows_per_rank'] <= p['target_shards'] * 2048

def test_shard_memory_limit_uses_least_available_device():
    a=admission(n=14,world=2,free=180_000_000_000)
    a['inventory'][1]['free_bytes']=3<<30
    limited=choose_segmented_shards(coxeter(14),a,target_rows=1<<20)
    a['inventory'][0]['free_bytes']=3<<30
    same=choose_segmented_shards(coxeter(14),a,target_rows=1<<20)
    assert limited['estimated_max_live_rows_in_vram']==same['estimated_max_live_rows_in_vram']
    assert limited['memory_limited']

def test_tiny_configured_segments_used_in_estimate():
    a=admission(n=4,world=2)
    p=choose_segmented_shards(coxeter(4),a,target_rows=1,segment_rows=1)
    assert p['segment_rows_used_for_estimate']==1
