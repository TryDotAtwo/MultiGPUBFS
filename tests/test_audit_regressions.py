"""Regressions for advisory caches and per-rank capacity selection."""
import errno
import json
import subprocess
from pathlib import Path
from unittest.mock import patch
import pytest
from multigpubfs import autotune
from multigpubfs.graph_definition import GraphDefinition

def measured(tmp_path, cache_read=None, fail_mkdir=False):
    cache=tmp_path/'profile-cache'
    native=tmp_path/'native'
    native.write_bytes(b'test')
    graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
    calls=[]
    def admission(*args, **kwargs):
        return {'devices':[0,1], 'inventory':[], 'plan':{'capacity':50000,'batch':64,'history_algorithm':'HASH','owner_lanes':0}}
    def pilot(g,target,**kwargs):
        calls.append(str(target))
        return {'layer_sizes':[1,20000,20000,20000,1], 'layer_seconds':[.1,.2,.3,.4],
                'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
    old_file=Path.is_file
    old_read=Path.read_text
    old_mkdir=Path.mkdir
    def isfile(p):
        if p.parent == cache and p.suffix == '.json' and cache_read is not None: return True
        return old_file(p)
    def read(p,*a,**kw):
        if p.parent == cache and p.suffix == '.json' and cache_read is not None:return cache_read
        return old_read(p,*a,**kw)
    def mkdir(p,*a,**kw):
        if p == cache and fail_mkdir: raise PermissionError(errno.EACCES,'permission denied')
        return old_mkdir(p,*a,**kw)
    with patch.dict('os.environ',{'MGBFS_PROFILE_CACHE':str(cache)}), \
         patch.object(autotune,'_admit',side_effect=admission), \
         patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],0,'ok','')), \
         patch.object(autotune,'_dependency_identity',return_value={'resolved_dependencies':[]}), \
         patch('multigpubfs.hardware_profiles.small_permutation_workload',return_value=False), \
         patch('multigpubfs.launch.run_graph',side_effect=pilot), \
         patch.object(Path,'is_file',isfile),patch.object(Path,'read_text',read),patch.object(Path,'mkdir',mkdir):
        result=autotune.choose_profile(graph,[0,1],None,90,str(native),{'MGBFS_GENERIC_HISTORY':'hash'},allow_specialized=False)
    return result,calls

@pytest.mark.parametrize('invalid',['[]','null','"text"'])
def test_non_mapping_cache_is_a_miss_and_measured_profile_survives(tmp_path,invalid):
    result,calls=measured(tmp_path,cache_read=invalid)
    assert result['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE'
    assert result['measured'] and len(calls)>=2

def test_cache_mkdir_failure_does_not_discard_measured_profile(tmp_path):
    result,calls=measured(tmp_path,fail_mkdir=True)
    assert result['measured'] and len(calls)>=2
    assert result['profile_cache_write_status']['stage']=='mkdir'
    assert result['profile_cache_write_status']['errno']==errno.EACCES

def test_equal_hard_cap_uses_speed_not_uncapped_estimate():
    pilots=[{'capacity_score_rows':9_000_000,'capacity_equivalent_rows':9_000_000,
             'capacity_score_rows_by_rank':[9_000_000]*2,'capacity_equivalent_rows_by_rank':[9_000_000]*2,
             'capacity_limit_per_rank_by_rank':[1_000_000]*2,'capacity_limit_scope':'one next layer per rank'},
            {'capacity_score_rows':8_000_000,'capacity_equivalent_rows':8_000_000,
             'capacity_score_rows_by_rank':[8_000_000]*2,'capacity_equivalent_rows_by_rank':[8_000_000]*2,
             'capacity_limit_per_rank_by_rank':[1_000_000]*2,'capacity_limit_scope':'one next layer per rank'}]
    assert autotune._pick_segmented_profile(pilots,[3.,1.],'max_capacity')==1

def test_capacity_score_uses_weakest_rank():
    def profile(rows):
        return {'capacity_score_rows':min(rows),'capacity_equivalent_rows':min(rows),
                'capacity_score_rows_by_rank':rows,'capacity_equivalent_rows_by_rank':rows,
                'capacity_limit_per_rank_by_rank':[None,None], 'capacity_limit_scope':'one next layer per rank'}
    assert autotune._pick_segmented_profile([profile([4_000_000,1_000_000]),profile([2_000_000,2_000_000])],[1.,2.],'max_capacity')==1

def test_capacity_scope_and_rank_receipts_are_not_mixed():
    base={'capacity_score_rows':10,'capacity_equivalent_rows':10,
          'capacity_score_rows_by_rank':[10,10],'capacity_equivalent_rows_by_rank':[10,10],
          'capacity_limit_per_rank_by_rank':[5,5],'capacity_limit_scope':'one next layer per rank'}
    other=dict(base,capacity_limit_scope='all visited states per rank')
    with pytest.raises(RuntimeError,match='CAPACITY.*SCOPE'):
        autotune._pick_segmented_profile([base,other],[1.,2.],'max_capacity')
    other=dict(base,capacity_limit_per_rank_by_rank=[5])
    with pytest.raises(RuntimeError,match='CAPACITY.*EVIDENCE'):
        autotune._pick_segmented_profile([base,other],[1.,2.],'max_capacity')
    other=dict(base,capacity_limit_per_rank_by_rank=[5,6])
    with pytest.raises(RuntimeError,match='CAPACITY.*LIMIT'):
        autotune._pick_segmented_profile([base,other],[1.,2.],'max_capacity')

@pytest.mark.parametrize('rank_cap',[None,50000])
def test_max_capacity_probes_all_admitted_candidates_without_overwriting_deadline(tmp_path,rank_cap):
    native=tmp_path/'native';native.write_bytes(b'native')
    graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
    calls=[]
    admission={'devices':[0,1],'inventory':[],'plan':{'capacity':100000,'batch':64,'history_algorithm':'HASH','owner_lanes':0,'queue_capacity':4096}}
    geometry={'admitted_shards':4,'minimum_shard_segment_budget':64}
    env={'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_GENERIC_HISTORY':'hash',
         'MGBFS_MEMORY_PRESET':'max_capacity','MGBFS_SEGMENT_QUEUE_CAP':'2048',
         'MGBFS_SEGMENT_OWNER_LANES':'1','_MGBFS_PRESET_IMPLICIT_QUEUE':'1',
         '_MGBFS_PRESET_IMPLICIT_LANES':'1'}
    def pilot(g,target,**kwargs):
        calls.append(str(target))
        e=kwargs['_native_env'];lane=int(e['MGBFS_SEGMENT_OWNER_LANES'])
        memory={'owner_lanes':lane,'exact_exchange':False,'actual_queue_capacity':2048,
                'storage_budget_bytes':10000,'bytes_per_segment':100,'rows_per_bank':1000,
                'capacity_limit_per_rank':rank_cap,'capacity_limit_is_hard':True,
                'capacity_limit_scope':'one next layer per rank'}
        return {'backend':'EXPERIMENTAL_SEGMENTED24','segment_memory_by_rank':[dict(memory),dict(memory)],
                'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[.1,.2,.3,.4],
                'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
    with patch.object(autotune,'_admit',return_value=admission), \
         patch.object(autotune,'_segmented_shard_plan',return_value=(geometry,admission)), \
         patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],1,'','')), \
         patch.object(autotune,'_dependency_identity',return_value=None), \
         patch('multigpubfs.hardware_profiles.small_permutation_workload',return_value=False), \
         patch('multigpubfs.launch.run_graph',side_effect=pilot):
        result=autotune.choose_profile(graph,[0,1],None,90,str(native),env,allow_specialized=False,fixed_shards=4)
    assert len(calls)>=2
    assert result['measured'] and result['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE'
