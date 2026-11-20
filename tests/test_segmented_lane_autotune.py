"""The segmented tuner measures owner lanes and carries the result into launch."""
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from multigpubfs import autotune
from multigpubfs.graph_definition import GraphDefinition


def _measure(tmp_path, *, forced=None, bitmap=False, fixed_shards=None):
    native=tmp_path/'native'
    native.write_bytes(b'native test executable')
    graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
    calls=[]
    def admit(graph,devices,capacity,shards,native,env,directory,**kwargs):
        return {'devices':[0,1], 'inventory':[], 'plan':{'capacity':100000,'batch':131072,'history_algorithm':'HASH','owner_lanes':0}}
    def pilot(graph,target,**kw):
        assert kw['capacity'] is None, 'segmented probes must not invent a per-rank cap or atomic counter'
        env=kw['_native_env']; lanes=int(env['MGBFS_SEGMENT_OWNER_LANES'])
        calls.append((kw['shards'],lanes,kw['_batch']))
        seconds=0.1 if (kw['shards'],lanes)==(8,8) else 1.0
        return {'backend':'EXPERIMENTAL_PERMUTATION_BITMAP' if bitmap else 'EXPERIMENTAL_SEGMENTED24',
                'segment_memory_by_rank':[{'owner_lanes':lanes,'exact_exchange':env.get('MGBFS_SEGMENT_EXACT_EXCHANGE','0')=='1'}]*2,
                'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[seconds]*4,
                'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
    env={'MGBFS_GENERIC_HISTORY':'hash','MGBFS_EXPERIMENTAL_SEGMENTED24':'1'}
    if bitmap:env['MGBFS_EXPERIMENTAL_BITMAP']='1'
    if forced is not None:env['MGBFS_SEGMENT_OWNER_LANES']=str(forced)
    policy={'shards':[1,4,8,16], 'probe_capacity_per_rank':50000}
    missing=subprocess.CompletedProcess([],1,'','')
    with patch.object(autotune,'_admit',side_effect=admit), patch.object(autotune,'_system_info',return_value=missing), patch.object(autotune,'_dependency_identity',return_value=None), patch('multigpubfs.hardware_profiles.startup_policy',return_value=policy), patch('multigpubfs.hardware_profiles.small_permutation_workload',return_value=False), patch('multigpubfs.launch.run_graph',side_effect=pilot):
        result=autotune.choose_profile(graph,[0,1],None,90,str(native),env,allow_specialized=False,fixed_shards=fixed_shards)
    return result,calls


@pytest.mark.parametrize('bitmap',[False,True])
def test_tuner_measures_lanes_and_carries_winner_to_launch(tmp_path,bitmap):
    result,calls=_measure(tmp_path,bitmap=bitmap)
    assert (8,4,8192) in calls and (8,8,8192) in calls
    assert result['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE'
    assert result['shards']==8 and result['segment_owner_lanes']==8
    assert result['pilots'][result['winner_index']]['segment_owner_lanes_by_rank']==[8,8]
    assert autotune._profile_environment({},result)['MGBFS_SEGMENT_OWNER_LANES']=='8'


def test_explicit_lanes_are_not_overridden_by_tuner(tmp_path):
    result,calls=_measure(tmp_path,forced=3)
    assert len(calls)>=2 and all(shards>=lanes==3 for shards,lanes,_ in calls)
    assert result['segment_owner_lanes']==3


@pytest.mark.parametrize('value',['0','-1','65','bad'])
def test_invalid_explicit_lanes_fail_before_probe(tmp_path,value):
    with pytest.raises(ValueError,match='SEGMENTED_OWNER_LANES'):
        _measure(tmp_path,forced=value)


@pytest.mark.parametrize('bitmap',[False,True])
def test_tuner_keeps_user_shards_and_still_measures_profiles(tmp_path,bitmap):
    result,calls=_measure(tmp_path,bitmap=bitmap,fixed_shards=4)
    assert len(calls)>=2 and all(shards==4 for shards,lanes,batch in calls)
    assert result['shards']==4
    assert result['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE'
    assert len({(lanes,batch) for _,lanes,batch in calls})>=2

def test_fixed_shards_and_forced_lanes_conflict_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='SEGMENTED_FIXED_SHARDS_LANES_CONFLICT'):
        _measure(tmp_path,forced=8,fixed_shards=4)
