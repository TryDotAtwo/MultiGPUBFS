"""Only validated and mutually consistent resource stops permit fallback."""
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch
import pytest
from multigpubfs import autotune
from multigpubfs.graph_definition import GraphDefinition

def choose(tmp_path, *, mismatch=None, reason='RESOURCE_10', fault=False):
    graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
    native=tmp_path/'native';native.write_bytes(b'native')
    calls=[]
    admission={'devices':[0,1],'inventory':[],'plan':{'capacity':100000,'batch':64,'history_algorithm':'HASH','owner_lanes':0}}
    geometry={'admitted_shards':4,'minimum_shard_segment_budget':64}
    def pilot(g,target,**kwargs):
        idx=len(calls);calls.append(str(target))
        if fault and idx==0:raise RuntimeError('simulated CUDA fault')
        sizes=[1,2,3]
        if mismatch=='prefix' and idx==1:sizes=[1,7]
        seconds=[0.1]*(len(sizes)-1)
        target.mkdir(parents=True)
        current=[[i,0,0] for i in range(sizes[-1])]
        previous=[[i,1,0] for i in range(sizes[-2])]
        if mismatch=='snapshot_disagree' and idx==1:current[0]=[99,0,0]
        if mismatch=='snapshot_count' and idx==0:current.pop()
        if mismatch=='snapshot_duplicate' and idx==0:current[1]=current[0]
        rank_states=[{'schema':2,'state_encoding':'signed_int64_vectors',
                      'current':current[r::2],'previous_small':previous[r::2]} for r in range(2)]
        states={'schema':2,'state_encoding':'signed_int64_vectors',
                'current':[v for part in rank_states for v in part['current']],
                'previous_small':[v for part in rank_states for v in part['previous_small']]}
        data=json.dumps(states).encode()
        target.joinpath('states.json').write_bytes(data)
        digest=hashlib.sha256(data).hexdigest()
        lane=int(kwargs['_native_env']['MGBFS_SEGMENT_OWNER_LANES'])
        memory=[{'rank':rank,'device':rank,'owner_lanes':lane,'exact_exchange':False,'segment_strategy':'atomic','capacity_limit_per_rank':3,'capacity_limit_is_hard':True,'capacity_limit_scope':'one next layer per rank'} for rank in range(2)]
        report={'graph_digest':g.digest(),'backend':'EXPERIMENTAL_SEGMENTED24','devices':[0,1],
                'rank_receipts':['rank-0/report.json','rank-1/report.json'],
                'rank_plans':[admission['plan']]*2,'segment_memory_by_rank':memory,
                'layer_sizes':sizes,'layer_seconds':seconds,'status':'INCOMPLETE','reason':reason,'states_sha256':digest}
        target.joinpath('report.json').write_text(json.dumps(report))
        for rank in range(2):
            directory=target/f'rank-{rank}';directory.mkdir()
            rank_snapshot=rank_states[rank]
            if mismatch=='snapshot_rank_disagree' and idx==0 and rank==1:
                rank_snapshot=dict(rank_snapshot,current=[[98,0,0]])
            rank_data=json.dumps(rank_snapshot).encode()
            directory.joinpath('states.json').write_bytes(rank_data)
            part=dict(report,rank=rank,world=2,device=rank,plan=admission['plan'],
                      states_sha256=hashlib.sha256(rank_data).hexdigest(),
                      segment_memory={k:v for k,v in memory[rank].items() if k not in ('rank','device')})
            if mismatch=='rank' and idx==0 and rank==1:part['layer_sizes']=[1,8,3]
            if mismatch=='graph' and idx==0 and rank==1:part['graph_digest']='wrong'
            if mismatch=='geometry' and idx==0 and rank==1:part['segment_memory']['owner_lanes']=lane+1
            if mismatch=='checksum' and idx==0 and rank==1:directory.joinpath('states.json').write_text('{}')
            directory.joinpath('report.json').write_text(json.dumps(part))
        return report
    env={'MGBFS_EXPERIMENTAL_SEGMENTED24':'1','MGBFS_GENERIC_HISTORY':'hash',
         'MGBFS_SEGMENT_MAX_RANK_STATES':'3','MGBFS_SEGMENT_EXACT_EXCHANGE':'0'}
    failure={'evidence':'test-captured','native_error':{'code':999},'error':'simulated','probe_retried':False}
    with patch.object(autotune,'_admit',return_value=admission), \
         patch.object(autotune,'_segmented_shard_plan',return_value=(geometry,admission)), \
         patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],1,'','')), \
         patch.object(autotune,'_dependency_identity',return_value=None), \
         patch.object(autotune,'_native_probe_failure',side_effect=lambda e:failure if str(e).startswith('simulated CUDA fault') else None), \
         patch('multigpubfs.hardware_profiles.small_permutation_workload',return_value=False), \
         patch('multigpubfs.launch.run_graph',side_effect=pilot):
        return autotune.choose_profile(graph,[0,1],None,90,str(native),env,allow_specialized=False,fixed_shards=4),calls

def test_all_clean_resource_stops_reach_explicit_unmeasured_profile(tmp_path):
    result,calls=choose(tmp_path)
    assert len(calls)>=2 and not result['measured']
    assert result['status']=='VALIDATED_RESOURCE_LIMIT_UNMEASURED_PROFILE'
    assert len(result['resource_rejections'])==len(calls)
    assert result['resource_rejections'][0]['validated_ranks']==2

@pytest.mark.parametrize('mismatch', ['prefix','rank','graph','geometry','checksum','snapshot_count','snapshot_duplicate','snapshot_disagree','snapshot_rank_disagree'])
def test_resource_prefix_and_rank_reports_are_checked(tmp_path,mismatch):
    with pytest.raises(RuntimeError,match='(PREFIX_CORRECTNESS|RESOURCE_RECEIPT|GRAPH_RECEIPT)'):
        choose(tmp_path,mismatch=mismatch)

def test_unknown_resource_code_is_not_reclassified(tmp_path):
    with pytest.raises(RuntimeError,match='SEGMENTED_TUNING_RESOURCE_UNSUPPORTED'):
        choose(tmp_path,reason='RESOURCE_999')

def test_resource_and_cuda_fault_do_not_become_healthy_fallback(tmp_path):
    with pytest.raises(RuntimeError,match='AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE'):
        choose(tmp_path,fault=True)
