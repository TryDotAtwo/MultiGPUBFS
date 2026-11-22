"""Multi-GPU opt-in packed runtime against independent full CPU layer multisets."""
import itertools,json,os,tempfile
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

native=os.environ['MGBFS_EXECUTABLE']
world=int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','2'))
devices=list(range(world))
root=Path(tempfile.mkdtemp(prefix='pack4-runtime-',dir='/workspace'))
def independent_layers(gens,start):
    front={tuple(start)};seen=set(front);layers=[]
    while front:
        layers.append(front)
        nxt={tuple(parent[i] for i in gen) for parent in front for gen in gens}-seen
        seen.update(nxt);front=nxt
    return layers

def adj(n):
    ans=[]
    for i in range(n-1):
        g=list(range(n));g[i],g[i+1]=g[i+1],g[i];ans.append(g)
    return ans

cases=[
 ('s4',adj(4),[0,1,2,3],'balanced',0,False),
 ('s4-two-bank',adj(4),[2,0,3,1],'max_capacity',8,True),
 ('signed',adj(4),[1000,-12,5,80],'balanced',8,True),
 ('repeated',adj(5),[0,0,1,2,3],'max_capacity',8,False),
 ('directed',[[1,2,3,4,5,0],[4,5,0,1,2,3]],[0,1,2,3,4,5],'max_capacity',0,True),
 ('coupled',[[1,0,3,2,4,5]],[-7,8,10,12,90,91],'balanced',0,False),
]
packed_env=dict(os.environ,MGBFS_SEGMENT_PACK4='1',MGBFS_SEGMENT_TEST_BATCH='1')
plain_env=dict(os.environ,MGBFS_SEGMENT_PACK4='0',MGBFS_SEGMENT_TEST_BATCH='1')
records=[]
with persistent_graph_workers() as pool:
    original=None
    for name,gens,start,preset,page,exact in cases:
        graph=GraphDefinition.permutation(gens,start);layers=independent_layers(gens,start)
        env=dict(packed_env,MGBFS_SEGMENT_EXACT_EXCHANGE=str(int(exact)))
        for depth,expected in enumerate(layers):
            out=root/f'{name}-layer-{depth}'
            report=run_graph(graph,out,devices=devices,autotune=False,shards=4,_batch=2,
               memory_backend='permutation_bitmap',memory_preset=preset,segment_rows=3,
               segment_strategy='atomic',bitmap_owner_page=page,executable=native,
               _profile_layers=depth,_native_env=env)
            assert report['status']=='INCOMPLETE' and report['reason']=='PROFILE_LAYER_LIMIT',report
            assert all(r.get('resident_codec')=='pack4-v1' for r in report['segment_memory_by_rank'])
            assert all(r['banks']==(2 if preset=='max_capacity' else 3) for r in report['segment_memory_by_rank'])
            assert all(r['bitmap_owner_page']==page for r in report['segment_memory_by_rank'])
            assert report['layer_sizes']==list(map(len,layers[:depth+1]))
            state=json.loads((out/'states.json').read_text())
            assert Counter(map(tuple,state['current']))==Counter(expected),(name,depth,state)
            if depth:assert Counter(map(tuple,state['previous_small']))==Counter(layers[depth-1]),(name,depth,state)
            ids=[w.process.pid for w in pool.workers]
            if original is None:original=ids
            assert ids==original,(name,ids,original)
        out=root/f'{name}-complete'
        report=run_graph(graph,out,devices=devices,autotune=False,shards=4,_batch=2,
           memory_backend='permutation_bitmap',memory_preset=preset,segment_rows=3,
           segment_strategy='atomic',bitmap_owner_page=page,executable=native,_native_env=env)
        assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,layers)),report
        state=json.loads((out/'states.json').read_text())
        assert Counter(map(tuple,state['current']))==Counter(layers[-1])
        assert Counter(map(tuple,state['previous_small']))==Counter(layers[-2])
        records.append({'case':name,'layers':len(layers),'states':sum(map(len,layers)),
                        'preset':preset,'page':page,'exact_exchange':exact,'worker_pids':ids})
        print('PASS exact layers',json.dumps(records[-1]),flush=True)
    # Full exact capacity boundary for cyclic S4. Root rank is 0 and the
    # lexicographic rank of the identity is 0, hence owner=ID modulo world.
    gens=adj(4);start=[0,1,2,3];graph=GraphDefinition.permutation(gens,start)
    layers=independent_layers(gens,start);ranks={p:i for i,p in enumerate(itertools.permutations(start))}
    cap=max(max(Counter(ranks[s]%world for s in layer).values()) for layer in layers)
    for requested in (cap,cap-1):
        out=root/f'cap-{requested}'
        report=run_graph(graph,out,devices=devices,autotune=False,shards=4,_batch=2,
           memory_backend='permutation_bitmap',memory_preset='max_capacity',segment_rows=3,
           segment_strategy='atomic',capacity=requested,executable=native,_native_env=packed_env)
        assert all(r['resident_codec']=='pack4-v1' for r in report['segment_memory_by_rank'])
        state=json.loads((out/'states.json').read_text())
        if requested==cap:
            assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,layers)),report
            depth=len(layers)-1
        else:
            first=next(d for d,layer in enumerate(layers) if max(Counter(ranks[s]%world for s in layer).values())>requested)
            depth=first-1
            assert report['status']=='INCOMPLETE' and report['reason']=='RESOURCE_10',report
            assert report['layer_sizes']==list(map(len,layers[:first])),(first,report)
        assert Counter(map(tuple,state['current']))==Counter(layers[depth])
        if depth:assert Counter(map(tuple,state['previous_small']))==Counter(layers[depth-1])
        assert [w.process.pid for w in pool.workers]==original
        print('PASS capacity',requested,'status',report['status'],'last_depth',depth,flush=True)
    # Different codecs and backends must not leak across jobs in the same workers.
    for backend,strategy,env in [('permutation_bitmap','atomic',plain_env),
                                  ('segmented24','warp',plain_env),
                                  ('permutation_bitmap','atomic',packed_env)]:
        report=run_graph(graph,root/f'switch-{backend}-{strategy}-{env["MGBFS_SEGMENT_PACK4"]}',
           devices=devices,autotune=False,shards=4,_batch=2,memory_backend=backend,
           segment_rows=3,segment_strategy=strategy,executable=native,_native_env=env)
        assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,layers))
        codec='pack4-v1' if env['MGBFS_SEGMENT_PACK4']=='1' else 'unpacked-soa'
        assert all(r['resident_codec']==codec for r in report['segment_memory_by_rank'])
        assert [w.process.pid for w in pool.workers]==original
        print('PASS switch',backend,strategy,codec,flush=True)
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('PACK4_RUNTIME_EXACT_LAYERS_PASS',root,flush=True)
