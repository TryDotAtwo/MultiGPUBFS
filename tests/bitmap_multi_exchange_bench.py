"""Matched multi-GPU bitmap exchange benchmark against an independent oracle.

This test retains only benchmark evidence on the machine running it.  It does
not claim that another graph, topology, GPU model or batch size has the same
performance.  Environment controls BITMAP_BENCH_WORLD, BITMAP_BENCH_N and
BITMAP_BENCH_ORDER (comma-separated padded/exact) and BITMAP_BENCH_BACKEND.
"""
import json, math, os, signal, subprocess, tempfile, time
from pathlib import Path
from multigpubfs import GraphDefinition

world = int(os.environ.get('BITMAP_BENCH_WORLD', '8'))
n = int(os.environ.get('BITMAP_BENCH_N', '13'))
order = os.environ.get('BITMAP_BENCH_ORDER', 'padded,exact,exact,padded').split(',')
backend = os.environ.get('BITMAP_BENCH_BACKEND', 'bitmap')
assert backend in ('bitmap', 'segmented')
assert 1 <= world <= 128 and n >= 2
assert order and all(x in ('padded', 'exact') for x in order)
root = Path(tempfile.mkdtemp(prefix='bitmap-multigpu-bench-', dir='/workspace'))
devices = ','.join(map(str, range(world)))
native = os.environ['MGBFS_EXECUTABLE']
shards, batch, rows, capacity = 8, 8192, 1048576, 200000
gens = []
for i in range(n-1):
    a = list(range(n)); a[i], a[i+1] = a[i+1], a[i]; gens.append(a)
graph = GraphDefinition.permutation(gens, list(range(n)))
source = root/'graph.json'; source.write_text(graph.to_json())
oracle = [1]
for degree in range(1,n+1):
    nxt = [0]*(len(oracle)+degree-1)
    for j,x in enumerate(oracle):
        for k in range(degree): nxt[j+k] += x
    oracle = nxt
assert sum(oracle) == math.factorial(n)
results=[]
base = dict(os.environ, MGBFS_GENERIC_GENERATOR='cuda', MGBFS_GENERIC_HISTORY='hash',
            MGBFS_GENERIC_TRANSPORT='full', MGBFS_GENERIC_SORT='none',
            MGBFS_EXPERIMENTAL_SEGMENTED24='1',
            MGBFS_SEGMENT_ROWS=str(rows), MGBFS_SEGMENT_OWNER_LANES='8',
            MGBFS_SEGMENT_TRANSPORT_TRACE='1')
if backend == 'bitmap': base['MGBFS_EXPERIMENTAL_BITMAP'] = '1'
else: base.pop('MGBFS_EXPERIMENTAL_BITMAP', None)
base.pop('LD_PRELOAD',None)
for index, variant in enumerate(order):
    case = root/f'{index}-{variant}';case.mkdir()
    env = dict(base, MGBFS_SEGMENT_EXACT_EXCHANGE='1' if variant=='exact' else '0')
    info = subprocess.run([native,'graph-info',str(source),devices,str(shards),str(capacity),str(batch)],
                          env=env,text=True,capture_output=True,timeout=120,check=True)
    config=json.loads(info.stdout);config['owner_cuts']=None
    cfg=case/'launch.json';cfg.write_text(json.dumps(config))
    procs=[];files=[];sampler=None;t=time.monotonic()
    print(f'START n={n} world={world} variant={variant} case={case}',flush=True)
    try:
        sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw',
                                  '--format=csv,noheader,nounits','-lms','1000','-f',str(case/'gpu.csv')],
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        for rank in range(world):
            out=(case/f'rank-{rank}.log').open('w');files.append(out)
            p=subprocess.Popen([native,'graph-rank',str(source),str(cfg),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],
                               env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            procs.append(p)
        deadline=float(os.environ.get('BITMAP_BENCH_DEADLINE_SECONDS','3600'))
        while any(p.poll() is None for p in procs):
            failed=[(i,p.returncode) for i,p in enumerate(procs) if p.poll() not in (None,0)]
            if failed: raise RuntimeError(f'rank_failure={failed}; see {case}')
            if time.monotonic()-t > deadline: raise TimeoutError(f'benchmark only deadline; see {case}')
            time.sleep(.1)
        parts=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(world)]
        for i,part in enumerate(parts):
            assert part['status']=='COMPLETE',(i,part.get('status'),part.get('reason'))
            assert part['layer_sizes']==oracle,(i,len(part['layer_sizes']),len(oracle))
        rec={'n':n,'world':world,'backend':backend,'variant':variant,'status':'COMPLETE','layers':len(oracle),
             'states':sum(oracle),'shards':shards,'batch':batch,'rows_per_bank':rows,
             'bfs_seconds':max(x['bfs_seconds'] for x in parts), 'wall_seconds':time.monotonic()-t,
             'retries':[x['source_retries'] for x in parts],
             'segment_memory':[x['segment_memory'] for x in parts],
             'oracle':'Mahonian inversion polynomial','case':str(case)}
        results.append(rec);(root/'summary.json').write_text(json.dumps(results,indent=2))
        print('RESULT',json.dumps(rec),flush=True)
    finally:
        for p in procs:
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        for p in procs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
        if sampler is not None and sampler.poll() is None:
            sampler.terminate();sampler.wait(timeout=10)
        for out in files:out.close()
print('BENCHMARK_COMPLETE',root,flush=True)
