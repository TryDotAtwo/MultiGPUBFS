"""Honest single-T4 S_n matrix BFS A/B with complete archived native output."""
import argparse, gc, hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path

def matrix_generators(n):
    import numpy as np
    permutations=[[(i+1)%n for i in range(n)],[(i+n-1)%n for i in range(n)],list(range(n))]
    permutations[2][0],permutations[2][1]=1,0
    result=[]
    for permutation in permutations:
        matrix=np.zeros((n,n),dtype=np.int64)
        matrix[range(n),permutation]=1
        result.append(matrix)
    return result

def baseline(n,batch,validate):
    import numpy as np, torch
    from cayleypy import CayleyGraph, CayleyGraphDef
    from cayleypy.cayley_graph_def import MatrixGenerator
    if torch.cuda.device_count()!=1: raise RuntimeError('exactly one visible GPU required')
    definition=CayleyGraphDef.for_matrix_group(
        generators=[MatrixGenerator.create(x,modulo=2) for x in matrix_generators(n)])
    assert definition.generators_inverse_closed
    graph=CayleyGraph(definition,device='cuda',num_gpus=1,batch_size=batch,random_seed=20260828,verbose=0)
    warm=graph.bfs(max_layer_size_to_store=1)
    assert warm.bfs_completed and sum(warm.layer_sizes)==math_factorial(n)
    del warm;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    before_free,total=torch.cuda.mem_get_info();started=time.perf_counter()
    result=graph.bfs(max_layer_size_to_store=None if validate else 1)
    torch.cuda.synchronize();seconds=time.perf_counter()-started;after_free,_=torch.cuda.mem_get_info()
    assert result.bfs_completed and sum(result.layer_sizes)==math_factorial(n)
    digests=[]
    if validate:
        for depth in range(len(result.layer_sizes)):
            states=np.ascontiguousarray(result.get_layer(depth).reshape(-1,n*n),dtype=np.uint8)
            keys=states.view(f'V{n*n}').reshape(-1);keys.sort()
            digests.append(hashlib.sha256(keys.tobytes()).hexdigest())
    return dict(status='COMPLETE',backend='cayleypy_single_matrix',group=f's{n}',batch=batch,
        verification_only=validate,search_complete_seconds=seconds,durable_run_commit_seconds=None,
        unique_states=sum(result.layer_sizes),layer_sizes=result.layer_sizes,layer_sha256=digests,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_before_used_bytes=total-before_free,cuda_after_used_bytes=total-after_free,
        output_contract='counts and requested verification layers; no archive')

def math_factorial(n):
    value=1
    for i in range(2,n+1):value*=i
    return value

def worker(command,out,label,env,timeout=7200):
    row=dict(label=label,command=command,status='INCOMPLETE')
    with (out/(label+'.log')).open('w') as log,(out/(label+'-smi.csv')).open('w') as smi:
        sampler=subprocess.Popen(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=timestamp,uuid,memory.used,utilization.gpu,utilization.memory,clocks.sm,power.draw','--format=csv,noheader,nounits','-lms','50'],stdout=smi,stderr=subprocess.STDOUT)
        try:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);started=time.monotonic()
            while process.poll() is None:
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f'RUNNING {label}: {time.monotonic()-started:.0f}s',flush=True)
                    if time.monotonic()-started>timeout:process.kill();process.wait();row['status']='TIMEOUT';break
            row['exit_code']=process.returncode
        finally:sampler.terminate();sampler.wait()
    if row['exit_code']==0:
        values=[json.loads(x) for x in (out/(label+'.log')).read_text().splitlines() if x.startswith('{')]
        if len(values)!=1:raise ValueError('worker JSON inventory')
        row.update(values[0])
    elif row['status']!='TIMEOUT':row['status']='FAILED'
    samples=[]
    for line in (out/(label+'-smi.csv')).read_text().splitlines():
        fields=line.split(',')
        if len(fields)==7:
            try:samples.append(float(fields[2]))
            except ValueError:pass
    row['smi_process_peak_mib']=max(samples) if samples else None;row['smi_samples']=len(samples)
    (out/(label+'.json')).write_text(json.dumps(row,indent=2));print(label,row['status'],row.get('search_complete_seconds'),flush=True)
    return row

def stats(rows):
    values=[r['search_complete_seconds'] for r in rows];median=statistics.median(values)
    return dict(median_seconds=median,mad_seconds=statistics.median(abs(x-median) for x in values),
        samples_seconds=values,repeats=len(rows),smi_process_peak_mib=max(r['smi_process_peak_mib'] for r in rows))

def suite(native,out,env):
    out.mkdir(parents=True,exist_ok=True)
    report=dict(schema=1,status='INCOMPLETE',scope='same S_n matrix states and three inverse-closed generators; native archive mandatory',rows=[],comparisons=[])
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2))
    def run(backend,n,batch,k,phase,rep=0):
        verify=phase=='verify';label=f's{n}-{backend}-b{batch}-k{k}-{phase}-{rep}'
        if backend=='native':
            archive=Path('/tmp')/(label+'.mgbfsar1')
            cmd=[str(native),f's{n}',str(batch),str(k),'1','verify' if verify else 'time',str(archive)]
            archive_rows=min(batch,16384);slots=(math_factorial(n)+archive_rows-1)//archive_rows+64
            runenv=dict(env,MGBFS_BENCH_CAPACITY=str(math_factorial(n)),MGBFS_FUTURE_CAPACITY=str(math_factorial(n)),MGBFS_ARCHIVE_ROWS=str(archive_rows),MGBFS_ARCHIVE_SLOTS=str(slots))
        else:
            cmd=[sys.executable,str(Path(__file__).resolve()),'baseline',str(n),str(batch),'verify' if verify else 'time'];runenv=env
        try: row=worker(cmd,out,label,runenv)
        finally:
            if backend=='native': archive.unlink(missing_ok=True)
        row.update(phase=phase,repetition=rep,config_backend=backend,macro_depth=k);report['rows'].append(row);save();return row
    try:
        native_verify=run('native',8,65536,2,'verify');base_verify=run('cayleypy',8,65536,1,'verify')
        if native_verify['status']!='COMPLETE' or base_verify['status']!='COMPLETE' or native_verify['layer_sizes']!=base_verify['layer_sizes'] or native_verify['layer_sha256']!=base_verify['layer_sha256']:raise ValueError('S8 exact verification mismatch')
        n=10;configs={};expected=None
        choices={'native':[(b,k) for k in (1,2) for b in (16384,65536,262144)],'cayleypy':[(b,1) for b in (65536,262144,1048576)]}
        for backend,variants in choices.items():
            trials=[]
            for batch,k in variants:
                row=run(backend,n,batch,k,'calibrate')
                if row['status']=='COMPLETE':
                    expected=expected or row['layer_sizes']
                    if row['layer_sizes']!=expected:raise ValueError('calibration layer mismatch')
                    trials.append((row['search_complete_seconds'],batch,k))
            if not trials:raise ValueError(f'no successful {backend} config')
            _,batch,k=min(trials);configs[backend]=(batch,k)
        measured={key:[] for key in configs}
        for rep in range(5):
            for backend in (['native','cayleypy'] if rep%2==0 else ['cayleypy','native']):
                batch,k=configs[backend];row=run(backend,n,batch,k,'measure',rep)
                if row['status']!='COMPLETE' or row['layer_sizes']!=expected:raise ValueError('measured run mismatch')
                measured[backend].append(row)
        comparison=dict(group=f's{n}',unique_states=math_factorial(n),configs=configs,layer_sizes=expected)
        for backend,rows in measured.items():comparison[backend]=stats(rows)
        report['comparisons'].append(comparison);report['status']='COMPLETE';return report
    finally:save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['baseline']);parser.add_argument('n',type=int);parser.add_argument('batch',type=int);parser.add_argument('operation',choices=['verify','time']);args=parser.parse_args()
    print(json.dumps(baseline(args.n,args.batch,args.operation=='verify')))
