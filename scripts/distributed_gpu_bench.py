"""Two-T4 native NCCL versus immutable CayleyPy torchrun matrix BFS."""
import argparse,gc,json,os,statistics,subprocess,sys,time
from pathlib import Path
from symmetric_gpu_bench import matrix_generators,math_factorial

def baseline_worker(n,batch,out):
 import numpy as np,torch,torch.distributed as dist
 from cayleypy import CayleyGraph,CayleyGraphDef
 from cayleypy.cayley_graph_def import MatrixGenerator
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
 definition=CayleyGraphDef.for_matrix_group(generators=[MatrixGenerator.create(x,modulo=2) for x in matrix_generators(n)])
 graph=CayleyGraph(definition,device='cuda',specific_devices=[local],batch_size=batch,random_seed=20260828,verbose=0)
 warm=graph.bfs(max_layer_size_to_store=1);assert warm.bfs_completed and sum(warm.layer_sizes)==math_factorial(n)
 del warm;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();dist.barrier();before_free,total=torch.cuda.mem_get_info();start=time.perf_counter();result=graph.bfs(max_layer_size_to_store=1);torch.cuda.synchronize();dist.barrier();seconds=time.perf_counter()-start;after_free,_=torch.cuda.mem_get_info();assert result.bfs_completed and sum(result.layer_sizes)==math_factorial(n)
 row=dict(status='COMPLETE',backend='cayleypy_torchrun',rank=rank,group=f's{n}',batch=batch,search_complete_seconds=seconds,durable_run_commit_seconds=None,layer_sizes=result.layer_sizes,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_before_used_bytes=total-before_free,cuda_after_used_bytes=total-after_free,output_contract='global counts; no archive')
 Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/f'rank-{rank}.json').write_text(json.dumps(row))

def run_group(command,out,label,env,timeout=7200):
 row=dict(label=label,command=command,status='INCOMPLETE');rank_out=out/(label+'-ranks');rank_out.mkdir()
 command=[x.replace('{RANK_OUT}',str(rank_out)) for x in command]
 with (out/(label+'.log')).open('w') as log,(out/(label+'-smi.csv')).open('w') as smi:
  sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,clocks.sm,power.draw','--format=csv,noheader,nounits','-lms','50'],stdout=smi,stderr=subprocess.STDOUT)
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
  ranks=[json.loads(x.read_text()) for x in rank_out.glob('rank-*.json')];ranks.sort(key=lambda x:x['rank'])
  if len(ranks)!=2:raise ValueError('rank result inventory')
  row.update(status='COMPLETE',backend=ranks[0]['backend'],rank_results=ranks,search_complete_seconds=max(x['search_complete_seconds'] for x in ranks),durable_run_commit_seconds=max((x.get('durable_run_commit_seconds') or 0) for x in ranks))
  if 'local_layer_sizes' in ranks[0]:row['layer_sizes']=[sum(values) for values in zip(*(x['local_layer_sizes'] for x in ranks))]
  else:
   if ranks[0]['layer_sizes']!=ranks[1]['layer_sizes']:raise ValueError('baseline rank count mismatch')
   row['layer_sizes']=ranks[0]['layer_sizes']
 else:row['status']='FAILED' if row['status']=='INCOMPLETE' else row['status']
 peaks={0:[],1:[]}
 for line in (out/(label+'-smi.csv')).read_text().splitlines():
  f=line.split(',')
  if len(f)==8:
   try:peaks[int(f[1])].append(float(f[3]))
   except ValueError:pass
 row['smi_peak_mib_per_rank']=[max(peaks[x]) if peaks[x] else None for x in range(2)];row['smi_peak_mib_total']=sum(x or 0 for x in row['smi_peak_mib_per_rank']);(out/(label+'.json')).write_text(json.dumps(row,indent=2));print(label,row['status'],row.get('search_complete_seconds'),flush=True);return row

def stats(rows):
 values=[x['search_complete_seconds'] for x in rows];median=statistics.median(values)
 return dict(median_seconds=median,mad_seconds=statistics.median(abs(x-median) for x in values),samples_seconds=values,repeats=len(rows),peak_mib_per_rank=[max(x['smi_peak_mib_per_rank'][r] for x in rows) for r in range(2)],peak_mib_total=max(x['smi_peak_mib_total'] for x in rows))

def suite(native,source,out,env):
 out.mkdir(parents=True,exist_ok=True);report=dict(schema=1,status='INCOMPLETE',scope='physical 2xT4, same S_n matrix states, native mandatory archive, CayleyPy torchrun no archive',rows=[],comparisons=[])
 def save():(out/'summary.json').write_text(json.dumps(report,indent=2))
 def run(backend,n,batch,phase,rep=0):
  label=f's{n}-{backend}-b{batch}-{phase}-{rep}'
  if backend=='native':
   bootstrap=Path('/tmp')/(label+'-bootstrap');prefix=str(Path('/tmp')/(label+'-archive'));rows=min(batch,16384);slots=(math_factorial(n)+rows-1)//rows+64;cfg=dict(env,MGBFS_BENCH_CAPACITY=str(math_factorial(n)),MGBFS_FUTURE_CAPACITY=str(math_factorial(n)),MGBFS_ARCHIVE_ROWS=str(rows),MGBFS_ARCHIVE_SLOTS=str(slots));command=['torchrun','--standalone','--nproc-per-node=2','--no-python',str(native),f's{n}',str(batch),str(bootstrap),prefix,'{RANK_OUT}']
  else:cfg=env;command=['torchrun','--standalone','--nproc-per-node=2',str(Path(__file__).resolve()),'baseline-worker',str(n),str(batch),'{RANK_OUT}']
  try:row=run_group(command,out,label,cfg,timeout=int(env.get('MGBFS_RUN_TIMEOUT','7200')))
  finally:
   if backend=='native':
    for rank in range(2):Path(f'{prefix}-rank-{rank}.mgbfsar1').unlink(missing_ok=True)
    bootstrap.unlink(missing_ok=True)
  row.update(phase=phase,repetition=rep,config_backend=backend,batch=batch);report['rows'].append(row);save();return row
 try:
  if env.get('MGBFS_DIAGNOSTIC')=='1':
   env=dict(env,MGBFS_TRACE_DEPTHS='1',MGBFS_RUN_TIMEOUT='600')
   row=run('native',8,1024,'diagnostic')
   report['status']=row['status']
   return report
  expected=None;n=10;configs={};choices={'native':[16384,65536,262144],'cayleypy':[65536,262144,1048576]}
  for backend,batches in choices.items():
   trials=[]
   for batch in batches:
    row=run(backend,n,batch,'calibrate')
    if row['status']=='COMPLETE':expected=expected or row['layer_sizes'];assert row['layer_sizes']==expected;trials.append((row['search_complete_seconds'],batch))
   if not trials:raise ValueError(f'no successful {backend} config')
   configs[backend]=min(trials)[1]
  measured={x:[] for x in configs}
  for rep in range(5):
   for backend in (['native','cayleypy'] if rep%2==0 else ['cayleypy','native']):
    row=run(backend,n,configs[backend],'measure',rep);assert row['status']=='COMPLETE' and row['layer_sizes']==expected;measured[backend].append(row)
  comparison=dict(group=f's{n}',unique_states=math_factorial(n),configs=configs,layer_sizes=expected)
  for backend,rows in measured.items():comparison[backend]=stats(rows)
  report['comparisons'].append(comparison);report['status']='COMPLETE';return report
 finally:save()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline-worker']);p.add_argument('n',type=int);p.add_argument('batch',type=int);p.add_argument('out');a=p.parse_args();baseline_worker(a.n,a.batch,a.out)
