"""Two-T4 native NCCL versus immutable CayleyPy torchrun matrix BFS."""
import argparse,gc,json,math,os,shutil,statistics,subprocess,sys,time
from pathlib import Path
from symmetric_gpu_bench import matrix_generators,math_factorial

class ProgressRelay:
 """Bounded, read-only tail of depth telemetry; retain the full log on disk."""
 def __init__(self,source,output):
  self.source=source;self.output=output;self.offset=0;self.pending=b'';self.dropping=False

 def emit(self,line):
  if line.startswith((b'MGBFS_DEPTH_',b'DISTRIBUTED_BENCH_INCOMPLETE')):
   print(line.decode('utf-8',errors='replace'),file=self.output,flush=True)

 def poll(self,final=False):
  while True:
   self.source.seek(self.offset);chunk=self.source.read(65536);self.offset+=len(chunk)
   parts=chunk.split(b'\n')
   for index,part in enumerate(parts):
    if not self.dropping:
     if len(self.pending)+len(part)>4096:self.pending=b'';self.dropping=True
     else:self.pending+=part
    if index<len(parts)-1:
     if not self.dropping:self.emit(self.pending)
     self.pending=b'';self.dropping=False
   if not final or not chunk:break
  if final:
   if self.pending and not self.dropping:self.emit(self.pending)
   self.pending=b'';self.dropping=False

def baseline_worker(n,batch,out):
 if int(os.environ['WORLD_SIZE'])==1:
  from symmetric_gpu_bench import baseline
  row=baseline(n,batch,False)
  row.update(rank=0,world_size=1,warmup_completed=True)
  Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/'rank-0.json').write_text(json.dumps(row))
  return
 import numpy as np,torch,torch.distributed as dist
 from cayleypy import CayleyGraph,CayleyGraphDef
 from cayleypy.cayley_graph_def import MatrixGenerator
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
 definition=CayleyGraphDef.for_matrix_group(generators=[MatrixGenerator.create(x,modulo=2) for x in matrix_generators(n)])
 graph=CayleyGraph(definition,device='cuda',specific_devices=[local],batch_size=batch,random_seed=20260828,verbose=0)
 warm=graph.bfs(max_layer_size_to_store=1);assert warm.bfs_completed and sum(warm.layer_sizes)==math_factorial(n)
 del warm;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();dist.barrier();before_free,total=torch.cuda.mem_get_info();start=time.perf_counter();result=graph.bfs(max_layer_size_to_store=1);torch.cuda.synchronize();dist.barrier();seconds=time.perf_counter()-start;after_free,_=torch.cuda.mem_get_info();assert result.bfs_completed and sum(result.layer_sizes)==math_factorial(n)
 row=dict(status='COMPLETE',backend='cayleypy_torchrun',rank=rank,group=f's{n}',batch=batch,warmup_completed=True,search_complete_seconds=seconds,durable_run_commit_seconds=None,layer_sizes=result.layer_sizes,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_before_used_bytes=total-before_free,cuda_after_used_bytes=total-after_free,output_contract='global counts; no archive')
 Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/f'rank-{rank}.json').write_text(json.dumps(row))

def smi_peaks(text,world=2):
 if world not in (1,2):raise ValueError('unsupported measurement world')
 peaks={r:[] for r in range(world)}
 for line in text.splitlines():
  fields=line.split(',')
  if len(fields)!=8:continue
  try:rank=int(fields[1]);value=float(fields[3])
  except ValueError:continue
  if rank in peaks and math.isfinite(value) and value>=0:peaks[rank].append(value)
 values=[max(peaks[rank]) if peaks[rank] else None for rank in range(world)]
 # Sum of per-rank peaks, not necessarily a simultaneous device peak.
 return values,sum(values) if all(x is not None for x in values) else None

def aggregate_rank_results(ranks,world=2):
 if world not in (1,2):raise ValueError('unsupported measurement world')
 ranks=sorted(ranks,key=lambda x:x['rank'])
 if [x['rank'] for x in ranks]!=list(range(world)):raise ValueError('rank result inventory')
 if any(x.get('world_size',world)!=world for x in ranks):raise ValueError('rank world mismatch')
 if any(x['status']!='COMPLETE' or x['backend']!=ranks[0]['backend'] for x in ranks):raise ValueError('rank result contract mismatch')
 # Legacy reports may omit a field everywhere; partial presence is not agreement.
 for key in ('group','batch','frontier_profile','owner_backend','pre_dedup',
             'capacity_mode','global_capacity_records','global_state_ring_records',
             'archive_enabled','archive_state_bytes','generation_variant',
             'hash_first_generation','warmup_completed'):
  if any(key in x for x in ranks) and (any(key not in x for x in ranks) or any(x[key]!=ranks[0][key] for x in ranks)):
   raise ValueError('rank configuration mismatch: '+key)
 for result in ranks:
  counts=result.get('local_layer_sizes',result.get('layer_sizes'))
  if not isinstance(counts,list) or not counts or any(type(x) is not int or x<0 for x in counts):
   raise ValueError('invalid rank layer counts')
 search=[x['search_complete_seconds'] for x in ranks]
 durable=[x.get('durable_run_commit_seconds') for x in ranks]
 if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in search):raise ValueError('invalid search timing')
 if all(x is None for x in durable):durable_max=None
 elif any(x is None or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in durable):raise ValueError('incomplete durable timing')
 else:durable_max=max(durable)
 row=dict(status='COMPLETE',world_size=world,backend=ranks[0]['backend'],rank_results=ranks,search_complete_seconds=max(search),durable_run_commit_seconds=durable_max)
 if 'local_layer_sizes' in ranks[0]:
  if any('local_layer_sizes' not in x or len(x['local_layer_sizes'])!=len(ranks[0]['local_layer_sizes']) for x in ranks):raise ValueError('rank depth mismatch')
  row['layer_sizes']=[sum(values) for values in zip(*(x['local_layer_sizes'] for x in ranks))]
 else:
  if any(x['layer_sizes']!=ranks[0]['layer_sizes'] for x in ranks):raise ValueError('baseline rank count mismatch')
  row['layer_sizes']=ranks[0]['layer_sizes']
 return row

def run_group(command,out,label,env,timeout=7200):
 world=int(env.get('MGBFS_BENCH_WORLD_SIZE','2'))
 if world not in (1,2):raise ValueError('unsupported measurement world')
 row=dict(label=label,command=command,status='INCOMPLETE');rank_out=out/(label+'-ranks');rank_out.mkdir()
 command=[x.replace('{RANK_OUT}',str(rank_out)) for x in command]
 with (out/(label+'.log')).open('w') as log,(out/(label+'-smi.csv')).open('w') as smi,(out/(label+'.log')).open('rb') as progress:
  relay=ProgressRelay(progress,sys.stdout)
  sampler=subprocess.Popen(['stdbuf','-oL','nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,clocks.sm,power.draw','--format=csv,noheader,nounits','-lms','50'],stdout=smi,stderr=subprocess.STDOUT)
  try:
   process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);started=time.monotonic()
   while process.poll() is None:
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
     relay.poll()
     print(f'RUNNING {label}: {time.monotonic()-started:.0f}s',flush=True)
     if time.monotonic()-started>timeout:process.kill();process.wait();row['status']='TIMEOUT';break
   row['exit_code']=process.returncode
   relay.poll(final=True)
  finally:sampler.terminate();sampler.wait()
 if row['exit_code']==0:
  ranks=[json.loads(x.read_text()) for x in rank_out.glob('rank-*.json')]
  row.update(aggregate_rank_results(ranks,world=world))
 else:row['status']='FAILED' if row['status']=='INCOMPLETE' else row['status']
 row['smi_peak_mib_per_rank'],row['smi_peak_mib_total']=smi_peaks((out/(label+'-smi.csv')).read_text(),world=world);row['smi_memory_complete']=row['smi_peak_mib_total'] is not None;(out/(label+'.json')).write_text(json.dumps(row,indent=2));print(label,row['status'],row.get('search_complete_seconds'),flush=True);return row

def stats(rows):
 if not rows:raise ValueError('EMPTY_MEASUREMENTS')
 world=len(rows[0]['smi_peak_mib_per_rank'])
 if world not in (1,2) or any(len(x['smi_peak_mib_per_rank'])!=world for x in rows):raise ValueError('MEMORY_WORLD_MISMATCH')
 if any(x.get('smi_peak_mib_total') is None or any(v is None for v in x['smi_peak_mib_per_rank']) for x in rows):raise ValueError('INCOMPLETE_MEMORY_SAMPLES')
 values=[x['search_complete_seconds'] for x in rows];median=statistics.median(values)
 return dict(median_seconds=median,mad_seconds=statistics.median(abs(x-median) for x in values),samples_seconds=values,repeats=len(rows),peak_mib_per_rank=[max(x['smi_peak_mib_per_rank'][r] for x in rows) for r in range(world)],peak_mib_total=max(x['smi_peak_mib_total'] for x in rows))

def suite(native,source,out,env):
 world=int(env.get('MGBFS_BENCH_WORLD_SIZE','2'))
 if world not in (1,2):raise ValueError('unsupported measurement world')
 capacity_mode=env.get('MGBFS_CAPACITY_MODE','max_per_rank')
 if capacity_mode not in ('equal_global','max_per_rank'):raise ValueError('BENCH_CAPACITY_MODE')
 # Fixed physical device inventory also matches the nvidia-smi index sampler.
 env=dict(env,CUDA_VISIBLE_DEVICES='0' if world==1 else '0,1',MGBFS_CAPACITY_MODE=capacity_mode)
 out.mkdir(parents=True,exist_ok=True);report=dict(schema=1,status='INCOMPLETE',world_size=world,scope=f'physical {world}xT4, same S_n matrix states, native mandatory archive, CayleyPy no archive',rows=[],comparisons=[],disk_events=[])
 archive_dir=Path(env.get('MGBFS_BENCH_ARCHIVE_DIR','/tmp'))
 report['archive_directory']=str(archive_dir)
 report['capacity_mode']=capacity_mode
 def save():(out/'summary.json').write_text(json.dumps(report,indent=2))
 def disk_event(label,stage):
  event=dict(label=label,stage=stage)
  for name,path in [('tmp',Path('/tmp')),('output',out),('archive',archive_dir)]:
   try:
    usage=shutil.disk_usage(path)
    event[name]=dict(total_bytes=usage.total,used_bytes=usage.used,free_bytes=usage.free)
   except OSError as error:event[name]=dict(error=str(error))
  report['disk_events'].append(event);save()
  print('MGBFS_BENCH_DISK '+json.dumps(event),flush=True)
 def run(backend,n,batch,phase,rep=0,selection=None):
  label=f's{n}-{backend}-b{batch}-{phase}-{rep}'
  if selection:label+='-'+selection[0]
  if backend=='native':
   bootstrap=archive_dir/(label+'-bootstrap');prefix=str(archive_dir/(label+'-archive'));rows=min(batch,16384);slots=(math_factorial(n)+rows-1)//rows+64;cfg=dict(env,MGBFS_RANK_MAP='0' if world==1 else env.get('MGBFS_RANK_MAP','0,1'),MGBFS_BENCH_WARMUP='1',MGBFS_BENCH_CAPACITY=str(math_factorial(n)),MGBFS_FUTURE_CAPACITY=str(math_factorial(n)),MGBFS_ARCHIVE_ROWS=str(rows),MGBFS_ARCHIVE_SLOTS=str(slots));command=['torchrun','--standalone',f'--nproc-per-node={world}','--no-python',str(native),f's{n}',str(batch),str(bootstrap),prefix,'{RANK_OUT}']
  else:cfg=env;command=['torchrun','--standalone',f'--nproc-per-node={world}',str(Path(__file__).resolve()),'baseline-worker',str(n),str(batch),'{RANK_OUT}']
  if selection:cfg=dict(cfg,**selection[1])
  disk_event(label,'before')
  try:
   row=run_group(command,out,label,cfg,timeout=int(env.get('MGBFS_RUN_TIMEOUT','7200')))
   if backend=='native' and row['status']=='COMPLETE' and env.get('MGBFS_VERIFY_ARCHIVE')=='1':
    checks=[]
    for rank in range(world):
     result=subprocess.run([str(native.parent.parent/'mgbfs'),'verify',f'{prefix}-rank-{rank}.mgbfsar1'],capture_output=True,text=True)
     (out/f'{label}-verify-{rank}.log').write_text(result.stdout+result.stderr)
     if result.returncode!=0:raise ValueError('NATIVE_ARCHIVE_VERIFY_FAILED')
     checks.append(json.loads(result.stdout))
    row['archive_verification']=checks
  finally:
   if backend=='native':
    for rank in range(world):Path(f'{prefix}-rank-{rank}.mgbfsar1').unlink(missing_ok=True)
    bootstrap.unlink(missing_ok=True)
   disk_event(label,'after_cleanup')
  if backend=='native':
   # Requested budgets, not measured allocations. Runtime rank_results remain
   # authoritative for admitted capacities and actual device allocation planes.
   multiplier=1 if capacity_mode=='equal_global' else world
   row['requested_global_capacity_records']=int(cfg['MGBFS_BENCH_CAPACITY'])*multiplier
   row['requested_global_state_ring_records']=int(cfg['MGBFS_FUTURE_CAPACITY'])*multiplier
  row.update(phase=phase,repetition=rep,config_backend=backend,batch=batch);report['rows'].append(row);save()
  print('MGBFS_BENCH_ROW '+json.dumps(dict(label=label,row=row)),flush=True)
  return row
 try:
  if env.get('MGBFS_DIAGNOSTIC')=='1':
   env=dict(env,MGBFS_TRACE_DEPTHS='1',MGBFS_RUN_TIMEOUT='600')
   row=run('native',8,1024,'diagnostic')
   report['status']=row['status']
   return report
  if env.get('MGBFS_PROFILE_SWEEP')=='1':
   n=int(env.get('MGBFS_PROFILE_SWEEP_N','10'));expected=None;trials=[];variants=[]
   # This panel declares n! records (global or per rank) through a u32 reference ABI.
   # Larger streaming-capacity experiments use their separate launcher.
   if not 2<=n<=12:raise ValueError('PROFILE_GROUP_CAPACITY')
   archive_codec=env.get('MGBFS_PROFILE_ARCHIVE_CODEC','matrix_u8')
   if archive_codec not in ('matrix_u8','permutation_u8'):raise ValueError('PROFILE_ARCHIVE_CODEC')
   for profile,generation in [('DENSE','SCALAR'),('HASH_FIRST','SCALAR'),('HASH_FIRST','INT_MMA_SM75')]:
    for owner in ['CUB_SORT_MERGE','BMMA_BUCKET']:
     for pre in ['OFF','ON']:
      key=f'{profile}-{generation}-{owner}-{pre}'
      variants.append((key,dict(MGBFS_PROFILE=profile,MGBFS_HASH_FIRST_GENERATION=generation,MGBFS_OWNER_BACKEND=owner,MGBFS_PRE_DEDUP=pre,MGBFS_STATE_CODEC='matrix_u8',MGBFS_ARCHIVE_CODEC=archive_codec,MGBFS_BENCH_SKIP_ARCHIVE='0',MGBFS_ARCHIVE_STREAM='0')))
   for batch in [65536,262144,1048576]:
    row=run('cayleypy',n,batch,'calibrate')
    if row['status']=='COMPLETE':
     expected=expected or row['layer_sizes']
     if row['layer_sizes']!=expected or sum(expected)!=math_factorial(n):raise ValueError('BASELINE_LAYER_MISMATCH')
     trials.append((row['search_complete_seconds'],batch))
   if not trials:raise ValueError('NO_BASELINE_CONFIG')
   baseline_batch=min(trials)[1];measured={key:[] for key,_ in variants};baseline=[]
   for rep in range(5):
    # Rotate the complete run order, retaining every sample, not the best run.
    jobs=[None]+variants;shift=rep%len(jobs);jobs=jobs[shift:]+jobs[:shift]
    for selection in jobs:
     row=run('native' if selection else 'cayleypy',n,65536 if selection else baseline_batch,'measure',rep,selection)
     if row['status']!='COMPLETE' or row['layer_sizes']!=expected:raise ValueError('PROFILE_RUN_NOT_COMPLETE')
     (measured[selection[0]] if selection else baseline).append(row)
   for key,selection in variants:
    report['comparisons'].append(dict(group=f's{n}',unique_states=math_factorial(n),config=selection,native_batch=65536,baseline_batch=baseline_batch,native=stats(measured[key]),cayleypy=stats(baseline),layer_sizes=expected))
   report['tuning_scope']='native fixed batch 65536 profile panel; baseline batch sweep; not final tuned Pareto acceptance'
   report['status']='COMPLETE';return report
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
