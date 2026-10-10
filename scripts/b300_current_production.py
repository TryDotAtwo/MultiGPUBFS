"""Prebuilt current run_graph production sweep. No compiler or installer at runtime."""
import argparse,hashlib,json,os,random,signal,sys,time,queue,threading,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.native_distribution import native_runtime
from bfs_tail_archive import atomic_json
from b300_production import inventory

def grid(n_min,n_max):
 if not 2<=n_min<=n_max<=128:raise ValueError('GRID_BOUNDS')
 return [(n,r) for n in range(n_min,n_max+1) for r in range(1,n+1)]

def definition(n,r,seed=0):
 labels=list(range(n-r+1))
 if seed:random.Random(seed).shuffle(labels)
 start=list(range(n-r+1))+[n-r]*(r-1)
 return GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],[labels[v] for v in start]),{v:k for k,v in enumerate(labels)}

def paired_status(reports):
 if len(reports)!=2:return 'PARTIAL_ONE_SEED'
 if all(v['status']=='COMPLETE' for v in reports):return 'COMPLETE'
 return 'INCOMPLETE'

def capacity_stop(report):
 # 0x100 marks owner errors; low bits 1/2/4 are table/arena/future capacity.
 # Other native errors, CUDA failures, time limits and cancellations cannot prune.
 return report.get('status')=='INCOMPLETE' and report.get('reason') in ('RESOURCE_257','RESOURCE_258','RESOURCE_260')

def payload(path,inverse):
 v=json.loads((Path(path)/'states.json').read_text())
 return {key:sorted(tuple(inverse[x] for x in row) for row in v[key]) for key in ('current','previous_small')}

def verify_oracle(report,g,path,limit):
 cpu=g.exact_layers(limit)
 if report['status']!='COMPLETE' or report['layer_sizes']!=list(map(len,cpu)):raise RuntimeError('PRODUCTION_CPU_ORACLE_MISMATCH')
 v=json.loads((Path(path)/'states.json').read_text())
 if sorted(map(tuple,v['current']))!=sorted(map(tuple,cpu[-1])):raise RuntimeError('PRODUCTION_TERMINAL_ORACLE_MISMATCH')
 if len(cpu)>1 and len(cpu[-2])<1000 and sorted(map(tuple,v['previous_small']))!=sorted(map(tuple,cpu[-2])):raise RuntimeError('PRODUCTION_PREVIOUS_ORACLE_MISMATCH')

class Publisher:
 def __init__(self,root,repo,token):
  from huggingface_hub import HfApi
  self.root,self.repo,self.token=root,repo,token;self.api=HfApi(token=token);self.pending=queue.Queue(maxsize=1);self.error=None;self.receipt=None;self.seen=set();self.entries=[]
  self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
 def enqueue(self,ledger,final=False):
  if self.error:raise RuntimeError('HF_BACKGROUND_FAILED') from self.error
  while True:
   if self.error:raise RuntimeError('HF_BACKGROUND_FAILED') from self.error
   try:self.pending.put((json.loads(json.dumps(ledger)),final),timeout=1);break
   except queue.Full:pass
 def work(self):
  try:
   while True:
    ledger,final=self.pending.get()
    try:self.publish(ledger,final)
    finally:self.pending.task_done()
    if final:return
  except BaseException as e:self.error=e
 def publish(self,ledger,final):
  import tarfile
  from huggingface_hub import CommitOperationAdd,hf_hub_download
  new=[];members=[];runs=[path for record in ledger['cases'].values() for path in record.get('runs',[]) if path not in self.seen]
  if runs:
   directory=self.root/'cohorts';directory.mkdir(exist_ok=True);name=f'cohorts/chunk-{len(self.entries):05d}.tar.gz';path=self.root/name
   with tarfile.open(path,'w:gz',compresslevel=1) as archive:
    for run in runs:
     for leaf in ('report.json','states.json','launch.json'):
      source=self.root/run/leaf
      if source.exists():
       member=run+'/'+leaf;archive.add(source,arcname=member,recursive=False);members.append(dict(path=member,bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
   entry=dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),members=members)
   self.entries.append(entry);new.append((name,path));self.seen.update(runs)
  for name in ('startup.json','hardware.txt','topology.txt','runtime-identity.json'):
   source=self.root/name
   if source.exists():new.append((name,source))
  ledger_bytes=json.dumps(ledger,indent=2).encode();manifest_bytes=json.dumps({'cohorts':self.entries,'ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),'retention':'compact current up to1000 and previous below1000, two seeds'},indent=2).encode()
  prefix='runs/'+self.root.name
  operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(path)) for name,path in new]+[CommitOperationAdd(path_in_repo=prefix+'/sweep.json',path_or_fileobj=ledger_bytes),CommitOperationAdd(path_in_repo=prefix+'/manifest.json',path_or_fileobj=manifest_bytes)]
  info=self.api.create_commit(repo_id=self.repo,repo_type='dataset',operations=operations,commit_message='Current compact BFS '+('final' if final else 'progress'))
  checks=[dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in new]+[{'path':'sweep.json','sha256':hashlib.sha256(ledger_bytes).hexdigest()},{'path':'manifest.json','sha256':hashlib.sha256(manifest_bytes).hexdigest()}]
  for entry in checks:
   downloaded=hf_hub_download(self.repo,prefix+'/'+entry['path'],repo_type='dataset',revision=info.oid,token=self.token,cache_dir=str(self.root/'readback-cache'))
   if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('HF_READBACK_CHECKSUM')
  for entry in self.entries:
   if entry['path'] in dict(new):entry.update(readback_verified=True,verified_revision=info.oid)
  self.receipt=dict(repo=self.repo,prefix=prefix,revision=info.oid,all_checksums_verified=all(v.get('readback_verified') for v in self.entries),final=final,cohorts=len(self.entries),scope='Every immutable cohort read back at its publication revision; final ledger and manifest read back at final revision')
  atomic_json(self.root/'publication-receipt.json',self.receipt)
 def finish(self,ledger):
  self.enqueue(ledger,True);self.thread.join()
  if self.error:raise RuntimeError('HF_FINAL_FAILED') from self.error
  if not self.receipt or not self.receipt['final']:raise RuntimeError('HF_FINAL_RECEIPT_MISSING')

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--repo-id',required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--token-file',type=Path,required=True)
 p.add_argument('--expected-gpus',type=int,choices=(1,2,4,8),default=4);p.add_argument('--n-min',type=int,default=2);p.add_argument('--n-max',type=int,default=128);p.add_argument('--preflight-only',action='store_true');p.add_argument('--allow-development-hardware',action='store_true')
 a=p.parse_args(argv)
 if not __import__('math').isfinite(a.deadline_unix) or a.deadline_unix-time.time()<120: p.error('finite deadline with at least120 seconds required')
 pairs=grid(a.n_min,a.n_max);a.root.mkdir(parents=True,exist_ok=True)
 lock=(a.root/'production.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (a.root/'sweep.json').exists():raise RuntimeError('EXISTING_SWEEP_NO_IMPLICIT_RESTART')
 devices=list(range(a.expected_gpus));native,env=native_runtime();os.environ['MGBFS_PROFILE_CACHE']=str(a.root/'profiles');env['MGBFS_PROFILE_CACHE']=str(a.root/'profiles')
 text=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap','--format=csv,noheader,nounits'],text=True)
 if a.allow_development_hardware:
  if len(text.strip().splitlines())!=a.expected_gpus:raise RuntimeError('DEVELOPMENT_GPU_COUNT')
 else:inventory(text,a.expected_gpus)
 (a.root/'hardware.txt').write_text(text);(a.root/'topology.txt').write_text(__import__('subprocess').check_output(['nvidia-smi','topo','-m'],text=True))
 from multigpubfs.native_distribution import runtime_identity
 atomic_json(a.root/'runtime-identity.json',runtime_identity(native,env))
 cancel=threading.Event()
 def stop(signum,frame):
  cancel.set()
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    cmd=(proc/'cmdline').read_bytes();ppid=int((proc/'stat').read_text().split(') ',1)[1].split()[1])
    if ppid==os.getpid() and str(native).encode() in cmd:os.kill(int(proc.name),signal.SIGTERM)
   except (OSError,ValueError):pass
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 publisher=Publisher(a.root,a.repo_id,a.token_file.read_text().strip())
 startup=[]
 for mode in ('cuda','gemm'):
  if cancel.is_set():raise RuntimeError('STARTUP_CANCELLED')
  g,_=definition(4,1);out=a.root/('startup-'+mode)
  v=run_graph(g,out,devices=devices,capacity=10000,shards=4,autotune=False,backend='generic',max_seconds=60,_batch=4096,_native_env=dict(env,MGBFS_GENERIC_GENERATOR=mode));verify_oracle(v,g,out,24);startup.append(v)
 g,_=definition(9,1);out=a.root/'startup-auto'
 v=run_graph(g,out,devices=devices,capacity=200000,max_seconds=min(180,max(1,int(a.deadline_unix-time.time()-60))),backend='auto',_native_env=env);verify_oracle(v,g,out,362880);startup.append(v)
 atomic_json(a.root/'startup.json',dict(status='VERIFIED_DEVELOPMENT_STARTUP' if a.allow_development_hardware else 'VERIFIED_B300_STARTUP',runs=startup))
 ledger=dict(configuration=dict(grid=pairs,devices=devices,backend='auto',retention='last_current_1000_previous_under1000',seeds=[0,13],pruning_policy='fixed-r-capacity-heuristic'),cases={},pending=[list(v) for v in pairs],status='RUNNING')
 atomic_json(a.root/'sweep.json',ledger);publisher.enqueue(ledger)
 blocked={};last_publication=time.monotonic()
 if not a.preflight_only:
  for n,rr in pairs:
   key=f'n{n}-r{rr}'
   if rr in blocked and n>blocked[rr]:
    ledger['cases'][key]=dict(status='RESOURCE_PRUNED_HEURISTIC',attempted=False,n=n,r=rr,pruned_by={'n':blocked[rr],'r':rr});continue
   if cancel.is_set() or time.time()>=a.deadline_unix-30:break
   runs=[];reports=[];inverses=[]
   for seed in (0,13):
    if cancel.is_set() or time.time()>=a.deadline_unix-30:break
    g,inverse=definition(n,rr,seed);path=f'{key}/seed-{seed}';out=a.root/path
    try:report=run_graph(g,out,devices=devices,max_seconds=max(1,int(a.deadline_unix-time.time()-25)),backend='auto',_native_env=env)
    except RuntimeError:
     if cancel.is_set():break
     raise
    runs.append(path);reports.append(report);inverses.append(inverse)
   if not reports:break
   verified=False
   if len(reports)==2:
    depth=min(len(v['layer_sizes']) for v in reports)
    if reports[0]['layer_sizes'][:depth]!=reports[1]['layer_sizes'][:depth]:raise RuntimeError('TWO_SEED_LAYER_MISMATCH')
    if all(v['status']=='COMPLETE' for v in reports):
     if reports[0]['layer_sizes']!=reports[1]['layer_sizes'] or payload(a.root/runs[0],inverses[0])!=payload(a.root/runs[1],inverses[1]):raise RuntimeError('TWO_SEED_TERMINAL_MISMATCH')
    verified=True
   record=dict(n=n,r=rr,attempted=True,status=paired_status(reports),runs=runs,two_seed_prefix_verified=verified,reasons=[v['reason'] for v in reports],layer_sizes=reports[0]['layer_sizes'],bfs_seconds=[v['bfs_seconds'] for v in reports])
   ledger['cases'][key]=record
   if len(reports)==2 and all(capacity_stop(v) for v in reports):blocked[rr]=n
   ledger['pending']=[list(v) for v in pairs if f'n{v[0]}-r{v[1]}' not in ledger['cases']];atomic_json(a.root/'sweep.json',ledger)
   if time.monotonic()-last_publication>60:publisher.enqueue(ledger);last_publication=time.monotonic()
   if publisher.error:raise RuntimeError('HF_BACKGROUND_FAILED') from publisher.error
 ledger['pending']=[list(v) for v in pairs if f'n{v[0]}-r{v[1]}' not in ledger['cases']]
 ledger['status']='PREFLIGHT_ONLY' if a.preflight_only else 'CLASSIFIED_GRID' if not ledger['pending'] else 'STOPPED_WITH_PENDING'
 ledger['all_graphs_complete']=not a.preflight_only and not ledger['pending'] and all(v['status']=='COMPLETE' for v in ledger['cases'].values())
 atomic_json(a.root/'sweep.json',ledger);publisher.finish(ledger)
 atomic_json(a.root/'automatic-report.json',dict(status='VERIFIED_PUBLICATION',sweep_status=ledger['status'],pending=len(ledger['pending']),attempted=sum(v['attempted'] for v in ledger['cases'].values()),pruned=sum(not v['attempted'] for v in ledger['cases'].values()),all_graphs_complete=ledger['all_graphs_complete'],publication=publisher.receipt))
 from huggingface_hub import CommitOperationAdd,hf_hub_download
 files=['automatic-report.json','publication-receipt.json'];prefix='runs/'+a.root.name
 info=publisher.api.create_commit(repo_id=a.repo_id,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(a.root/name)) for name in files],commit_message='Verified final compact BFS report')
 for name in files:
  path=hf_hub_download(a.repo_id,prefix+'/'+name,repo_type='dataset',revision=info.oid,token=publisher.token,cache_dir=str(a.root/'readback-cache'))
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=hashlib.sha256((a.root/name).read_bytes()).hexdigest():raise RuntimeError('HF_REPORT_READBACK_MISMATCH')
 atomic_json(a.root/'final-evidence-receipt.json',dict(repo=a.repo_id,revision=info.oid,prefix=prefix,all_checksums_verified=True,final_report_verified=True,cohort_revision=publisher.receipt['revision']))
if __name__=='__main__':main()
