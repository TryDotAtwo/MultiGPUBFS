"""Actual two-GPU unequal owner arenas and measured free-memory admission."""
import ctypes as C,json,os,subprocess,time,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path(os.environ.get('MGBFS_HETEROGENEOUS_GATE_ROOT','/root/universal/heterogeneous-gate'));r.mkdir(exist_ok=True);native=Path('/root/universal/src/target/release/mgbfs');example=native.parent/'examples/generic_distributed_gate';env=dict(os.environ)
g=GraphDefinition.permutation([[1,2,3,4,5,0],[1,0,2,3,4,5]],list(range(6)));oracle=g.exact_layers(1000);checks=[]
for codec in (1,8):
 for bits in (0,64):
  out=r/f'weighted-{codec}-{bits}';out.mkdir();definition=out/'graph.json';definition.write_text(g.to_json());jobs=[];logs=[];cuts=[0,1431655766,4294967296]
  try:
   for rank,capacity in enumerate((384,768)):
    log=(out/f'rank-{rank}.log').open('w');logs.append(log);jobs.append(subprocess.Popen([str(example),str(rank),str(out),str(definition),str(bits),'4',str(capacity),str(codec),'32',json.dumps(cuts)],env=env,stdout=log,stderr=subprocess.STDOUT))
   start=time.monotonic()
   while any(p.poll() is None for p in jobs):
    if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('WEIGHTED_RANK_FAILURE')
    if time.monotonic()-start>90:raise RuntimeError('WEIGHTED_OBSERVATION_TIMEOUT_NO_RESTART')
    time.sleep(.02)
   assert all(p.returncode==0 for p in jobs)
  finally:
   for p in jobs:
    if p.poll() is None:p.terminate()
   for p in jobs:
    if p.poll() is None:
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:p.kill();p.wait()
   for log in logs:log.close()
  parts=[json.loads((out/f'rank-{i}.json').read_text()) for i in range(2)]
  # Forced zero hashes all belong to rank zero: 384 capacity deliberately
  # exhausts before 720; retain exact completed layers and previous frontier.
  if bits==0:
   assert all(v['fatal']!=0 for v in parts);depth=len(parts[0]['layers']);assert sorted(parts[0]['previous_small']+parts[1]['previous_small'])==oracle[depth-2]
  else:assert all(v['fatal']==0 for v in parts)
  actual=[sorted(parts[0]['layers'][i]+parts[1]['layers'][i]) for i in range(len(parts[0]['layers']))];assert actual==oracle[:len(actual)]
  checks.append({'codec':codec,'hash_bits':bits,'capacities':[384,768],'owner_cuts':cuts,'all_completed_layers_exact':True,'resource_snapshot_exact':bits==0,'root_rank':next(i for i,v in enumerate(parts) if v['layers'][0])})
# Public automatic launch uses cut-based root placement, including device swap.
for devices in ([0,1],[1,0]):
 report=run_graph(g,r/('public-'+''.join(map(str,devices))),devices=devices,shards=4,max_seconds=30,autotune=False)
 assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,oracle));assert len(report['rank_plans'])==2 and report['owner_cuts'][0]==0 and report['owner_cuts'][-1]==1<<32
 checks.append({'public_devices':devices,'weighted_public_launch_exact':True})
# Hold 4 GiB on physical GPU1 while child inventories actual free memory.
large=GraphDefinition.permutation([list(range(1,14))+[0],[1,0]+list(range(2,14))],list(range(14)));path=r/'admission-large.json';path.write_text(large.to_json());cuda=C.CDLL('libcudart.so.12');cuda.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t];cuda.cudaFree.argtypes=[C.c_void_p];assert cuda.cudaSetDevice(1)==0;reserve=C.c_void_p();assert cuda.cudaMalloc(C.byref(reserve),4<<30)==0
try:
 probe=subprocess.run([str(native),'graph-info',str(path),'0,1','4'],env=env,capture_output=True,text=True,check=True);admission=json.loads(probe.stdout);plans=admission['rank_plans'];assert plans[0]['capacity']>plans[1]['capacity'];assert plans[0]['batch']==plans[1]['batch'] and plans[0]['queue_capacity']==plans[1]['queue_capacity'];assert admission['owner_cuts'][1]>(1<<31)
 (r/'unequal-vram-admission.json').write_text(json.dumps(admission));checks.append({'held_gpu1_bytes':4<<30,'actual_free_vram_admission':True,'capacities':[p['capacity'] for p in plans]})
finally:assert cuda.cudaFree(reserve)==0
receipt={'status':'VERIFIED_WEIGHTED_UNEQUAL_CAPACITY_TWO_GPU','checks':checks,'scope':'two identical RTX3060 with unequal arenas and reserved VRAM; different GPU models, multihost and larger ranks unverified'};(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
