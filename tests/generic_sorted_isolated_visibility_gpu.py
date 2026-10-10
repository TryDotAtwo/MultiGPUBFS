"""Two actual GPUs with independent files and network rank control on one host."""
import os,json,subprocess,time,socket,hashlib,sys
from pathlib import Path
from multigpubfs import GraphDefinition
r=Path(os.environ.get('MGBFS_NETWORK_GATE_ROOT','/root/universal/sorted-network'));r.mkdir(exist_ok=True);checks=[]
worker=r/'worker.py';worker.write_text("from multigpubfs import GraphDefinition,run_graph\nfrom pathlib import Path\nimport sys,os,json\ng=GraphDefinition.from_dict(json.loads(Path(sys.argv[1]).read_text()))\nv=run_graph(g,sys.argv[2],devices=[int(os.environ['LOCAL_RANK'])],capacity=None if sys.argv[3]=='auto' else int(sys.argv[3]),shards=None if sys.argv[4]=='auto' else int(sys.argv[4]),autotune=sys.argv[4]=='auto',transport='auto' if sys.argv[4]=='auto' else 'parent',max_seconds=30)\nprint(json.dumps(v))\n")
def case(name,g,capacity='auto',shards='4',mismatch=False,history_mismatch=False):
 folder=r/name;folder.mkdir();jobs=[];logs=[]
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 try:
  for rank in (0,1):
   local=folder/f'local-{rank}';local.mkdir();definition=local/'definition.json';definition.write_text((GraphDefinition.permutation([[1,0]],[0,1]) if mismatch and rank==1 else g).to_json());env=dict(os.environ,RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK='0',CUDA_VISIBLE_DEVICES=str(rank),MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port-1),MGBFS_CONTROL_PORT=str(port),MGBFS_RUN_ID=name,MGBFS_CONTROL_TOKEN='fixture-token')
   if history_mismatch:env['MGBFS_GENERIC_HISTORY']='sorted' if rank==0 else 'hash'
   log=(folder/f'rank-{rank}.log').open('w');logs.append(log);jobs.append(subprocess.Popen([sys.executable,str(worker),str(definition),str(local/'result'),str(capacity),str(shards)],env=env,stdout=log,stderr=subprocess.STDOUT))
  start=time.monotonic()
  while any(p.poll() is None for p in jobs):
   if time.monotonic()-start>100:raise RuntimeError('NETWORK_GATE_OBSERVATION_TIMEOUT_NO_RESTART '+name)
   time.sleep(.05)
  if mismatch or history_mismatch:
   assert all(p.returncode!=0 for p in jobs);assert 'MISMATCH' in (folder/'rank-0.log').read_text();checks.append({'case':name,'parameter_mismatch_rejected_all_ranks':True});return
  assert all(p.returncode==0 for p in jobs),[p.returncode for p in jobs]
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for log in logs:log.close()
 expected=g.exact_layers(50000);reports=[]
 for rank in (0,1):
  out=folder/f'local-{rank}/result';v=json.loads((out/'report.json').read_text());raw=(out/'states.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==v['states_sha256'];states=json.loads(raw);assert v['layer_sizes']==list(map(len,expected[:len(v['layer_sizes'])]));depth=len(v['layer_sizes']);assert sorted(states['current'])==expected[depth-1];assert sorted(states['previous_small'])==expected[depth-2];reports.append(v)
 assert reports[0]==reports[1]
 if g.inverse_closed():assert reports[0]['plan']['history_layers']==3
 else:assert reports[0]['plan']['history_layers']==1
 if capacity=='auto':assert reports[0]['status']=='COMPLETE'
 else:assert reports[0]['status']=='INCOMPLETE'
 if shards=='auto':
  profile=reports[0]['autotune'];pilots=profile['pilots'];assert profile['common_depth']>=4
  forced=os.environ.get('MGBFS_GENERIC_HISTORY')
  if forced=='sorted':assert len(pilots)==6 and all(p['history_algorithm']=='SORTED_RUNS' for p in pilots)
  elif forced is None:
   assert {p['history_algorithm'] for p in pilots}=={'HASH','SORTED_RUNS'}
   assert {p['owner_lanes'] for p in pilots if p['history_algorithm']=='SORTED_RUNS'}=={1,2,4,8}
  assert reports[0]['plan']['history_algorithm']==profile['history_algorithm']
  if profile['history_algorithm']=='SORTED_RUNS':assert reports[0]['plan']['owner_lanes']==profile['owner_lanes']
 checks.append({'case':name,'states':sum(reports[0]['layer_sizes']),'status':reports[0]['status'],'all_layer_counts_and_terminal_states_exact':True,'collective_autotune':shards=='auto'})

g=GraphDefinition.permutation([[1,2,3,4,5,0],[5,0,1,2,3,4],[1,0,2,3,4,5]],list(range(6)))
case('isolated-visible-sorted',g,shards='8')
reports=[json.loads((r/'isolated-visible-sorted'/f'local-{i}'/'result/report.json').read_text()) for i in range(2)]
assert all(v['devices']==[0,0] and v['plan']['history_algorithm']=='SORTED_RUNS' for v in reports)
(r/'verification.json').write_text(json.dumps({'status':'VERIFIED_LOCAL_GPU_ONLY_SORTED_NETWORK','checks':checks,'scope':'Each worker exposes only its own physical RTX3060 as local device0, separate directories, actual TCP/NCCL; not physical multi-host hardware.'},indent=2))
