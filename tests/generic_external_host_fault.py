"""Actual two-process HTTP peer failure; injected host RAM, no GPU BFS."""
import os,sys,socket,time,subprocess,json
from pathlib import Path
r=Path(os.environ['MGBFS_HOST_FAULT_GATE_ROOT']);r.mkdir()
worker=r/'worker.py';worker.write_text("from multigpubfs import GraphDefinition,run_graph\nfrom unittest.mock import patch\nimport os,sys\ng=GraphDefinition.permutation([[1,0]],[0,1])\nif os.environ['RANK']=='1':\n with patch('multigpubfs.host_memory.inventory',return_value={'available_bytes':100,'sources':['injected']}):run_graph(g,sys.argv[1],devices=[0],shards=1,autotune=False,max_seconds=3)\nelse:run_graph(g,sys.argv[1],devices=[0],shards=1,autotune=False,max_seconds=3)\n")
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
jobs=[];logs=[];start=time.monotonic()
try:
 for rank in (0,1):
  env=dict(os.environ,RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK='0',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port-1),MGBFS_CONTROL_PORT=str(port),MGBFS_RUN_ID='host-fault',MGBFS_CONTROL_TOKEN='fixture-token')
  f=(r/f'rank-{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([sys.executable,str(worker),str(r/f'result-{rank}')],env=env,stdout=f,stderr=subprocess.STDOUT))
 for p in jobs:p.wait(timeout=15)
 assert all(p.returncode!=0 for p in jobs)
 assert time.monotonic()-start<15
 for f in logs:f.flush()
 for rank in (0,1):assert 'HOST_MEMORY_ADMISSION_FAILED' in (r/f'rank-{rank}.log').read_text()
 assert not list(r.glob('result-*/rank-*'))
 receipt={'status':'VERIFIED_COLLECTIVE_HOST_REFUSAL','seconds':time.monotonic()-start,'no_native_rank_launched':True,'scope':'two processes on one host; RAM fault injected before any GPU work'}
 (r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
finally:
 for p in jobs:
  if p.poll() is None:p.terminate()
 for p in jobs:
  if p.poll() is None:p.wait(timeout=5)
 for f in logs:f.close()
