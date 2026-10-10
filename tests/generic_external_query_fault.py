"""Two real HTTP peers; one cold native query hangs, no GPU work."""
import os,sys,socket,time,subprocess,json
from pathlib import Path
r=Path(os.environ['MGBFS_QUERY_FAULT_GATE_ROOT']);r.mkdir();native=r/'native';pidfile=r/'query.pid'
native.write_text('#!/usr/bin/env python3\nimport os,time,json\nfrom pathlib import Path\nif os.environ.get("RANK")=="1":\n Path('+repr(str(pidfile))+').write_text(str(os.getpid()))\n time.sleep(120)\nelse:print(json.dumps({"inventory":[{"device":0,"free_bytes":1073741824,"total_bytes":2147483648}]}))\n');native.chmod(0o755)
worker=r/'worker.py';worker.write_text("from multigpubfs import GraphDefinition,run_graph\nfrom unittest.mock import patch\nimport sys\ng=GraphDefinition.permutation([[1,0]],[0,1])\nwith patch('multigpubfs.distributed_launch._dependency_identity',return_value={'cuda_library_sha256':'0'*64}):run_graph(g,sys.argv[1],devices=[0],shards=1,autotune=False,max_seconds=3,backend='generic',executable=sys.argv[2])\n")
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
jobs=[];logs=[];start=time.monotonic()
try:
 for rank in (0,1):
  env=dict(os.environ,RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK='0',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port-1),MGBFS_CONTROL_PORT=str(port),MGBFS_RUN_ID='query-fault',MGBFS_CONTROL_TOKEN='fixture-token',MGBFS_NATIVE_QUERY_SECONDS='0.3')
  f=(r/f'rank-{rank}.log').open('w');logs.append(f);jobs.append(subprocess.Popen([sys.executable,str(worker),str(r/f'result-{rank}'),str(native)],env=env,stdout=f,stderr=subprocess.STDOUT))
 for p in jobs:p.wait(timeout=15)
 assert all(p.returncode!=0 for p in jobs)
 for f in logs:f.flush()
 for rank in (0,1):assert 'NATIVE_STARTUP_QUERY_TIMEOUT' in (r/f'rank-{rank}.log').read_text()
 assert not (Path('/proc')/pidfile.read_text()).exists()
 assert not list(r.glob('result-*/rank-*')) and not list(r.glob('result-*/report.json'))
 receipt={'status':'VERIFIED_COLLECTIVE_NATIVE_QUERY_TIMEOUT','seconds':time.monotonic()-start,'query_child_absent':True,'no_native_rank_launched':True,'scope':'two HTTP peers on one host; injected native query, no GPU or physical multi-node acceptance'}
 (r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
finally:
 for p in jobs:
  if p.poll() is None:p.terminate()
 for p in jobs:
  if p.poll() is None:p.wait(timeout=5)
 for f in logs:f.close()
