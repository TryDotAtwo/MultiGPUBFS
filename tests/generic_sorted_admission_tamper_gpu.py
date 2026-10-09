import os,json,subprocess
from pathlib import Path
from multigpubfs import GraphDefinition
r=Path(os.environ.get('MGBFS_SORTED_TAMPER_GATE_ROOT','/tmp/mgbfs-sorted-admission-tamper'));r.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,CUDA_MODULE_LOADING='EAGER',MGBFS_GENERIC_HISTORY='sorted');native=env['MGBFS_EXECUTABLE']
g=GraphDefinition.permutation([[1,2,3,0],[1,0,2,3]],list(range(4)));definition=r/'graph.json';definition.write_text(g.to_json())
probe=subprocess.run([native,'graph-info',str(definition),'0','1','24','2'],env=env,capture_output=True,text=True,check=True);launch=json.loads(probe.stdout)
for p in [launch['plan']]+launch['rank_plans']:p['sorted_owner_bytes']+=256;p['device_bytes']+=256
configuration=r/'launch.json';configuration.write_text(json.dumps(launch))
worker=subprocess.run([native,'graph-rank',str(definition),str(configuration),'0',str(r/'id'),str(r/'result'),'10'],env=env,capture_output=True,text=True,timeout=45)
(r/'worker.log').write_text(worker.stderr)
assert worker.returncode!=0 and 'SORTED_OWNER_SERIALIZED_SHAPE_MISMATCH' in worker.stderr
assert not (r/'result/report.json').exists()
receipt={'status':'VERIFIED_SERIALIZED_SHAPE_TAMPER_REJECTED_BEFORE_STATE_ALLOCATION','error':'SORTED_OWNER_SERIALIZED_SHAPE_MISMATCH','returncode':worker.returncode};(r/'verification.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
