"""Public cold memory/disk and resolved-runtime provenance on real GPUs."""
import os,json,hashlib,subprocess,sys
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path(os.environ['MGBFS_COLD_IDENTITY_GATE_ROOT']);r.mkdir()
g=GraphDefinition.permutation([[1,0,2],[1,2,0],[2,0,1]],[0,1,2]);expected=g.exact_layers(100);checks=[]
for devices in ([0],[1,0]):
 out=r/('public-'+str(len(devices)));v=run_graph(g,out,devices=devices,shards=4,autotune=False,backend='generic',max_seconds=10)
 assert v['status']=='COMPLETE' and v['layer_sizes']==list(map(len,expected))
 raw=(out/'states.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==v['states_sha256'];states=json.loads(raw)
 assert sorted(states['current'])==expected[-1] and sorted(states['previous_small'])==expected[-2]
 identity=v['runtime_identity'];assert len(identity['native_sha256'])==64 and len(identity['python_runtime_sha256'])==64 and len(identity['cuda_library_sha256'])==64
 source=os.environ.get('MGBFS_EXPECT_PACKAGE_SOURCE')
 if source:assert identity['source_commit']==source
 assert v['host_memory'] and v['host_memory'][0]['disk'];assert all(x['free_bytes']>=x['estimated_required_bytes'] for x in v['host_memory'][0]['disk'])
 checks.append({'devices':devices,'states':sum(v['layer_sizes']),'terminal_states_exact':True,'runtime_identity':identity})
definition=r/'definition.json';definition.write_text(g.to_json());q=subprocess.run([sys.executable,'-m','multigpubfs',str(definition),str(r/'console'),'--no-autotune','--backend','generic','--seconds','10'],capture_output=True,text=True,timeout=60);(r/'console.log').write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr
report=json.loads(q.stdout);assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,expected)) and report['runtime_identity']['native_sha256']==checks[0]['runtime_identity']['native_sha256']
receipt={'status':'VERIFIED_PUBLIC_COLD_ADMISSION_AND_RUNTIME_IDENTITY','checks':checks,'console_verified':True,'scope':'one/two physical GPUs; compact output'}
(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
