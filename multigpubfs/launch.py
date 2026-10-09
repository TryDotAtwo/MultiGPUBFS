"""Native graph launch; mathematical CPU oracles are never runtime fallbacks."""
import hashlib,json,os,shutil,subprocess,tempfile
from pathlib import Path
from .graph_definition import GraphDefinition,from_cayleypy

def run_graph(graph,output,*,device=0,capacity=None,max_seconds=3600,executable=None):
 """Run the general exact single-device engine and validate its output receipt.

 Distributed SHARD_AB dispatch is being integrated; this entry point does not
 silently distribute or ignore a requested topology. No HF account is needed.
 """
 if not isinstance(graph,GraphDefinition):graph=from_cayleypy(graph)
 for name,value in [('device',device),('max_seconds',max_seconds)]:
  if type(value) is not int or value<(0 if name=='device' else 1):raise ValueError('INVALID_'+name.upper())
 if capacity is not None and (type(capacity) is not int or not 1<=capacity<=1<<28):raise ValueError('INVALID_CAPACITY')
 if os.environ.get('WORLD_SIZE','1')!='1':raise RuntimeError('GENERIC_DISTRIBUTED_INTEGRATION_NOT_READY')
 native=executable or os.environ.get('MGBFS_EXECUTABLE') or shutil.which('mgbfs')
 if not native:raise RuntimeError('MGBFS_EXECUTABLE_NOT_FOUND: install the native Linux CUDA runtime or set MGBFS_EXECUTABLE')
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='mgbfs-definition-',dir=output.parent) as temporary:
  definition=Path(temporary)/'graph.json';definition.write_text(graph.to_json(),encoding='utf-8')
  command=[str(native),'graph',str(definition),str(output),'--device',str(device),'--seconds',str(max_seconds)]
  if capacity is not None:command+=['--capacity',str(capacity)]
  process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  if process.returncode:raise RuntimeError('NATIVE_GRAPH_FAILED: '+process.stderr[-4000:])
 report=json.loads((output/'report.json').read_text(encoding='utf-8'))
 digest=graph.digest()
 if report['graph_digest']!=digest:raise RuntimeError('GRAPH_RECEIPT_IDENTITY_MISMATCH')
 if report['status'] not in ('COMPLETE','INCOMPLETE'):raise RuntimeError('GRAPH_RECEIPT_STATUS')
 if hashlib.sha256((output/'states.json').read_bytes()).hexdigest()!=report['states_sha256']:raise RuntimeError('GRAPH_RECEIPT_CHECKSUM')
 return report
