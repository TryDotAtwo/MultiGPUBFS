import json,subprocess,os,time,tempfile,signal
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path('/root/universal');native=r/'src/target/release/mgbfs';os.environ['MGBFS_EXECUTABLE']=str(native)
g=GraphDefinition.matrix(2,1,[([1,1,0,1],0)],[0,1])
with tempfile.TemporaryDirectory(prefix='stop-gate-',dir=r) as temporary:
 root=Path(temporary);report=run_graph(g,root/'deadline',capacity=1048576,max_seconds=1)
 assert report['status']=='INCOMPLETE' and report['reason']=='DEADLINE',report
 states=json.loads((root/'deadline/states.json').read_text());depth=len(report['layer_sizes'])-1
 assert states['current']==[[depth,1]];assert states['previous_small']==[[depth-1,1]];assert all(n==1 for n in report['layer_sizes'])
 definition=root/'graph.json';definition.write_text(g.to_json());output=root/'cancel';log=root/'cancel.log'
 with log.open('w') as stream:
  process=subprocess.Popen([str(native),'graph',str(definition),str(output),'--capacity','1048576','--seconds','60'],stdout=stream,stderr=subprocess.STDOUT)
  started=time.monotonic()
  while 'MGBFS_GRAPH_READY' not in log.read_text():
   assert process.poll() is None,log.read_text()
   if time.monotonic()-started>15:process.terminate();raise AssertionError('startup readiness timeout')
   time.sleep(.01)
  time.sleep(.05);process.send_signal(signal.SIGTERM);assert process.wait(timeout=15)==0,log.read_text()
  cancelled=json.loads((output/'report.json').read_text());assert cancelled['status']=='INCOMPLETE' and cancelled['reason']=='CANCELLED'
  states=json.loads((output/'states.json').read_text());depth=len(cancelled['layer_sizes'])-1;assert states['current']==[[depth,1]]
print(json.dumps({'status':'VERIFIED_GENERIC_DEADLINE_AND_SIGTERM','deadline_seconds':report['bfs_seconds'],'deadline_layers':len(report['layer_sizes']),'cancel_layers':len(cancelled['layer_sizes']),'scope':'single-device CLI with compact state readout; no resumed damaged future'}))
