from pathlib import Path
import json,os,sys,subprocess,time,signal
r=Path('/root/universal/unified-cancel-gate')
if len(sys.argv)>1:
 from multigpubfs import GraphDefinition,run_graph
 graph=GraphDefinition.matrix(2,1,[([1,1,0,1],0)],[0,1]);report=run_graph(graph,r,devices=[0,1],capacity=16384,max_seconds=30);assert report['status']=='INCOMPLETE' and report['reason']=='CANCELLED';states=json.loads((r/'states.json').read_text());depth=len(report['layer_sizes'])-1;assert states['current']==[[depth,1]] and states['previous_small']==[[depth-1,1]];(r/'verification.json').write_text(json.dumps({'status':'VERIFIED_UNIFIED_ONE_RANK_SIGTERM_GLOBAL_STOP','depth':depth,'last_two_layers_exact':True}));print('VERIFIED_ONE_RANK_SIGTERM',depth)
else:
 child=subprocess.Popen([sys.executable,__file__,'worker']);started=time.monotonic();target=None
 while time.monotonic()-started<30:
  log=r/'rank-1.log'
  if log.exists() and 'MGBFS_GRAPH_READY rank=1' in log.read_text():
   for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:argv=path.read_bytes().split(bytes([0]))
    except (OSError,ProcessLookupError):continue
    if b'graph-rank' in argv and str(r/'rank-1').encode() in argv:
     target=int(path.parent.name);os.kill(target,signal.SIGTERM);break
   if target is not None:break
  if child.poll() is not None:raise RuntimeError('LAUNCH_EXITED_BEFORE_STOP')
  time.sleep(.01)
 if target is None:raise RuntimeError('TARGET_NOT_FOUND_NO_SIGNAL_SENT')
 code=child.wait(timeout=30);assert code==0;receipt=json.loads((r/'verification.json').read_text());receipt['signalled_rank']=1;(r/'verification.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
