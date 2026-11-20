"""Skewed source queues keep exact BFS and retain a learned batch bound."""
import json, os, subprocess, tempfile, time
from pathlib import Path
from multigpubfs import GraphDefinition

native=os.environ['MGBFS_EXECUTABLE']
root=Path(tempfile.mkdtemp(prefix='mgbfs-retry-feedback-'))
def swap(i):
 p=list(range(5));p[i],p[i+1]=p[i+1],p[i];return p
g=GraphDefinition.permutation([swap(i) for i in range(4)],list(range(5)))
expected=g.exact_layers(1000)
for world in (1,2):
 for bitmap in (False,True):
  name=f"{'bitmap' if bitmap else 'segmented'}-{world}";case=root/name;case.mkdir()
  definition=case/'definition.json';definition.write_text(g.to_json())
  env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP=str(int(bitmap)),MGBFS_SEGMENT_ROWS='3',MGBFS_SEGMENT_TEST_QUEUE_CAP='4',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
  devices=','.join(map(str,range(world)))
  query=subprocess.run([native,'graph-info',str(definition),devices,'1','1024','4'],env=env,text=True,capture_output=True,timeout=60,check=True)
  plan=json.loads(query.stdout);plan['owner_cuts']=None
  config=case/'launch.json';config.write_text(json.dumps(plan));jobs=[];logs=[]
  try:
   for rank in range(world):
    f=(case/f'rank-{rank}.log').open('w');logs.append(f)
    jobs.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=f,stderr=subprocess.STDOUT))
   until=time.monotonic()+90
   while any(p.poll() is None for p in jobs) and time.monotonic()<until:
    if any(p.poll() not in (None,0) for p in jobs):break
    time.sleep(.03)
   for f in logs:f.flush()
   assert all(p.poll()==0 for p in jobs),[p.poll() for p in jobs]+[(case/f'rank-{i}.log').read_text()[-3000:] for i in range(world)]
   reports=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(world)]
   snapshots=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(world)]
   assert all(x['status']=='COMPLETE' and x['layer_sizes']==list(map(len,expected)) for x in reports),reports
   assert sorted(sum((s['current'] for s in snapshots),[]))==expected[-1],snapshots
   retries=[x['source_retries'] for x in reports]
   print(json.dumps({'mode':name,'source_retries':retries,'layers':reports[0]['layer_sizes'],'bfs_seconds':reports[0]['bfs_seconds']}),flush=True)
   if os.environ.get('MGBFS_RETRY_BASELINE')!='1':assert max(retries)<=2,reports
  finally:
   for p in jobs:
    if p.poll() is None:p.terminate()
   for p in jobs:
    if p.poll() is None:
     try:p.wait(timeout=8)
     except subprocess.TimeoutExpired:p.kill();p.wait()
   for f in logs:f.close()
print('EVIDENCE',root,flush=True)
