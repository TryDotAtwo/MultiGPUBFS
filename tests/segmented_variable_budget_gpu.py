"""Real multi-GPU allocations under a controlled small storage byte budget.

This is not a physical capacity measurement and makes no B200 claim. It
forces nonuniform segments and checks every completed layer and saved samples
against an independent CPU oracle. Set MGBFS_EXECUTABLE and LD_LIBRARY_PATH.
"""
import json, os, subprocess, tempfile, time
from pathlib import Path
from multigpubfs import GraphDefinition

native=os.environ['MGBFS_EXECUTABLE']
root=Path(tempfile.mkdtemp(prefix='segmented-variable-budget-',dir='/workspace'))
def swap(i):
 p=list(range(5));p[i],p[i+1]=p[i+1],p[i];return p
g=GraphDefinition.permutation([swap(i) for i in range(4)],list(range(5)))
expected=g.exact_layers(1000)
assert list(map(len,expected))==[1,4,9,15,20,22,20,15,9,4,1]
def rows(v):return sorted(map(tuple,v))

def run(bitmap,budget,label=''):
 name=('bitmap' if bitmap else 'hash')+'-'+str(budget)+(('-'+label) if label else '');case=root/name;case.mkdir()
 definition=case/'graph.json';definition.write_text(g.to_json())
 env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP=str(int(bitmap)),MGBFS_SEGMENT_ROWS='4',MGBFS_SEGMENT_TEST_BYTES=str(budget),MGBFS_SEGMENT_TEST_QUEUE_CAP='4',MGBFS_SEGMENT_OWNER_LANES='2',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash')
 query=subprocess.run([native,'graph-info',str(definition),'0,1','4','1','1'],env=env,text=True,capture_output=True,timeout=60,check=True)
 plan=json.loads(query.stdout);plan['owner_cuts']=None
 config=case/'launch.json';config.write_text(json.dumps(plan));jobs=[];logs=[]
 try:
  for rank in range(2):
   log=(case/f'rank-{rank}.log').open('w');logs.append(log)
   jobs.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=log,stderr=subprocess.STDOUT))
  until=time.monotonic()+90
  while any(p.poll() is None for p in jobs) and time.monotonic()<until:
   if any(p.poll() not in (None,0) for p in jobs):break
   time.sleep(.04)
  for f in logs:f.flush()
  assert all(p.poll()==0 for p in jobs),[p.poll() for p in jobs]+[(case/f'rank-{i}.log').read_text()[-2500:] for i in range(2)]
  parts=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  snaps=[json.loads((case/f'rank-{i}'/'states.json').read_text()) for i in range(2)]
  assert len(set(p['status'] for p in parts))==1,parts
  depth=len(parts[0]['layer_sizes'])-1
  assert all(p['layer_sizes']==list(map(len,expected[:depth+1])) for p in parts),parts
  assert rows(sum((s['current'] for s in snaps),[]))==rows(expected[depth]),name
  if depth>0:assert rows(sum((s['previous_small'] for s in snaps),[]))==rows(expected[depth-1]),name
  if parts[0]['status']=='INCOMPLETE':assert parts[0]['reason']=='RESOURCE_10',parts
  actual=[[n for bucket in part['segment_memory']['segment_rows_per_bank_by_shard'] for n in bucket] for part in parts]
  peak=[p['segment_memory']['peak_live_rows'] for p in parts]
  return {'mode':name,'budget_bytes_per_rank':budget,'status':parts[0]['status'],'depth':depth,'rows_by_rank':actual,'peak_live_rows':peak,'legacy_plan_capacity':plan['plan']['capacity'],'resource_detail_by_rank':[p['segment_memory']['resource_detail'] for p in parts]}
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in logs:f.close()

if __name__ == "__main__":
    results=[]
    for bitmap in (False,True):
     full=(3*4*5+24) if bitmap else (32*8+3*4*5+3*4*8+24)
     observed_tail=observed_complete=False
     for multiple in range(3,22):
      outcome=run(bitmap,full*multiple+full//2);results.append(outcome)
      print(json.dumps(outcome),flush=True)
      observed_tail |= any(0<n<4 for rank in outcome['rows_by_rank'] for n in rank)
      observed_complete |= outcome['status']=='COMPLETE'
      if observed_tail and observed_complete:break
     assert observed_tail and observed_complete,('missing variable tail or completion',bitmap,results)
     assert any(max(x['peak_live_rows'])>x['legacy_plan_capacity'] for x in results if x['mode'].startswith('bitmap' if bitmap else 'hash')),results
    print('VARIABLE_BUDGET_TWO_GPU_PASS',root,flush=True)
