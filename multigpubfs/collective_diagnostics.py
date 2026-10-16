"""Joint diagnostics for one allocated GPU set; never alter device visibility.

Independent CUDA primitives run concurrently with one child per selected device.
BFS/NCCL cases use the real distributed implementation with the entire device set.
The diagnostic supervisor owns every child; no child detaches from its group.
"""
import json,re,subprocess,sys,time,traceback
from pathlib import Path

def group_cases(cases,devices):
 devices=list(devices)
 if not devices or any(type(d) is not int or d<0 for d in devices) or len(set(devices))!=len(devices):raise ValueError('INVALID_COLLECTIVE_DEVICES')
 result=[];seen=set()
 for case in cases:
  config=dict(case['config']);config.pop('device',None);config.pop('devices',None)
  identity=json.dumps(config,sort_keys=True)
  if identity in seen:continue
  seen.add(identity);config['devices']=devices
  name=re.sub(r'gpu-?\d+','all-gpus',case['name'])
  result.append(dict(case,name=name,config=config))
 return result

def run_group(config,root):
 from .deep_diagnostics import write
 root=Path(root);devices=list(config['devices']);group_cases([],devices)
 state={'schema':1,'status':'RUNNING','devices':devices,'ranks':[],
        'scope':'Concurrent per-rank API/kernel evidence on the whole selected GPU set; NCCL and distributed BFS require their separate actual joint tests.'}
 jobs=[];logs=[]
 try:
  for rank,device in enumerate(devices):
   folder=root/('rank-'+str(rank));folder.mkdir();child=dict(config,device=device);child.pop('devices')
   write(folder/'case-input.json',child)
   log_path=root/('rank-'+str(rank)+'.log');log=log_path.open('w');logs.append(log)
   row={'rank':rank,'device':device,'status':'STARTING','log':log_path.name};state['ranks'].append(row);write(root/'probe.json',state)
   process=subprocess.Popen([sys.executable,'-m','multigpubfs.deep_diagnostics','--child',str(folder/'case-input.json'),'--output',str(folder)],stdout=log,stderr=subprocess.STDOUT)
   row.update(status='RUNNING',pid=process.pid);jobs.append((process,folder,row));write(root/'probe.json',state)
  pending=list(jobs)
  while pending:
   for process,folder,row in pending[:]:
    code=process.poll()
    if code is None:continue
    row['exit_code']=code
    try:
     result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
     row['result']=result
     row['status']='PASS' if code==0 and result.get('status')=='PASS' else 'FAIL'
     if row['status']=='PASS' and result.get('data',{}).get('status')=='UNAVAILABLE':row['status']='SKIPPED'
    except (OSError,ValueError) as error:row.update(status='FAIL',error='Missing or invalid child result: '+str(error))
    row['log_tail']=(root/row['log']).read_text(encoding='utf-8',errors='replace')[-4000:]
    pending.remove((process,folder,row));write(root/'probe.json',state)
   if pending:time.sleep(.01)
  failed=[(r['rank'],r.get('exit_code')) for r in state['ranks'] if r['status']=='FAIL']
  if failed:raise RuntimeError('COLLECTIVE_DIAGNOSTIC_RANK_FAILED '+repr(failed)+'; details: '+str(root/'probe.json'))
  state['status']='UNAVAILABLE' if all(r['status']=='SKIPPED' for r in state['ranks']) else 'PASS'
  if state['status']=='UNAVAILABLE':state['reason']='No supported catalog representatives on the selected GPU set'
  write(root/'probe.json',state);return state
 except BaseException as error:
  state.update(status='FAIL',error=str(error),exception_type=type(error).__name__);write(root/'probe.json',state);raise
 finally:
  for process,_,_ in jobs:
   if process.poll() is None:process.terminate()
  for process,_,_ in jobs:
   if process.poll() is None:
    try:process.wait(timeout=1)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=1)
  for log in logs:log.close()

def catalog_batch(config,root):
 from .deep_diagnostics import catalog_probe,write
 root=Path(root);state={'schema':1,'status':'RUNNING','device':config['device'],'cases':[],
                      'scope':'One process and reused CUDA context per selected rank; finite successor checks, not full puzzle BFS.'}
 for index,case in enumerate(config['catalog']):
  row={'family':case['family'],'constructor':case['constructor']};state['cases'].append(row)
  if case.get('unsupported'):row.update(status='SKIPPED',reason=case['unsupported'])
  else:
   folder=root/('catalog-'+str(index).zfill(3));folder.mkdir()
   try:
    data=catalog_probe(dict(config,kind='catalog',catalog=case),folder)
    if data.get('all_generators_exact_readback_verified') is not True:raise RuntimeError('CATALOG_READBACK_NOT_VERIFIED')
    row.update(status='PASS',data=data)
   except Exception as error:row.update(status='FAIL',error=str(error),exception_type=type(error).__name__,traceback=traceback.format_exc())
  write(root/'probe.json',state)
 state.update(passed=sum(x['status']=='PASS' for x in state['cases']),skipped=sum(x['status']=='SKIPPED' for x in state['cases']),failed=sum(x['status']=='FAIL' for x in state['cases']))
 state['status']='FAIL' if state['failed'] else 'PASS' if state['passed'] else 'UNAVAILABLE';write(root/'probe.json',state)
 if state['failed']:raise RuntimeError('COLLECTIVE_CATALOG_FAILED '+str(state['failed'])+'; details: '+str(root/'probe.json'))
 return state
