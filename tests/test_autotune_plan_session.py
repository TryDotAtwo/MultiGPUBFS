import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import generic_session as session

RESOURCE='REQUESTED_CAPACITY_EXCEEDS_SORTED_ADMISSION'
def failed_job(command='graph-local-plan',error=RESOURCE,sequence=0,schema=1):
 response={'schema':schema,'sequence':sequence,'ok':False,'error':error}
 return SimpleNamespace(command=command,sequence=0,returncode=1,response=response,poll=lambda:1)

class NativePlanningRejections(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def worker(self,job,dead=False):
  w=object.__new__(session.Worker);w.root=self.root;w.rank=0;w.device=0;w.sequence=1;w.job=job;w.process=SimpleNamespace(pid=1001,poll=lambda:1 if dead else None);return w
 def test_live_worker_accepts_next_request_after_valid_native_capacity_plan_rejection(self):
  job=failed_job();original=json.dumps(job.response,sort_keys=True);w=self.worker(job)
  try:next_job=w.submit('inventory',[],{})
  except RuntimeError as e:self.fail('Plan-only capacity refusal poisoned a live worker: '+str(e))
  self.assertEqual(next_job.pid,1001);self.assertEqual(w.sequence,2)
  self.assertEqual(json.dumps(job.response,sort_keys=True),original);self.assertEqual(job.returncode,1)
  self.assertEqual(json.loads((self.root/'job-00000001.json').read_text())['command'],'inventory')
 def test_identical_error_from_bfs_job_remains_fatal(self):
  w=self.worker(failed_job(command='graph-rank'))
  with self.assertRaisesRegex(RuntimeError,'GENERIC_SESSION_FAILED_NO_RESTART'):w.submit('inventory',[],{})
 def test_unknown_or_invalid_plan_response_is_not_benign(self):
  for job in [failed_job(error='GENERIC_CUDA_700'),failed_job(sequence=99),failed_job(schema=2),failed_job(command=None)]:
   with self.subTest(response=job.response,command=job.command),self.assertRaisesRegex(RuntimeError,'GENERIC_SESSION_FAILED_NO_RESTART'):self.worker(job).submit('inventory',[],{})
 def test_exited_worker_is_not_reused_even_after_capacity_refusal(self):
  with self.assertRaisesRegex(RuntimeError,'GENERIC_SESSION_FAILED_NO_RESTART'):self.worker(failed_job(),dead=True).submit('inventory',[],{})
 def test_pool_reuses_same_group_after_native_plan_refusal(self):
  class FakeWorker:
   def __init__(self,native,env,rank,device,root,kind):
    self.root=root;root.mkdir();self.job=None;self.process=SimpleNamespace(pid=1000+rank,poll=lambda:None)
   def close(self):pass
  pool=session.Pool()
  try:
   with patch('multigpubfs.autotune._dependency_identity',return_value=None),patch('multigpubfs.autotune._stat_identity',return_value=('fixed',)),patch.object(session,'Worker',side_effect=FakeWorker) as factory:
    pool.select('native',[0,1],{});workers=tuple(pool.workers);generation=pool.generation
    for worker in workers:worker.job=failed_job(command='graph-plan')
    try:pool.select('native',[0,1],{})
    except RuntimeError as e:self.fail('A plan rejection poisoned group selection: '+str(e))
    self.assertEqual(tuple(pool.workers),workers);self.assertEqual(pool.generation,generation);self.assertEqual(factory.call_count,2)
  finally:pool.close()
 def query_with_local_error(self,error):
  self.waited=[False,False];test=self
  class FakeWorker:
   def __init__(self,rank):self.rank=rank
   def submit(self,command,args,env):
    rank=self.rank
    response={'schema':1,'sequence':0,'ok':False,'error':error} if rank==0 else {'schema':1,'sequence':0,'ok':True,'result':{'free_bytes':100000,'plan':{'batch':64}}}
    def wait(seconds):test.waited[rank]=True;return 1 if rank==0 else 0
    return SimpleNamespace(command=command,sequence=0,returncode=1 if rank==0 else 0,response=response,wait=wait)
  pool=SimpleNamespace(inventory=lambda *args:[{'device':0},{'device':1}],workers=[FakeWorker(0),FakeWorker(1)])
  definition=self.root/'definition.json';definition.write_text(json.dumps({'action':{'generators':[0]}}))
  with patch.object(session,'_active',pool):
   return session.query(['native','graph-info',str(definition),'0,1','4','100000'],{'MGBFS_GENERIC_HISTORY':'sorted'},10,{})
 def test_all_submitted_local_plans_are_drained_before_resource_refusal_returns(self):
  with self.assertRaisesRegex(RuntimeError,'GENERIC_SESSION_LOCAL_PLAN_FAILED'):self.query_with_local_error(RESOURCE)
  self.assertEqual(self.waited,[True,True],'Peer plan must finish before reusing the group')
 def test_unknown_local_plan_failure_keeps_fail_fast_behavior(self):
  with self.assertRaisesRegex(RuntimeError,'GENERIC_SESSION_LOCAL_PLAN_FAILED'):self.query_with_local_error('GENERIC_CUDA_700')
  self.assertEqual(self.waited,[True,False])

if __name__=='__main__':unittest.main()
