import json,subprocess,tempfile,unittest,warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import autotune,generic_session
from multigpubfs.graph_definition import GraphDefinition

class MeasuredSelectionAfterCandidateFailure(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.native=self.root/'native';self.native.write_bytes(b'test-native')
  self.graph=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
  self.calls=[];self.closes=0;self.after_fault=False
  self.pool=SimpleNamespace(workers=['old0','old1'],identity='old',launch_admission='stale')
  def close():self.closes+=1;self.pool.workers=[];self.pool.identity=None;self.after_fault=True
  self.pool.close_workers=close
  self.env={'MGBFS_GENERIC_HISTORY':'hash','MGBFS_TUNING_EVIDENCE_ROOT':str(self.root/'evidence'),'_MGBFS_PROFILE_INVENTORY':'stale'}
 def tearDown(self):self.tmp.cleanup()
 def admit(self,g,devices,capacity,shards,native,env,directory,**kwargs):
  if self.after_fault:
   self.assertNotIn('_MGBFS_PROFILE_INVENTORY',env,'A destroyed rank group needs fresh admission')
   self.assertEqual(self.pool.workers,[])
   self.assertIsNone(self.pool.launch_admission)
  self.assertEqual(devices,[0,1])
  history='SORTED_RUNS' if env.get('MGBFS_GENERIC_HISTORY')=='sorted' else 'HASH'
  return {'devices':[0,1],'inventory':[],'plan':{'capacity':50000,'batch':64,'history_algorithm':history,'owner_lanes':int(env.get('MGBFS_GENERIC_OWNER_LANES',0))}}
 def native_fault(self,target,code=999):
  evidence=Path(target)/'failure-evidence';(evidence/'rank-1').mkdir(parents=True)
  (evidence/'rank-1/phase.json').write_text(json.dumps({'native_error':{'code':code,'call':'cudaMemcpyAsync','rank':1,'device':1,'layer':2}}))
  error='NATIVE_DISTRIBUTED_RANK_FAILED [(1, 1)]; evidence='+str(evidence)
  (evidence/'summary.json').write_text(json.dumps({'captures':[],'error':error}))
  raise RuntimeError(error)
 def report(self,index):
  return {'layer_sizes':[1,20000,20000,20000,1],'layer_seconds':[.03]+[1-index*.1]*3,'status':'INCOMPLETE','reason':'PROFILE_LAYER_LIMIT'}
 def choose(self,pilot):
  with patch.object(autotune,'_admit',side_effect=self.admit),patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],1,'','')),patch.object(autotune,'_dependency_identity',return_value=None),patch('multigpubfs.launch.run_graph',side_effect=pilot),patch.object(generic_session,'_active',self.pool),warnings.catch_warnings(record=True):
   return autotune.choose_profile(self.graph,[0,1],None,90,str(self.native),self.env,allow_specialized=False)
 def test_failed_first_candidate_is_excluded_and_remaining_profiles_are_measured(self):
  def pilot(g,target,**kwargs):
   self.calls.append(Path(target).name)
   if len(self.calls)==1:self.native_fault(target)
   return self.report(len(self.calls))
  try:result=self.choose(pilot)
  except RuntimeError as e:self.fail_test('One candidate aborted all measurements: '+str(e))
  self.assertTrue(result['measured']);self.assertEqual(result['status'],'MEASURED_EQUAL_PREFIX_GPU_PROFILE')
  self.assertEqual((result['shards'],result['candidate_order']),(4,'radix'))
  self.assertEqual(len(result['pilots']),4);self.assertEqual(len(result['candidate_failures']),1)
  self.assertEqual(self.calls.count('pilot-0'),1);self.assertEqual(self.closes,1)
  failure=result['candidate_failures'][0]
  self.assertEqual(failure['candidate'],'pilot-0');self.assertFalse(failure['probe_retried'])
  self.assertTrue(Path(failure['evidence']).is_dir());self.assertIn(str(failure['evidence']),failure['error'])
 def fail_test(self,message):raise AssertionError(message)
 def test_all_failed_candidates_raise_instead_of_baseline_or_measured_claim(self):
  def pilot(g,target,**kwargs):self.calls.append(Path(target).name);self.native_fault(target)
  with self.assertRaisesRegex(RuntimeError,'AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE') as caught:self.choose(pilot)
  self.assertEqual(len(self.calls),5);self.assertEqual(len(caught.exception.candidate_failures),5)
  self.assertTrue(all(Path(c['evidence']).is_dir() for c in caught.exception.candidate_failures))
 def test_one_survivor_is_not_an_empirical_comparison(self):
  def pilot(g,target,**kwargs):
   self.calls.append(Path(target).name)
   if len(self.calls)!=2:self.native_fault(target)
   return self.report(2)
  with self.assertRaisesRegex(RuntimeError,'AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE'):self.choose(pilot)
 def test_unproven_error_and_timeout_are_not_skipped(self):
  for error in (RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED [(1, 1)]'),subprocess.TimeoutExpired('pilot',5)):
   self.calls=[]
   def pilot(*args,**kwargs):self.calls.append('one');raise error
   with self.subTest(error=type(error).__name__),self.assertRaises(type(error)):self.choose(pilot)
   self.assertEqual(len(self.calls),1)
 def test_other_cuda_error_is_not_reclassified_as_the_known_probe_failure(self):
  def pilot(g,target,**kwargs):self.calls.append(Path(target).name);self.native_fault(target,700)
  with self.assertRaisesRegex(RuntimeError,'NATIVE_DISTRIBUTED_RANK_FAILED'):self.choose(pilot)
  self.assertEqual(len(self.calls),1)
 def test_sorted_candidates_still_run_when_all_hash_candidates_fail(self):
  self.env.pop('MGBFS_GENERIC_HISTORY')
  def pilot(g,target,**kwargs):
   self.calls.append(Path(target).name)
   if kwargs['_native_env'].get('MGBFS_GENERIC_HISTORY')!='sorted':self.native_fault(target)
   return self.report(len(self.calls))
  result=self.choose(pilot)
  self.assertTrue(result['measured']);self.assertEqual(result['history_algorithm'],'SORTED_RUNS')
  self.assertEqual(len(result['pilots']),4);self.assertEqual(len(result['candidate_failures']),5)
  self.assertEqual(self.closes,5)
 def test_measured_geometry_survives_unmeasured_size_tiers(self):
  p=self.report(2);p.update(backend='generic',shards=4,transport='full',candidate_order='radix',history_algorithm='HASH',owner_lanes=0,batch=16,generator_backend='cuda')
  global_profile=dict(p,status='MEASURED_EQUAL_PREFIX_GPU_PROFILE',measured=True,batch_fraction=.25,pilots=[p])
  admission={'size_profile_capability':1,'devices':[0,1],'inventory':[],'plan':{'capacity':50000,'batch':64,'history_algorithm':'HASH','generator_backend':'cuda'}}
  with patch.object(autotune,'choose_profile',return_value=global_profile),patch.object(autotune,'_admit',return_value=admission),patch.object(autotune,'_dependency_identity',return_value=None),patch.object(autotune,'_system_info',return_value=subprocess.CompletedProcess([],1,'','')):
   result=autotune._choose_size_profile_uncached(self.graph,[0,1],None,90,str(self.native),self.env,allow_specialized=False)
  self.assertTrue(result['measured'],'Measured geometry must not be replaced by an unmeasured selection')
  self.assertFalse(result['size_profiles_measured'])
  self.assertTrue(all(not p['measured'] for p in result['size_profiles']))
 def test_size_entry_does_not_replace_a_terminal_tuning_error_with_baseline(self):
  root=self.root/'saved';ev=root/'pilot-0/failure-evidence';(ev/'rank-1').mkdir(parents=True)
  (ev/'rank-1/phase.json').write_text(json.dumps({'native_error':{'code':999}}));(ev/'summary.json').write_text('{}')
  error=RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED [(1, 1)]; evidence='+str(ev));error.tuning_evidence_root=str(root)
  with patch.object(generic_session,'_active',None),patch.object(autotune,'_choose_size_profile_uncached',side_effect=error),patch.object(autotune,'_admit') as admit,warnings.catch_warnings(record=True):
   with self.assertRaisesRegex(RuntimeError,'NATIVE_DISTRIBUTED_RANK_FAILED'):autotune.choose_size_profile(self.graph,[0,1],None,90,'native',{},allow_specialized=False)
   admit.assert_not_called()

if __name__=='__main__':unittest.main()
