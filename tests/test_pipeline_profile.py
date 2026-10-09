import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_profile import select_pipeline,FLAGS

class PipelineProfileTests(unittest.TestCase):
 def base(self):return {'env':{'MGBFS_OWNER_BACKEND':'SHARD_AB','MGBFS_PRE_DEDUP':'ON'}}
 def test_key_first_freezes_all_gates_before_admission(self):
  b=self.base();x=select_pipeline(b,{},'key-first');self.assertEqual(b,self.base());self.assertEqual(x['env']['MGBFS_PRE_DEDUP'],'OFF')
  for key in FLAGS[:6]:self.assertEqual(x['env'][key],'1')
  for key in FLAGS[6:-1]:self.assertEqual(x['env'][key],'0')
  self.assertEqual(x['env']['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS'],'1')
 def test_baseline_does_not_silently_enable_experiments(self):
  x=select_pipeline(self.base(),{},'baseline');self.assertEqual(x['pipeline_profile'],'baseline');self.assertTrue(all(x['env'][k]=='0' for k in FLAGS))
 def test_runtime_key_first_infers_profile(self):
  self.assertEqual(select_pipeline(self.base(),{'MGBFS_SHARD_AB_KEY_FIRST':'1'})['pipeline_profile'],'key-first')
 def test_resume_preserves_settings(self):
  x=select_pipeline(self.base(),{},'key-first');self.assertEqual(select_pipeline(x,{},resume=True),x)
 def test_resume_rejects_profile_or_flag_change(self):
  x=select_pipeline(self.base(),{},'key-first')
  for rt,profile in [({},'baseline'),({'MGBFS_SHARD_AB_KEY_FIRST':'0'},None),({'MGBFS_PRE_DEDUP':'ON'},None),({'MGBFS_SHARD_AB_DEDUP':'SORT_MERGE'},None)]:
   with self.assertRaises(ValueError):select_pipeline(x,rt,profile,resume=True)
 def test_legacy_resume_unchanged(self):
  x=self.base();self.assertEqual(select_pipeline(x,{},resume=True),x)
  with self.assertRaises(ValueError):select_pipeline(x,{},'key-first',resume=True)
 def test_wrong_backend_rejected(self):
  x=self.base();x['env']['MGBFS_OWNER_BACKEND']='CUCO_RANK'
  with self.assertRaises(ValueError):select_pipeline(x,{},'key-first')
 def test_conflicting_runtime_flag_rejected(self):
  with self.assertRaises(ValueError):select_pipeline(self.base(),{'MGBFS_SHARD_AB_REUSE_HISTORY':'1'},'key-first')
 def test_sort_mode_is_frozen(self):
  x=select_pipeline(self.base(),{'MGBFS_SHARD_AB_DEDUP':'SORT_MERGE'},'key-first');self.assertEqual(x['env']['MGBFS_SHARD_AB_DEDUP'],'SORT_MERGE')
 def test_key_first_rejects_intermediate_prededup(self):
  with self.assertRaises(ValueError):select_pipeline(self.base(),{'MGBFS_PRE_DEDUP':'ON'},'key-first')
 def test_baseline_freezes_explicit_prededup(self):
  self.assertEqual(select_pipeline(self.base(),{'MGBFS_PRE_DEDUP':'OFF'},'baseline')['env']['MGBFS_PRE_DEDUP'],'OFF')
 def test_status_reuse_only_for_hash_key_first(self):
  x=select_pipeline(self.base(),{'MGBFS_SHARD_AB_DEDUP':'SORT_MERGE'},'key-first')
  self.assertEqual(x['env']['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS'],'0')
 def test_old_profile_resume_retains_status_reuse_off(self):
  x=select_pipeline(self.base(),{},'key-first');del x['env']['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS']
  y=select_pipeline(x,{},resume=True);self.assertEqual(y['env']['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS'],'0')
  with self.assertRaises(ValueError):select_pipeline(x,{'MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS':'1'},resume=True)
 def test_new_profile_resume_rejects_status_reuse_change(self):
  x=select_pipeline(self.base(),{},'key-first')
  with self.assertRaises(ValueError):select_pipeline(x,{'MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS':'0'},resume=True)
 def test_legacy_resume_rejects_new_status_reuse(self):
  with self.assertRaises(ValueError):select_pipeline(self.base(),{'MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS':'1'},resume=True)
 def test_new_shard_run_defaults_to_measured_key_first(self):
  x=select_pipeline(self.base(),{});self.assertEqual(x,select_pipeline(self.base(),{},'key-first'));self.assertEqual(x['pipeline_profile'],'key-first');self.assertEqual(x['env']['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS'],'1')
 def test_explicit_runtime_baseline_remains_available(self):
  self.assertEqual(select_pipeline(self.base(),{'MGBFS_SHARD_AB_KEY_FIRST':'0'})['pipeline_profile'],'baseline')
 def test_other_backend_default_unchanged(self):
  x=self.base();x['env']['MGBFS_OWNER_BACKEND']='CUCO_RANK';self.assertEqual(select_pipeline(x,{}),x)
if __name__=='__main__':unittest.main()
