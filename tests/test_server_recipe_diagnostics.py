"""Working server recipes require exact BFS and an executable-command check."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
try:
 from multigpubfs import server_recipe as recipe
except ImportError:
 recipe=None

class ServerRecipeTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(recipe,'Missing automatic verified server recipe and launcher')

 def settings(self,**kw):
  return dict(variant='shared-sm120',devices=[0,1],worker='persistent',module='inherited',data_loading='inherited',transport='inherited',**kw)

 def good(self,settings=None):
  s=settings or self.settings()
  return {'status':'PASS','result':{'status':'PASS','data':{'cpu_oracles_verified':True,'settings':s,'graphs':[{'fixture':n,'status':'COMPLETE','layers_verified':True,'terminal_states_verified':True} for n in ('permutation','repeated','matrix')]}},'partial_evidence':{'cleanup.json':{'complete':True}}}

 def test_kernel_or_inventory_alone_cannot_become_working_recipe(self):
  self.assertFalse(recipe.accepted({'status':'PASS','result':{'data':{'kernel':True}}},self.settings()))

 def test_exact_combined_gpu_bfs_and_cleanup_are_required(self):
  s=self.settings();step=self.good(s)
  self.assertTrue(recipe.accepted(step,s))
  for key in ('layers_verified','terminal_states_verified'):
   bad=copy.deepcopy(step);bad['result']['data']['graphs'][1][key]=False
   self.assertFalse(recipe.accepted(bad,s))
  bad=copy.deepcopy(step);bad['partial_evidence']['cleanup.json']['complete']=False
  self.assertFalse(recipe.accepted(bad,s))
  bad=copy.deepcopy(step);bad['result']['data']['settings']['devices']=[0]
  self.assertFalse(recipe.accepted(bad,s))

 def test_incomplete_timeout_and_missing_proof_never_pass(self):
  for status in ('FAIL','TIMEOUT','SKIPPED','INCOMPLETE'):
   bad=self.good();bad['status']=status
   self.assertFalse(recipe.accepted(bad,self.settings()))
  bad=self.good();bad['result']['data']['graphs'].pop()
  self.assertFalse(recipe.accepted(bad,self.settings()))

 def test_search_prefers_same_full_gpu_set_and_current_production(self):
  rows=list(recipe.candidates([0,1]))
  self.assertEqual(rows[0],self.settings())
  first_single=next(i for i,c in enumerate(rows) if len(c['devices'])==1)
  self.assertTrue(all(c['devices']==[0,1] for c in rows[:first_single]))
  self.assertTrue(any(c['variant']=='static-sm120' for c in rows[:first_single]))
  self.assertTrue(any(c['worker']=='fresh' for c in rows[:first_single]))
  self.assertTrue(any(c['module']=='LAZY' for c in rows[:first_single]))
  self.assertTrue(any(c['transport']=='socket' for c in rows[:first_single]))

 def test_only_allowlisted_environment_adjustments_are_emitted(self):
  s=self.settings();s.update(module='LAZY',data_loading='LAZY',transport='socket')
  e=recipe.overrides(s)
  self.assertEqual(e['CUDA_MODULE_LOADING'],'LAZY')
  self.assertEqual(e['NCCL_IB_DISABLE'],'1')
  s['variant']='../../outside'
  with self.assertRaises(ValueError):recipe.overrides(s)
  s=self.settings();s['devices']=[0,0]
  with self.assertRaises(ValueError):recipe.overrides(s)

 def test_report_command_is_argv_quoted_for_paths_with_spaces(self):
  import shlex
  path=Path('/tmp/server job/working-config.json')
  command=recipe.command(path)
  self.assertEqual(shlex.split(command),recipe.command_argv(path))
  self.assertIn('--check',shlex.split(command))
  self.assertIn(str(path),shlex.split(command))

 def test_profile_proof_is_bound_to_settings_and_file_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);settings=self.settings();proof=p/'proof.json'
   proof.write_text(json.dumps(self.good(settings)['result']))
   value={'schema':1,'settings':settings,'proof':{'path':'proof.json','sha256':recipe.digest(proof)}}
   self.assertEqual(recipe.read_proof(value,p)['settings'],settings)
   value['settings']=dict(settings,worker='fresh')
   with self.assertRaises(RuntimeError):recipe.read_proof(value,p)
   value['settings']=settings;proof.write_text('{}')
   with self.assertRaises(RuntimeError):recipe.read_proof(value,p)

 def test_profile_rejects_external_proof_and_unknown_fields(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaises(RuntimeError):recipe.read_proof({'proof':{'path':'../other.json','sha256':'0'*64}},p)
  s=self.settings();s['LD_PRELOAD']='arbitrary.so'
  with self.assertRaises(ValueError):recipe.overrides(s)

 def test_no_candidates_after_unreaped_child_and_no_success_command(self):
  class Runner:
   deep_cleanup_blocked=True
   report={'steps':[]}
  with tempfile.TemporaryDirectory() as d:
   runner=Runner();runner.root=Path(d)
   value=recipe.find_working(runner,[0,1],0)
   self.assertEqual(value['status'],'BLOCKED')
   self.assertNotIn('command',value)
   self.assertFalse((runner.root/'working-config.json').exists())

 def test_no_unchecked_command_on_exhausted_budget(self):
  class Runner:
   deep_cleanup_blocked=False
   report={'steps':[]}
  with tempfile.TemporaryDirectory() as d:
   runner=Runner();runner.root=Path(d)
   value=recipe.find_working(runner,[0,1],0)
   self.assertEqual(value['status'],'INCOMPLETE')
   self.assertNotIn('command',value)

 def test_server_identity_change_invalidates_profile(self):
  with patch.object(recipe,'server_identity',return_value={'host':'new'}):
   with self.assertRaisesRegex(RuntimeError,'SERVER_CHANGED'):
    recipe.validate_identity({'server':{'host':'old'}})

 def test_environment_change_invalidates_profile(self):
  original=recipe.server_identity()
  with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'changed'}):
   self.assertNotEqual(original,recipe.server_identity())


class RecipeArtifactRegression(unittest.TestCase):
 def test_probe_rejects_runtime_library_different_from_selected_build(self):
  import contextlib
  from multigpubfs.deep_diagnostics import oracle
  from multigpubfs import launch,native_distribution,diagnostics,runtime_variants
  settings=dict(variant='shared-sm120',devices=[0],worker='fresh',module='inherited',data_loading='inherited',transport='inherited')
  def run(graph,output,**kwargs):
   output=Path(output);output.mkdir();sizes,last=oracle(graph)
   (output/'states.json').write_text(json.dumps({'current':last}))
   return {'status':'COMPLETE','layer_sizes':sizes,'states_sha256':'not-the-test-subject','bfs_seconds':.01,'runtime_identity':{'cuda_library_sha256':'wrong-library'}}
  inv={'status':'AVAILABLE','variants':[{'id':'shared-sm120','sha256':'selected-library'}]}
  with tempfile.TemporaryDirectory() as d,patch.object(launch,'run_graph',side_effect=run),patch.object(native_distribution,'native_runtime',return_value=('/bin/false',{})),patch.object(runtime_variants,'inventory',return_value=inv),patch.object(diagnostics,'_paths',return_value=('/bin/false',{},{})),patch.object(runtime_variants,'confirm_resolution',return_value={'sha256':'selected-library'}):
   with self.assertRaisesRegex(RuntimeError,'LIBRARY_IDENTITY'):
    recipe.probe(settings,Path(d))


class NativeRecipePackagingRegression(unittest.TestCase):
 def test_native_payload_is_extracted_from_standard_wheel_platlib(self):
  import io,zipfile
  from scripts import repackage_verified_native as package
  self.assertTrue(hasattr(package,'extract_native'),'Missing wheel platlib extraction')
  data=io.BytesIO()
  with zipfile.ZipFile(data,'w') as z:
   z.writestr('multigpubfs-0.2.0.dev16.data/platlib/multigpubfs/_native/bin/mgbfs',b'native-binary')
   z.writestr('multigpubfs-0.2.0.dev16.data/platlib/multigpubfs/_native/lib/libmgbfs_cuda.so',b'native-library')
   z.writestr('multigpubfs-0.2.0.dev16.data/platlib/multigpubfs/server_recipe.py',b'old-python-must-not-be-copied')
  with tempfile.TemporaryDirectory() as d,zipfile.ZipFile(data) as z:
   package.extract_native(z,Path(d))
   self.assertEqual((Path(d)/'multigpubfs/_native/bin/mgbfs').read_bytes(),b'native-binary')
   self.assertFalse((Path(d)/'multigpubfs/server_recipe.py').exists())

if __name__=='__main__':unittest.main()
