"""The selected GPU set is one diagnostic allocation, never a serial GPU sweep."""
import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import deep_diagnostics as deep,diagnostics,runtime_variants as variants

class CollectiveDiagnostics(unittest.TestCase):
 def group(self,cases,devices):
  from multigpubfs.collective_diagnostics import group_cases
  return group_cases(cases,devices)

 def test_build_checks_cover_eight_ranks_together_in_three_cases(self):
  devices=list(range(8));cases=self.group(variants.cases(devices,production_only=True),devices)
  self.assertEqual(len(cases),3)
  self.assertTrue(all(c['config']['devices']==devices and 'device' not in c['config'] for c in cases))
  self.assertEqual({c['config']['kind'] for c in cases},{'api','component','bfs'})
  self.assertTrue(all(c['config']['variant']==variants.PRODUCTION for c in cases))

 def test_loading_matrix_is_fourteen_joint_cases_not_112_separate_cases(self):
  devices=list(range(8));cases=self.group(deep.startup_cases(devices),devices)
  self.assertEqual(len(cases),14)
  self.assertEqual(len({c['name'] for c in cases}),14)
  self.assertTrue(all(c['config']['devices']==devices for c in cases))
  self.assertFalse(any('CUDA_VISIBLE_DEVICES' in c['config'] for c in cases))

 def test_noncontiguous_logical_gpu_selection_is_preserved(self):
  devices=[2,5];cases=self.group(deep.startup_cases(devices),devices)
  self.assertTrue(all(c['config']['devices']==devices for c in cases))
  self.assertEqual(len(cases),14)

 def test_all_gpu_cases_start_before_wait_and_keep_rank_evidence(self):
  from multigpubfs.collective_diagnostics import run_group
  real_popen=subprocess.Popen
  script=r'''
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);rank=int(root.name.split('-')[-1]);config=json.loads((root/'case-input.json').read_text())
(root.parent/('ready-'+str(rank))).write_text('ready');deadline=time.monotonic()+8
while len(list(root.parent.glob('ready-*')))!=8:
 if time.monotonic()>deadline:raise RuntimeError('Workers were launched serially')
 time.sleep(.005)
(root/'result.json').write_text(json.dumps({'status':'PASS','data':{'device':config['device'],'executed':True}}))
'''
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);seen=[]
   def launch(command,**kwargs):
    self.assertNotIn('start_new_session',kwargs)
    folder=command[command.index('--output')+1];seen.append(folder)
    return real_popen([sys.executable,'-c',script,folder],**kwargs)
   with patch('multigpubfs.collective_diagnostics.subprocess.Popen',side_effect=launch):result=run_group({'kind':'api','devices':list(range(8))},root)
   self.assertEqual(len(seen),8);self.assertEqual(result['devices'],list(range(8)))
   self.assertEqual([x['device'] for x in result['ranks']],list(range(8)))
   self.assertTrue(all(x['status']=='PASS' and x['result']['data']['executed'] for x in result['ranks']))
   self.assertEqual(json.loads((root/'probe.json').read_text())['status'],'PASS')

 def test_rank_five_failure_is_retained_and_never_passes_group(self):
  from multigpubfs.collective_diagnostics import run_group
  real_popen=subprocess.Popen
  script=r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);rank=int(root.name.split('-')[-1]);failed=rank==5
(root/'result.json').write_text(json.dumps({'status':'FAIL' if failed else 'PASS','error':'specific rank five failure' if failed else None}))
raise SystemExit(1 if failed else 0)
'''
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   def launch(command,**kwargs):return real_popen([sys.executable,'-c',script,command[command.index('--output')+1]],**kwargs)
   with patch('multigpubfs.collective_diagnostics.subprocess.Popen',side_effect=launch):
    with self.assertRaisesRegex(RuntimeError,'COLLECTIVE_DIAGNOSTIC_RANK_FAILED'):run_group({'kind':'api','devices':list(range(8))},root)
   report=json.loads((root/'probe.json').read_text());self.assertEqual(report['status'],'FAIL')
   self.assertEqual(len(report['ranks']),8);self.assertEqual(report['ranks'][5]['exit_code'],1)
   self.assertIn('specific rank five failure',json.dumps(report['ranks'][5]))

 def test_catalog_runs_once_per_rank_in_same_process_and_keeps_each_result(self):
  from multigpubfs.collective_diagnostics import catalog_batch
  cases=[{'family':'PermutationGroups','constructor':n,'kwargs':{'n':4},'unsupported':None} for n in ('coxeter','all_cycles')]
  cases.append({'family':'Puzzles','constructor':'unavailable','kwargs':{},'unsupported':'no bounded arguments'})
  calls=[]
  def probe(config,root):calls.append((config['catalog']['constructor'],os.getpid()));return {'all_generators_exact_readback_verified':True,'device':config['device']}
  with tempfile.TemporaryDirectory() as d,patch.object(deep,'catalog_probe',side_effect=probe):
   result=catalog_batch({'kind':'catalog-batch','catalog':cases,'device':3},Path(d))
   self.assertEqual(result['status'],'PASS');self.assertEqual(result['passed'],2);self.assertEqual(result['skipped'],1)
   self.assertEqual(len({pid for _,pid in calls}),1);self.assertEqual(len(result['cases']),3)

 def test_catalog_failure_does_not_hide_later_cases(self):
  from multigpubfs.collective_diagnostics import catalog_batch
  cases=[{'family':'PermutationGroups','constructor':str(i),'kwargs':{},'unsupported':None} for i in range(3)]
  def probe(config,root):
   if config['catalog']['constructor']=='1':raise RuntimeError('family-specific-error')
   return {'all_generators_exact_readback_verified':True}
  with tempfile.TemporaryDirectory() as d,patch.object(deep,'catalog_probe',side_effect=probe):
   with self.assertRaisesRegex(RuntimeError,'COLLECTIVE_CATALOG_FAILED'):catalog_batch({'catalog':cases,'device':0},Path(d))
   report=json.loads((Path(d)/'probe.json').read_text());self.assertEqual(len(report['cases']),3)
   self.assertEqual(report['cases'][1]['status'],'FAIL');self.assertEqual(report['cases'][2]['status'],'PASS')

 def test_full_allocation_recipe_never_falls_back_to_one_gpu(self):
  from multigpubfs.server_recipe import candidates
  devices=list(range(8));cases=list(candidates(devices,full_gpu_only=True))
  self.assertTrue(cases);self.assertTrue(all(c['devices']==devices for c in cases))

 def test_small_remaining_budget_skips_instead_of_shortening_probe(self):
  with tempfile.TemporaryDirectory() as d:
   runner=diagnostics.Runner(Path(d)/'out',60)
   with patch.object(runner,'run',side_effect=AssertionError('A 20-second case must not start with 4 seconds remaining')):
    result=deep.run_case(runner,{'name':'catalog-group','config':{'kind':'api'},'seconds':20},time.monotonic()+12)
   self.assertEqual(result['status'],'SKIPPED');self.assertEqual(result['reason'],'GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED')

 def test_deep_scheduler_has_only_full_allocation_gpu_cases(self):
  devices=list(range(8));recorded=[]
  def run(runner,case,deadline):
   recorded.append(case);kind=case['config'].get('stage');data={}
   if kind=='count':data={'visible_devices':8}
   if case['config'].get('kind')=='build-inventory':data={'status':'AVAILABLE','lazy_delivery':False}
   result={'name':case['name'],'status':'PASS','result':{'status':'PASS','data':data},'diagnostic_case':case['config']};runner.report['steps'].append(result);return result
  catalog=[{'family':'PermutationGroups','constructor':'coxeter','kwargs':{'n':4},'unsupported':None}]
  with tempfile.TemporaryDirectory() as d,patch.object(deep,'run_case',side_effect=run),patch.object(deep,'catalog_configurations',return_value=catalog),patch('multigpubfs.server_recipe.find_working',return_value={'status':'NOT_FOUND'}) as search:
   report=deep.run_deep(Path(d)/'out',devices=devices,total_timeout=1800)
  gpu=[c for c in recorded if c['config'].get('kind') in ('api','bfs','catalog-batch') or c['config'].get('stage') in ('context','meminfo','hardware','memory','kernel','persistent','nccl_exchange')]
  self.assertTrue(gpu);self.assertTrue(all(c['config'].get('devices')==devices and 'device' not in c['config'] for c in gpu),[(c['name'],c['config']) for c in gpu])
  self.assertEqual(len([c for c in gpu if c['config'].get('kind')=='catalog-batch']),1)
  self.assertTrue(search.call_args.kwargs['full_gpu_only'])
  self.assertEqual(report['devices'],devices)

 def test_basic_scheduler_uses_same_full_allocation_policy(self):
  devices=list(range(8));recorded=[]
  def component(runner,name,stage,**kwargs):
   data={'visible_devices':8} if stage=='count' else {};value={'name':name,'status':'PASS','result':{'status':'PASS','data':data}};runner.report['steps'].append(value);return value
  def run(runner,case,deadline):
   recorded.append(case);value={'name':case['name'],'status':'PASS','result':{'status':'PASS','data':{}},'diagnostic_case':case['config']};runner.report['steps'].append(value);return value
  with tempfile.TemporaryDirectory() as d,patch.object(diagnostics.Runner,'component',component),patch.object(deep,'run_case',side_effect=run),patch('multigpubfs.server_recipe.find_working',return_value={'status':'NOT_FOUND'}):
   diagnostics.run_diagnostics(Path(d)/'out',devices=devices)
  self.assertTrue(recorded)
  self.assertTrue(all(c['config']['devices']==devices and 'device' not in c['config'] for c in recorded))
