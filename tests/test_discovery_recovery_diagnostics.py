import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs import deep_diagnostics as deep,diagnostics
from multigpubfs.diagnostic_analysis import analyze
from test_universal_architecture_diagnostics import driver_step

class DeepRecoveryRegression(unittest.TestCase):
 def test_deep_native_count_timeout_does_not_prevent_actual_recipe_search(self):
  def run(runner,case,deadline):
   name=case['name'];stage=case['config'].get('stage')
   result=driver_step() if stage=='driver' else {'status':'TIMEOUT' if stage=='count' else 'PASS','result':{'status':'PASS','data':{}}}
   if stage=='count':result.update(error='fixture native timeout')
   value=dict(result,name=name,diagnostic_case=case['config']);runner.report['steps'].append(value);return value
  with tempfile.TemporaryDirectory() as d,patch.object(deep,'run_case',side_effect=run),patch.object(deep,'startup_cases',return_value=[]),patch.object(deep,'catalog_configurations',return_value=[]),patch('multigpubfs.server_recipe.find_working',return_value={'status':'NOT_FOUND','reason':'fixture'}) as search:
   report=deep.run_deep(Path(d)/'out',total_timeout=30)
  search.assert_called_once();self.assertEqual(search.call_args.args[1],[0,1]);self.assertEqual(report['status'],'FAIL')
 def test_ready_recipe_does_not_say_bfs_was_never_reached(self):
  report={'steps':[{'name':'cuda-discovery','status':'TIMEOUT'}],'working_configuration':{'status':'READY','selected':{'devices':[0,1]},'command':'verified-command','calculation_example':'verified-example'}}
  result=analyze(report)
  self.assertFalse(any('BFS and NCCL were not reached' in x['statement'] for x in result['ruled_out_for_this_observation']))
 def test_missing_native_sm_and_unknown_ptx_are_not_certain_cause(self):
  report={'steps':[{'name':'cuda-discovery','status':'TIMEOUT'}],'architecture_support':{'devices':[{'logical_id':0,'sm':100,'code':'UNKNOWN_PTX'}]}}
  result=analyze(report)
  self.assertIn('GPU_NATIVE_TARGET_NOT_LISTED',[x['code'] for x in result['confirmed_findings']])
  self.assertFalse(any(x['code']=='NATIVE_GPU_CODE_UNAVAILABLE' for x in result['confirmed_findings']))
 def test_compatible_ptx_can_only_explain_first_load_as_hypothesis(self):
  report={'steps':[{'name':'cuda-discovery','status':'TIMEOUT'}],'architecture_support':{'devices':[{'logical_id':0,'sm':100,'code':'PTX_JIT'}]}}
  result=analyze(report)
  self.assertIn('FIRST_LOAD_PTX_JIT',[x['code'] for x in result['hypotheses']])
  self.assertFalse(any('root cause' in x['statement'] and 'confirmed' in x['statement'] for x in result['confirmed_findings']))
