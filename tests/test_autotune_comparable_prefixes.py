import unittest
from pathlib import Path
import test_autotune_candidate_failures as harness

class ComparableMeasuredPrefixes(unittest.TestCase):
 def setUp(self):
  self.h=harness.MeasuredSelectionAfterCandidateFailure();self.h.setUp()
  self.sizes=[1,6,22,64,162,378,819,1688,3400,5856,8270,9550,9696,9562,9266,8680,7528,5256,2612,412,4]
 def tearDown(self):self.h.tearDown()
 def report(self,index,depth=20):
  return {'layer_sizes':self.sizes[:depth+1],'layer_seconds':[1.0-index*.1]*depth,'status':'COMPLETE' if depth==20 else 'INCOMPLETE','reason':'FRONTIER_EXHAUSTED' if depth==20 else 'DEADLINE'}
 def choose(self,depths,fault=False,mismatch=False):
  def pilot(g,target,**kwargs):
   index=len(self.h.calls);self.h.calls.append(Path(target).name)
   if fault and index==0:self.h.native_fault(target)
   value=self.report(index,depths[index])
   if mismatch and index==1:value['layer_sizes'][3]+=1
   return value
  return self.h.choose(pilot)
 def test_short_deadline_candidate_does_not_poison_four_measured_candidates(self):
  result=self.choose([20,10,20,20,20])
  self.assertTrue(result['measured']);self.assertEqual(result['common_depth'],20)
  self.assertEqual(len(result['pilots']),4)
  self.assertEqual([x['candidate'] for x in result['excluded_prefix_candidates']],['pilot-1'])
  self.assertEqual(result['excluded_prefix_candidates'][0]['completed_depth'],10)
  self.assertEqual(result['excluded_prefix_candidates'][0]['reason'],'INSUFFICIENT_COMPLETED_PREFIX')
 def test_adequate_incomplete_prefix_still_participates_at_its_completed_depth(self):
  result=self.choose([20,10,14,20,20])
  self.assertTrue(result['measured']);self.assertEqual(result['common_depth'],14)
  self.assertEqual(len(result['pilots']),4)
  self.assertIn('pilot-2',[p['candidate'] for p in result['pilots']])
  self.assertEqual(result['common_layer_sizes'],self.sizes[:15])
 def test_faulted_candidate_and_short_candidate_are_separate_from_measured_survivors(self):
  try:result=self.choose([20,10,20,20,20],fault=True)
  except RuntimeError as error:self.fail('An insufficient bounded probe aborted adequate survivors: '+str(error))
  self.assertTrue(result['measured']);self.assertEqual(len(result['pilots']),3)
  self.assertEqual(len(result['candidate_failures']),1)
  self.assertEqual(len(result['excluded_prefix_candidates']),1)
  self.assertEqual(len(self.h.calls),5)
 def test_short_candidate_is_still_checked_for_correctness_before_exclusion(self):
  with self.assertRaisesRegex(RuntimeError,'AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH'):self.choose([20,10,20,20,20],mismatch=True)
 def test_one_adequate_candidate_is_not_a_measured_comparison(self):
  result=self.choose([20,10,10,10,10])
  self.assertFalse(result['measured'])
  self.assertIn('excluded_prefix_candidates',result)
  self.assertEqual(len(result['excluded_prefix_candidates']),4)
 def test_fault_with_only_one_adequate_candidate_reports_no_measured_profile(self):
  with self.assertRaisesRegex(RuntimeError,'AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE'):self.choose([20,20,10,10,10],fault=True)

if __name__=='__main__':unittest.main()
