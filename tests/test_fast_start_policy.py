import unittest
from unittest.mock import patch
from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.autotune import _choose_size_profile_uncached
class FastPolicy(unittest.TestCase):
 def graph(self):return GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
 def test_coxeter_skips_replayed_pilots(self):
  admission={'plan':{'batch':64},'inventory':[{'sm_count':40}]}
  with patch('multigpubfs.autotune._admit',return_value=admission),patch('multigpubfs.autotune.choose_profile',side_effect=AssertionError('cold replay')):
   p=_choose_size_profile_uncached(self.graph(),[0,1],64,60,'native',{})
  self.assertEqual(p['status'],'FAST_START_LIVE_SIZE_TUNING');self.assertTrue(p['online_size_tuning']);self.assertFalse(p['measured'])
 def test_explicit_history_preserves_selector(self):
  expected={'backend':'generic','status':'SMALL_OR_SHORT_WORKLOAD_CONSERVATIVE_PROFILE'}
  with patch('multigpubfs.autotune._admit',side_effect=AssertionError('unexpected fast admission')),patch('multigpubfs.autotune.choose_profile',return_value=expected) as select:
   p=_choose_size_profile_uncached(self.graph(),[0,1],64,60,'native',{'MGBFS_GENERIC_HISTORY':'sorted'})
  self.assertEqual(p,expected);select.assert_called_once()
