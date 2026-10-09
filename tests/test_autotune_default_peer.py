import unittest
from unittest.mock import patch
from multigpubfs import GraphDefinition
from multigpubfs.autotune import choose_profile
class Defaults(unittest.TestCase):
 def test_omitted_auto_normalized_without_mutating_input(self):
  g=GraphDefinition.permutation([[1,0]],[0,1]);seen=[]
  def admit(graph,devices,capacity,shards,native,env,directory):seen.append(dict(env));return {'devices':[0],'plan':{'capacity':2}}
  first={};second={'MGBFS_PEER_TRANSPORT':'auto'}
  with patch('multigpubfs.autotune._admit',side_effect=admit):
   choose_profile(g,[0],None,5,'native',first);choose_profile(g,[0],None,5,'native',second)
  self.assertEqual(seen[0],seen[1]);self.assertEqual(first,{});self.assertEqual(second,{'MGBFS_PEER_TRANSPORT':'auto'})
 def test_forced_peer_kept(self):
  g=GraphDefinition.permutation([[1,0]],[0,1])
  with patch('multigpubfs.autotune._admit',return_value={'devices':[0],'plan':{'capacity':2}}) as admit:
   choose_profile(g,[0],None,5,'native',{'MGBFS_PEER_TRANSPORT':'host'})
  self.assertEqual(admit.call_args.args[5]['MGBFS_PEER_TRANSPORT'],'host')
