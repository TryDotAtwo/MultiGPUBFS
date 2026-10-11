import unittest
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import generic_session as g
class Transaction(unittest.TestCase):
 def fixture(self):
  worker=SimpleNamespace(sequence=10,process=SimpleNamespace(pid=100))
  pool=SimpleNamespace(generation=1,identity=('runtime',),workers=[worker],select=lambda *args:None)
  graph=SimpleNamespace(to_json=lambda:'exact-graph')
  admission={'devices':[0],'plan':{'batch':8}}
  return pool,graph,admission
 def test_one_shot(self):
  p,graph,a=self.fixture()
  with patch.object(g,'_active',p):
   g.remember_launch_admission(graph,'/bin/true',{},[0],100,1,8,a)
   self.assertEqual(g.take_launch_admission(graph,'/bin/true',{},[0],100,1,8),a)
   self.assertIsNone(g.take_launch_admission(graph,'/bin/true',{},[0],100,1,8))
 def test_intervening_job_release_generation_and_options(self):
  for mutation in ('sequence','generation','env','graph'):
   p,graph,a=self.fixture()
   with patch.object(g,'_active',p):
    g.remember_launch_admission(graph,'/bin/true',{},[0],100,1,8,a)
    env={}
    if mutation=='sequence':p.workers[0].sequence+=1
    if mutation=='generation':p.generation+=1
    if mutation=='env':env={'MGBFS_GENERIC_HISTORY':'sorted'}
    if mutation=='graph':graph=SimpleNamespace(to_json=lambda:'different-graph')
    self.assertIsNone(g.take_launch_admission(graph,'/bin/true',env,[0],100,1,8))
 def test_failure_propagates_and_consumes_ticket(self):
  p,graph,a=self.fixture()
  with patch.object(g,'_active',p):
   g.remember_launch_admission(graph,'/bin/true',{},[0],100,1,8,a)
   def fail(*args):raise RuntimeError('failed rank')
   p.select=fail
   with self.assertRaisesRegex(RuntimeError,'failed rank'):g.take_launch_admission(graph,'/bin/true',{},[0],100,1,8)
   self.assertIsNone(p.launch_admission)
