import unittest
from multigpubfs.nccl_preflight import reserve_bytes, alternatives
class ReserveTests(unittest.TestCase):
 def test_b200_measured_over_two_gib(self):
  self.assertEqual(reserve_bytes([dict(verified=True,nccl_device_bytes=3<<30)]), (3<<30)+(512<<20))
 def test_no_percentage_and_no_lowering_override(self):
  self.assertEqual(reserve_bytes([dict(verified=True,nccl_device_bytes=1)],'1'),2<<30)
 def test_unverified_rejected(self):
  with self.assertRaises(RuntimeError):reserve_bytes([dict(verified=False)])
 def test_explicit_transport_is_respected(self):
  self.assertEqual(list(alternatives({'NCCL_IB_DISABLE':'0'})),[('inherited',{})])
 def test_fallbacks_are_ordered(self):
  self.assertEqual([n for n,p in alternatives({})],['inherited','host-staged-ib','local-shm'])


class CoordinatorTests(unittest.TestCase):
 def test_failed_transport_is_not_accepted(self):
  import tempfile, json
  from pathlib import Path
  from unittest.mock import patch
  from multigpubfs.nccl_preflight import prepare
  class Failed:
   returncode=1
   def poll(self): return 1
   def wait(self,timeout=None): return 1
  with tempfile.TemporaryDirectory() as d, patch('multigpubfs.nccl_preflight._identity',return_value='test'), patch('multigpubfs.nccl_preflight.subprocess.Popen',return_value=Failed()):
   with self.assertRaisesRegex(RuntimeError,'NO_VERIFIED_TRANSPORT'):
    prepare('fake',[0,1],{},Path(d)/'run')
   r=json.loads((Path(d)/'run-communication-evidence/report.json').read_text())
   self.assertEqual(len(r['attempts']),3)
   self.assertEqual(r['status'],'FAILED')
 def test_successful_retry_requires_readback_and_records_failure(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from multigpubfs.nccl_preflight import prepare
  class Process:
   def __init__(self,code):self.returncode=code
   def poll(self):return self.returncode
   def wait(self,timeout=None):return self.returncode
  def start(args,**kwargs):
   root=Path(args[-2]);rank=int(args[-1]);failed=root.name=='inherited'
   if not failed:(root/f'rank-{rank}.json').write_text(json.dumps(dict(verified=True,rank=rank,device=rank,nccl_device_bytes=3<<30)))
   return Process(1 if failed else 0)
  with tempfile.TemporaryDirectory() as d,patch('multigpubfs.nccl_preflight._identity',return_value='test'),patch('multigpubfs.nccl_preflight.subprocess.Popen',side_effect=start):
   env,r=prepare('fake',[0,1],{},Path(d)/'run')
   self.assertEqual(r['selected'],'host-staged-ib')
   self.assertEqual(len(r['prior_attempts']),1)
   self.assertEqual(env['NCCL_NET_GDR_LEVEL'],'LOC')
   self.assertEqual(int(env['MGBFS_GENERIC_VRAM_RESERVE_BYTES']),(3<<30)+(512<<20))
 def test_explicit_transport_failure_does_not_override_operator(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from multigpubfs.nccl_preflight import prepare
  class Failed:
   returncode=1
   def poll(self):return 1
   def wait(self,timeout=None):return 1
  with tempfile.TemporaryDirectory() as d,patch('multigpubfs.nccl_preflight._identity',return_value='test'),patch('multigpubfs.nccl_preflight.subprocess.Popen',return_value=Failed()):
   with self.assertRaisesRegex(RuntimeError,'NO_VERIFIED_TRANSPORT'):prepare('fake',[0,1],{'NCCL_IB_DISABLE':'0'},Path(d)/'run')
   r=json.loads((Path(d)/'run-communication-evidence/report.json').read_text())
   self.assertEqual(len(r['attempts']),1)
 def test_all_rank_memory_must_be_nonnegative(self):
  with self.assertRaises(ValueError):reserve_bytes([dict(verified=True,nccl_device_bytes=-1),dict(verified=True,nccl_device_bytes=100)])

 def test_exchange_success_does_not_mask_bfs_prefix_failure(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from multigpubfs.nccl_preflight import prepare
  class Graph:
   def to_json(self):return '{}'
   def digest(self):return 'fixture'
  class Process:
   def __init__(self,code):self.returncode=code
   def poll(self):return self.returncode
   def wait(self,timeout=None):return self.returncode
  def start(args,**kwargs):
   root=Path(args[-2])
   if args[-1]=='--prefix':return Process(1)
   rank=int(args[-1]);(root/f'rank-{rank}.json').write_text(json.dumps(dict(verified=True,rank=rank,device=rank,nccl_device_bytes=0)))
   return Process(0)
  with tempfile.TemporaryDirectory() as d,patch('multigpubfs.nccl_preflight._identity',return_value='test'),patch('multigpubfs.nccl_preflight.subprocess.Popen',side_effect=start):
   with self.assertRaisesRegex(RuntimeError,'NO_VERIFIED_TRANSPORT'):prepare('fake',[0,1],{},Path(d)/'run',Graph())
   r=json.loads((Path(d)/'run-communication-evidence/report.json').read_text())
   self.assertEqual(len(r['attempts']),3)
   self.assertTrue(all(v['error']=='NCCL_PREFLIGHT_BFS_PREFIX_FAILED' for v in r['attempts']))
