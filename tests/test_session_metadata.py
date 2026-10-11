import hashlib,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from multigpubfs import autotune as a,generic_session as g
class Metadata(unittest.TestCase):
 def test_generation_and_visibility_invalidate(self):
  pool=SimpleNamespace(generation=1)
  cmd=['nvidia-smi','topo','-m']
  with patch.object(g,'_active',pool),patch.object(a.subprocess,'run',return_value=subprocess.CompletedProcess(cmd,0,'topology','')) as run:
   a._system_info(cmd);a._system_info(cmd);self.assertEqual(run.call_count,1)
   pool.generation=2;a._system_info(cmd);self.assertEqual(run.call_count,2)
   with patch.dict(a.os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):a._system_info(cmd)
   self.assertEqual(run.call_count,3)
 def test_free_memory_is_never_cached(self):
  cmd=['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader']
  with patch.object(g,'_active',SimpleNamespace(generation=1)),patch.object(a.subprocess,'run',return_value=subprocess.CompletedProcess(cmd,0,'100','')) as run:
   a._system_info(cmd);a._system_info(cmd);self.assertEqual(run.call_count,2)
 def test_failures_are_not_cached(self):
  cmd=['nvidia-smi','topo','-m']
  with patch.object(g,'_active',SimpleNamespace(generation=1)),patch.object(a.subprocess,'run',return_value=subprocess.CompletedProcess(cmd,1,'','error')) as run:
   a._system_info(cmd);a._system_info(cmd);self.assertEqual(run.call_count,2)
 def test_outside_session_is_live(self):
  cmd=['nvidia-smi','topo','-m']
  with patch.object(g,'_active',None),patch.object(a.subprocess,'run',return_value=subprocess.CompletedProcess(cmd,0,'ok','')) as run:
   a._system_info(cmd);a._system_info(cmd);self.assertEqual(run.call_count,2)
 def test_digest_invalidation_and_order(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,'_active',SimpleNamespace(generation=1)):
   p=Path(d)/'a';q=Path(d)/'b';p.write_bytes(b'a');q.write_bytes(b'b')
   self.assertEqual(a._content_digest(p,q),hashlib.sha256(b'ab').hexdigest())
   self.assertNotEqual(a._content_digest(q,p),a._content_digest(p,q))
   p.write_bytes(b'changed');self.assertEqual(a._content_digest(p,q),hashlib.sha256(b'changedb').hexdigest())
