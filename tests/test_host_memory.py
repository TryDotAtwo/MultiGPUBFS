import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from multigpubfs.host_memory import inventory,admit
from multigpubfs.graph_definition import GraphDefinition
class HostMemoryTests(unittest.TestCase):
 def fixture(self,v2=True):
  tmp=tempfile.TemporaryDirectory();p=Path(tmp.name);(p/'proc/self').mkdir(parents=True);m=p/'mount';(m/'child').mkdir(parents=True)
  (p/'proc/meminfo').write_text('MemAvailable: 1048576 kB\n')
  (p/'proc/self/cgroup').write_text('0::/root/child\n' if v2 else '3:memory:/root/child\n')
  (p/'proc/self/mountinfo').write_text('1 0 0:1 /root '+str(m)+' rw - '+('cgroup2 cgroup rw' if v2 else 'cgroup cgroup rw,memory')+'\n')
  return tmp,p,m
 def test_nested_v2_parent_limit(self):
  tmp,p,m=self.fixture()
  with tmp:
   (m/'child/memory.max').write_text('max');(m/'child/memory.current').write_text('10')
   (m/'memory.max').write_text('500');(m/'memory.current').write_text('120')
   self.assertEqual(inventory(p/'proc')['available_bytes'],380)
 def test_v1_unlimited_and_child(self):
  tmp,p,m=self.fixture(False)
  with tmp:
   (m/'memory.limit_in_bytes').write_text(str(1<<62));(m/'memory.usage_in_bytes').write_text('0')
   (m/'child/memory.limit_in_bytes').write_text('100');(m/'child/memory.usage_in_bytes').write_text('101')
   self.assertEqual(inventory(p/'proc')['available_bytes'],0)
 def test_unknown(self):
  with tempfile.TemporaryDirectory() as p:self.assertIsNone(inventory(Path(p))['available_bytes'])
 def test_insufficient_memory_rejected(self):
  graph=GraphDefinition.permutation([[0]],[0])
  with patch('multigpubfs.host_memory.inventory',return_value={'available_bytes':100,'sources':[]}):
   with self.assertRaisesRegex(RuntimeError,'HOST_MEMORY_ADMISSION_FAILED'):admit(graph)
if __name__=='__main__':unittest.main()
