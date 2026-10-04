"""Mapping-order contract only; full two-GPU sanitizer gates remain required."""
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'patches/nccl-2.29.7-local-first-map.patch'

class LocalFirstMapping(unittest.TestCase):
    def test_pinned_patch_is_mapping_order_only(self):
        data = PATCH.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
            '95a042db1698504182c0cf0ee34c8c74dbaf5cb376d1aeeae93d464b0378b5b1')
        text = data.decode()
        self.assertEqual(text.count('@@'), 2)
        changed = [line[1:] for line in text.splitlines()
                   if line.startswith('+') and not line.startswith('+++')]
        self.assertEqual(len(changed), 4)
        self.assertNotRegex('\n'.join(changed), r'Synchronize|Memset|GetProcAddress|barrierCount|resourceRequirements')

    def test_compiled_actual_patch_loop_maps_each_rank_once_local_first(self):
        compiler = shutil.which('g++')
        if not compiler:
            self.fail('mapping-order gate requires g++')
        added = '\n'.join(line[1:] for line in PATCH.read_text().splitlines()
                          if line.startswith('+') and not line.startswith('+++'))
        loop = re.search(r'for \(int order[^\n]+\n\s+const int r[^\n]+;', added)
        self.assertIsNotNone(loop)
        code = '#include <vector>\nstruct S {int lsaSelf,lsaSize;};\nint main(){ for(int n=1;n<=128;n++) for(int self=0;self<n;self++){ S state{self,n}; S* devr=&state; std::vector<int> seen(n); int first=-1; '+loop.group()+' if(first<0) first=r; if(r<0||r>=n||seen[r]++) return 1; } if(first!=self) return 2; for(int count:seen) if(count!=1) return 3; } return 0;}\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'loop.cc').write_text(code)
            subprocess.run([compiler,'-std=c++17','-Wall','-Werror',str(path/'loop.cc'),'-o',str(path/'loop')],check=True)
            subprocess.run([str(path/'loop')],check=True)

    def test_gate_build_applies_and_attests_the_patch(self):
        text = (ROOT/'kaggle/lsa-bfs-gate/kernel.py').read_text()
        self.assertIn('NCCL_LOCAL_FIRST_PATCH_DIGEST_MISMATCH',text)
        self.assertIn('local_first_map_patch_sha256=local_first_digest',text)
        self.assertIn('"nccl-local-first-check", cwd=vendor',text)
        self.assertIn('"nccl-local-first-patch", cwd=vendor',text)

if __name__ == '__main__':
    unittest.main()
