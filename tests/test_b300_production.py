import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from b300_production import inventory
class AdmissionTests(unittest.TestCase):
    def rows(self,n=8):
        return '\n'.join(f'{i}, NVIDIA B300, GPU-{i}, 288000, 280000, 10.3' for i in range(n))
    def test_eight_distinct_b300(self):
        self.assertEqual(len(inventory(self.rows(),8)),8)
    def test_wrong_hardware_and_count_fail(self):
        for text in (self.rows(1),self.rows().replace('B300','H200'),self.rows().replace('10.3','10.0'),self.rows().replace('GPU-7','GPU-0'),self.rows().replace('280000','0')):
            with self.assertRaises(ValueError):inventory(text,8)
    def test_single_b300_staging_is_explicit(self):
        self.assertEqual(len(inventory(self.rows(1),1)),1)
if __name__=='__main__':unittest.main()
