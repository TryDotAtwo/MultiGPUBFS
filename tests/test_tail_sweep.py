import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sweep_tail_bfs import execute,pairs


class SweepTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(pairs(3,4,2,3),[(3,2),(3,3),(4,2),(4,3)])
        with self.assertRaises(ValueError): pairs(4,3)

    def test_resume_and_failure(self):
        calls=[]
        def fake(config,source,case,runtime):
            calls.append((config['n'],config['r']))
            if config['r']==2: raise RuntimeError('capacity')
            case.mkdir()
            path=case/'manifest.json'
            path.write_text(json.dumps(dict(status='COMPLETE',last_completed_layer=4,stop_reason='exhausted')))
            return path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base={'run_id':'unit'};grid=pairs(3,3,1,2)
            ledger=execute(base,root,root,{},grid,10,fake)
            self.assertEqual(ledger['cases']['n3-m2']['status'],'INCOMPLETE')
            self.assertEqual(ledger['pending'],[])
            execute(base,root,root,{},grid,10,fake)
            self.assertEqual(calls,[(3,1),(3,2)])
            with self.assertRaises(ValueError): execute({'run_id':'changed'},root,root,{},grid,10,fake)

    def test_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=execute({},Path(tmp),Path(tmp),{},[(32,1)],10,
                           lambda *args: self.fail('unsupported case executed'))
            self.assertFalse(ledger['cases']['n32-m1']['attempted'])

if __name__=='__main__': unittest.main()
