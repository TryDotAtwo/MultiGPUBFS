import sys
import unittest
import threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_auto_tail import device_budget,pair_config,SweepPublisher


class AutomaticPlanningTests(unittest.TestCase):
    def test_uses_smallest_available_gpu_and_retains_headroom(self):
        p=device_budget([{'free_bytes':12<<30},{'free_bytes':8<<30}])
        self.assertEqual(p['free_bytes'],8<<30)
        self.assertLess(p['usable_bytes'],8<<30)
        self.assertLessEqual(p['library_pool_bytes']+2048*p['max_rows_per_rank'],p['usable_bytes'])

    def test_pairs_resize_without_mutating_persisted_budget(self):
        base=dict(resource_plan=device_budget([{'free_bytes':12<<30}]),env={},batch=123)
        small=pair_config(base,4,3);large=pair_config(base,16,4)
        self.assertEqual(small['env']['MGBFS_BENCH_CAPACITY'],'256')
        self.assertEqual(int(large['env']['MGBFS_BENCH_CAPACITY']),base['resource_plan']['max_rows_per_rank'])
        self.assertEqual(base['env'],{})
        self.assertEqual(int(large['env']['MGBFS_FUTURE_CAPACITY']),2*base['resource_plan']['max_rows_per_rank'])

    def test_unavailable_hardware_rejected(self):
        for inventory in ([],[{'free_bytes':0}],[{'free_bytes':128<<20}]):
            with self.assertRaises(ValueError):device_budget(inventory)

    def test_background_keeps_inflight_and_freezes_latest_queued_ledger(self):
        started=threading.Event();release=threading.Event();seen=[]
        def publish(root,repo,api,deadline,*,ledger):
            seen.append(ledger)
            if len(seen)==1:
                started.set()
                if not release.wait(5):raise TimeoutError('test worker gate')
            return {'receipt':'ok'}
        base=dict(configuration=dict(grid=[[2,1],[2,2],[3,1]]),cases={})
        worker=SweepPublisher(None,None,None,None,publish)
        worker.enqueue(base);self.assertTrue(started.wait(5))
        base['cases']['n2-m1']=dict(attempted=True)
        worker.enqueue(base)
        base['cases']['n2-m2']=dict(attempted=True)
        worker.enqueue(base)
        base['cases']['n3-m1']=dict(attempted=True)
        release.set();worker.finish()
        self.assertEqual(len(seen),2)
        self.assertEqual(seen[0]['cases'],{})
        self.assertEqual(set(seen[1]['cases']),{'n2-m1','n2-m2'})
        self.assertEqual(seen[1]['pending'],[[3,1]])


if __name__=='__main__':unittest.main()
