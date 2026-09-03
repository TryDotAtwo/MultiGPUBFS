"""Statistics-contract fixtures, not GPU measurements."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('mode_report',Path(__file__).parents[1]/'scripts/mode_sweep_report.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class ModeStatistics(unittest.TestCase):
    def rows(self):
        return [dict(status='COMPLETE',repetition=i,search_seconds=t,
                     durable_run_commit_seconds=t+10,smi_process_peak_mib=100+i,
                     requested_device_bytes=80,pinned_bytes=200)
                for i,t in enumerate([1.,2.,3.,4.,20.])]
    def test_median_and_mad_are_not_best_run(self):
        s=module.summarize(self.rows())
        self.assertEqual((s['search_s'],s['mad_s'],s['durable_s'],s['peak_mib']), (3.,1.,13.,104))
    def test_failure_is_not_silently_discarded(self):
        rows=self.rows(); rows[-1]['status']='FAILED'
        s=module.summarize(rows)
        self.assertEqual(s['status'],'FAILED'); self.assertIsNone(s['search_s'])
    def test_missing_and_duplicate_repetitions_are_not_five_trials(self):
        for rows in (self.rows()[:4], self.rows()[:4]+[self.rows()[0]]):
            self.assertEqual(module.summarize(rows)['status'],'INCOMPLETE')
    def test_baseline_has_no_fake_durable_time(self):
        rows=self.rows()
        for r in rows: del r['durable_run_commit_seconds']
        self.assertIsNone(module.summarize(rows)['durable_s'])

if __name__=='__main__': unittest.main()
