import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from graph_profile_selection import select_graph_profile


def measurements(graph_seconds):
    return [dict(configuration_identity='same-geometry-and-binary', pair=p,
                 graph_batches=g, search_seconds=graph_seconds if g else 1.,
                 status='COMPLETE', full_state_parity=True,
                 full_windows_per_rank=[25, 25] if g else [0, 0])
            for p in range(3) for g in (0, 32)]


class SelectionTests(unittest.TestCase):
    def test_gain_and_slowdown(self):
        self.assertEqual(select_graph_profile(measurements(.9552))['graph_batches'],32)
        self.assertEqual(select_graph_profile(measurements(1.0251))['graph_batches'],0)
        self.assertEqual(select_graph_profile(measurements(.995))['graph_batches'],0)

    def test_no_extrapolation_or_unverified_success(self):
        for field, value in [('configuration_identity','other-batch'),
                             ('status','INCOMPLETE'),('full_state_parity',False),
                             ('search_seconds',float('nan')),
                             ('full_windows_per_rank',[25,0])]:
            samples=measurements(.95);samples[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                select_graph_profile(samples)

    def test_pairing_and_no_mutation(self):
        samples=measurements(.95);original=copy.deepcopy(samples)
        select_graph_profile(samples);self.assertEqual(samples,original)
        samples[1]['pair']=1
        with self.assertRaises(ValueError):select_graph_profile(samples)
        with self.assertRaises(ValueError):select_graph_profile(original[:-1])

    def test_one_regression_keeps_direct_launch(self):
        samples=measurements(.9);samples[1]['search_seconds']=1.01
        self.assertEqual(select_graph_profile(samples)['graph_batches'],0)


if __name__=='__main__':unittest.main()
