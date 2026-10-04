import unittest
from scripts.benchmark_tail_modes import compare


class MatchedTimingTests(unittest.TestCase):
    def test_unequal_coverage_and_incomplete_pairs_cannot_inflate_speedup(self):
        baseline = dict(name='original-end', pairs={
            'a': dict(status='COMPLETE', runner_wall_seconds=10, production_search_seconds=4),
            'b': dict(status='COMPLETE', runner_wall_seconds=90, production_search_seconds=20),
            'c': dict(status='INCOMPLETE', runner_wall_seconds=50)})
        mode = dict(name='five-end', pairs={
            'a': dict(status='COMPLETE', runner_wall_seconds=5, production_search_seconds=4),
            'c': dict(status='COMPLETE', runner_wall_seconds=1),
            'd': dict(status='COMPLETE', runner_wall_seconds=100)})
        results = compare([mode, baseline])
        self.assertEqual([r['common_complete_pairs'] for r in results], [['a'], ['a']])
        self.assertEqual([r['speedup'] for r in results], [2, 1])

    def test_missing_rank_report_remains_unknown(self):
        rows = [dict(name=name, pairs={'a': dict(status='COMPLETE',
                runner_wall_seconds=1, production_search_seconds=None)})
                for name in ['original-end', 'five-graph']]
        report = compare(rows)
        self.assertIsNone(report[1]['speedup'])
        self.assertEqual(report[1]['common_complete_pairs'], [])


if __name__ == '__main__':
    unittest.main()
