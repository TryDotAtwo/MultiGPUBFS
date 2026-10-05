import unittest
from scripts.summarize_sweep_timing import summarize

class TimingTests(unittest.TestCase):
    def test_both_seeds_and_failed_prefix_have_distinct_scopes(self):
        cases={
            'gpu':dict(attempted=True,status='COMPLETE',timing=dict(execution_backend='GPU',runner_wall_seconds=25,transition_before_seconds=1),comparison=dict(runs=[dict(production_search_seconds=10),dict(production_search_seconds=12)])),
            'failed':dict(attempted=True,status='INCOMPLETE',timing=dict(execution_backend='GPU',runner_wall_seconds=7,completed_layer_seconds=3,transition_before_seconds=2)),
            'skip':dict(attempted=False,status='INCOMPLETE'),
            'cpu':dict(attempted=True,status='COMPLETE',timing=dict(execution_backend='CPU_EXACT_PACKED',runner_wall_seconds=4),comparison=dict(runs=[dict(production_search_seconds=1),dict(production_search_seconds=1.5)]))}
        result=summarize(dict(cases=cases));gpu=result['by_backend']['GPU']
        self.assertEqual(result['attempted_pairs'],3)
        self.assertEqual(result['skipped_pairs'],1)
        self.assertEqual(gpu['runner_seconds'],32)
        self.assertEqual(gpu['complete_search_seconds'],22)
        self.assertEqual(gpu['complete_nonsearch_runner_seconds'],3)
        self.assertEqual(gpu['incomplete_committed_layer_seconds'],3)
        self.assertEqual(gpu['transition_seconds'],3)
        self.assertEqual(result['by_backend']['CPU_EXACT_PACKED']['complete_search_seconds'],2.5)

    def test_missing_second_seed_time_does_not_become_zero(self):
        result=summarize(dict(cases={'x':dict(attempted=True,status='COMPLETE',timing=dict(runner_wall_seconds=4),comparison=dict(runs=[dict(production_search_seconds=1),dict(production_search_seconds=None)]))}))
        group=result['by_backend']['UNKNOWN']
        self.assertEqual(group['complete_pairs_with_missing_search_timing'],1)
        self.assertEqual(group['timed_complete_pairs'],0)
        self.assertEqual(group['complete_runner_seconds'],0)

if __name__=='__main__':unittest.main()
