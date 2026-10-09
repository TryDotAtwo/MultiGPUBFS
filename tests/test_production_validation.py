import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paired_tail import run_pair
from sweep_tail_bfs import execute, resource_stop
from tail_validation import validation_summary
from run_tail_bfs import launch_environment


class ProductionValidationTests(unittest.TestCase):
    def test_native_environment_removes_pool_from_all_sources(self):
        with patch.dict(os.environ, {'MGBFS_LIBRARY_POOL_BYTES': '1'}):
            for config, runtime in [({'env': {'MGBFS_OWNER_BACKEND': 'SHARD_AB'}}, {}),
                                    ({'env': {'MGBFS_OWNER_BACKEND': 'SHARD_AB', 'MGBFS_LIBRARY_POOL_BYTES': '3'}},
                                     {'MGBFS_LIBRARY_POOL_BYTES': '2'})]:
                self.assertNotIn('MGBFS_LIBRARY_POOL_BYTES', launch_environment(config, runtime))
            self.assertEqual(launch_environment({'env': {'MGBFS_OWNER_BACKEND': 'CUCO_RANK'}}, {})['MGBFS_LIBRARY_POOL_BYTES'], '1')

    def test_source_confirmed_capacity_only(self):
        for reason in ['SHARD_AB_PREPARE_FATAL_11', 'SHARD_AB_PREPARE_FATAL_12',
                       'SHARD_AB_PREPARE_FATAL_16', 'SHARD_AB_PREPARE_FATAL_112',
                       'GROUP_STATE_RING_RETIRE_FATAL_112']:
            self.assertTrue(resource_stop(dict(status='INCOMPLETE', attempted=True, reason=reason)))
        for reason in ['SHARD_AB_PREPARE_FATAL_2', 'SHARD_AB_PREPARE_FATAL_1120',
                       'GROUP_STATE_RING_RETIRE_FATAL_7', 'SSH timeout', 'SIGTERM', 'CUDA_STATUS_2']:
            self.assertFalse(resource_stop(dict(status='INCOMPLETE', attempted=True, reason=reason)))

    def test_pair_ledger_and_report_gate(self):
        for mode in ['positive', 'counts', 'status', 'seed', 'missing_seed', 'empty', 'exception']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def native(cfg, source, case, runtime):
                    (case / 'saved').mkdir(parents=True)
                    (case / 'result').mkdir()
                    m = dict(graph={'n': 4}, packing={'width': 1}, status='COMPLETE',
                             last_completed_layer=1, stop_reason='exhausted',
                             layers=[dict(depth=0, states=1, seconds=0.1), dict(depth=1, states=3, seconds=0.2)])
                    if cfg['repetition'] == 2:
                        if mode == 'counts': m['layers'][1]['states'] = 4
                        if mode == 'status': m['status'] = 'INCOMPLETE'
                        if mode == 'empty': m['layers'] = []
                    path = case / 'saved/manifest.json'
                    path.write_text(json.dumps(m))
                    seed = cfg['env']['MGBFS_HASH_SEED_HEX']
                    if mode == 'seed' and cfg['repetition'] == 2: seed = 'wrong'
                    if mode != 'missing_seed' or cfg['repetition'] != 2:
                        (case / 'result/rank-0.json').write_text(json.dumps(dict(hash_seed_hex=seed)))
                    if mode == 'exception' and cfg['repetition'] == 2:
                        raise RuntimeError('archive finalization failed')
                    return path
                def paired(cfg, source, case, runtime):
                    return run_pair(cfg, source, case, runtime, native, None)
                ledger = execute(dict(world=1, run_id='gate', two_seeds=True), root, root, {}, [(4, 1)], 10, paired)
                record = ledger['cases']['n4-m1']
                summary = validation_summary(ledger)
                self.assertEqual(record['status'], 'COMPLETE')
                self.assertEqual(summary['search_complete'], 1)
                self.assertEqual(summary['complete'], int(mode == 'positive'))
                self.assertEqual(record['validation_status'], 'VERIFIED_LAYER_COUNTS' if mode == 'positive' else 'FAILED')
                if mode == 'exception':
                    self.assertEqual(record['replicas'][0]['runner_error']['message'], 'archive finalization failed')

    def test_direct_runner_exception_retains_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def runner(cfg, source, case, runtime):
                (case / 'saved').mkdir(parents=True)
                (case / 'saved/manifest.json').write_text(json.dumps(dict(status='COMPLETE', last_completed_layer=7, stop_reason='exhausted')))
                raise OSError('publication unavailable')
            ledger = execute({}, root, root, {}, [(4, 1)], 10, runner)
            record = ledger['cases']['n4-m1']
            self.assertEqual(record['last_completed_layer'], 7)
            self.assertEqual(record['search_status'], 'COMPLETE')
            self.assertEqual(record['runner_error']['type'], 'OSError')
            self.assertEqual(record['validation_status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
