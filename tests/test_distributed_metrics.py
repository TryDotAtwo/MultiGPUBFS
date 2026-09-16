import sys
import unittest
import tempfile
import json
import os
import subprocess
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from distributed_gpu_bench import smi_peaks, aggregate_rank_results, suite, stats, baseline_worker, run_group


class RankMetrics(unittest.TestCase):
    def test_launcher_collects_eight_rank_files_from_real_child_process(self):
        # Only the unavailable GPU sampler is substituted; launch, file IO,
        # process lifecycle and aggregation execute normally on the host.
        popen = subprocess.Popen
        class Sampler:
            def terminate(self): pass
            def wait(self): return 0
        def launch(command, **kwargs):
            if command[0] == 'stdbuf':
                kwargs['stdout'].write('\n'.join(
                    f't,{r},uuid,{100+r},0,0,0,0' for r in range(8)))
                kwargs['stdout'].flush()
                return Sampler()
            return popen(command, **kwargs)
        worker = (
            "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "[(p/f'rank-{r}.json').write_text(json.dumps(dict(rank=r,"
            "status='COMPLETE',backend='fixture',local_layer_sizes=[int(r==0),r],"
            "search_complete_seconds=r+1,durable_run_commit_seconds=r+2))) "
            "for r in range(8)]")
        with tempfile.TemporaryDirectory() as directory, patch(
                'distributed_gpu_bench.subprocess.Popen', side_effect=launch):
            row = run_group([sys.executable, '-c', worker, '{RANK_OUT}'],
                            Path(directory), 'eight',
                            dict(os.environ, MGBFS_BENCH_WORLD_SIZE='8'), timeout=10)
            self.assertEqual(row['status'], 'COMPLETE')
            self.assertEqual(row['layer_sizes'], [1, 28])
            self.assertEqual(row['smi_peak_mib_total'], 828)
            self.assertEqual(len(row['rank_results']), 8)

    def test_eight_rank_counts_timings_and_memory_include_last_rank(self):
        ranks = [dict(rank=r, status='COMPLETE', backend='native_test',
                      local_layer_sizes=[int(r == 0), r],
                      search_complete_seconds=r + 1, durable_run_commit_seconds=r + 2)
                 for r in reversed(range(8))]
        row = aggregate_rank_results(ranks, world=8)
        self.assertEqual(row['layer_sizes'], [1, 28])
        self.assertEqual(row['search_complete_seconds'], 8)
        self.assertEqual(row['durable_run_commit_seconds'], 9)
        text = '\n'.join(f't,{r},uuid,{100+r},0,0,0,0' for r in range(8))
        memory, total = smi_peaks(text, world=8)
        self.assertEqual(memory, [100,101,102,103,104,105,106,107])
        self.assertEqual(total, 828)
        row.update(smi_peak_mib_per_rank=memory, smi_peak_mib_total=total)
        self.assertEqual(stats([row])['peak_mib_total'], 828)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            aggregate_rank_results(ranks[:-1], world=8)

    def test_profiled_rows_cannot_enter_performance_statistics(self):
        row = dict(search_complete_seconds=1, durable_run_commit_seconds=2,
                   smi_peak_mib_per_rank=[100], smi_peak_mib_total=100,
                   profiled=True)
        with self.assertRaisesRegex(ValueError, 'PROFILED_MEASUREMENTS'):
            stats([row])

    def test_rank_durable_cannot_precede_its_search_even_if_global_max_hides_it(self):
        rows = [dict(rank=0, status='COMPLETE', backend='native_test',
                     local_layer_sizes=[1, 2], search_complete_seconds=10,
                     durable_run_commit_seconds=11),
                dict(rank=1, status='COMPLETE', backend='native_test',
                     local_layer_sizes=[0, 3], search_complete_seconds=3,
                     durable_run_commit_seconds=2)]
        with self.assertRaisesRegex(ValueError, 'durable precedes search'):
            aggregate_rank_results(rows)
        rows[1]['durable_run_commit_seconds'] = 3
        result = aggregate_rank_results(rows)
        self.assertEqual(result['search_complete_seconds'], 10)
        self.assertEqual(result['durable_run_commit_seconds'], 11)

    def test_invalid_capacity_mode_is_rejected_before_any_gpu_launch(self):
        for mode in ('', 'equal-global', 'auto'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with patch('distributed_gpu_bench.run_group', return_value=dict(status='COMPLETE')) as launch:
                    with self.assertRaisesRegex(ValueError, 'BENCH_CAPACITY_MODE'):
                        suite(Path('native'), Path('source'), Path(directory),
                              {'MGBFS_DIAGNOSTIC':'1', 'MGBFS_CAPACITY_MODE':mode})
                    launch.assert_not_called()

    def test_capacity_budget_semantics_are_explicit_for_one_and_two_ranks(self):
        for world, mode, global_records in ((1, 'equal_global', 40320),
                                             (2, 'equal_global', 40320),
                                             (2, 'max_per_rank', 80640)):
            with self.subTest(world=world, mode=mode), tempfile.TemporaryDirectory() as directory:
                def worker(command, out, label, env, timeout):
                    self.assertEqual(env['MGBFS_CAPACITY_MODE'], mode)
                    self.assertEqual(env['MGBFS_BENCH_CAPACITY'], '40320')
                    self.assertEqual(env['MGBFS_FUTURE_CAPACITY'], '40320')
                    return dict(status='COMPLETE')
                with patch('distributed_gpu_bench.run_group', worker):
                    report = suite(Path('native'), Path('source'), Path(directory),
                                   {'MGBFS_DIAGNOSTIC':'1', 'MGBFS_CAPACITY_MODE':mode,
                                    'MGBFS_BENCH_WORLD_SIZE':str(world)})
                self.assertEqual(report.get('capacity_mode'), mode)
                self.assertEqual(report['rows'][0].get('requested_global_capacity_records'), global_records)
                self.assertEqual(report['rows'][0].get('requested_global_state_ring_records'), global_records)

    def test_selected_archive_filesystem_is_used_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def worker(command, out, label, env, timeout):
                # GPU launch is the only substituted boundary. Paths and cleanup are real.
                self.assertEqual(Path(command[-2]).parent, root)
                self.assertEqual(Path(command[-3]).parent, root)
                Path(command[-2]+'-rank-0.mgbfsar1').write_bytes(b'fixture')
                return dict(status='COMPLETE')
            with patch('distributed_gpu_bench.run_group', worker):
                report = suite(Path('native'), Path('source'), root/'logs',
                               {'MGBFS_DIAGNOSTIC':'1','MGBFS_BENCH_WORLD_SIZE':'1',
                                'MGBFS_BENCH_ARCHIVE_DIR':str(root)})
            self.assertEqual(report['archive_directory'], str(root))
            self.assertFalse(list(root.glob('*.mgbfsar1')))
            self.assertTrue(all('free_bytes' in e['archive'] for e in report['disk_events']))

    def test_one_rank_baseline_uses_single_gpu_measurement_without_collectives(self):
        def measured(n, batch, validate):
            self.assertEqual((n, batch, validate), (4, 65536, False))
            return dict(status='COMPLETE', backend='cayleypy_single_matrix',
                        layer_sizes=[1, 23], search_complete_seconds=1.0)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'WORLD_SIZE':'1', 'RANK':'0', 'LOCAL_RANK':'0'}), \
             patch('symmetric_gpu_bench.baseline', measured):
            baseline_worker(4, 65536, directory)
            result = json.loads((Path(directory)/'rank-0.json').read_text())
        self.assertEqual(result['backend'], 'cayleypy_single_matrix')
        self.assertEqual(result['world_size'], 1)
        self.assertTrue(result['warmup_completed'])

    def test_one_gpu_panel_launches_one_rank_and_records_topology(self):
        launches = []
        def worker(command, out, label, env, timeout):
            launches.append(command)
            self.assertIn('--nproc-per-node=1', command)
            self.assertEqual(env['MGBFS_BENCH_WORLD_SIZE'], '1')
            if '--no-python' in command:
                self.assertEqual(env['MGBFS_RANK_MAP'], '0')
            return dict(status='COMPLETE', layer_sizes=[1, 3628799],
                        search_complete_seconds=1.0, smi_peak_mib_total=100,
                        smi_peak_mib_per_rank=[100])
        with tempfile.TemporaryDirectory() as directory, patch('distributed_gpu_bench.run_group', worker):
            report = suite(Path('native'), Path('source'), Path(directory),
                           {'MGBFS_PROFILE_SWEEP': '1', 'MGBFS_BENCH_WORLD_SIZE': '1'})
        self.assertEqual(len(launches), 68)
        self.assertEqual(report['world_size'], 1)
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(report['comparisons'][0]['native']['peak_mib_per_rank'], [100])

    def test_one_rank_counts_require_explicit_world_inventory(self):
        row = self.rows()[0]
        result = aggregate_rank_results([row], world=1)
        self.assertEqual(result['layer_sizes'], [1, 2])
        self.assertEqual(result['search_complete_seconds'], 2)
        self.assertEqual(result['world_size'], 1)
        row = dict(rank=0, status='COMPLETE', backend='cayleypy_single_gpu',
                   layer_sizes=[1, 5], search_complete_seconds=3)
        self.assertEqual(aggregate_rank_results([row], world=1)['layer_sizes'], [1, 5])
        with self.assertRaises(ValueError):
            aggregate_rank_results([row])  # missing rank 1 is not a single GPU run
        with self.assertRaises(ValueError):
            aggregate_rank_results([dict(row, world_size=2)], world=1)

    def test_profile_group_rejected_before_launch_when_full_capacity_exceeds_u32(self):
        for value in ("0", "1", "-1", "13", "1000000"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                with patch("distributed_gpu_bench.run_group", side_effect=AssertionError("unexpected launch")):
                    with self.assertRaisesRegex(ValueError, "PROFILE_GROUP_CAPACITY"):
                        suite(Path("native"), Path("source"), Path(directory),
                              {"MGBFS_PROFILE_SWEEP": "1", "MGBFS_PROFILE_SWEEP_N": value})

    def test_profile_suite_runs_requested_group_not_fixed_s10(self):
        launches = []
        def worker(command, out, label, env, timeout):
            launches.append(command)
            if "--no-python" in command:
                self.assertEqual(env["MGBFS_ARCHIVE_CODEC"], "permutation_u8")
                self.assertEqual(env["MGBFS_STATE_CODEC"], "matrix_u8")
            return dict(status="COMPLETE", layer_sizes=[1, 39916799],
                        search_complete_seconds=2.0, smi_peak_mib_total=200,
                        smi_peak_mib_per_rank=[100, 100])
        with tempfile.TemporaryDirectory() as directory, patch("distributed_gpu_bench.run_group", worker):
            report = suite(Path("native"), Path("source"), Path(directory),
                           {"MGBFS_PROFILE_SWEEP": "1", "MGBFS_PROFILE_SWEEP_N": "11",
                            "MGBFS_PROFILE_ARCHIVE_CODEC": "permutation_u8"})
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(len(launches), 68)
        self.assertTrue(all("s11" in cmd or "11" in cmd for cmd in launches))
        self.assertTrue(all(x["group"] == "s11" and x["unique_states"] == 39916800
                            for x in report["comparisons"]))

    def test_profile_suite_retains_all_modes_and_repetitions(self):
        # GPU processes are unavailable to this CPU orchestration test. The
        # boundary returns fixed measurements; real files/aggregation execute.
        def worker(command, out, label, env, timeout):
            return dict(status="COMPLETE", layer_sizes=[1, 3628799],
                        search_complete_seconds=1.0, smi_peak_mib_total=200,
                        smi_peak_mib_per_rank=[100, 100])
        with tempfile.TemporaryDirectory() as directory, patch("distributed_gpu_bench.run_group", worker):
            report = suite(Path("native"), Path("source"), Path(directory), {"MGBFS_PROFILE_SWEEP": "1"})
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(len(report["comparisons"]), 12)
        self.assertEqual(len(report["rows"]), 68)  # three baseline calibration + 13*5
        self.assertEqual(len(report['disk_events']), 136)
        self.assertEqual([x['stage'] for x in report['disk_events'][:2]], ['before', 'after_cleanup'])
        self.assertTrue(all('free_bytes' in x['output'] for x in report['disk_events']))
        self.assertTrue(all(x["native"]["repeats"] == 5 for x in report["comparisons"]))
        self.assertTrue(all(x["cayleypy"]["repeats"] == 5 for x in report["comparisons"]))

    def rows(self):
        return [dict(rank=r, status="COMPLETE", backend="native", search_complete_seconds=2+r,
                     durable_run_commit_seconds=4+r, local_layer_sizes=v)
                for r, v in enumerate(([1, 2], [0, 3]))]

    def test_native_counts_and_slowest_rank_times(self):
        result = aggregate_rank_results(list(reversed(self.rows())))
        self.assertEqual(result["layer_sizes"], [1, 5])
        self.assertEqual(result["search_complete_seconds"], 3)
        self.assertEqual(result["durable_run_commit_seconds"], 5)

    def test_mixed_rank_configuration_is_rejected(self):
        for key in ('group', 'batch', 'frontier_profile', 'owner_backend',
                    'pre_dedup', 'capacity_mode', 'global_capacity_records',
                    'global_state_ring_records', 'archive_enabled', 'archive_state_bytes',
                    'generation_variant', 'hash_first_generation', 'warmup_completed',
                    'library_pool_reserved_bytes'):
            rows = self.rows()
            rows[0][key] = 1
            rows[1][key] = 2
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate_rank_results(rows)
            del rows[1][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                aggregate_rank_results(rows)

    def test_layer_counts_are_nonnegative_integers_on_every_rank(self):
        for invalid in (-1, 0.5, True, float('nan')):
            rows = self.rows()
            rows[1]['local_layer_sizes'][0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                aggregate_rank_results(rows)

    def test_absent_baseline_archive_time_stays_unknown(self):
        rows = self.rows()
        for row in rows:
            del row["local_layer_sizes"]
            row.update(backend="cayleypy_torchrun", layer_sizes=[1, 5], durable_run_commit_seconds=None)
        result = aggregate_rank_results(rows)
        self.assertIsNone(result["durable_run_commit_seconds"])
        self.assertEqual(result["layer_sizes"], [1, 5])

    def test_invalid_rank_results_cannot_produce_success(self):
        for mutation in (
            lambda x: x[1].update(rank=0),
            lambda x: x[1].update(local_layer_sizes=[0]),
            lambda x: x[1].update(status="INCOMPLETE"),
            lambda x: x[1].update(backend="different"),
            lambda x: x[1].update(search_complete_seconds=float("nan")),
            lambda x: x[1].update(durable_run_commit_seconds=None),
        ):
            rows = self.rows()
            mutation(rows)
            with self.assertRaises(ValueError):
                aggregate_rank_results(rows)


class MemoryMetrics(unittest.TestCase):
    def test_invalid_search_or_memory_cannot_enter_comparison(self):
        row = dict(search_complete_seconds=1, smi_peak_mib_per_rank=[100],
                   smi_peak_mib_total=100)
        for invalid in (-1, float('nan'), float('inf'), True, '2'):
            for key in ('search_complete_seconds', 'smi_peak_mib_total',
                        'smi_peak_mib_per_rank'):
                value = [invalid] if key == 'smi_peak_mib_per_rank' else invalid
                with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                    stats([dict(row, **{key: value})])

    def test_durable_statistics_retain_every_sample(self):
        rows = [dict(search_complete_seconds=1, durable_run_commit_seconds=t,
                     smi_peak_mib_per_rank=[100], smi_peak_mib_total=100)
                for t in (2, 9, 4)]
        result = stats(rows)
        self.assertEqual(result.get('durable_median_seconds'), 4)
        self.assertEqual(result.get('durable_mad_seconds'), 2)
        self.assertEqual(result.get('durable_samples_seconds'), [2, 9, 4])
        for row in rows:
            row['durable_run_commit_seconds'] = None
        result = stats(rows)
        self.assertIsNone(result['durable_median_seconds'])
        self.assertIsNone(result['durable_mad_seconds'])
        self.assertEqual(result['durable_samples_seconds'], [None, None, None])

    def test_partial_or_invalid_durable_samples_are_rejected(self):
        row = dict(search_complete_seconds=1, durable_run_commit_seconds=2,
                   smi_peak_mib_per_rank=[100], smi_peak_mib_total=100)
        for invalid in (None, -1, float('nan'), float('inf'), True, '2'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                stats([row, dict(row, durable_run_commit_seconds=invalid)])

    def test_single_gpu_does_not_charge_unused_device(self):
        text = 't, 0, a, 500, 1, 1, 1, 1\nt, 1, b, 9000, 1, 1, 1, 1\n'
        self.assertEqual(smi_peaks(text, world=1), ([500.0], 500.0))
        self.assertEqual(smi_peaks('', world=1), ([None], None))
        rows = [dict(search_complete_seconds=t, smi_peak_mib_per_rank=[m],
                     smi_peak_mib_total=m) for t, m in [(1, 500), (3, 600)]]
        result = stats(rows)
        self.assertEqual(result['median_seconds'], 2)
        self.assertEqual(result['peak_mib_per_rank'], [600])
        with self.assertRaises(ValueError):
            stats(rows + [dict(rows[0], smi_peak_mib_per_rank=[500, 500])])

    def test_missing_rank_is_unknown_not_zero(self):
        self.assertEqual(smi_peaks(""), ([None, None], None))
        self.assertEqual(smi_peaks("t, 0, uuid, 500, 1, 1, 1, 1\n"), ([500.0, None], None))

    def test_both_rank_maxima_and_sum(self):
        text = "\n".join([
            "t, 0, a, 500, 1, 1, 1, 1",
            "t, 1, b, 400, 1, 1, 1, 1",
            "t, 0, a, 600, 1, 1, 1, 1",
            "t, 3, c, 999, 1, 1, 1, 1",
            "t, 1, b, N/A, 1, 1, 1, 1",
            "t, 1, b, nan, 1, 1, 1, 1",
        ])
        self.assertEqual(smi_peaks(text), ([600.0, 400.0], 1000.0))


if __name__ == "__main__":
    unittest.main()
