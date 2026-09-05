import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from distributed_gpu_bench import smi_peaks, aggregate_rank_results


class RankMetrics(unittest.TestCase):
    def rows(self):
        return [dict(rank=r, status="COMPLETE", backend="native", search_complete_seconds=2+r,
                     durable_run_commit_seconds=4+r, local_layer_sizes=v)
                for r, v in enumerate(([1, 2], [0, 3]))]

    def test_native_counts_and_slowest_rank_times(self):
        result = aggregate_rank_results(list(reversed(self.rows())))
        self.assertEqual(result["layer_sizes"], [1, 5])
        self.assertEqual(result["search_complete_seconds"], 3)
        self.assertEqual(result["durable_run_commit_seconds"], 5)

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
