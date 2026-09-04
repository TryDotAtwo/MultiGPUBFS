import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PrepareRunSummary(unittest.TestCase):
    def test_derives_symmetric_group_metadata_without_s10_constants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw.json"
            raw.write_text(json.dumps({
                "status": "COMPLETE", "group": "s8", "batch": 1024,
                "macro_depth": 1, "macro_move_count": 3, "prededup": True,
                "unique_states": 40320, "layer_sizes": [1, 7, 40312],
                "per_depth_seconds": [0.1, 0.2, 0.3],
                "search_complete_seconds": 0.6, "durable_run_commit_seconds": 1.0,
            }), encoding="utf-8")
            output = root / "summary.json"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/prepare_hf_run_summary.py"),
                str(raw), str(output), "--source-commit", "a" * 40,
                "--hardware", "Tesla T4", "--run-id", "s8-native-k1-seed-20260828",
            ], check=True)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["run_id"], "s8-native-k1-seed-20260828")
            self.assertEqual(summary["group_id"], "S_8 cycle-inverse-transposition matrix Cayley graph over F2")
            self.assertEqual(summary["config"]["state_bytes"], 64)
            self.assertEqual(summary["layers"]["1"]["generated_candidates"], 21)

    def test_rejects_unknown_group_or_inconsistent_macro_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw.json"
            raw.write_text(json.dumps({"status": "COMPLETE", "group": "x", "unique_states": 1,
                                       "layer_sizes": [1], "macro_move_count": 0}), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/prepare_hf_run_summary.py"),
                str(raw), str(root / "out"), "--source-commit", "a" * 40,
                "--hardware", "Tesla T4", "--run-id", "x",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
