import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("verify_owner", Path(__file__).parents[1] / "scripts/verify_bounded_owner_gate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SOURCE = "0c27d0a491cc733bc6d87d688de8adc396db880c"

class GateTests(unittest.TestCase):
    def test_state_gate_requires_its_own_test_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            s = self.fixture(p)
            (p / "summary.json").write_text(json.dumps(s))
            with self.assertRaises(ValueError): MODULE.verify(p, SOURCE, marker="STATE_COMMIT_PASS")
            for log in p.glob("*.log"):
                log.write_text(log.read_text().replace("BOUNDED_OWNER_PASS", "STATE_COMMIT_PASS"))
            self.assertEqual(MODULE.verify(p, SOURCE, marker="STATE_COMMIT_PASS"), "VERIFIED_STATE_COMMIT_GATE 10/10")
    def fixture(self, path):
        gpus = [dict(index=i, uuid=f"GPU-{i}", name="Tesla T4") for i in range(2)]
        summary = dict(source=SOURCE, status="COMPLETE", gpus=gpus, checks=[])
        for gpu in gpus:
            for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
                name = f"gpu{gpu['index']}-{tool}.log"
                text = "BOUNDED_OWNER_PASS\n"
                if tool == "racecheck":
                    text += "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
                elif tool != "plain":
                    text += "========= ERROR SUMMARY: 0 errors\n"
                (path / name).write_text(text)
                summary["checks"].append(dict(gpu=gpu["index"], uuid=gpu["uuid"], tool=tool, log=name))
        return summary

    def test_complete_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "summary.json").write_text(json.dumps(self.fixture(p)))
            self.assertEqual(MODULE.verify(p, SOURCE), "VERIFIED_BOUNDED_OWNER_GATE 10/10")

    def test_reject_incomplete_wrong_source_duplicate_device_or_tool(self):
        for fault in range(5):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as d:
                p = Path(d)
                s = self.fixture(p)
                if fault == 0: s["checks"].pop()
                elif fault == 1: s["source"] = "0" * 40
                elif fault == 2: s["gpus"][1]["uuid"] = "GPU-0"
                elif fault == 3: s["checks"][1] = s["checks"][0]
                else: s["checks"][0]["uuid"] = "GPU-wrong"
                (p / "summary.json").write_text(json.dumps(s))
                with self.assertRaises(ValueError): MODULE.verify(p, SOURCE)

    def test_reject_missing_test_marker_and_nonzero_final_sanitizer(self):
        for content in ("", "BOUNDED_OWNER_PASS\n", "BOUNDED_OWNER_PASS\n========= ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 1 errors\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as d:
                p = Path(d)
                s = self.fixture(p)
                (p / "summary.json").write_text(json.dumps(s))
                (p / "gpu0-memcheck.log").write_text(content)
                with self.assertRaises(ValueError): MODULE.verify(p, SOURCE)

if __name__ == "__main__":
    unittest.main()
