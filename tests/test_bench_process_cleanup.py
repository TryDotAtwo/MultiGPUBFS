"""Real POSIX descendants: killing only torchrun must fail this test."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import distributed_gpu_bench as bench


@unittest.skipUnless(os.name == "posix", "Linux process-group contract")
class ProcessCleanup(unittest.TestCase):
    def test_timeout_stops_child_rank_not_only_launcher(self):
        self.check_cleanup("time.sleep(300)", "TIMEOUT")

    def test_failed_launcher_does_not_leave_child_rank(self):
        self.check_cleanup("sys.exit(9)", "FAILED")

    def check_cleanup(self, finish, expected_status):
        real_popen = subprocess.Popen
        # Replace only the unavailable GPU telemetry executable; processes,
        # process groups, timeout, logs and result serialization remain real.
        def launch(command, **kwargs):
            if command[0] == "stdbuf":
                command = [sys.executable, "-c", "import time; time.sleep(300)"]
            return real_popen(command, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "child.pid"
            program = (
                "import subprocess,sys,time,pathlib; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); " + finish
            )
            try:
                with patch.object(bench.subprocess, "Popen", launch):
                    result = bench.run_group(
                        [sys.executable, "-c", program, str(pid_file)], root,
                        "timeout", dict(os.environ, MGBFS_BENCH_WORLD_SIZE="1"), timeout=1)
                self.assertEqual(result["status"], expected_status)
                self.assertTrue(pid_file.exists(), "child must have started")
                stat = Path("/proc") / pid_file.read_text() / "stat"
                state = stat.read_text().split(") ", 1)[1].split()[0] if stat.exists() else None
                self.assertIn(state, (None, "Z"), "child rank still running after timeout")
            finally:
                if pid_file.exists():
                    try:
                        os.kill(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
