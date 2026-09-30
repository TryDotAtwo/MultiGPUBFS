"""Real subreaper exit contract; requires Linux, not mocked process ownership."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(sys.platform == 'linux', 'Linux subreaper required')
class ProcessScopeLinuxTests(unittest.TestCase):
    def run_supervised(self, program):
        script = Path(__file__).resolve().parents[1] / 'scripts/process_scope.py'
        return subprocess.run([sys.executable, str(script), '--', sys.executable,
                               '-c', program], capture_output=True, text=True,
                              timeout=20)

    def test_normal_failure_preserves_exit_without_forced_cleanup(self):
        result = self.run_supervised('raise SystemExit(7)')
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertNotIn('PROCESS_SCOPE_FORCED_CLEANUP', result.stderr)

    def test_live_orphan_is_reaped_but_not_reported_as_success(self):
        result = self.run_supervised(
            'import subprocess,sys; '
            'p=subprocess.Popen([sys.executable,"-c",'
            '"import time; time.sleep(30)"],start_new_session=True); '
            'print("ORPHAN_PID",p.pid,flush=True)')
        self.assertEqual(result.returncode, 99, result.stderr)
        self.assertIn('PROCESS_SCOPE_FORCED_CLEANUP', result.stderr)
        pid = int(result.stdout.strip().split()[-1])
        self.assertFalse(Path(f'/proc/{pid}').exists(), result.stdout)


if __name__ == '__main__':
    unittest.main()
