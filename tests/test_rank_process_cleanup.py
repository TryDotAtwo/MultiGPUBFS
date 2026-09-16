"""Real Linux regression: torchrun-like ranks leave the launcher's session."""
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from eight_gpu_gate import run_command

@unittest.skipUnless(sys.platform == 'linux', 'requires Linux process sessions')
class RankProcessCleanup(unittest.TestCase):
    def check_rank(self, parent_exits):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); pidfile=root/'rank.pid'
            rank="import os,pathlib,time; pathlib.Path("+repr(str(pidfile))+").write_text(str(os.getpid())); time.sleep(60)"
            parent=("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(rank)+"],start_new_session=True); time.sleep("+('0.3' if parent_exits else '60')+")")
            rank_pid=None
            try:
                if parent_exits:
                    run_command([sys.executable,'-c',parent],root/'log',dict(os.environ),5)
                else:
                    with self.assertRaises(TimeoutError):
                        run_command([sys.executable,'-c',parent],root/'log',dict(os.environ),1)
                rank_pid=int(pidfile.read_text())
                # A zombie is also a leak: the supervisor must reap adopted ranks.
                self.assertFalse(Path(f'/proc/{rank_pid}').exists(), 'independent-session rank leaked')
            finally:
                if rank_pid is None and pidfile.exists(): rank_pid=int(pidfile.read_text())
                if rank_pid is not None:
                    try: os.kill(rank_pid,signal.SIGKILL)
                    except ProcessLookupError: pass

    def test_timeout_reaps_rank_in_its_own_session(self): self.check_rank(False)
    def test_exited_launcher_does_not_abandon_rank(self): self.check_rank(True)

if __name__=='__main__': unittest.main()
