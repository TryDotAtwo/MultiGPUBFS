"""Linux subreaper for launchers whose ranks create independent sessions.

Only descendants of this dedicated supervisor are adopted/signalled. No global
process-name matching, PID files, or unrelated process-group cleanup.
"""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def spawn_group(command, **kwargs):
    if sys.platform == 'linux':
        command = [sys.executable, str(Path(__file__).resolve()), '--', *command]
    return subprocess.Popen(command, start_new_session=True, **kwargs)


def stop_group(process):
    if process.poll() is None:
        try:
            if os.name == 'posix': os.killpg(process.pid, signal.SIGTERM)
            else: process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                if os.name == 'posix': os.killpg(process.pid, signal.SIGKILL)
                else: process.kill()
            except ProcessLookupError:
                pass
    process.wait()


def supervise(command):
    # PR_SET_CHILD_SUBREAPER: orphaned grandchildren become our children even
    # when they called setsid(), as torchrun --no-python ranks do.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = subprocess.Popen(command, start_new_session=True)
    code = 143
    try:
        while not stopping:
            result = child.poll()
            if result is not None:
                code = result if result >= 0 else 128-result
                break
            time.sleep(0.01)
    finally:
        children_file = Path(f'/proc/self/task/{os.getpid()}/children')
        deadline = time.monotonic()+10
        while True:
            children = [int(pid) for pid in children_file.read_text().split()]
            if not children:
                break
            for pid in children:
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
            while True:
                try:
                    if os.waitpid(-1, os.WNOHANG)[0] == 0: break
                except ChildProcessError:
                    break
            if time.monotonic() >= deadline:
                print('PROCESS_SCOPE_CLEANUP_FAILED', file=sys.stderr, flush=True)
                code = 98
                break
            time.sleep(0.01)
    return code


if __name__ == '__main__':
    if sys.platform != 'linux' or len(sys.argv)<3 or sys.argv[1]!='--':
        raise SystemExit('process_scope.py -- command ... requires Linux')
    raise SystemExit(supervise(sys.argv[2:]))
