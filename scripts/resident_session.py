"""One torchrun rank group, sequential jobs, unchanged per-case archive readers.

The process facade ends at a job boundary; the rank processes remain resident.
A native failure kills this session and the next eligible pair starts a fresh
rank group. Ordinary completion and query-only jobs keep CUDA/NCCL warm.
"""
import contextlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time

_active = None


class JobProcess:
    def __init__(self, session, sequence):
        self.session, self.sequence = session, sequence
        self.pid = session.process.pid
        self.returncode = None
        self.codes = {}
        self.errors = {}
        self.lines = queue.Queue()
        self.stdout = self._lines()

    def _lines(self):
        while True:
            line = self.lines.get()
            if line is None:
                return
            yield line

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic()+timeout
        while self.returncode is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('resident job', timeout)
            self.session.done.wait(.001)
        return self.returncode

    def communicate(self, timeout=None):
        # Query callers need a bounded collector, not a blocking stdout iterator.
        output = []
        deadline = None if timeout is None else time.monotonic()+timeout
        while True:
            remaining = None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired('resident query', timeout)
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise subprocess.TimeoutExpired('resident query', timeout)
            if line is None:
                return ''.join(output), None
            output.append(line)


class Session:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.process = self.job = None
        self.sequence = 0
        self.generation = 0
        self.done = threading.Event()
        self.router = None
        self.world = None
        self.capacity_profiles = {}
        self.metadata = {}

    def _start(self, command, env):
        self.world = int(env['MGBFS_BENCH_WORLD_SIZE'])
        self.identity = self._identity(command, env)
        self.directory = self.root/f'generation-{self.generation}'
        self.directory.mkdir()
        self.generation += 1
        cli = command.index('--no-python')+1
        worker = command[:cli+1]+['session', str(self.directory)]
        self.process = subprocess.Popen(worker, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.sequence = 0
        self.router = threading.Thread(target=self._route, daemon=True)
        self.router.start()

    @staticmethod
    def _identity(command, env):
        cli = command.index('--no-python')+1
        return (tuple(command[:cli+1]), env['MGBFS_BENCH_WORLD_SIZE'],
            env.get('MGBFS_TRANSPORT_BACKEND'),env.get('CUDA_VISIBLE_DEVICES'),
            tuple(sorted((k,v) for k,v in env.items() if k.startswith('NCCL_'))))

    def _route(self):
        process = self.process
        with (self.directory/'worker.log').open('w', buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                job = self.job
                if job is None:
                    continue
                job.lines.put(line)
                try:
                    record = json.loads(line)
                except ValueError:
                    record = {}
                if record.get('status') == 'ERROR' and 'rank' in record:
                    job.errors[record['rank']] = record.get('error')
                match = re.search(r'MGBFS_SESSION_DONE sequence=(\d+) rank=(\d+) code=(\d+)', line)
                if match and int(match[1]) == job.sequence:
                    job.codes[int(match[2])] = int(match[3])
                    if len(job.codes) == self.world:
                        job.returncode = max(job.codes.values())
                        job.lines.put(None)
                        self.done.set()
            process.wait()
            if self.job is not None and self.job.returncode is None:
                self.job.returncode = process.returncode or 1
                self.job.lines.put(None)
                self.done.set()

    def launch(self, command, env):
        if self.job is not None and self.job.returncode is None:
            raise RuntimeError('resident session already has an active job')
        if self.process is not None and self.identity != self._identity(command, env):
            self._stop()
        # Query error sentinel is recoverable; all other failures exit workers.
        if self.job is not None and self.job.returncode:
            if not (self.query_job and self.job.errors ==
                    {rank:'MEMORY_QUERY_DONE' for rank in range(self.world)}):
                self._stop()
        if self.process is None or self.process.poll() is not None:
            self._stop()
            self._start(command, env)
        cli = command.index('--no-python')+1
        args = command[cli+3:cli+8]
        if len(args) != 5:
            raise ValueError('resident bench command shape')
        self.done.clear()
        job = JobProcess(self, self.sequence)
        self.job = job
        self.query_job = env.get('MGBFS_MEMORY_QUERY') == '1'
        payload = dict(schema=1, sequence=self.sequence, args=['resident-bench', *args],
            env={k:v for k,v in env.items() if k.startswith(('MGBFS_', 'NCCL_'))})
        path = self.directory/f'job-{self.sequence:08}.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(payload), encoding='utf-8')
        os.replace(temp, path)
        self.sequence += 1
        return job

    def _stop(self):
        if self.process is None:
            return
        (self.directory/'shutdown').touch()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self.router:
            self.router.join(timeout=5)
            if self.router.is_alive():raise RuntimeError('resident log router did not retire')
        self.process.stdout.close()
        self.process = self.job = None
        self.capacity_profiles.clear()
        self.metadata.clear()

    def close(self):
        self._stop()


@contextlib.contextmanager
def resident(root):
    global _active
    if _active is not None:
        raise RuntimeError('nested resident session')
    session = Session(root)
    _active = session
    try:
        yield session
    finally:
        _active = None
        session.close()


def launch(command, **kwargs):
    if _active is not None:
        return _active.launch(command, kwargs['env'])
    return subprocess.Popen(command, **kwargs)


def active_session():
    return _active
