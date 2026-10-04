"""Linux native BFS launcher: completed-layer SSD tail and optional HF worker."""
import argparse
import hashlib
import json
import os
import queue
import re
import select
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from bfs_tail_archive import TailArchive, atomic_json
from tail_wire import consume


class FifoReader:
    def __init__(self, path, stopped):
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self.stopped = stopped

    def read(self, size):
        while True:
            try:
                data = os.read(self.fd, size)
                if data:
                    return data
            except BlockingIOError:
                pass
            if self.stopped.is_set():
                return b''
            select.select([self.fd], [], [], .05)

    def close(self):
        os.close(self.fd)


def chunks(paths):
    for path in paths:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                yield chunk


def release_rank_spools(root, world, threads):
    """Discard uncommitted rank parts only after every reader has stopped.

    Saved snapshots live elsewhere. Never follow a replaced spool directory or
    unlink through a symlink, and preserve all input while a reader can write.
    """
    if any(thread.is_alive() for thread in threads):
        return False
    root = Path(root).resolve()
    paths = []
    for rank in range(world):
        directory = root / f'spool-{rank}'
        if directory.is_symlink():
            raise ValueError('rank spool directory is a symlink')
        if not directory.exists():
            continue
        for path in directory.glob('layer-*.bin'):
            if path.is_symlink() or path.resolve().parent != directory.resolve():
                raise ValueError('rank spool file escapes directory')
            if path.is_file():
                paths.append(path)
    for path in paths:
        path.unlink()
    return True


def native_failure(line):
    match=re.search(r'MGBFS_(?:RUNTIME|ARCHIVE_WORKER)_FATAL (?:rank|device)=\d+ error=(.*)',line)
    if match:
        return match[1].strip()[:1024]
    try:
        record=json.loads(line)
    except ValueError:
        return None
    if isinstance(record,dict) and record.get('status')=='ERROR' and isinstance(record.get('error'),str):
        return record['error'][:1024]
    return None


@contextmanager
def cancellation_signals():
    """Signals request cleanup; they never interrupt a snapshot or upload."""
    reason = [None]
    def request(signum, _frame):
        reason[0] = 'requested cancellation: '+signal.Signals(signum).name
    previous = {sig: signal.signal(sig, request) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield lambda: reason[0]
    finally:
        for sig, handler in previous.items():signal.signal(sig, handler)


def native_completion(reports, calibration_limit, completed_layers):
    if not reports:
        raise ValueError('native reports missing')
    if all(report['status']=='COMPLETE' for report in reports):
        return True, 'graph exhausted'
    if calibration_limit is not None:
        limit=int(calibration_limit)
        if limit>0 and completed_layers==limit and all(
                report['status']=='INCOMPLETE'
                and report.get('stop_reason')=='calibration layer limit'
                and report.get('calibration_layers')==limit
                and report.get('last_completed_layer')==limit-1 for report in reports):
            return False, 'calibration layer limit'
    raise ValueError('native result incomplete')


def finish_run(root, final, reason, failure, commit, binary_sha, publisher, *, complete=True):
    """Always record the local outcome, including a failed final upload."""
    upload_error = None
    if publisher:
        try:publisher.enqueue(final)
        except BaseException as error:upload_error = error
        try:publisher.finish()
        except BaseException as error:upload_error = upload_error or error
    if upload_error:
        failure = failure or upload_error
        reason += '; HF publication failed; local snapshots and pinned inputs retained'
    atomic_json(root/'run-summary.json', dict(status='INCOMPLETE' if failure or not complete else 'COMPLETE',
        reason=reason, manifest=str(final), program_commit=commit, binary_sha256=binary_sha,
        publication_status='FAILED' if upload_error else ('COMPLETE' if publisher else 'NOT_REQUESTED')))
    return failure, reason


def run(config, source, root, runtime_env, *, publisher_api=None, cancelled=None):
    if os.name != 'posix':
        raise ValueError('Linux required')
    n, r, world = config['n'], config['r'], config.get('world', 2)
    if not 2 <= n <= 128 or not 1 <= r <= n or world not in (1, 2, 4, 8):
        raise ValueError('graph/topology')
    root.mkdir(parents=True, exist_ok=False)
    source = source.resolve()
    from resident_session import active_session
    session = active_session()
    metadata = session.metadata if session else {}
    identity_key = str(source)
    commit = metadata.get(identity_key+'commit')
    if commit is None:
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        metadata[identity_key+'commit'] = commit
    cli = (source/config.get('binary_path','target/release/mgbfs')).resolve()
    if not cli.is_relative_to(source):
        raise ValueError('binary must belong to the source checkout')
    binary_sha = metadata.get(str(cli)+'sha')
    if binary_sha is None:
        binary_sha = hashlib.sha256(cli.read_bytes()).hexdigest()
        metadata[str(cli)+'sha'] = binary_sha
    env = dict(os.environ, **runtime_env)
    env.update(config.get('env', {}))
    env.update(MGBFS_BENCH_WORLD_SIZE=str(world), MGBFS_RANK_MAP=','.join(map(str,range(world))),
        MGBFS_STATE_CODEC='permutation_u8', MGBFS_ARCHIVE_CODEC='permutation_u8',
        MGBFS_ARCHIVE_STREAM='1', MGBFS_BENCH_SKIP_ARCHIVE='0', MGBFS_TRACE_DEPTHS='1',
        MGBFS_BENCH_WARMUP='0')
    command = ['torchrun', '--standalone', f'--nproc-per-node={world}', '--no-python',
        str(cli), 'bench', '--reference', f'lrx{n}r{r}', str(config['batch']),
        str(root/'bootstrap'), str(root/'archive'), str(root/'result/rank-{rank}.json')]
    # Native CLI writes rank-N.json into this shared output directory.
    command[-1] = str(root/'result')
    start = list(range(n-r+1)) + [n-r]*(r-1)
    from state_layout import orbit_layout
    inventory = metadata.get('gpu_inventory')
    if inventory is None:
        inventory = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,memory.total,driver_version',
            '--format=csv,noheader'],text=True).splitlines()
        metadata['gpu_inventory'] = inventory
    saved = dict(config, command=command, binary_sha256=binary_sha,
        orbit_layout=orbit_layout(n,r),
        runtime_paths=runtime_env,
        gpu_inventory=inventory,
        runtime_configuration={k:v for k,v in env.items() if k.startswith(('MGBFS_', 'NCCL_'))})
    archive = TailArchive(root/'saved', n=n, r=r, start=start,
        actions=dict(L='cyclic left rotation', R='cyclic right rotation', X='swap positions 0 and 1'),
        program_commit=commit, launch_config=saved, sample_interval_seconds=.05)
    archive.manifest['vram_observation'] = dict(
        source='separate nvidia-smi monitor', timestamp='host receipt time',
        window='earliest rank BEGIN to latest rank END',
        missing_sample=None, caveat='short layers may contain no sample; peaks between samples may be missed')
    archive.manifest['layer_time_scope'] = 'maximum rank whole-layer duration at existing advance boundary'
    publisher = None
    if config.get('repo_id'):
        from tail_upload import Publisher
        token=None
        if publisher_api is None:
            from huggingface_hub import get_token
            token=get_token()
        publisher = Publisher(root/'upload-pins', config['repo_id'], config['run_id'], token, api=publisher_api)
    messages, stopped = queue.Queue(), threading.Event()
    begins, ends, samples, native_errors, lock = {}, {}, [], [], threading.Lock()
    readers, threads, pending, receipts = [], [], {}, {}
    for rank in range(world):
        fifo = root/f'archive-rank-{rank}.mgbfsar1'
        os.mkfifo(fifo)
        reader = FifoReader(fifo, stopped)
        readers.append(reader)
        def read_rank(rank=rank, reader=reader):
            try:
                receipt = consume(reader, root/f'spool-{rank}', n,
                    lambda depth,count,path,digest: messages.put(('layer',rank,depth,count,path,digest)),
                    bits_per_symbol=archive.manifest['packing']['bits_per_symbol'])
                messages.put(('receipt',rank,receipt))
            except BaseException as error:
                messages.put(('error',rank,str(error)))
        thread = threading.Thread(target=read_rank, daemon=True)
        thread.start()
        threads.append(thread)
    memory = subprocess.Popen(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits', '-lms', '50'], stdout=subprocess.PIPE, text=True)
    def monitor():
        for line in memory.stdout:
            try:
                index, mib = [int(x.strip()) for x in line.split(',')]
                with lock:
                    samples.append((time.time(), str(index), mib*1024*1024))
            except ValueError:
                pass
    threading.Thread(target=monitor, daemon=True).start()
    from resident_session import launch
    process = launch(command, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True)
    if session:
        archive.manifest['launch_config']['resident_session'] = dict(
            root=str(session.root), generation=session.generation-1,
            sequence=process.sequence, rank_group_pid=process.pid,
            worker_command=['torchrun','--standalone',f'--nproc-per-node={world}',
                '--no-python',str(cli),'session',str(session.directory)])
    def log_reader():
        with (root/'native.log').open('w', buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                fatal = native_failure(line)
                if fatal:
                    native_errors.append(fatal)
                match = re.search(r'MGBFS_DEPTH_(BEGIN|END) rank=(\d+) depth=(\d+) (.*)', line)
                if match:
                    fields = dict(re.findall(r'(\w+)=([^\s]+)', match[4]))
                    if 'unix' in fields:
                        key = (int(match[2]), int(match[3]))
                        with lock:
                            (begins if match[1]=='BEGIN' else ends)[key] = fields
            stopped.set()
    logthread = threading.Thread(target=log_reader, daemon=True)
    logthread.start()
    deadline = time.monotonic()+config.get('timeout_seconds',300)
    reason, failure, deferred_error = 'running', None, None
    traversal_complete=True
    cancellation_error, cancellation_started = None, None
    try:
        while True:
            now = time.monotonic()
            request = cancelled() if cancelled else None
            if cancellation_error is None and ((request and process.poll() is None) or now > deadline):
                cancellation_error = RuntimeError(request) if request else TimeoutError('search deadline')
                cancellation_started = now
                if process.poll() is None:
                    try:os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:pass
            if cancellation_started is not None and now > cancellation_started+10 and process.poll() is None:
                try:os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:pass
            if publisher and publisher.error:
                raise RuntimeError('HF background upload failed')
            try:
                message = messages.get(timeout=.05)
            except queue.Empty:
                message = None
            if message:
                if message[0] == 'error':
                    deferred_error = f'archive rank {message[1]}: {message[2]}'
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                if message[0] == 'receipt':
                    receipts[message[1]] = message[2]
                elif message[0] == 'layer':
                    _,rank,depth,count,path,digest = message
                    pending.setdefault(depth,{})[rank] = (count,path,digest)
            depth = archive.manifest['last_completed_layer']+1
            parts = pending.get(depth,{})
            with lock:
                ready = len(parts)==world and all((rank,depth) in ends and (rank,depth) in begins for rank in range(world))
                if ready:
                    begin = min(float(begins[rank,depth]['unix']) for rank in range(world))
                    end = max(float(ends[rank,depth]['unix']) for rank in range(world))
                    seconds = max(float(ends[rank,depth]['seconds']) for rank in range(world))
                    peaks = {str(gpu): max((used for at,index,used in samples if index==str(gpu) and begin<=at<=end), default=None) for gpu in range(world)}
            if ready:
                if len({p[2] for p in parts.values()}) != 1:
                    raise ValueError('rank configuration mismatch')
                archive.completed_layer(depth, sum(p[0] for p in parts.values()),
                    chunks([parts[rank][1] for rank in range(world)]), seconds, peaks)
                for _,path,_ in parts.values():
                    path.unlink()
                del pending[depth]
                if publisher:
                    # Pin immediately in this writer thread before next snapshot cleanup.
                    manifest = json.loads((archive.root/'manifest.json').read_text())
                    generation = archive.root/Path(manifest['files'][0]['path']).parent if manifest['files'] else None
                    if generation:
                        publisher.enqueue(generation/'manifest.json')
            if stopped.is_set() and process.poll() is not None:
                drained = messages.empty() and all(not thread.is_alive() for thread in threads)
                # All queued rank parts must be committed before cancellation
                # or a native error seals the final prefix snapshot.
                if drained and ready:continue
                if drained and cancellation_error:
                    raise cancellation_error
                if drained and (process.returncode != 0 or deferred_error):
                    primary = 'native fatal: '+','.join(dict.fromkeys(native_errors)) if native_errors else None
                    raise RuntimeError(primary or deferred_error or f'native exit {process.returncode}')
                if len(receipts)==world and not pending:
                    expected = archive.manifest['last_completed_layer']+1
                    if any(receipt['depths']!=expected for receipt in receipts.values()):
                        raise ValueError('rank depths mismatch')
                    reports = [json.loads((root/f'result/rank-{rank}.json').read_text()) for rank in range(world)]
                    traversal_complete,reason=native_completion(reports,
                        env.get('MGBFS_CALIBRATION_LAYERS'),expected)
                    if any(len(report['local_layer_sizes'])!=expected for report in reports):
                        raise ValueError('native rank layer counts differ')
                    actual_counts = [sum(row) for row in zip(*(report['local_layer_sizes'] for report in reports))]
                    if actual_counts != [layer['states'] for layer in archive.manifest['layers']]:
                        raise ValueError('archived counts differ from native reports')
                    break
        final = archive.snapshot(traversal_complete, reason)
    except BaseException as error:
        failure, reason = error, str(error)
        final = archive.snapshot(False, reason)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stopped.set()
        memory.terminate()
        memory.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=5)
        for reader, thread in zip(readers, threads):
            # Closing a descriptor while read/select is using it can race with
            # descriptor reuse. A slow reader keeps its inputs for recovery.
            if not thread.is_alive():
                reader.close()
        release_rank_spools(root, world, threads)
    failure, reason = finish_run(root, final, reason, failure, commit, binary_sha, publisher,
                               complete=traversal_complete)
    archive.release_working_tail()
    if failure:
        raise failure
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime-env', type=Path, required=True)
    args = parser.parse_args()
    with cancellation_signals() as cancelled:
        print(run(json.loads(args.config.read_text()), args.source, args.root,
            json.loads(args.runtime_env.read_text()), cancelled=cancelled), flush=True)


if __name__ == '__main__':
    main()
