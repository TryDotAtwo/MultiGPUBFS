"""One remote build + exact graph oracle; publish metadata directly from host.

No rental creation, retries of execution, or state downloads. Token is read from
a private file only by the final publisher, never passed to child processes.
"""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def run_group(command, *, env, stdout, timeout):
    """A deadline owns the whole CUDA child group, including cargo test binaries."""
    process = subprocess.Popen(command, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def evidence_files(work):
    """Allowlist diagnostic text; exclude binaries, states and credential files."""
    candidates = [work/'build-summary.json', work/'gate-summary.json',
                  work/'graph-oracle.log', work/'build-driver.log']
    candidates.extend(sorted((work/'logs').glob('*.log')))
    return [p for p in candidates if p.is_file() and not p.is_symlink()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--nccl-lsa-root', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--repo', default='TryDotAtwo/multigpubfs-bfs-results')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--cuda-architecture', default='86')
    args = parser.parse_args()
    if (sys.platform != 'linux' or args.work.exists() or
            not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.run_id)):
        parser.error('Linux, fresh work directory and safe run id required')
    # A clean pinned checkout is checked by remote_build before downloads.
    report = dict(status='INCOMPLETE', source_commit=args.commit, started_at=time.time(),
                  scope='native graph smoke and two-rank full-state multiset oracle')
    env = {k:v for k,v in os.environ.items()
           if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN') and not k.startswith('MGBFS_')}
    driver_log = args.work.parent/(args.work.name+'-driver.log')
    try:
        with driver_log.open('x') as output:
            run_group([sys.executable, str(args.source/'scripts/remote_build.py'),
                '--source', str(args.source), '--commit', args.commit, '--work', str(args.work),
                '--timeout-seconds', '1800', '--jobs', '8', '--cuda-architecture', args.cuda_architecture,
                '--nccl-lsa-root', str(args.nccl_lsa_root), '--graph-smoke-test'],
                env=env, stdout=output, timeout=1860)
        summary = json.loads((args.work/'build-summary.json').read_text())
        if summary.get('status') != 'BUILT' or summary.get('graph_smoke_test') != 'PASS':
            raise RuntimeError('BUILD_OR_SMOKE_GATE_NOT_PASSED')
        env.update(json.loads((args.work/'runtime-env.json').read_text()))
        env['MGBFS_CUDA_GRAPH_BATCHES'] = '32'
        with (args.work/'graph-oracle.log').open('x') as output:
            run_group(['cargo', 'test', '--manifest-path', str(args.source/'Cargo.toml'),
                '--locked', '--release', '-p', 'mgbfs-runtime', '--features', 'cuda,library-owner',
                '--test', 'lrx_multiset_gpu', 'lrx_multiset_two_rank_graph_windows_full_state_oracle',
                '--', '--ignored', '--exact', '--nocapture', '--test-threads=1'],
                env=env, stdout=output, timeout=600)
        oracle = (args.work/'graph-oracle.log').read_text()
        if '1 passed' not in oracle or 'GRAPH_WINDOW_ORACLE rank=0' not in oracle or 'GRAPH_WINDOW_ORACLE rank=1' not in oracle:
            raise RuntimeError('EXACT_ORACLE_EVIDENCE_MISSING')
        report['status'] = 'PASS'
    except Exception as error:
        report['error_type'] = type(error).__name__
        # Do not persist exception arguments from credential/network libraries.
    finally:
        args.work.mkdir(exist_ok=True)
        if driver_log.exists():
            driver_log.rename(args.work/'build-driver.log')
        report['finished_at'] = time.time()
        (args.work/'gate-summary.json').write_text(json.dumps(report, indent=2))
        from huggingface_hub import HfApi, CommitOperationAdd
        token = args.token_file.read_text().strip()
        files = evidence_files(args.work)
        inventory = {str(p.relative_to(args.work)): dict(bytes=p.stat().st_size,
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
        inventory_path = args.work/'evidence-inventory.json'
        inventory_path.write_text(json.dumps(inventory, indent=2))
        files.append(inventory_path)
        revision = HfApi(token=token).create_commit(repo_id=args.repo, repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo='evidence/'+args.run_id+'/'+str(p.relative_to(args.work)),
                        path_or_fileobj=str(p)) for p in files],
            commit_message='Remote graph hardware gate '+args.run_id).oid
        print(json.dumps(dict(status=report['status'], evidence_revision=revision)), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
