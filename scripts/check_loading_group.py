"""Actual two-process startup-loading fault gate, not mocked driver evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from replay_lsa_cancel_candidate import (
    cleanup_rank_processes, failure_has_no_complete, rank_arguments, rank_environment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--deadline', type=float, default=45)
    args = parser.parse_args()
    if not 1 <= args.deadline <= 120:
        parser.error('deadline must be 1..120 seconds')
    config = json.loads(args.config.read_text())
    if config['topology']['world_size'] != 2:
        parser.error('requires two independent ranks')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'scope': 'two physical ranks: asymmetric loading admission', 'cases': []}
    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    save()
    for loading_variable in ('CUDA_MODULE_LOADING', 'CUDA_MODULE_DATA_LOADING'):
        for failed_rank in (0, 1):
            case = args.output / (loading_variable + '-rank-' + str(failed_rank))
            case.mkdir()
            processes, streams = [], []
            env = dict(os.environ, CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER',
                       TORCHELASTIC_RUN_ID='loading-gate-' + uuid.uuid4().hex)
            started = time.monotonic()
            try:
                for rank in (0, 1):
                    rank_env = rank_environment(env, rank)
                    if rank == failed_rank:
                        rank_env[loading_variable] = 'LAZY'
                    stream = (case / ('rank-' + str(rank) + '.log')).open('w')
                    streams.append(stream)
                    command = [str(args.binary.resolve()), *rank_arguments(case.resolve(), 's4', config['parent_batch'], args.config.resolve())]
                    processes.append(subprocess.Popen(command, env=rank_env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True))
                deadline = started + args.deadline
                while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
                    time.sleep(.05)
                codes = [p.poll() for p in processes]
                forced = any(code is None for code in codes)
                for stream in streams:
                    stream.flush()
                error = (case / ('rank-' + str(failed_rank) + '.log')).read_text(errors='replace')
                rejected = 'CUDA_LOADING_POLICY' in error and loading_variable in error
                no_complete = failure_has_no_complete(case)
                row = {'variable': loading_variable, 'failed_rank': failed_rank,
                       'returncodes': codes, 'forced_cleanup': forced,
                       'seconds': time.monotonic() - started, 'deadline_seconds': args.deadline,
                       'policy_rejection_reached': rejected, 'no_complete': no_complete,
                       'pass': not forced and all(code not in (None, 0) for code in codes) and rejected and no_complete}
                report['cases'].append(row)
                save()
            finally:
                cleanup_errors = cleanup_rank_processes(processes)
                for stream in streams:
                    stream.close()
                if cleanup_errors:
                    report['cleanup_errors'] = cleanup_errors
                    report['status'] = 'FAIL'
                    save()
                    raise RuntimeError('LOADING_GATE_CLEANUP_FAILED')
            if not row['pass']:
                report['status'] = 'FAIL'
                save()
                raise RuntimeError('ASYMMETRIC_LOADING_ADMISSION_FAILED')
    report['status'] = 'PASS'
    save()


if __name__ == '__main__':
    main()
