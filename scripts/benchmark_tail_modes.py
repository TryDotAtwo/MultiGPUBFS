"""GPU-host comparison of full automatic sweeps; no rental lifecycle here.

Each configuration gets its own immutable run root and HF run ID. The caller
owns the global cost deadline. A deadline-limited sweep is never called full.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

MODES = [('five-end', 'end', 5), ('original-end', 'end', None),
         ('five-graph', 'graph', 5), ('five-background', 'background', 5),
         ('five-search', 'search', 5)]


def compare(reports):
    """Compare only identical COMPLETE pairs, separately from unequal coverage."""
    baseline = next((r for r in reports if r['name'] == 'original-end'), None)
    result = []
    if baseline is None:
        return result
    for report in reports:
        if report is baseline:
            continue
        common = sorted(set(baseline['pairs']) & set(report['pairs']))
        common = [k for k in common if all(r['pairs'][k]['status'] == 'COMPLETE'
                                         for r in (baseline, report))]
        for field in ('runner_wall_seconds', 'production_search_seconds'):
            keys = [k for k in common if all(isinstance(r['pairs'][k].get(field), (int, float))
                                             for r in (baseline, report))]
            old = sum(baseline['pairs'][k][field] for k in keys)
            new = sum(report['pairs'][k][field] for k in keys)
            result.append(dict(name=report['name'], baseline=baseline['name'],
                metric=field, common_complete_pairs=keys, baseline_seconds=old,
                mode_seconds=new, speedup=old/new if new > 0 else None,
                scope='Matched complete pairs only; upload can overlap native work.'))
    return result


def summarize(root, name, started, events, error):
    ledger_path = root/'sweep.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    records = ledger.get('cases', {})
    pairs = {k: dict(status=r['status'], reason=r.get('reason'), comparison=r.get('comparison'), **r.get('timing', {}))
             for k, r in records.items() if r.get('attempted')}
    auto_path = root/'automatic-report.json'
    auto = json.loads(auto_path.read_text()) if auto_path.exists() else {}
    gap = [r['transition_before_seconds'] for r in pairs.values()
           if isinstance(r.get('transition_before_seconds'), (int, float))]
    publication_verified = auto.get('status') in ('VERIFIED', 'VERIFIED_PUBLICATION')
    return dict(name=name, status='VERIFIED' if publication_verified and auto.get('validation_status') != 'FAILED' else 'INCOMPLETE',
        validation_status=auto.get('validation_status', 'NOT_VALIDATED'),
        error=error, program_wall_seconds=time.monotonic()-started, events=events,
        grid_pairs=len(ledger.get('configuration', {}).get('grid', [])),
        attempted=len(pairs), complete=sum(r['status'] == 'COMPLETE' for r in pairs.values()),
        incomplete=sum(r['status'] != 'COMPLETE' for r in pairs.values()),
        pruned=sum('pruned_by' in r for r in records.values()),
        pending=ledger.get('pending'), all_eligible_pairs_terminal=bool(ledger) and not ledger.get('pending'),
        stop_reason=ledger.get('global_stop_reason'), pairs=pairs, automatic_report=auto,
        ledger_transition=dict(sum=sum(gap), median=statistics.median(gap) if gap else None,
                               scope='Ledger bookkeeping only; not actual search-to-search gap.'),
        scope='Native search reports exclude archive drain; missing failed-rank timing is unknown. '
              'Publication can overlap search; event durations must not be summed as disjoint phases.')


def worker(args):
    import run_auto_tail as app
    from run_tail_bfs import cancellation_signals
    from bfs_tail_archive import atomic_json
    name, mode, layers = next(x for x in MODES if x[0] == args.worker)
    if args.matched_pairs:
        selected = [tuple(map(int, value.split(','))) for value in args.matched_pairs]
        if any(len(pair) != 2 or not 2 <= pair[0] <= 128 or not 1 <= pair[1] <= pair[0]
               for pair in selected):
            raise ValueError('invalid matched calibration pair')
        app.automatic_pairs = lambda: selected
    root = args.root/(args.root.name+'-'+name)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    events, lock = [], threading.Lock()
    def measure(kind, function):
        def wrapped(*pos, **kw):
            at, begin = time.time(), time.monotonic()
            try:
                return function(*pos, **kw)
            finally:
                row = dict(kind=kind, started_at=at, seconds=time.monotonic()-begin)
                with lock:
                    events.append(row)
                    with (root/'performance-events.jsonl').open('a') as file:
                        file.write(json.dumps(row)+'\n')
        return wrapped
    app.execute = measure('compute_sweep_with_archive_and_backpressure', app.execute)
    app.verify_hf = measure('final_hf_readback', app.verify_hf)
    original = app.SweepPublisher.__init__
    def publisher(self, root, repo, api, deadline, publish):
        return original(self, root, repo, api, deadline,
                        measure('pack_upload_readback_release', publish))
    app.SweepPublisher.__init__ = publisher
    command = ['run_auto_tail', '--source', str(args.source), '--runtime-env', str(args.runtime_env),
        '--root', str(root), '--repo-id', args.repo_id, '--upload-mode', mode,
        '--deadline-unix', str(args.deadline_unix)]
    if args.two_seeds:
        command += ['--two-seeds']
    if layers is not None:
        command += ['--retained-layers', str(layers)]
    sys.argv = command
    error = None
    try:
        with (root/'program.log').open('a') as log, contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log), cancellation_signals() as cancelled:
            app.main(cancelled)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    result = summarize(root, name, started, events, error)
    atomic_json(root/'performance.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('pairs', 'automatic_report')}), flush=True)
    return 0 if result['status'] == 'VERIFIED' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--two-seeds',action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--runtime-env', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--worker', choices=[x[0] for x in MODES])
    parser.add_argument('--matched-pairs', nargs='+', metavar='N,R',
                        help='Separate matched calibration only; omit for the entire automatic grid')
    parser.add_argument('--order', nargs='+', choices=[x[0] for x in MODES],
                        default=[x[0] for x in MODES])
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if os.name != 'posix':
        parser.error('GPU execution requires Linux')
    from bfs_tail_archive import atomic_json
    args.root.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, name in enumerate(args.order):
        remaining = args.deadline_unix-time.time()
        if remaining < 600:
            break
        # No artificial graph or mode timeout: only the externally owned cap.
        deadline = args.deadline_unix
        command = [sys.executable, __file__, '--worker', name, '--source', str(args.source),
            '--runtime-env', str(args.runtime_env), '--root', str(args.root),
            '--repo-id', args.repo_id, '--deadline-unix', str(deadline)]
        if args.two_seeds:
            command += ['--two-seeds']
        if args.matched_pairs:
            command += ['--matched-pairs', *args.matched_pairs]
        child = subprocess.Popen(command, start_new_session=True)
        atomic_json(args.root/'active-worker.json', dict(pid=child.pid, name=name, deadline=deadline))
        child.wait()
        report_path = args.root/(args.root.name+'-'+name)/'performance.json'
        if report_path.exists():
            reports.append(json.loads(report_path.read_text()))
        atomic_json(args.root/'comparison.json', dict(reports=reports, matched=compare(reports),
            all_requested_modes_verified=(len(reports)==len(args.order) and
                                          all(r['status']=='VERIFIED' for r in reports)),
            all_requested_grids_terminal=(len(reports)==len(args.order) and
                                          all(r['all_eligible_pairs_terminal'] for r in reports)),
            requested_order=args.order, not_started=args.order[index+1:],
            scope='Each mode uses the entire automatic grid unless matched calibration was requested. '
                  'Only the external global cost deadline limits runs. '
                  'Pending pairs indicate incomplete coverage, not resource pruning.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
