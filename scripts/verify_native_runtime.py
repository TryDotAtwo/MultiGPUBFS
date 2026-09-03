"""Audit downloaded native runtime gates and raw benchmark records."""
import argparse
import json
import re
from pathlib import Path

TOOLS = ('plain', 'memcheck', 'racecheck', 'initcheck', 'synccheck')

def verify_gate(path, source, require_async_archive=False):
    summary = json.loads((path/'summary.json').read_text())
    if not re.fullmatch('[0-9a-f]{40}', source) or summary['source_commit'] != source:
        raise ValueError('SOURCE_MISMATCH')
    gpus = summary['gpus']
    if len(gpus) != 2 or {g['index'] for g in gpus} != {0, 1} or len({g['uuid'] for g in gpus}) != 2:
        raise ValueError('PHYSICAL_GPU_INVENTORY')
    if any(g['name'] not in ('Tesla T4', 'NVIDIA T4', 'NVIDIA Tesla T4') for g in gpus):
        raise ValueError('T4_REQUIRED')
    expected = {(g['uuid'], t) for g in gpus for t in TOOLS}
    actual = [(r['gpu'], r['tool']) for r in summary['tests'] if r['status'] == 'PASS']
    if len(actual) != 10 or set(actual) != expected:
        raise ValueError('GATE_MATRIX')
    for gpu in gpus:
        for tool in TOOLS:
            log = (path/f"gpu{gpu['index']}-{tool}.log").read_text(errors='replace')
            fixtures = ('native_archive_roundtrip', 'native_feedback_full_layers', 'layer_capacity_failure_is_terminal')
            if require_async_archive:
                fixtures += ('archive_overlap_survives_blocked_disk_and_ring_wrap', 'asynchronous_archive_disk_error_is_not_complete')
            for test in fixtures:
                if not re.search(r'^test '+test+r' \.\.\. ok\s*$', log, re.M):
                    raise ValueError('MISSING_FIXTURE_'+test)
            if not re.search(r'test result: ok\. \d+ passed; 0 failed;', log):
                raise ValueError('INCOMPLETE_RUST_TESTS')
            if tool == 'plain':
                if any(f'FULL_STATE_PASS m={m} pre={pre}' not in log for m in range(5,9) for pre in ('false','true')):
                    raise ValueError('LARGE_FULL_STATE_MATRIX')
            elif tool == 'racecheck':
                totals = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', log)
                if not totals or any(t != ('0','0','0') for t in totals):
                    raise ValueError('RACECHECK_FAILED')
            else:
                totals = re.findall(r'ERROR SUMMARY: (\d+) errors', log)
                if not totals or any(t != '0' for t in totals):
                    raise ValueError('SANITIZER_FAILED')
    return summary

def verify_measurements(path, summary):
    if summary['status'] != 'COMPLETE_SINGLE_RANK_AB':
        raise ValueError('INCOMPLETE_AB')
    measured = {'native': [], 'cayleypy': []}
    for row in summary['rows']:
        if row['status'] != 'COMPLETE':
            # Failed capacity/calibration rows stay in the report, never samples.
            if row['phase'] in ('time', 'verify'): raise ValueError('REQUIRED_RUN_FAILED')
            continue
        raw = [json.loads(line) for line in (path/(row['label']+'.log')).read_text().splitlines() if line.startswith('{')]
        if len(raw) != 1 or any(raw[0].get(key) != row.get(key) for key in raw[0]):
            raise ValueError('RAW_RESULT_MISMATCH')
        if not row.get('smi_samples') or row.get('smi_process_peak_mib') is None:
            raise ValueError('MISSING_DEVICE_SAMPLER')
        if row['backend_requested'] == 'native':
            if not (row['durable_run_commit_seconds'] >= row['search_complete_seconds'] > 0):
                raise ValueError('DURABILITY_TIMER')
            if row['requested_device_bytes'] <= 0 or row['pinned_bytes'] <= 0:
                raise ValueError('MISSING_MEMORY_ACCOUNTING')
        if row['phase'] == 'time': measured[row['backend_requested']].append(row)
    for rows in measured.values():
        if len(rows) != 5 or {r['repetition'] for r in rows} != set(range(5)):
            raise ValueError('FIVE_FRESH_RUNS_REQUIRED')
    if len({tuple(r['layer_sizes']) for rows in measured.values() for r in rows}) != 1:
        raise ValueError('LAYER_COUNTS_DIFFER')
    return measured

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--source', required=True)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--require-async-archive', action='store_true')
    args = parser.parse_args()
    summary = verify_gate(args.path, args.source, args.require_async_archive)
    if not args.gate_only: verify_measurements(args.path, summary)
    print('VERIFIED_NATIVE_RUNTIME_GATE_10/10' + ('' if args.gate_only else '_AND_AB'))
