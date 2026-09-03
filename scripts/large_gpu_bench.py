"""Size escalation until a minute-scale search or actual capacity boundary.

Not repeated small BFS disguised as a large graph. Each worker warms and then
exhausts one full graph. No algorithm fallback, no within-run capacity changes.
"""
import json
from pathlib import Path
import sys

from single_gpu_bench import timing_stats, worker

NATIVE_BATCHES = [65536, 262144, 524280]
BASELINE_BATCHES = [65536, 262144, 1048576]


def suite(native, out, env, moduli=(16, 20, 24, 28, 32)):
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, status='INCOMPLETE', rows=[], comparisons=[],
        scope='single-GPU scaling; layer counts and total cardinality verified, not full-state digests on large graphs',
        capacity_policy='min(m**6, 32000000) state/hash rows, fixed before each worker; 1 GiB VRAM reserve',
        requested_minimum_search_seconds=60, repeats=5,
        native_batch_limit_reason='Sm75 GEMM grid.y <= 65535, n=4: at most 524280 parents')
    def save():
        (out/'summary.json').write_text(json.dumps(report, indent=2))
    def run(backend, m, batch, phase, rep=0):
        capacity = min(m**6, 32_000_000)
        label = f'm{m}-{backend}-b{batch}-{phase}-{rep}'
        device_env = dict(env, MGBFS_BENCH_CAPACITY=str(capacity), MGBFS_BENCH_RESERVE_GIB='1')
        command = ([str(native), str(m), str(batch), '1', 'time'] if backend == 'native' else
            [sys.executable, str(Path(__file__).with_name('single_gpu_bench.py')), 'baseline', str(m), str(batch), 'time'])
        row = worker(command, out, label, device_env, timeout=1800)
        row.update(phase=phase, repetition=rep, config_backend=backend,
                   modulus=m, batch=batch, native_capacity=capacity if backend == 'native' else None)
        report['rows'].append(row)
        save()
        return row
    try:
        last_common = None
        for m in moduli:
            # Independent configuration trials, never fallback inside a BFS.
            # One failed configuration does not establish a graph capacity limit.
            probes = {}
            for backend, batches in [('native', NATIVE_BATCHES), ('cayleypy', BASELINE_BATCHES)]:
                for batch in reversed(batches):
                    row = run(backend, m, batch, 'size_probe')
                    probes[backend] = row
                    if row['status'] == 'COMPLETE':
                        break
            a, b = probes['native'], probes['cayleypy']
            if a['status'] != 'COMPLETE' or b['status'] != 'COMPLETE':
                report['first_common_capacity_boundary'] = m
                break
            if a['layer_sizes'] != b['layer_sizes']:
                raise ValueError('large layer count mismatch')
            last_common = (m, a['layer_sizes'])
            if min(a['search_seconds'], b['search_seconds']) >= 60:
                break
        if last_common is None:
            report['status'] = 'NO_COMMON_COMPLETED_GRAPH'
            return report
        m, expected = last_common
        configs = {}
        for backend in ['native', 'cayleypy']:
            trials = []
            for batch in (NATIVE_BATCHES if backend == 'native' else BASELINE_BATCHES):
                row = run(backend, m, batch, 'calibrate')
                if row['status'] == 'COMPLETE':
                    if row['layer_sizes'] != expected:
                        raise ValueError('large calibration count mismatch')
                    trials.append((row['search_seconds'], batch))
            if not trials:
                raise ValueError('no successful large configuration')
            configs[backend] = min(trials)[1]
        measured = {b: [] for b in configs}
        for rep in range(5):
            for backend in (['native', 'cayleypy'] if rep % 2 == 0 else ['cayleypy', 'native']):
                row = run(backend, m, configs[backend], 'measure', rep)
                if row['status'] != 'COMPLETE' or row['layer_sizes'] != expected:
                    raise ValueError('large measured run failed')
                measured[backend].append(row)
        result = dict(modulus=m, states=m**6, configs=configs, layer_sizes=expected)
        for backend, rows in measured.items():
            result[backend] = timing_stats(rows)
            result[backend]['smi_process_peak_mib'] = max(r['smi_process_peak_mib'] for r in rows)
        report['comparisons'].append(result)
        report['minute_scale_both_backends'] = all(result[b]['median_seconds'] >= 60 for b in configs)
        report['status'] = 'COMPLETE' if report['minute_scale_both_backends'] else 'CAPACITY_OR_SIZE_LIMIT_BEFORE_MINUTE_SCALE'
        return report
    finally:
        save()
