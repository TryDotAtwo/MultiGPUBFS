"""Single-device A/B orchestration and CayleyPy worker; no GPU code changes.

Every worker is a fresh process, both implementations warm the full workload.
Verification runs include readback/digests and are NEVER performance samples.
Native fixed capacity is m**6, not hindsight peak-frontier sizing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def verify_pair(a, b):
    if a['status'] != 'COMPLETE' or b['status'] != 'COMPLETE':
        raise ValueError('incomplete BFS')
    if a['layer_sizes'] != b['layer_sizes']:
        raise ValueError('layer count mismatch')
    if not a['layer_sha256'] or a['layer_sha256'] != b['layer_sha256']:
        raise ValueError('full-state layer digest mismatch')


def timing_stats(rows):
    if not rows or any(r['status'] != 'COMPLETE' for r in rows):
        raise ValueError('failed samples must not be silently discarded')
    values = [r['search_seconds'] for r in rows]
    median = statistics.median(values)
    return dict(median_seconds=median, mad_seconds=statistics.median(abs(x-median) for x in values), repeats=len(rows))


def baseline(m, batch, validate):
    import gc
    import numpy as np
    import torch
    from cayleypy import CayleyGraph, CayleyGraphDef
    from cayleypy.cayley_graph_def import MatrixGenerator
    if torch.cuda.device_count() != 1:
        raise RuntimeError('worker must see exactly one GPU')
    generators = []
    for delta in [1, m-1]:
        for i in range(3):
            a = np.eye(4, dtype=np.int64)
            a[i, i+1] = delta
            generators.append(MatrixGenerator(a, modulo=m))
    definition = CayleyGraphDef.for_matrix_group(generators=generators)
    assert definition.generators_inverse_closed
    graph = CayleyGraph(definition, device='cuda', num_gpus=1, batch_size=batch,
                        random_seed=20260828, verbose=0)
    assert graph.num_gpus == 1
    # 0 is NOT no-storage in this baseline: it means all. Use 1 explicitly.
    warm = graph.bfs(max_layer_size_to_store=1)
    assert warm.bfs_completed and sum(warm.layer_sizes) == m**6
    del warm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_free, total = torch.cuda.mem_get_info()
    start = time.perf_counter()
    result = graph.bfs(max_layer_size_to_store=None if validate else 1)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    after_free, _ = torch.cuda.mem_get_info()
    assert result.bfs_completed and sum(result.layer_sizes) == m**6
    digests = []
    if validate:
        for depth in range(len(result.layer_sizes)):
            states = np.ascontiguousarray(result.get_layer(depth).reshape(-1, 16), dtype=np.uint8)
            keys = states.view('V16').reshape(-1)
            keys.sort()
            digests.append(hashlib.sha256(keys.tobytes()).hexdigest())
    return dict(status='COMPLETE', backend='cayleypy_single', modulus=m, batch=batch,
                verification_only=validate, search_seconds=seconds,
                layer_sizes=result.layer_sizes, layer_sha256=digests,
                torch_peak_allocated_bytes=peak_alloc, torch_peak_reserved_bytes=peak_reserved,
                cuda_before_used_bytes=total-before_free, cuda_after_used_bytes=total-after_free,
                state_dtype=str(graph.central_state.dtype), torch_version=torch.__version__,
                torch_cuda_version=torch.version.cuda, hash_method=graph.hasher.make_hashes.__name__)


def worker(command, out, label, env, timeout=600):
    """Keep failures as rows; sampler covers warmup and search (process peak)."""
    row = dict(label=label, command=command, status='INCOMPLETE')
    with (out/(label+'.log')).open('w') as log, (out/(label+'-smi.csv')).open('w') as smi:
        sampler = subprocess.Popen(['nvidia-smi', '-i', env['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm,power.draw',
            '--format=csv,noheader,nounits', '-lms', '50'], stdout=smi, stderr=subprocess.STDOUT)
        try:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f'RUNNING {label}: {time.monotonic()-started:.0f}s', flush=True)
                    if time.monotonic()-started > timeout:
                        process.kill()
                        process.wait()
                        row['status'] = 'TIMEOUT'
                        break
            row['exit_code'] = process.returncode
        finally:
            sampler.terminate()
            sampler.wait()
    if row['exit_code'] == 0:
        outputs = [json.loads(line) for line in (out/(label+'.log')).read_text().splitlines() if line.startswith('{')]
        if len(outputs) != 1:
            raise ValueError('worker JSON inventory')
        row.update(outputs[0])
    elif row['status'] != 'TIMEOUT':
        row['status'] = 'FAILED'
    samples = []
    for line in (out/(label+'-smi.csv')).read_text().splitlines():
        parts = line.split(',')
        if len(parts) == 6:
            try:
                samples.append(float(parts[2]))
            except ValueError:
                pass
    row['smi_process_peak_mib'] = max(samples) if samples else None
    row['smi_samples'] = len(samples)
    (out/(label+'.json')).write_text(json.dumps(row, indent=2))
    print(f'{label}: {row["status"]}, {row.get("search_seconds")} s', flush=True)
    return row


def suite(native, out, env, moduli=(5, 8, 12)):
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, scope='experimental single-bucket ping-pong vs ordinary single-GPU CayleyPy; neither archives',
                  status='INCOMPLETE', rows=[], comparisons=[])
    def save():
        (out/'summary.json').write_text(json.dumps(report, indent=2))
    def run(backend, m, batch, pre, phase, rep=0):
        verify = phase == 'verify'
        label = f'm{m}-{backend}-b{batch}-p{int(pre)}-{phase}-{rep}'
        command = ([str(native), str(m), str(batch), str(int(pre)), 'verify' if verify else 'time']
                   if backend == 'native' else
                   [sys.executable, str(Path(__file__).resolve()), 'baseline', str(m), str(batch), 'verify' if verify else 'time'])
        row = worker(command, out, label, env)
        row.update(phase=phase, repetition=rep, config_backend=backend)
        report['rows'].append(row)
        save()
        return row
    try:
        for m in moduli:
            a = run('native', m, 65536, True, 'verify')
            b = run('cayleypy', m, 65536, False, 'verify')
            verify_pair(a, b)
            configs = {}
            for backend, batches in [('native', [4096, 16384, 65536]), ('cayleypy', [65536, 262144, 1048576])]:
                trials = []
                for batch in batches:
                    for pre in ([False, True] if backend == 'native' else [False]):
                        row = run(backend, m, batch, pre, 'calibrate')
                        if row['status'] == 'COMPLETE':
                            if row['layer_sizes'] != a['layer_sizes']:
                                raise ValueError('calibration layer mismatch')
                            trials.append((row['search_seconds'], batch, pre))
                if not trials:
                    raise ValueError('no successful configuration')
                _, batch, pre = min(trials)
                configs[backend] = (batch, pre)
            measured = {backend: [] for backend in configs}
            for rep in range(5):
                # Alternate process order to reduce systematic thermal/order bias.
                for backend in (['native', 'cayleypy'] if rep % 2 == 0 else ['cayleypy', 'native']):
                    batch, pre = configs[backend]
                    row = run(backend, m, batch, pre, 'measure', rep)
                    if row['status'] != 'COMPLETE' or row['layer_sizes'] != a['layer_sizes']:
                        raise ValueError('measured run failed verification')
                    measured[backend].append(row)
            comparison = dict(modulus=m, states=m**6, layer_sizes=a['layer_sizes'], configs=configs)
            for backend, rows in measured.items():
                comparison[backend] = timing_stats(rows)
                comparison[backend]['smi_process_peak_mib'] = max(r['smi_process_peak_mib'] for r in rows)
            report['comparisons'].append(comparison)
            save()
        report['status'] = 'COMPLETE'
    finally:
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['baseline'])
    parser.add_argument('modulus', type=int)
    parser.add_argument('batch', type=int)
    parser.add_argument('operation', choices=['verify', 'time'])
    args = parser.parse_args()
    print(json.dumps(baseline(args.modulus, args.batch, args.operation == 'verify')))
