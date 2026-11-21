"""Installed-wheel acceptance on two GPUs, with a transport-safe stub mode.

This script is an experimental verification harness, not a BFS timeout policy.
It writes its evidence to a new output directory and never modifies the source.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

PREFIXES = (
    'strategy-layer-oracle-', 'warp-race-oracle-', 'batch-bitmap-layers-',
    'memory-presets-api-',
)
TESTS = (
    ('hash-layer-multisets', 'segmented_warp_race_layers_gpu.py', 1800),
    ('bitmap-atomic-warp', 'segmented_strategy_layers_gpu.py', 1200),
    ('bitmap-batch', 'segmented_batch_bitmap_layers_gpu.py', 1200),
    ('public-presets', 'segmented_memory_presets_gpu.py', 1200),
)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def oracle(generators, start):
    frontier = {tuple(start)}
    visited = set(frontier)
    layers = []
    while frontier:
        layers.append(Counter({state: 1 for state in frontier}))
        nxt = {tuple(parent[i] for i in generator)
               for parent in frontier for generator in generators} - visited
        visited.update(nxt)
        frontier = nxt
    return layers

def check_counter(actual, expected):
    if actual != expected or any(n != 1 for n in actual.values()):
        raise AssertionError('LAYER_MULTISET_MISMATCH')

def synthetic(output, fail):
    layers = oracle([[1, 0, 2], [0, 2, 1]], [0, 1, 2])
    assert [sum(c.values()) for c in layers] == [1, 2, 2, 1]
    check_counter(layers[0], Counter({(0, 1, 2): 1}))
    try:
        check_counter(Counter({(0, 1, 2): 2}), layers[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('DUPLICATE_FIXTURE_ACCEPTED')
    (output / 'partial-evidence.txt').write_text('oracle and duplicate fixture checked\n')
    if fail:
        raise RuntimeError('SYNTHETIC_INTENTIONAL_FAILURE')
    return {'oracle': 'PASS', 'duplicate_fixture': 'REJECTED'}

API_CODE = r"""
import json
import os
from pathlib import Path
from cayleypy import CayleyGraph, PermutationGroups
import multigpubfs
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.generic_session import persistent_graph_workers

def mahonian(n):
    a=[1]
    for k in range(1,n+1):
        b=[0]*(len(a)+k-1)
        for i,x in enumerate(a):
            for j in range(k):
                b[i+j]+=x
        a=b
    return a

root=Path('public-api')
root.mkdir()
multigpubfs.enable_native(recovery=False)
try:
    with persistent_graph_workers() as pool:
        pids=None
        for n,backend,strategy,preset in [
            (4,'segmented','atomic','balanced'),
            (4,'permutation_bitmap','warp','max_capacity'),
            (5,'segmented','batch','max_capacity'),
            (5,'permutation_bitmap','batch','balanced'),
            (4,'segmented','warp','balanced'),
            (4,'permutation_bitmap','atomic','max_capacity'),
        ]:
            graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
            options=NativeOptions(output_root=root,devices=(0,1),autotune=False,
                memory_backend=backend,memory_preset=preset,segment_rows=3,
                segment_strategy=strategy)
            result=graph.bfs(native_options=options)
            assert result.bfs_completed and result.layer_sizes==mahonian(n),result.report
            wanted='EXPERIMENTAL_SEGMENTED24' if backend=='segmented' else 'EXPERIMENTAL_PERMUTATION_BITMAP'
            assert result.report['backend']==wanted,result.report
            assert result.report['segment_strategy']==strategy,result.report
            now=[w.process.pid for w in pool.workers]
            if pids is None:pids=now
            assert now==pids,(pids,now)
            library=os.environ['MGBFS_EXPECTED_LIB_PATH']
            for pid in now:
                assert library in Path(f'/proc/{pid}/maps').read_text(),(pid,library)
            print('PUBLIC_API_PASS',n,backend,strategy,preset,now,flush=True)
        graph=CayleyGraph(PermutationGroups.coxeter(4),device='cpu')
        options=NativeOptions(output_root=root,devices=(0,1),autotune=True,
            memory_backend='permutation_bitmap',memory_preset='balanced',segment_rows=3)
        result=graph.bfs(native_options=options)
        assert result.bfs_completed and result.layer_sizes==mahonian(4),result.report
        assert result.report['backend']=='EXPERIMENTAL_PERMUTATION_BITMAP',result.report
        assert [w.process.pid for w in pool.workers]==pids
        print('PUBLIC_API_AUTOTUNE_SMALL_PASS',flush=True)
finally:
    multigpubfs.disable_native()
print('PUBLIC_API_ALL_PASS',flush=True)
"""

def run_process(label, command, env, output, timeout):
    before = {p for prefix in PREFIXES for p in Path('/workspace').glob(prefix + '*')}
    log = output / (label + '.log')
    started = time.monotonic()
    timed_out = False
    with log.open('wb') as stream:
        process = subprocess.Popen(command, cwd=output, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = process.returncode
    found = {p for prefix in PREFIXES for p in Path('/workspace').glob(prefix + '*')}
    evidence = output / 'evidence'
    for path in sorted(found - before):
        if path.is_dir() and not path.is_symlink():
            evidence.mkdir(exist_ok=True)
            shutil.copytree(path, evidence / path.name)
    return {'name': label, 'exit_code': code, 'timed_out': timed_out,
            'seconds': time.monotonic() - started, 'log': log.name,
            'captured_directories': sorted(p.name for p in found - before)}

def gpu(source, output):
    if os.environ.get('MGBFS_NATIVE_OFFLINE') != '1':
        raise RuntimeError('GPU_ACCEPTANCE_REQUIRES_OFFLINE_PREFETCHED_CACHE')
    if os.environ.get('MGBFS_EXECUTABLE'):
        raise RuntimeError('GPU_ACCEPTANCE_FORBIDS_EXTERNAL_EXECUTABLE_OVERRIDE')
    import multigpubfs
    from multigpubfs.native_distribution import native_runtime

    version = importlib.metadata.version('multigpubfs')
    if version != '0.2.0.dev41':
        raise RuntimeError('UNEXPECTED_INSTALLED_VERSION ' + version)
    package = Path(multigpubfs.__file__).resolve()
    if package.is_relative_to(source.resolve()):
        raise RuntimeError('SOURCE_TREE_SHADOWS_INSTALLED_WHEEL')
    binary, native_env = native_runtime()
    native = Path(binary).resolve()
    bundle = native.parent.parent
    manifest = json.loads((bundle / 'manifest.json').read_text())
    library = bundle / 'lib/libmgbfs_cuda.so'
    for key, path in (('bin/mgbfs', native), ('lib/libmgbfs_cuda.so', library)):
        if digest(path) != manifest['sha256'][key]:
            raise RuntimeError('INSTALLED_COMPONENT_DIGEST_MISMATCH ' + key)
    expected_source = os.environ.get('MGBFS_EXPECTED_SOURCE_SHA', '')
    info = json.loads(package.with_name('_package_info.json').read_text())
    if len(expected_source) != 40 or any(c not in '0123456789abcdef' for c in expected_source):
        raise RuntimeError('EXPECTED_SOURCE_SHA_REQUIRED')
    if any(value != expected_source for value in (info.get('source_commit'),
            manifest.get('native_binary_source_commit'), manifest.get('cuda_library_source_commit'))):
        raise RuntimeError('INSTALLED_SOURCE_IDENTITY_MISMATCH')
    selected = manifest['native_delivery']['selected_component']
    if selected != 'cuda-sm86':
        raise RuntimeError('EXPECTED_SM86_TEST_DEVICE got ' + selected)
    env = dict(native_env)
    env.pop('PYTHONPATH', None)
    env['MGBFS_EXECUTABLE'] = str(native)
    report = {'version': version, 'package': str(package), 'native': str(native),
              'native_sha256': digest(native), 'library_sha256': digest(library),
              'selected_component': selected, 'source_commit': expected_source, 'tests': []}
    for label, filename, limit in TESTS:
        path = source / 'tests' / filename
        if not path.is_file():
            raise RuntimeError('SOURCE_TEST_MISSING ' + filename)
        row = run_process(label, [sys.executable, str(path)], env, output, limit)
        report['tests'].append(row)
        (output / 'progress.json').write_text(json.dumps(report, indent=2))
    api_env = dict(env)
    api_env.pop('MGBFS_EXECUTABLE', None)
    api_env['MGBFS_EXPECTED_LIB_PATH'] = str(library.resolve())
    row = run_process('public-api-default-native', [sys.executable, '-c', API_CODE],
                      api_env, output, 900)
    report['tests'].append(row)
    (output / 'progress.json').write_text(json.dumps(report, indent=2))
    if any(item['exit_code'] != 0 or item['timed_out'] for item in report['tests']):
        raise RuntimeError('GPU_ACCEPTANCE_TEST_FAILED ' +
                           ','.join(x['name'] for x in report['tests']
                                    if x['exit_code'] != 0 or x['timed_out']))
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('synthetic-pass', 'synthetic-fail', 'gpu'), required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = {'mode': args.mode, 'status': 'FAIL'}
    try:
        if args.mode == 'gpu':
            if args.source is None or not args.source.is_dir():
                raise RuntimeError('SOURCE_DIRECTORY_REQUIRED')
            report['data'] = gpu(args.source, output)
        else:
            report['data'] = synthetic(output, args.mode == 'synthetic-fail')
        report['status'] = 'PASS'
    except BaseException as error:
        report['error'] = str(error)
        (output / 'traceback.txt').write_text(traceback.format_exc())
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print('INSTALLED_CANDIDATE_STATUS', report['status'], output, flush=True)
    return 0 if report['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
