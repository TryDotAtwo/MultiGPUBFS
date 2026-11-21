"""Verify single-host all-peer traffic before admitting large BFS allocations.

Calibration runs in disposable processes, never restarts a BFS, and preserves
all attempts. Transport alternatives are accepted only after exact readback.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MiB = 1 << 20
POLICY_KEYS = ('NCCL_NET_GDR_LEVEL', 'NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE',
               'NCCL_SHM_DISABLE', 'NCCL_NET', 'NCCL_SOCKET_IFNAME')


class ResidentPreflightCleanupError(Exception):
    """A failed diagnostic pool could not be safely discarded."""



def reserve_bytes(ranks, explicit=None):
    if not ranks or any(not r.get('verified') for r in ranks):
        raise RuntimeError('NCCL_PREFLIGHT_UNVERIFIED')
    # Additional absolute allowance covers context differences and late small
    # allocations. Never treat the calibration scratch as NCCL-owned memory.
    measurements = [int(r['nccl_device_bytes']) for r in ranks]
    if any(v < 0 for v in measurements):
        raise ValueError('NCCL_PREFLIGHT_MEMORY_RANGE')
    measured = max(measurements)
    if measured < 0:
        raise ValueError('NCCL_PREFLIGHT_MEMORY_RANGE')
    result = max(2 << 30, measured + (512 << 20))
    if explicit is not None:
        if int(explicit) < 0:
            raise ValueError('GENERIC_VRAM_RESERVE_BYTES')
        result = max(result, int(explicit))
    return result


def alternatives(env):
    yield 'inherited', {}
    # Explicit operator transport choices are authoritative.
    if any(k in env for k in POLICY_KEYS):
        return
    yield 'host-staged-ib', {'NCCL_NET_GDR_LEVEL': 'LOC'}
    yield 'local-shm', {'NCCL_IB_DISABLE': '1', 'NCCL_NET_GDR_LEVEL': 'LOC'}


def _write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _identity(native, devices, env):
    from .autotune import _stat_identity, _dependency_identity
    dependency = _dependency_identity(str(native), env)
    libraries = None if dependency is None else (
        dependency.get('cuda_library_sha256'),
        tuple(tuple(item) for item in dependency.get('resolved_dependencies', ())))
    return (str(native), _stat_identity(native), tuple(devices),
            tuple(sorted((k, v) for k, v in env.items()
                         if k.startswith(('CUDA_', 'NCCL_', 'MGBFS_GENERIC_VRAM'))
                         or k in ('LD_LIBRARY_PATH', 'LD_PRELOAD'))), libraries)


def _group_token(pool):
    from .generic_session import _job_failed
    workers = getattr(pool, 'workers', ()) if pool is not None else ()
    if not workers or any(w.process.poll() is not None or _job_failed(w.job) for w in workers):
        return None
    return (pool.generation, pool.identity, tuple(w.process.pid for w in workers))


def bind_communication_group(pool):
    # Called only after a real production graph job returned successfully.
    # A cold standalone probe cannot certify an unrelated future worker group.
    if pool is not None:
        pool.communication_bound_token = _group_token(pool)


def _bound_cache(pool):
    if pool is None:
        return {}
    token = _group_token(pool)
    if token is None or token != getattr(pool, 'communication_bound_token', None):
        pool.communication_profiles = {}
    return getattr(pool, 'communication_profiles', {})


def _requested_memory_backend(env):
    segmented = env.get('MGBFS_EXPERIMENTAL_SEGMENTED24') == '1'
    bitmap = env.get('MGBFS_EXPERIMENTAL_BITMAP') == '1'
    if bitmap and not segmented:
        raise ValueError('BITMAP_REQUIRES_SEGMENTED_ENGINE')
    return 'permutation_bitmap' if bitmap else 'segmented24' if segmented else 'legacy'


def _preflight_cache_key(native, devices, env, graph):
    # A transport receipt includes a real BFS prefix. Reusing it across a
    # different storage backend or graph would misreport which path was tried.
    # Transport discovery itself is reusable, but a prefix is evidence for
    # the backend *and* the settings under which that prefix was run.  Keep
    # these settings separate from the transport identity so a changed mode
    # can rerun just its prefix without recalibrating NCCL.
    execution_settings = tuple(sorted((k, v) for k, v in env.items()
        if k.startswith(('MGBFS_SEGMENT_', 'MGBFS_GENERIC_', 'MGBFS_EXPERIMENTAL_'))))
    return (_identity(native, devices, env), _requested_memory_backend(env),
            None if graph is None else graph.digest(), execution_settings)


def _run_graph_prefix(root):
    from . import GraphDefinition
    from .launch import run_graph
    graph = GraphDefinition.from_dict(json.loads((root / 'graph.json').read_text()))
    request = json.loads((root / 'request.json').read_text())
    backend = request['memory_backend']
    if backend not in ('legacy', 'segmented24', 'permutation_bitmap'):
        raise ValueError('PREFLIGHT_MEMORY_BACKEND')
    return run_graph(graph, root / 'prefix', devices=request['devices'], capacity=20000,
                     max_seconds=20, shards=1, autotune=False, backend='generic',
                     memory_backend=backend, _profile_layers=4)

def _cached_prefix(root, native, devices, env, patch, report, graph, memory_backend):
    # Reuse the already measured all-rank transport and NCCL reserve, but
    # rehearse this graph with the requested backend.  A cache hit is not
    # evidence that an unrelated graph prefix was run.
    attempt = root / 'cached-transport'
    attempt.mkdir()
    candidate = dict(env, **patch)
    candidate['MGBFS_EXECUTABLE'] = str(native)
    candidate['NCCL_DEBUG'] = 'INFO'
    candidate['NCCL_DEBUG_FILE'] = str(attempt / 'bfs-nccl-%h-%p.log')
    candidate['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
    _write(attempt / 'request.json', {'devices': devices, 'policy': report['policy'],
                                      'memory_backend': memory_backend})
    (attempt / 'graph.json').write_text(graph.to_json())
    # A healthy persistent pool can execute the same bounded graph rehearsal
    # itself. This is still a real BFS prefix on every selected GPU; no
    # transport or graph validation is skipped. Preserve the production NCCL
    # environment, because a per-attempt NCCL_DEBUG_FILE would change pool
    # identity and force all workers to restart.
    from .generic_session import _active
    if _active is not None and getattr(_active, 'workers', None):
        from .launch import run_graph
        try:
            rehearsal = run_graph(graph, attempt / 'prefix', devices=devices,
                                  capacity=20000, max_seconds=20, shards=1,
                                  autotune=False, backend='generic',
                                  memory_backend=memory_backend, executable=str(native),
                                  _native_env=dict(env, **patch), _profile_layers=4)
            if rehearsal['status'] != 'COMPLETE' and len(rehearsal.get('layer_sizes', [])) < 4:
                raise RuntimeError('NCCL_PREFLIGHT_INSUFFICIENT_BFS_PREFIX')
            _write(attempt / 'prefix-execution.json', {'mode': 'resident', 'status': 'VERIFIED'})
            return {k: rehearsal.get(k) for k in ('status', 'reason', 'layer_sizes')}
        except BaseException as error:
            # Only a bounded diagnostic prefix has run at this point. A bad
            # resident job may poison its CUDA context: discard the entire
            # pool before the ordinary fallback tries a transport candidate.
            # No production BFS is restarted or resumed. Invalidate the
            # shared transport evidence too: it predates the failed prefix.
            try:
                _active.close_workers()
                getattr(_active, 'communication_profiles', {}).clear()
            except Exception as cleanup:
                raise ResidentPreflightCleanupError('NCCL_PREFLIGHT_RESIDENT_CLEANUP_FAILED') from cleanup
            _write(attempt / 'prefix-execution.json', {'mode': 'resident', 'status': 'FAILED', 'error': str(error)})
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError('NCCL_PREFLIGHT_PREFIX_STARTUP_TIMEOUT') from error
            raise
    own_groups = candidate.get('MGBFS_DIAGNOSTIC_SUPERVISED') != '1'
    process = None
    try:
        with (attempt / 'prefix.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m',
                'multigpubfs.nccl_preflight', str(attempt), '--prefix'],
                env=candidate, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=own_groups)
            try:
                code = process.wait(timeout=120)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('NCCL_PREFLIGHT_PREFIX_STARTUP_TIMEOUT') from error
        if code:
            raise RuntimeError('NCCL_PREFLIGHT_BFS_PREFIX_FAILED')
        rehearsal = json.loads((attempt / 'prefix' / 'report.json').read_text())
        if rehearsal['status'] != 'COMPLETE' and len(rehearsal.get('layer_sizes', [])) < 4:
            raise RuntimeError('NCCL_PREFLIGHT_INSUFFICIENT_BFS_PREFIX')
        return {k: rehearsal.get(k) for k in ('status', 'reason', 'layer_sizes')}
    finally:
        if process is not None and process.poll() is None:
            try:
                if own_groups: os.killpg(process.pid, signal.SIGTERM)
                else: process.terminate()
            except ProcessLookupError: pass
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    if own_groups: os.killpg(process.pid, signal.SIGKILL)
                    else: process.kill()
                except ProcessLookupError: pass
                process.wait(timeout=3)


def prepare(native, devices, env, output, graph=None):
    if len(devices) < 2:
        return dict(env), {'status': 'SINGLE_GPU'}
    root = Path(str(output) + '-communication-evidence').absolute()
    from .generic_session import _active
    # Family changes require a new worker epoch before consulting the cache.
    # A transport receipt from the previous communicator must not certify it.
    memory_backend = _requested_memory_backend(env)
    if _active is not None:_active.prepare_family(env)
    key = _preflight_cache_key(native, devices, env, graph)
    cache = _bound_cache(_active)
    # If ldd could not identify the CUDA dependency, no later invocation may
    # accept a receipt as if the binary identity had been verified.
    if isinstance(key[0], tuple) and key[0][-1] is None:
        cache.clear()
    if key in cache:
        patch, report = cache[key]
        # Retain the source cost separately: this invocation did no new
        # transport probe or graph-prefix rehearsal.
        return dict(env, **patch), dict(report, cache_hit=True,
                                      source_preflight_seconds=report.get('seconds'),
                                      seconds=0.0)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    # Only transport and reserve are graph-independent.  A different graph
    # still gets its own real backend-specific BFS prefix before admission.
    previous = next(((patch, report) for cached_key, (patch, report) in cache.items()
                     if len(cached_key) in (3, 4) and cached_key[0] == key[0]
                     and report.get('status') == 'VERIFIED'), None)
    if previous is not None and graph is not None:
        patch, old_report = previous
        started = time.monotonic()
        try:
            prefix = _cached_prefix(root, native, devices, env, patch, old_report,
                                    graph, memory_backend)
            report = dict(old_report, cache_hit=True, transport_cache_hit=True,
                          cache_source_graph=old_report.get('rehearsal_graph'),
                          rehearsal_graph=graph.digest(), rehearsal_backend=memory_backend,
                          rehearsal_geometry={'shards':1,'profile_max_layers':4,'capacity':20000},
                          production_geometry_status='NOT_VERIFIED_BY_PREFLIGHT',
                          bfs_prefix=prefix, seconds=time.monotonic()-started,
                          evidence=str(root))
            _write(root / 'report.json', report)
            cache[key] = (patch, report)
            _active.communication_profiles = cache
            return dict(env, **patch), report
        except (RuntimeError, TimeoutError, ValueError, OSError, KeyError) as error:
            reports.append({'mode':'cached-transport-prefix','status':'FAILED',
                            'error':str(error),'seconds':time.monotonic()-started})
            _write(root / 'cached-transport-failure.json', reports[-1])
    for name, policy in alternatives(env):
        attempt = root / name
        attempt.mkdir()
        candidate = dict(env, **policy)
        candidate['MGBFS_EXECUTABLE'] = str(native)
        candidate['NCCL_DEBUG'] = 'INFO'
        candidate['NCCL_DEBUG_FILE'] = str(attempt / 'nccl-%h-%p.log')
        # Force this package in subprocesses, including source-tree validation.
        candidate['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
        _write(attempt / 'request.json', {'devices': devices, 'policy': policy, 'memory_backend': memory_backend})
        processes, logs = [], []
        own_groups = candidate.get("MGBFS_DIAGNOSTIC_SUPERVISED") != "1"
        started = time.monotonic()
        try:
            for rank in range(len(devices)):
                log = (attempt / f'rank-{rank}.log').open('w')
                logs.append(log)
                processes.append(subprocess.Popen(
                    [sys.executable, '-u', '-m', 'multigpubfs.nccl_preflight',
                     str(attempt), str(rank)], env=candidate, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=own_groups))
            deadline = started + 90
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('NCCL_PREFLIGHT_RANK_FAILED')
                if time.monotonic() >= deadline:
                    raise TimeoutError('NCCL_PREFLIGHT_STARTUP_TIMEOUT')
                time.sleep(.02)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError('NCCL_PREFLIGHT_RANK_FAILED')
            ranks = [json.loads((attempt / f'rank-{r}.json').read_text())
                     for r in range(len(devices))]
            if ([r.get('rank') for r in ranks] != list(range(len(devices)))
                    or [r.get('device') for r in ranks] != list(devices)):
                raise RuntimeError('NCCL_PREFLIGHT_RANK_IDENTITY')
            reserve = reserve_bytes(ranks, env.get('MGBFS_GENERIC_VRAM_RESERVE_BYTES'))
            patch = dict(policy, MGBFS_GENERIC_VRAM_RESERVE_BYTES=str(reserve))
            rehearsal = None
            if graph is not None:
                # Exercise the graph kernels plus the same NCCL exchange sequence
                # that reported CUDA999. A communication-only pass is insufficient.
                (attempt / 'graph.json').write_text(graph.to_json())
                rehearsal_env = dict(candidate, **patch)
                rehearsal_env['NCCL_DEBUG_FILE'] = str(attempt / 'bfs-nccl-%h-%p.log')
                log = (attempt / 'prefix.log').open('w')
                logs.append(log)
                process = subprocess.Popen([sys.executable, '-u', '-m',
                    'multigpubfs.nccl_preflight', str(attempt), '--prefix'],
                    env=rehearsal_env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=own_groups)
                processes.append(process)
                try:
                    code = process.wait(timeout=120)
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError('NCCL_PREFLIGHT_PREFIX_STARTUP_TIMEOUT') from error
                if code:
                    raise RuntimeError('NCCL_PREFLIGHT_BFS_PREFIX_FAILED')
                rehearsal = json.loads((attempt / 'prefix' / 'report.json').read_text())
                if rehearsal['status'] != 'COMPLETE' and len(rehearsal.get('layer_sizes', [])) < 4:
                    raise RuntimeError('NCCL_PREFLIGHT_INSUFFICIENT_BFS_PREFIX')
            report = {'status': 'VERIFIED', 'selected': name, 'policy': policy,
                      'reserve_bytes': reserve, 'ranks': ranks,
                      'rehearsal_graph': graph.digest() if graph is not None else None,
                      'rehearsal_backend': memory_backend if graph is not None else None,
                      'rehearsal_geometry': None if graph is None else {'shards':1,'profile_max_layers':4,'capacity':20000},
                      'production_geometry_status': 'NOT_VERIFIED_BY_PREFLIGHT',
                      'bfs_prefix': None if rehearsal is None else {k:rehearsal.get(k) for k in ('status','reason','layer_sizes')},
                      'prior_attempts': reports, 'cache_hit': False,
                      'seconds': time.monotonic() - started,
                      'evidence': str(root),
                      'scope': 'all selected GPUs; transport and one-shard bounded BFS prefix only; production geometry, all future BFS kernels and node health are not verified'}
            _write(root / 'report.json', report)
            if _active is not None:
                cache[key] = (patch, report)
                _active.communication_profiles = cache
            return dict(env, **patch), report
        except (RuntimeError, TimeoutError) as error:
            report = {'mode': name, 'status': 'FAILED', 'error': str(error),
                      'seconds': time.monotonic() - started}
            reports.append(report)
            _write(attempt / 'failure.json', report)
        finally:
            for process in processes:
                if process.poll() is None:
                    try:
                        if own_groups: os.killpg(process.pid, signal.SIGTERM)
                        else: process.terminate()
                    except ProcessLookupError: pass
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        if own_groups: os.killpg(process.pid, signal.SIGKILL)
                        else: process.kill()
                    except ProcessLookupError: pass
                    process.wait(timeout=3)
            for log in logs:
                log.close()
    _write(root / 'report.json', {'status': 'FAILED', 'attempts': reports})
    raise RuntimeError('NCCL_PREFLIGHT_NO_VERIFIED_TRANSPORT: evidence=' + str(root))


def _rank(root, rank):
    from .diagnostics import _cuda
    request = json.loads((root / 'request.json').read_text())
    devices = request['devices']
    world = len(devices)
    cuda, lib, ck = _cuda(devices[rank], True)
    cuda.cudaMemGetInfo.argtypes = [C.POINTER(C.c_size_t)] * 2
    cuda.cudaMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
    def free():
        a, b = C.c_size_t(), C.c_size_t()
        ck(cuda.cudaMemGetInfo(C.byref(a), C.byref(b)))
        return a.value
    before = free()
    uid = C.create_string_buffer(128)
    lib.mgbfs_nccl_unique_id.argtypes = [C.c_void_p]
    if rank == 0:
        ck(lib.mgbfs_nccl_unique_id(uid))
        tmp = root / 'id.tmp'
        tmp.write_bytes(uid.raw)
        tmp.replace(root / 'id.bin')
    deadline = time.monotonic() + 60
    while not (root / 'id.bin').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('NCCL_ID_TIMEOUT')
        time.sleep(.01)
    uid = C.create_string_buffer((root / 'id.bin').read_bytes(), 128)
    comm, error = C.c_void_p(), C.create_string_buffer(1024)
    lib.mgbfs_nccl_create.argtypes = [C.c_uint32]*3 + [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p, C.c_size_t]
    code = lib.mgbfs_nccl_create(rank, world, devices[rank], uid, C.byref(comm), error, len(error))
    if code:
        raise RuntimeError('NCCL_INIT: ' + str(code) + ' ' + error.value.decode())
    q = max(16, min(524288, (1024*MiB)//(world*160)))
    sizes = [world*4, world*q*32, world*q*128]
    allocations = []
    for size in sizes*2:
        ptr = C.c_void_p()
        ck(cuda.cudaMalloc(C.byref(ptr), size))
        allocations.append(ptr)
    fn = lib.mgbfs_nccl_exchange_bounded_triplets
    fn.argtypes = [C.c_void_p] + [C.c_uint32]*5 + [C.c_uint64, C.c_uint32] + [C.c_void_p]*7
    fn.restype = C.c_int
    maximum = 0
    for used in (1, 16, q, q):
        print('all-peer used=' + str(used), flush=True)
        for ptr, size in zip(allocations[:3], sizes):
            ck(cuda.cudaMemset(ptr, rank+1, size))
        for ptr, size in zip(allocations[3:], sizes):
            ck(cuda.cudaMemset(ptr, 0, size))
        code = fn(comm, rank, world, 1, q, used, 128, 8, *allocations, None)
        if code:
            raise RuntimeError('NCCL_ALL_PEER: ' + str(code))
        ck(cuda.cudaDeviceSynchronize())
        for peer in range(world):
            if peer == rank:
                continue
            offsets = [(allocations[3], peer*4, 4),
                       (allocations[4], peer*q*32, 32),
                       (allocations[4], peer*q*32+(used-1)*32, 32)]
            for element in range(16):
                offsets.extend([(allocations[5], peer*q*128+element*q*8, 8),
                                (allocations[5], peer*q*128+element*q*8+(used-1)*8, 8)])
            for ptr, offset, size in offsets:
                host = C.create_string_buffer(size)
                ck(cuda.cudaMemcpy(host, C.c_void_p(ptr.value+offset), size, 2))
                if host.raw != bytes([peer+1])*size:
                    raise RuntimeError('NCCL_PAYLOAD_MISMATCH peer='+str(peer))
        maximum = max(maximum, before-free()-sum(sizes)*2)
    # Exercise the same scalar collective families as BFS, not only point-to-point.
    send, recv = allocations[0], allocations[3]
    value = C.c_uint32(rank+1)
    ck(cuda.cudaMemcpy(send, C.byref(value), 4, 1))
    for name, expected in [('mgbfs_nccl_all_reduce_max_u32', [world]),
                           ('mgbfs_nccl_all_gather_u32', list(range(1,world+1)))]:
        collective = getattr(lib, name)
        collective.argtypes = [C.c_void_p]*4
        ck(collective(comm, send, recv, None))
        ck(cuda.cudaDeviceSynchronize())
        host = (C.c_uint32*len(expected))()
        ck(cuda.cudaMemcpy(host, recv, C.sizeof(host), 2))
        if list(host) != expected:
            raise RuntimeError('NCCL_COLLECTIVE_MISMATCH '+name)
        maximum = max(maximum, before-free()-sum(sizes)*2)
    # Rank process exit releases only its own CUDA resources. Explicit abort
    # cannot block indefinitely in normal teardown; parent enforces startup bound.
    (root / f'rank-{rank}.readback-done').touch()
    while not all((root / f'rank-{r}.readback-done').exists() for r in range(world)):
        time.sleep(.01)
    lib.mgbfs_nccl_abort.argtypes = [C.c_void_p]
    ck(lib.mgbfs_nccl_abort(comm))
    _write(root / f'rank-{rank}.json', {'verified': True, 'rank': rank,
           'device': devices[rank], 'nccl_device_bytes': max(0,maximum),
           'scratch_bytes': sum(sizes)*2, 'queue_capacity': q,
           'passes': 4, 'payload_readback': True})


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if sys.argv[2] == '--prefix':
        _run_graph_prefix(root)
    else:
        _rank(root, int(sys.argv[2]))
