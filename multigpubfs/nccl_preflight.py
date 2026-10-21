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
    from .autotune import _stat_identity
    return (str(native), _stat_identity(native), tuple(devices),
            tuple(sorted((k, v) for k, v in env.items()
                         if k.startswith(('CUDA_', 'NCCL_', 'MGBFS_GENERIC_VRAM'))
                         or k in ('LD_LIBRARY_PATH', 'LD_PRELOAD'))))


def prepare(native, devices, env, output, graph=None):
    if len(devices) < 2:
        return dict(env), {'status': 'SINGLE_GPU'}
    root = Path(str(output) + '-communication-evidence').absolute()
    from .generic_session import _active
    key = _identity(native, devices, env)
    cache = getattr(_active, 'communication_profiles', {}) if _active else {}
    if key in cache:
        patch, report = cache[key]
        return dict(env, **patch), dict(report, cache_hit=True)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for name, policy in alternatives(env):
        attempt = root / name
        attempt.mkdir()
        candidate = dict(env, **policy)
        candidate['MGBFS_EXECUTABLE'] = str(native)
        candidate['NCCL_DEBUG'] = 'INFO'
        candidate['NCCL_DEBUG_FILE'] = str(attempt / 'nccl-%h-%p.log')
        # Force this package in subprocesses, including source-tree validation.
        candidate['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
        _write(attempt / 'request.json', {'devices': devices, 'policy': policy})
        processes, logs = [], []
        started = time.monotonic()
        try:
            for rank in range(len(devices)):
                log = (attempt / f'rank-{rank}.log').open('w')
                logs.append(log)
                processes.append(subprocess.Popen(
                    [sys.executable, '-u', '-m', 'multigpubfs.nccl_preflight',
                     str(attempt), str(rank)], env=candidate, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True))
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
                    start_new_session=True)
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
                      'bfs_prefix': None if rehearsal is None else {k:rehearsal.get(k) for k in ('status','reason','layer_sizes')},
                      'prior_attempts': reports, 'cache_hit': False,
                      'seconds': time.monotonic() - started,
                      'evidence': str(root),
                      'scope': 'all selected GPUs; checked all-peer payloads and collectives; not proof of all future BFS kernels or node health'}
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
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
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
        from . import GraphDefinition
        from .launch import run_graph
        graph = GraphDefinition.from_dict(json.loads((root / 'graph.json').read_text()))
        devices = json.loads((root / 'request.json').read_text())['devices']
        run_graph(graph, root / 'prefix', devices=devices, capacity=20000,
                  max_seconds=20, shards=1, autotune=False, backend='generic',
                  _profile_layers=4)
    else:
        _rank(root, int(sys.argv[2]))
