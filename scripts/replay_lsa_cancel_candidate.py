"""Incremental Linux replay on an already provisioned diagnostic work tree.

No provisioning, rental, download, publication or hardware-gate substitution.
Records the dirty source digest: this is a candidate test, not a commit replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import re
import signal


def cleanup_rank_processes(processes):
    """Cancel all owned sessions before bounded reaping; attempt every rank."""
    errors = []
    for process in processes:
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Child exited between poll and signal; still reap it below.
        except OSError as error:
            errors.append(f'RANK_SIGNAL_{process.pid}: {error}')
    for process in processes:
        try:
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f'RANK_REAP_{process.pid}: {error}')
    return errors


def process_case_timeout(tool, requested):
    if requested is not None:
        if type(requested) is not int or not 1 <= requested <= 3600:
            raise ValueError('CASE_TIMEOUT_CONFIG')
        return requested
    # The local full-state racecheck already takes >360s. A 120s cutoff
    # would kill a healthy two-rank sanitizer gate before collecting evidence.
    return 600 if tool == 'racecheck' else (120 if tool else 45)


def configure_epoch_window(env, requested=None):
    value = requested if requested is not None else env.get('MGBFS_EPOCH_WINDOW', '2')
    text = str(value)
    if re.fullmatch(r'[0-9]+', text) is None or not 2 <= int(text) <= 0xffffffff:
        raise ValueError('EPOCH_WINDOW_CONFIG')
    window = int(text)
    env['MGBFS_EPOCH_WINDOW'] = str(window)
    return window


def configure_hash_seed(env, value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{32}', value) is None:
        raise ValueError('HASH_SEED_HEX_32')
    seed = value.lower()
    env['MGBFS_HASH_SEED_HEX'] = seed
    return seed


def configure_run_epoch_window(env, config, requested=None):
    window = config.get('completion_epoch_window', 2)
    if type(window) is not int or not 2 <= window <= 0xffffffff:
        raise ValueError('RUN_REPLAY_EPOCH_WINDOW_INVALID')
    if requested is not None and requested != window:
        raise ValueError('RUN_REPLAY_EPOCH_WINDOW_MISMATCH')
    return configure_epoch_window(env, window)


def configure_route_banks(env, requested=None):
    value = requested if requested is not None else env.get('MGBFS_ROUTE_BANKS', '2')
    if isinstance(value, bool) or str(value) not in ('2', '3', '4'):
        raise ValueError('ROUTE_BANK_CONFIG')
    banks = int(value)
    env['MGBFS_ROUTE_BANKS'] = str(banks)
    return banks


def configure_run_route_banks(env, config, requested=None):
    banks = config.get('capacities', {}).get('route_slot_count')
    if type(banks) is not int or not 2 <= banks <= 4:
        raise ValueError('RUN_REPLAY_ROUTE_BANKS_INVALID')
    if requested is not None and requested != banks:
        raise ValueError('RUN_REPLAY_ROUTE_BANKS_MISMATCH')
    return configure_route_banks(env, banks)


def rank_arguments(case, reference_group, batch, run_config=None):
    paths = [str(case / 'bootstrap'), str(case / 'archive'), str(case / 'result')]
    if run_config is not None:
        return ['run', str(run_config), *paths]
    return ['bench', '--reference', reference_group, str(batch), *paths]


def configure_owner_environment(env, backend, rank_map, run_config=None):
    if backend not in ('CUCO_RANK', 'CUB_SORT_MERGE', 'BMMA_BUCKET'):
        raise ValueError('UNKNOWN_OWNER_BACKEND')
    if rank_map not in ('0,1', '1,0'):
        raise ValueError('INVALID_TWO_RANK_MAP')
    pool = 64 << 20
    if run_config is not None:
        if run_config['owner_backend'] != backend:
            raise ValueError('RUN_REPLAY_OWNER_MISMATCH')
        pool = run_config.get('library_pool_bytes')
        if backend == 'CUCO_RANK':
            if type(pool) is not int or not 0 < pool <= 0xffffffffffffffff or pool % 256:
                raise ValueError('RUN_REPLAY_LIBRARY_POOL_REQUIRED_ALIGNED')
        elif pool is not None:
            raise ValueError('RUN_REPLAY_UNUSED_LIBRARY_POOL')
    env.update(MGBFS_OWNER_BACKEND=backend, MGBFS_RANK_MAP=rank_map)
    if backend == 'CUCO_RANK':
        env['MGBFS_LIBRARY_POOL_BYTES'] = str(pool)
    else:
        env.pop('MGBFS_LIBRARY_POOL_BYTES', None)


def instrument_rank_command(binary, arguments, tool, report_prefix):
    target = [str(binary), *arguments]
    if tool is None:
        return target
    if tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
        return ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
                '--error-exitcode', '97', *target]
    if tool == 'nsys':
        return ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=process-tree',
                '--cuda-trace-all-apis=true',
                '--cudabacktrace=memory:0,sync:0,other:0', '--cpuctxsw=none',
                '-o', str(report_prefix), *target]
    raise ValueError('UNKNOWN_PROCESS_INSTRUMENTATION')


def instrumentation_clean(text, tool):
    if tool == 'racecheck':
        rows = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', text)
        return bool(rows) and not any(int(n) for row in rows for n in row)
    rows = re.findall(r'ERROR SUMMARY: (\d+) errors', text)
    return bool(rows) and not any(int(n) for n in rows)


def record_archive_verification(row, case, **kwargs):
    """Keep process outcomes even when verification fails; never promote to PASS."""
    try:
        row['full_state_oracle'] = verify_process_archives(case, **kwargs)
    except Exception as error:
        row['pass'] = False
        row['verification_error'] = dict(type=type(error).__name__, detail=str(error))


def failure_has_no_complete(case):
    """A nonzero exit alone cannot rule out false per-rank completion."""
    result = case / 'result'
    if (result / 'group-complete.json').exists():
        return False
    for path in result.glob('rank-*.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        if not isinstance(record, dict) or record.get('status') != 'INCOMPLETE':
            return False
    return True


def s4_reference_layers():
    return s_reference_layers(4)


def s_reference_layers(n):
    """Independent full-state oracle: row permutations, no GPU hash/dedup code."""
    if not 2 <= n <= 8:
        raise ValueError('BOUNDED_REFERENCE_SIZE')
    frontier = {tuple(range(n))}
    visited, layers = set(frontier), []
    while frontier:
        layers.append({bytes(int(column == state[row])
                             for row in range(n) for column in range(n))
                       for state in frontier})
        children = set()
        for state in frontier:
            children.update((state[1:] + state[:1], state[-1:] + state[:-1],
                             (state[1], state[0], *state[2:])))
        frontier = children - visited
        visited.update(frontier)
    return layers


def u_reference_layers(n, modulus):
    """Exact full bytes; elementary left row operations, no GPU hashes."""
    if not 2 <= n <= 4 or not 2 <= modulus <= 6:
        raise ValueError('BOUNDED_REFERENCE_UNITRIANGULAR')
    frontier = {bytes(int(i == j) for i in range(n) for j in range(n))}
    visited, layers = set(frontier), []
    while frontier:
        layers.append(frontier)
        children = set()
        for state in frontier:
            for row in range(n - 1):
                for delta in (1, modulus - 1):
                    child = bytearray(state)
                    for column in range(n):
                        index = row * n + column
                        child[index] = (state[index] + delta * state[index + n]) % modulus
                    children.add(bytes(child))
        frontier = children - visited
        visited.update(frontier)
    return layers


def verify_process_archives(case, frame_reader=None, n=4, world=2, modulus=None, expected_seed=None,
                            expected_epoch_window=None, expected_config_digest=None,
                            expected_run_contract=None, expected_route_banks=None,
                            require_bank_reuse=False, expected_owner_backend=None):
    """Reuse the checksummed archive reader, then compare every state/depth."""
    if frame_reader is None:
        from export_hf_dataset import frames
        frame_reader = frames
    expected = s_reference_layers(n) if modulus is None else u_reference_layers(n, modulus)
    actual = [set() for _ in expected]
    visited, digest = set(), None
    bank_reuses = []
    rank_hashes = []
    rank_digests = []
    for rank in range(world):
        local = [0] * len(expected)
        for depth, width, count, payload, config in frame_reader(
                case / f'archive-rank-{rank}.mgbfsar1'):
            if width != n * n or not 0 <= depth < len(expected):
                raise ValueError('PROCESS_ORACLE_SHAPE')
            if digest is not None and config != digest:
                raise ValueError('PROCESS_ORACLE_CONFIG')
            if expected_config_digest is not None and config != expected_config_digest:
                raise ValueError('PROCESS_ORACLE_REQUESTED_CONFIG')
            digest = config
            for index in range(count):
                state = payload[index * width:(index + 1) * width]
                if state in visited:
                    raise ValueError('PROCESS_ORACLE_DUPLICATE')
                visited.add(state)
                actual[depth].add(state)
                local[depth] += 1
        result_bytes = (case / f'result/rank-{rank}.json').read_bytes()
        result = json.loads(result_bytes)
        rank_hashes.append(list(hashlib.sha256(result_bytes).digest()))
        rank_digests.append(result.get('bootstrap_digest'))
        if expected_owner_backend is not None and result.get('owner_backend') != expected_owner_backend:
            raise ValueError('PROCESS_ORACLE_OWNER_BACKEND')
        if expected_seed is not None and result.get('hash_seed_hex') != expected_seed:
            raise ValueError('PROCESS_ORACLE_HASH_SEED')
        if expected_epoch_window is not None and result.get('epoch_window') != expected_epoch_window:
            raise ValueError('PROCESS_ORACLE_EPOCH_WINDOW')
        if expected_route_banks is not None and result.get('route_banks') != expected_route_banks:
            raise ValueError('PROCESS_ORACLE_ROUTE_BANKS')
        reuse = result.get('route_bank_reuses')
        if require_bank_reuse and (type(reuse) is not int or reuse < 0):
            raise ValueError('PROCESS_ORACLE_ROUTE_BANK_REUSE')
        bank_reuses.append(reuse)
        if expected_run_contract is not None and result.get('run_contract') != expected_run_contract:
            raise ValueError('PROCESS_ORACLE_RUN_CONTRACT')
        if expected_config_digest is not None and result.get('bootstrap_digest') != list(bytes.fromhex(expected_config_digest)):
            raise ValueError('PROCESS_ORACLE_BOOTSTRAP_CONFIG')
        if result.get('status') != 'COMPLETE' or result.get('local_layer_sizes') != local:
            raise ValueError('PROCESS_ORACLE_RANK_COUNTS')
        if (type(result.get('rank')) is not int or result['rank'] != rank or
                type(result.get('world_size')) is not int or result['world_size'] != world or
                result.get('archive_commit_scope') != 'file_fsync'):
            raise ValueError('PROCESS_ORACLE_RANK_IDENTITY_OR_SCOPE')
    if actual != expected:
        raise ValueError('PROCESS_ORACLE_LAYER')
    if require_bank_reuse and not any(bank_reuses):
        raise ValueError('PROCESS_ORACLE_ROUTE_BANK_REUSE')
    try:
        marker = json.loads((case / 'result/group-complete.json').read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError('PROCESS_ORACLE_GROUP_COMMIT') from error
    if (not isinstance(marker, dict) or marker.get('schema') != 'mgbfs-group-run-commit-v1' or
            marker.get('status') != 'COMPLETE' or type(marker.get('world_size')) is not int or
            marker['world_size'] != world or marker.get('archive_commit_scope') != 'file_fsync' or
            marker.get('bootstrap_digest') != list(bytes.fromhex(digest)) or
            marker.get('rank_sha256') != rank_hashes or
            any(value != list(bytes.fromhex(digest)) for value in rank_digests)):
        raise ValueError('PROCESS_ORACLE_GROUP_COMMIT')
    return dict(unique_states=len(visited), layer_sizes=list(map(len, actual)), route_bank_reuses=bank_reuses,
                scope=f'{world} independent rank-process archives; full canonical states at every depth')


def install_termination_handler():
    # The notebook supervisor must be able to cancel the replay while its
    # independent rank sessions are alive. Unwind through the existing
    # per-case finally, which kills and reaps every started rank.
    def terminate_replay(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate_replay)


def main():
    install_termination_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--sanitizers", action="store_true")
    parser.add_argument('--instrument-processes', choices=(
        'memcheck', 'racecheck', 'initcheck', 'synccheck', 'nsys'))
    parser.add_argument('--healthy-only', action='store_true')
    parser.add_argument('--case-timeout-seconds', type=int,
                        help='explicit bounded per-case deadline (1..3600s), recorded in report')
    parser.add_argument('--capacity-faults', action='store_true',
                        help='also exhaust actual owner capacity on either independent rank')
    parser.add_argument('--reference-size', type=int, choices=range(2, 9), default=4)
    parser.add_argument('--unitriangular-modulus', type=int, choices=range(2, 7),
                        help='full-state U4 oracle instead of symmetric permutation matrices')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--run-config', type=Path,
                        help='test the typed run dispatcher with this immutable config instead of bench')
    parser.add_argument('--epoch-window', type=int,
                        help='bounded completion credits; inherits environment or defaults to 2, not payload slots')
    parser.add_argument('--route-banks', type=int, choices=(2, 3, 4),
                        help='physical source payload banks, independent of completion credits and receive slot')
    parser.add_argument('--require-bank-reuse', action='store_true',
                        help='healthy oracle must observe within-depth physical bank reuse on at least one rank')
    parser.add_argument('--profile', choices=('DENSE', 'HASH_FIRST'), default='DENSE')
    parser.add_argument('--pre-dedup', choices=('ON', 'OFF'), default='ON')
    parser.add_argument('--rank-map', choices=('0,1', '1,0'), default='0,1')
    parser.add_argument('--hash-seed-hex', default='000000000000000000000000013527dc',
                        help='explicit 128-bit seed, 32 hex digits; overrides inherited environment')
    parser.add_argument('--owner-backend', choices=('CUCO_RANK', 'CUB_SORT_MERGE', 'BMMA_BUCKET'),
                        default='CUCO_RANK')
    args = parser.parse_args()
    try:
        case_timeout = process_case_timeout(args.instrument_processes, args.case_timeout_seconds)
        seed_environment = {}
        seed_hex = configure_hash_seed(seed_environment, args.hash_seed_hex)
        epoch_environment = dict(os.environ)
        epoch_window = (2 if args.run_config is not None else
                        configure_epoch_window(epoch_environment, args.epoch_window))
        route_banks = (2 if args.run_config is not None else
                       configure_route_banks(epoch_environment, args.route_banks))
    except ValueError as error:
        parser.error(str(error))
    if args.batch < 1:
        parser.error('--batch must be positive')
    if args.unitriangular_modulus is not None and args.reference_size != 4:
        parser.error('unitriangular replay requires --reference-size 4')
    production_config = None
    replay_config = None
    if args.run_config is not None:
        try:
            production_config = args.run_config.read_bytes()
            config = json.loads(production_config)
            replay_config = config
            if config['topology']['world_size'] != 2 or config.get('macro_depth', 1) != 1:
                raise ValueError('RUN_REPLAY_REQUIRES_TWO_RANK_UNIT_DEPTH')
            if config['graph']['rows'] != args.reference_size:
                raise ValueError('RUN_REPLAY_ORACLE_DEGREE_MISMATCH')
            if args.unitriangular_modulus is not None and config['graph']['modulus'] != args.unitriangular_modulus:
                raise ValueError('RUN_REPLAY_ORACLE_MODULUS_MISMATCH')
            seed_hex = f"{int.from_bytes(bytes(config['seed']), 'little'):032x}"
            configure_hash_seed(seed_environment, seed_hex)
            epoch_window = configure_run_epoch_window(epoch_environment, config, args.epoch_window)
            route_banks = configure_run_route_banks(epoch_environment, config, args.route_banks)
            args.batch = config['parent_batch']
            args.profile = config['frontier_profile']
            args.owner_backend = config['owner_backend']
            args.pre_dedup = 'ON' if config['local_pre_dedup'] else 'OFF'
            args.rank_map = ','.join(map(str, config['topology']['logical_owner_to_rank']))
            configure_owner_environment({}, args.owner_backend, args.rank_map, replay_config)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
    work, output = args.work.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_snapshot = None
    if production_config is not None:
        config_snapshot = output / 'run-config.json'
        config_snapshot.write_bytes(production_config)
    source = work / "source"
    env = dict(os.environ)
    env.update(seed_environment)
    env['MGBFS_EPOCH_WINDOW'] = str(epoch_window)
    env['MGBFS_ROUTE_BANKS'] = str(route_banks)
    if args.instrument_processes == 'nsys':
        # Batch/archive attribution must be present in a full runtime trace.
        # This enables ranges only, never TRACE_ROUTE's diagnostic host waits.
        env['MGBFS_TRACE_RANGES'] = '1'
    site = next((work / "venv/lib").glob("python*/site-packages"))
    libdirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
    env.update(CARGO_HOME=str(work / "cargo"), RUSTUP_HOME=str(work / "rustup"),
               MGBFS_CUDA_LIB_DIR=str(work / "native-build"),
               MGBFS_LIBRARY_OWNER_LIB_DIR=str(work / "library-build"),
               MGBFS_CUDART_LIB_DIR=str(work / "cuda-12.9/lib"))
    env["LD_LIBRARY_PATH"] = ":".join([str(work / "native-build"),
        str(work / "library-build"), str(work / "nccl/nvidia/nccl/lib"),
        str(work / "cuda-12.9/lib"), *libdirs, env.get("LD_LIBRARY_PATH", "")])
    env["PATH"] = str(work / "cargo/bin") + ":" + env.get("PATH", "")
    report = {"scope": "dirty candidate; hardware diagnostic, not T4 acceptance",
        "epoch_window": epoch_window, "route_banks": route_banks,
        "run_contract": "RunConfigV1" if production_config is not None else "reference_bench",
        "run_config_sha256": hashlib.sha256(production_config).hexdigest() if production_config is not None else None,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=source, text=True).strip(), "cases": []}
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    (output / "source.patch").write_bytes(diff)
    report["source_patch_sha256"] = hashlib.sha256(diff).hexdigest()
    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    save()
    with (output / "build.log").open("w") as log:
        subprocess.run([str(work / "cargo/bin/cargo"), "build", "--locked",
            "-p", "mgbfs-cli", "--features", "cuda,library-owner"],
            cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=600, check=True)
    expected_config_digest = None
    if config_snapshot is not None:
        preflight = subprocess.check_output([str(source / 'target/debug/mgbfs'),
            'preflight', '--offline', str(config_snapshot)], cwd=source, env=env, text=True)
        admission = json.loads(preflight)
        if admission.get('status') != 'CONFIG_VALIDATED':
            raise ValueError('RUN_REPLAY_CONFIG_NOT_VALIDATED')
        expected_config_digest = admission['config_digest']
        report['expected_config_digest'] = expected_config_digest
        save()
    env.update(MGBFS_OWNER_BACKEND=args.owner_backend, MGBFS_TRACE_FAILURE_TEARDOWN="1",
        MGBFS_PROFILE=args.profile,
        MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128", MGBFS_BUCKETS="8",
        MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2", MGBFS_BUCKET_CAPACITY="32",
        MGBFS_STATE_CODEC="matrix_u8", MGBFS_ARCHIVE_CODEC="matrix_u8",
        MGBFS_ARCHIVE_ROWS="3", MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0",
        MGBFS_PRE_DEDUP=args.pre_dedup, MGBFS_BENCH_SKIP_ARCHIVE="0", MGBFS_ARCHIVE_STREAM="0",
        MGBFS_CAPACITY_MODE="max_per_rank",
        MGBFS_TRANSPORT_BACKEND="NCCL_LSA", NCCL_CUMEM_ENABLE="1")
    configure_owner_environment(env, args.owner_backend, args.rank_map, replay_config)
    if args.reference_size != 4 or args.unitriangular_modulus is not None:
        # Capacity is deliberately conservative for this bounded full-state
        # oracle, not a prediction of unknown production frontiers.
        import math
        capacity = max(64, math.factorial(args.reference_size) if args.unitriangular_modulus is None
                       else args.unitriangular_modulus ** 6)
        env.update(MGBFS_BENCH_CAPACITY=str(capacity),
                   MGBFS_FUTURE_CAPACITY=str(capacity * 2),
                   MGBFS_BUCKET_CAPACITY=str(capacity),
                   MGBFS_ARCHIVE_ROWS='512', MGBFS_ARCHIVE_SLOTS='128')
    report['reference_size'] = args.reference_size
    reference_group = (f's{args.reference_size}' if args.unitriangular_modulus is None
                       else f'u4m{args.unitriangular_modulus}')
    report['reference_group'] = reference_group
    report['batch'] = args.batch
    report['profile'] = args.profile
    report['pre_dedup'] = args.pre_dedup
    report['owner_backend'] = args.owner_backend
    report['rank_map'] = args.rank_map
    report['hash_seed_hex'] = seed_hex
    report['owner_dag_capture_requested'] = 'MGBFS_TEST_OWNER_DAG_CAPTURE' in env
    faults = [("startup", "MGBFS_TEST_NCCL_STARTUP_FAULT_RANK"),
              ("constructor", "MGBFS_TEST_CONSTRUCTOR_FAULT_RANK"),
              ("constructor_late", "MGBFS_TEST_CONSTRUCTOR_LATE_FAULT_RANK"),
              ("owner", "MGBFS_TEST_OWNER_HOST_FAULT_RANK"),
              ("admission", "MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK"),
              ("worker_write", "MGBFS_TEST_ARCHIVE_WORKER_WRITE_FAULT_RANK"),
              ("worker_sync", "MGBFS_TEST_ARCHIVE_WORKER_SYNC_FAULT_RANK"),
              ("finish", "MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK")]
    cases = [("healthy", None, None)] + [(name, key, rank)
        for name, key in faults for rank in (0, 1)]
    if args.capacity_faults:
        cases += [('capacity', 'MGBFS_TEST_OWNER_CAPACITY_RANK', rank) for rank in (0, 1)]
    if args.healthy_only:
        cases = cases[:1]
    report['process_instrumentation'] = args.instrument_processes
    report['case_timeout_seconds'] = case_timeout
    for name, key, fault_rank in cases:
        case = output / f"{name}-{fault_rank}"
        case.mkdir()
        case_env = dict(env)
        case_env.pop('MGBFS_TEST_OWNER_CAPACITY_RANK', None)
        for _, fault in faults:
            case_env.pop(fault, None)
        if key:
            case_env[key] = str(fault_rank)
        case_env["TORCHELASTIC_RUN_ID"] = "lsa-cancel-" + uuid.uuid4().hex
        processes, streams = [], []
        started = time.monotonic()
        try:
            for rank in (0, 1):
                stream = (case / f"rank-{rank}.log").open("w")
                streams.append(stream)
                rank_env = dict(case_env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
                command = instrument_rank_command(source / 'target/debug/mgbfs',
                    rank_arguments(case, reference_group, args.batch, config_snapshot),
                    args.instrument_processes, case / f'rank-{rank}')
                processes.append(subprocess.Popen(command, cwd=source,
                    env=rank_env, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True))
            deadline = started + case_timeout
            while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
                time.sleep(.05)
            forced = any(p.poll() is None for p in processes)
            row = {"name": name, "fault_rank": fault_rank, "forced_cleanup": forced,
                   "returncodes": [p.poll() for p in processes],
                   "seconds": time.monotonic() - started,
                   "deadline_seconds": case_timeout,
                   "group_complete": (case / "result/group-complete.json").exists()}
            row["pass"] = (not forced and (all(c == 0 for c in row["returncodes"])
                if key is None else all(c not in (None, 0) for c in row["returncodes"])
                and failure_has_no_complete(case)))
            if key is None:
                row["pass"] &= row["group_complete"]
                if row["pass"]:
                    record_archive_verification(row, case, n=args.reference_size,
                        modulus=args.unitriangular_modulus, expected_seed=seed_hex,
                        expected_epoch_window=epoch_window, expected_config_digest=expected_config_digest,
                        expected_run_contract='RunConfigV1' if config_snapshot is not None else None,
                        expected_route_banks=route_banks, require_bank_reuse=args.require_bank_reuse,
                        expected_owner_backend=args.owner_backend)
            for stream in streams:
                stream.flush()
            text = "\n".join((case / f"rank-{rank}.log").read_text(errors="replace")
                             for rank in (0, 1))
            if report['owner_dag_capture_requested'] and name == 'healthy':
                row['owner_dag_capture_launches'] = [
                    (case / f'rank-{rank}.log').read_text(errors='replace')
                    .count('MGBFS_OWNER_DAG_CAPTURE launched') for rank in (0, 1)]
                row['pass'] &= all(row['owner_dag_capture_launches'])
            if args.instrument_processes and args.instrument_processes != 'nsys':
                row['instrumentation_clean'] = all(instrumentation_clean(
                    (case / f'rank-{rank}.log').read_text(errors='replace'),
                    args.instrument_processes) for rank in (0, 1))
                row['pass'] &= row['instrumentation_clean']
            expected = {"startup": "TEST_INJECTED_NCCL_STARTUP_ERROR",
                        "constructor": "TEST_INJECTED_CONSTRUCTOR_ERROR",
                        "constructor_late": "TEST_INJECTED_CONSTRUCTOR_LATE_ERROR",
                        "owner": "TEST_INJECTED_OWNER_HOST_ERROR",
                        "admission": "TEST_INJECTED_ARCHIVE_ADMISSION_ERROR",
                        "worker_write": "TEST_INJECTED_ARCHIVE_WORKER_WRITE_ERROR",
                        "worker_sync": "TEST_INJECTED_ARCHIVE_WORKER_SYNC_ERROR",
                        "finish": "TEST_INJECTED_ARCHIVE_FINISH_ERROR"}
            if key:
                if name == 'capacity':
                    row['fault_reached'] = any(marker in text for marker in (
                        'LIBRARY_RANK_DEPTH_FATAL', 'GROUP_OWNER_OR_PRE_OWNER_FATAL'))
                    row['capacity_profile'] = args.profile
                else:
                    row["fault_reached"] = expected[name] in text
                row["pass"] &= row["fault_reached"]
            report["cases"].append(row)
            save()
        finally:
            cleanup_errors = cleanup_rank_processes(processes)
            for stream in streams:
                stream.close()
            if cleanup_errors:
                report['cleanup_errors'] = cleanup_errors
                save()
                raise RuntimeError('CANDIDATE_CLEANUP_FAILED: ' + '; '.join(cleanup_errors))
        if not row["pass"]:
            raise RuntimeError("CANDIDATE_CASE_FAILED: " + name)
    if args.oracle:
        with (output / "full-state-oracle.log").open("w") as log:
            checked = subprocess.run([str(work / "cargo/bin/cargo"), "test", "--locked",
                "-p", "mgbfs-runtime", "--features", "cuda,library-owner", "--test",
                "library_multi_gpu", ("cuco_rank_lsa_hash_first_layers_and_archives_match_oracle"
                    if args.profile == 'HASH_FIRST' else
                    "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle"),
                "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        report["oracle_returncode"] = checked.returncode
        report["oracle_scope"] = "two physical GPUs, threads in one process; full state/archive; U/S, maps, pre-dedup"
        save()
        if checked.returncode or "test result: ok. 1 passed; 0 failed" not in (
            output / "full-state-oracle.log").read_text():
            raise RuntimeError("FULL_STATE_ORACLE_FAILED")
        with (output / "capacity-fault.log").open("w") as log:
            checked = subprocess.run([str(work / "cargo/bin/cargo"), "test", "--locked",
                "-p", "mgbfs-runtime", "--features", "cuda,library-owner", "--test",
                "library_multi_gpu", "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=45)
        report["capacity_fault_returncode"] = checked.returncode
        report["capacity_fault_profile"] = "DENSE"
        save()
        if checked.returncode:
            raise RuntimeError("CAPACITY_FAULT_FAILED")
    if args.sanitizers:
        compiled = subprocess.check_output([str(work / "cargo/bin/cargo"), "test",
            "--locked", "-p", "mgbfs-runtime", "--features", "cuda,library-owner",
            "--test", "library_multi_gpu", "--no-run", "--message-format=json"],
            cwd=source, env=env, text=True, timeout=300)
        artifacts = [json.loads(line) for line in compiled.splitlines() if line.startswith("{")]
        binaries = [a["executable"] for a in artifacts if a.get("reason") == "compiler-artifact"
                    and a.get("executable") and a["target"]["name"] == "library_multi_gpu"]
        if len(binaries) != 1:
            raise RuntimeError("TEST_BINARY_INVENTORY")
        report["sanitizers"] = []
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
            log_path = output / (tool + ".log")
            command = ["/usr/local/cuda/bin/compute-sanitizer", "--tool", tool,
                "--error-exitcode", "97", binaries[0],
                "cuco_rank_lsa_single_fixture_for_sanitizer", "--ignored", "--exact",
                "--nocapture", "--test-threads=1"]
            try:
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=source, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        returncode = process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                        raise
                text = log_path.read_text(errors="replace")
                summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", text)
                races = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", text)
                clean = (bool(summaries) and not any(int(n) for n in summaries)) or (
                    tool == "racecheck" and bool(races) and not any(int(n) for row in races for n in row))
                report["sanitizers"].append({"tool": tool, "returncode": returncode,
                    "pass": returncode == 0 and clean and
                    "test result: ok. 1 passed; 0 failed" in text})
            except subprocess.TimeoutExpired:
                report["sanitizers"].append({"tool": tool, "pass": False, "status": "TIMEOUT"})
                save()
                raise
            save()
            if not report["sanitizers"][-1]["pass"]:
                raise RuntimeError("SANITIZER_FAILED: " + tool)
    report["status"] = "DIAGNOSTIC_CASES_PASS"
    save()


if __name__ == "__main__":
    main()
