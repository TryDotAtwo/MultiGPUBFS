
def focused_validation_plan(suite):
    baseline = [('DENSE', 'CUCO_RANK'), ('DENSE', 'CUB_SORT_MERGE'), ('HASH_FIRST', 'CUCO_RANK'), ('HASH_FIRST', 'CUB_SORT_MERGE'), ('HASH_FIRST', 'BMMA_BUCKET')]
    if suite == 'baseline':
        return dict(oracle_pairs=baseline, capture_pairs=[], fault_pairs=[baseline[0], baseline[1], baseline[3], baseline[4]], sanitizer_pairs=baseline[:2], expected_panels=32)
    if suite == 'owner_capture_and_missing_modes':
        return dict(oracle_pairs=[('DENSE', 'BMMA_BUCKET')], capture_pairs=baseline + [('DENSE', 'BMMA_BUCKET')], fault_pairs=[('DENSE', 'BMMA_BUCKET'), ('HASH_FIRST', 'CUCO_RANK')], sanitizer_pairs=[('DENSE', 'BMMA_BUCKET')] + baseline[2:], expected_panels=46)
    if suite == 'host_full_regression':
        pairs = baseline + [('DENSE', 'BMMA_BUCKET')]
        return dict(oracle_pairs=pairs, capture_pairs=pairs, fault_pairs=pairs, sanitizer_pairs=pairs, expected_panels=78)
    if suite == 'lsa_batch_capture':
        return dict(oracle_pairs=[], capture_pairs=[], batch_capture_pairs=[('DENSE', 'CUCO_RANK')], fault_pairs=[('DENSE', 'CUCO_RANK')], sanitizer_pairs=[], expected_panels=5)
    if suite == 'lsa_batch_pipeline':
        pair = [('DENSE', 'CUCO_RANK')]
        return dict(oracle_pairs=[], capture_pairs=[], batch_capture_pairs=pair, fault_pairs=pair, sanitizer_pairs=pair, timeline_pairs=pair, expected_panels=10)
    if suite == 'lsa_batch_profiles':
        pairs = baseline + [('DENSE', 'BMMA_BUCKET')]
        return dict(oracle_pairs=[], capture_pairs=[], batch_capture_pairs=pairs, fault_pairs=pairs, sanitizer_pairs=pairs, timeline_pairs=pairs, expected_panels=60)
    if suite == 'lsa_scoped_pipeline':
        return dict(oracle_pairs=[], capture_pairs=[], fault_pairs=[], sanitizer_pairs=[], timeline_pairs=baseline + [('DENSE', 'BMMA_BUCKET')], expected_panels=6)
    raise ValueError('Unknown focused validation suite: ' + str(suite))

"""Exact-source, two-T4 CUCO_RANK/LSA full-state BFS correctness gate."""
import hashlib
import ctypes
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# Explicit cache ingress, independent of Kaggle output mounting. Runtime/test
# source and all existing cache recipe/digest checks remain unchanged.
CACHE_INPUT_ROOT = Path(os.environ.get("MGBFS_CACHE_INPUT_ROOT", "/kaggle/input"))
_cache_downloads = globals().pop("_MGBFS_CACHE_DOWNLOADS", None)
if _cache_downloads is not None:
    import urllib.request
    if CACHE_INPUT_ROOT != Path("/tmp/mgbfs-verified-input"):
        raise RuntimeError("CACHE_DOWNLOAD_ROOT")
    CACHE_INPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print("CACHE_INGRESS_STARTED", flush=True)
    for _entry in _cache_downloads:
        _name, _url, _expected = _entry
        if Path(_name).name != _name or len(_expected) != 64:
            raise RuntimeError("CACHE_DOWNLOAD_IDENTITY")
        _target = CACHE_INPUT_ROOT / _name
        _digest = hashlib.sha256()
        _bytes = 0
        try:
            with urllib.request.urlopen(_url, timeout=120) as _response, _target.open("xb") as _out:
                while True:
                    _chunk = _response.read(1024 * 1024)
                    if not _chunk:
                        break
                    _bytes += len(_chunk)
                    if _bytes > 4 * 1024**3:
                        raise RuntimeError("CACHE_DOWNLOAD_CAPACITY")
                    _digest.update(_chunk)
                    _out.write(_chunk)
        except Exception:
            # Never expose temporary signed URLs in logs or retained evidence.
            raise RuntimeError("CACHE_DOWNLOAD_FAILED_" + _name) from None
        if _digest.hexdigest() != _expected:
            raise RuntimeError("CACHE_DOWNLOAD_SHA256_" + _name)
        print("CACHE_INGRESS_VERIFIED", _name, _bytes, _expected, flush=True)
    del _cache_downloads, _entry, _name, _url, _expected

SOURCE = "6f35c433ebd3269da45c8569fba1424f4e20a1d6"
BEFORE_SOURCE = "c64c869fb403d755682c859eb1c72597a3ba88b9"
LIBRARY_CACHE_REFERENCE_SOURCE = "3d5c736f713604c51deef92c94119ed1ed0e4772"
CUCO = "532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4"
MODE = "weighted_library_gate"
# Explicit changed-path inventory; reuse the shared weighted gate, not a parallel harness.
WEIGHTED_FULL_OWNERS = ("CUCO_RANK",)
FOCUSED_GATE_SUITE = "lsa_batch_profiles"
FOCUSED_TRANSPORT = "NCCL_LSA"
HARDWARE = "T4"  # A4000 is an explicit diagnostic, never T4 acceptance.
NCCL_VARIANT = "minimum_arch_guard_posix"
# Exact immutable cache consumer; native/library/NCCL misses fail without compilation.
KAGGLE_DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
NCCL_CACHE_MODE = "required"
NCCL_CACHE_ARCHIVE_SHA256 = "c4f5f5368fb3644ec5d31419a9a9aa3f9c2f1f4e28886bc850253faceaea3761"
# Attach completed trydotatwo/mgbfs-lsa-full-bfs-gate-t4 v131 outputs
# to a different consumer notebook; do not overwrite the sole cache producer.
NCCL_CACHE_HELPER_COMMIT = "a71fc9b768f095d8241db2c226b1da8fc3b45f6f"



# publish is an explicit first-producer mode. Consumers must use required:
# a missing/mismatched cache then terminates, never falls back to compilation.
COMPILED_CACHE_MODE = "required"
WEIGHTED_OWNER_SANITIZERS = True
# Required consumers pin these from the retained producer summary before launch.
COMPILED_CACHE_EXPECTED_ARCHIVES = {"compiled-mgbfs_library_owner-265b27c600ecc648be75ef4809ee46420bb497a1f61da2f2370ab6ec55143456.tar.gz":"92427b56398b0a2ee893be7eb1851cc2ee8eb2b3f8127685a56d983d4a37a573","compiled-mgbfs_cuda-2114f80f3e37ea18c6ea7132030755ddc4ac6b773e9347d9dfaace5a04e9fdf4.tar.gz":"eb2ea5987110d9277b8437731f6c8ad53699a48fbfa7f66d762beea0912920af"}


def compiled_cache_key(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_compiled_cache(path, recipe, library_name):
    import tarfile
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != 2 or {m.name for m in members} != {'manifest.json', library_name}:
            raise RuntimeError('COMPILED_CACHE_MEMBERS')
        if any(not m.isfile() or m.size > 128 * 1024 * 1024 for m in members):
            raise RuntimeError('COMPILED_CACHE_MEMBER_TYPE_SIZE')
        meta = next(m for m in members if m.name == 'manifest.json')
        if meta.size > 1024 * 1024:
            raise RuntimeError('COMPILED_CACHE_MANIFEST_SIZE')
        manifest = json.loads(archive.extractfile(meta).read())
        if manifest.get('schema') != 1 or manifest.get('recipe') != recipe:
            raise RuntimeError('COMPILED_CACHE_RECIPE')
        payload = archive.extractfile(library_name).read()
        if not payload.startswith(b'\x7fELF') or manifest.get('sha256') != hashlib.sha256(payload).hexdigest():
            raise RuntimeError('COMPILED_CACHE_PAYLOAD')
        return payload


def write_compiled_cache(path, recipe, library_name, payload):
    import io, tarfile, gzip
    if not payload.startswith(b'\x7fELF'):
        raise RuntimeError('COMPILED_CACHE_EXPORT_NOT_ELF')
    manifest = dict(schema=1, recipe=recipe, sha256=hashlib.sha256(payload).hexdigest())
    with Path(path).open('wb') as output, gzip.GzipFile(filename='', fileobj=output, mode='wb', mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as archive:
        for name, data in [('manifest.json', json.dumps(manifest, sort_keys=True).encode()), (library_name, payload)]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(data))


def compiled_cache_fixtures():
    import tarfile
    # Pure admission tests, not a CUDA correctness gate or ELF loader test.
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / 'cache.tar.gz'
        recipe = {'source': 'fixture', 'flags': ['sm75']}
        payload = b'\x7fELF' + bytes(64)
        write_compiled_cache(path, recipe, 'fixture.so', payload)
        if read_compiled_cache(path, recipe, 'fixture.so') != payload:
            raise RuntimeError('COMPILED_CACHE_FIXTURE_ROUNDTRIP')
        for bad_recipe, bad_name in [({'source': 'changed', 'flags': ['sm75']}, 'fixture.so'),
                                     (recipe, '../fixture.so'),
                                     ({'source': 'fixture', 'flags': ['sm80']}, 'fixture.so')]:
            try:
                read_compiled_cache(path, bad_recipe, bad_name)
            except RuntimeError:
                pass
            else:
                raise RuntimeError('COMPILED_CACHE_FIXTURE_REJECTION')
        write_compiled_cache(path, recipe, 'fixture.so', payload)
        first_digest = file_digest(path)
        write_compiled_cache(path, recipe, 'fixture.so', payload)
        if file_digest(path) != first_digest:
            raise RuntimeError('COMPILED_CACHE_FIXTURE_NONDETERMINISTIC')
        # Correct manifest + changed payload must fail, even with ELF magic intact.
        import io
        manifest = dict(schema=1, recipe=recipe, sha256=hashlib.sha256(payload).hexdigest())
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in [('manifest.json', json.dumps(manifest).encode()),
                               ('fixture.so', payload + b'corrupt')]:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        try:
            read_compiled_cache(path, recipe, 'fixture.so')
        except RuntimeError:
            pass
        else:
            raise RuntimeError('COMPILED_CACHE_FIXTURE_CHECKSUM')
        write_compiled_cache(path, recipe, 'fixture.so', payload)
        # Truncation must not be mistaken for a usable archive.
        path.write_bytes(path.read_bytes()[:20])
        try:
            read_compiled_cache(path, recipe, 'fixture.so')
        except (RuntimeError, EOFError, OSError, tarfile.TarError):
            pass
        else:
            raise RuntimeError('COMPILED_CACHE_FIXTURE_TRUNCATION')
    return 'PASS_ADMISSION_ONLY'


def oracle_dependency_packages(mode):
    return ['pyarrow==19.0.1'] if mode in ('focused_fault_replay', 'device_protocol_replay', 'native_rank_gate',
        'typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate', 'typed_warmup_gate',
        'typed_sanitizer_version_gate', 'typed_matrix_gate', 'single_rank_production_gate',
        'weighted_multi_rank_gate', 'weighted_timeline', 'owner_graph_gate', 'owner_graph_full_gate', 'owner_graph_timeline', 'weighted_library_gate') else []


def macro_capture_command():
    return ['cargo', 'test', '--locked', '-p', 'mgbfs-runtime',
        '--features', 'cuda,library-owner', '--lib',
        'macro_native::producer_capture_tests::',
        'weighted_reclamation_keeps_an_older_future_allocation_live',
                '--', '--nocapture', '--test-threads=1']


def typed_rank_configs(base):
    """Complete-packet candidate: both profiles; unsupported admission is explicit."""
    cases = []
    for profile in ('DENSE', 'HASH_FIRST'):
        for backend in ('CUB_SORT_MERGE', 'CUCO_RANK', 'BMMA_BUCKET'):
            for banks in (2, 3):
                for prededup in (True, False):
                    for mapping in ([0, 1], [1, 0]):
                        config = copy.deepcopy(base)
                        config.update(frontier_profile=profile, completion_epoch_window=3,
                            local_pre_dedup=prededup, owner_backend=backend,
                            transport_backend='HOST_SIZED_NCCL',
                            library_pool_bytes=(96 << 20) if backend == 'CUCO_RANK' else None)
                        config['topology'].update(shards_per_rank=4, buckets_per_shard=8,
                            logical_owner_to_rank=mapping.copy())
                        config['capacities']['route_slot_count'] = banks
                        cases.append(config)
    return cases


def typed_reuse_configs(base):
    """U4(F2): a 13-state layer forces >4 parent batches on some rank."""
    config = copy.deepcopy(base)
    generators = []
    for row in range(3):
        matrix = base['graph']['start'].copy()
        matrix[row * 4 + row + 1] = 1
        generators.append(matrix)
    config['graph'].update(generators=generators + copy.deepcopy(generators),
        inverse_map=[3, 4, 5, 0, 1, 2], expected_max_unique_states=64)
    config['capacities']['route_slot_records'] = 6
    return [candidate for candidate in typed_rank_configs(config)
        if candidate['completion_epoch_window'] == 3 and candidate['local_pre_dedup']
        and candidate['topology']['logical_owner_to_rank'] == [0, 1]]


def typed_stress_configs(base, modulus):
    """Bounded full-state U4 gate; not an end-to-end performance benchmark."""
    if type(modulus) is not int or not 2 <= modulus <= 6:
        raise ValueError('BOUNDED_REFERENCE_UNITRIANGULAR')
    config = copy.deepcopy(base)
    generators, inverses = [], []
    identity = [int(row == col) for row in range(4) for col in range(4)]
    for row in range(3):
        forward, inverse = identity.copy(), identity.copy()
        forward[row * 4 + row + 1] = 1
        inverse[row * 4 + row + 1] = modulus - 1
        generators.append(forward)
        inverses.append(inverse)
    states = modulus ** 6
    records = 1 << (states - 1).bit_length()
    config['graph'].update(rows=4, cols=4, modulus=modulus, start=identity,
        generators=generators + inverses, inverse_map=[3, 4, 5, 0, 1, 2],
        expected_max_unique_states=states)
    config['parent_batch'] = 8
    config['capacities'].update(state_ring_records=2 * records,
        state_extent_descriptors=2 * records, layer_hash_records_per_arena=records,
        next_bucket_capacity_records=records, route_slot_records=48,
        pinned_archive_slots=2 * records, pinned_archive_slot_bytes=512)
    cases = []
    for candidate in typed_rank_configs(config):
        if candidate['completion_epoch_window'] != 3:
            continue
        # typed_rank_configs already enumerates each owner. Re-enumerating
        # owners here repeated every labelled case three times, overwriting
        # its raw output directory and invalidating an exhaustive matrix count.
        cases.append(candidate)
    return cases


def typed_matrix_cases(base):
    """Full U4/m=2..6 state oracle matrix through the existing rank runtime."""
    cases = []
    for modulus in range(2, 7):
        for config in typed_stress_configs(base, modulus):
            if config['capacities']['route_slot_count'] != 3:
                continue
            config['parent_batch'] = 256
            config['capacities']['route_slot_records'] = 1536
            config['capacities']['pinned_archive_slot_bytes'] = 8192
            config['capacities']['pinned_archive_slots'] = 256
            for seed in (0, 1, 20260828):
                selected = copy.deepcopy(config)
                selected['seed'] = list(seed.to_bytes(16, 'little'))
                label = (f'u4m{modulus}-{selected["owner_backend"]}-{selected["frontier_profile"]}'
                    f'-pre{int(selected["local_pre_dedup"])}'
                    f'-map{"".join(map(str, selected["topology"]["logical_owner_to_rank"]))}-seed{seed}')
                cases.append(dict(config=selected, label=label,
                    extra=['--healthy-only', '--unitriangular-modulus', str(modulus)]))
    labels = [case['label'] for case in cases]
    if len(cases) != 360 or len(set(labels)) != len(labels):
        raise RuntimeError('TYPED_MATRIX_DUPLICATE_OR_MISSING_CASE')
    # Includes explicit unsupported HASH_FIRST/CUCO admission records; those
    # are never counted as successful GPU searches by the replay verifier.
    return cases


def typed_followup_cases(base):
    """Unfiltered registration replays and full BFS traces; not performance."""
    configs = [config for config in typed_rank_configs(base)
        if config['completion_epoch_window'] == 3 and config['local_pre_dedup']
        and config['topology']['logical_owner_to_rank'] == [0, 1]]
    cases = []
    for config in configs:
        for repeat in range(3):
            label = (f"initcheck-{config['frontier_profile']}-banks-"
                     f"{config['capacities']['route_slot_count']}-repeat-{repeat}")
            cases.append(dict(config=copy.deepcopy(config), repeat=repeat, label=label,
                tool='initcheck', extra=['--healthy-only', '--instrument-processes', 'initcheck']))
    for config in typed_reuse_configs(base):
        if config['capacities']['route_slot_count'] == 3:
            cases.append(dict(config=config, repeat=0, label='timeline-' + config['frontier_profile'],
                tool='nsys', extra=['--healthy-only', '--unitriangular-modulus', '2',
                    '--require-bank-reuse', '--instrument-processes', 'nsys']))
    return cases


def typed_paired_config(base, n):
    """Explicit matrix L/R/X production input; no reference-dispatch defaults."""
    if type(n) is not int or not 2 <= n <= 20:
        raise ValueError('PAIRED_MATRIX_SIZE')
    config = copy.deepcopy(base)
    identity = list(range(n))
    permutations = [identity[1:] + identity[:1], identity[-1:] + identity[:-1],
                    [1, 0, *identity[2:]]]
    order = 1
    for value in range(2, n + 1):
        order *= value
    config['graph'] = dict(schema=1, rows=n, cols=n, modulus=2,
        start=[int(i == j) for i in range(n) for j in range(n)],
        generators=[[int(j == permutation[i]) for i in range(n) for j in range(n)]
                    for permutation in permutations], inverse_map=[1, 0, 2],
        expected_max_unique_states=order)
    config.update(owner_backend='CUCO_RANK', library_pool_bytes=96 << 20,
                  parent_batch=32768, frontier_profile='DENSE', local_pre_dedup=True,
                  macro_depth=1, completion_epoch_window=3)
    config['topology'].update(world_size=2, shards_per_rank=4, buckets_per_shard=256,
                              logical_owner_to_rank=[0, 1])
    config['capacities'].update(state_ring_records=1_000_000,
        state_extent_descriptors=1_000_000, layer_hash_records_per_arena=1_000_000,
        next_bucket_capacity_records=1_000_000, route_slot_records=98304,
        route_slot_count=3, pinned_archive_slots=256,
        pinned_archive_slot_bytes=32768 * (n * n + 16),
        disk_extent_bytes_per_rank=1 << 30, untouched_vram_reserve_bytes=1 << 30)
    return config


def cuda_build_target(hardware):
    """Match the admitted physical GPU; never reuse another major's SASS."""
    targets = {"T4": "75", "RTX2070": "75", "A4000": "86"}
    if hardware not in targets:
        raise ValueError("UNSUPPORTED_CUDA_BUILD_HARDWARE: " + hardware)
    return targets[hardware]


def typed_warmup_cases(base):
    cases = []
    for profile in ('DENSE', 'HASH_FIRST'):
        config = copy.deepcopy(base)
        config['frontier_profile'] = profile
        config['owner_backend'] = 'CUCO_RANK'
        config['library_pool_bytes'] = 96 << 20
        config['completion_epoch_window'] = 3
        config['capacities']['route_slot_count'] = 3
        cases.append(dict(config=config, label='warmup-' + profile,
                          extra=['--bench-warmup', '--capacity-faults']))
    return cases


def typed_sanitizer_version_cases(base):
    return [dict(config=copy.deepcopy(case['config']), version=version, tool=tool,
                 label=f"toolchain-{case['config']['frontier_profile']}-{version}-{tool}",
                 extra=['--healthy-only', '--instrument-processes', tool])
            for case in typed_warmup_cases(base)
            for version in ('host', 'cuda129')
            for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck')]


def sanitizer_version_environment(env, host, pinned, version):
    if version not in ('host', 'cuda129'):
        raise ValueError('SANITIZER_VERSION_SELECTION')
    selected = host if version == 'host' else pinned
    result = dict(env)
    result['MGBFS_COMPUTE_SANITIZER'] = str(selected)
    result['PATH'] = Path(selected).parent.as_posix() + ':' + result.get('PATH', '')
    return result


def sanitizer_selection_matches(actual, executable, sha256):
    return (isinstance(actual, dict) and actual.get('executable') == executable
        and actual.get('sha256') == sha256 and isinstance(actual.get('version'), str)
        and bool(actual['version'].strip()))


def vmm_probe_cases():
    return [dict(mode=mode, tool=tool, runtime_init=runtime_init, symmetric=symmetric,
                 label=f'{mode}-{tool or "plain"}' + ('-runtime-init' if runtime_init else '')
                       + ('-symmetric' if symmetric else ''))
            for mode in ('local', 'import')
            for runtime_init, symmetric in ((False, False), (True, False), (True, True))
            for tool in (None, 'memcheck', 'racecheck', 'initcheck', 'synccheck')]


def run_window_process_pair(command, cwd, env, output, timeout=120, required_stage=None,
                            require_window=True):
    """Reduced vendor probe, independent ranks, bounded whole process trees."""
    if not require_window and required_stage not in (
            'device_comm_only_create', 'device_comm_zero_create', 'vmm_local', 'vmm_import'):
        raise ValueError('WINDOWLESS_PROBE_REQUIRES_DEVICE_ONLY_STAGE')
    # Kaggle relocates the uploaded script; cwd is the pinned source checkout.
    scripts = str(Path(cwd).resolve() / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from process_scope import spawn_group, stop_group
    output.mkdir(parents=True, exist_ok=True)
    bootstrap = output.resolve() / 'bootstrap'
    processes, handles = [], []
    timed_out = False
    deadline = time.monotonic() + timeout
    try:
        for rank in (0, 1):
            log = (output / f'rank-{rank}.log').open('w')
            handles.append(log)
            rank_env = dict(env, MGBFS_WINDOW_RANK=str(rank),
                            MGBFS_WINDOW_BOOTSTRAP=str(bootstrap))
            processes.append(spawn_group(command, cwd=cwd, env=rank_env,
                stdout=log, stderr=subprocess.STDOUT))
        for process in processes:
            try:
                process.wait(timeout=max(0.001, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                break
    finally:
        for process in processes:
            stop_group(process)
        for log in handles:
            log.close()
    texts = [(output / f'rank-{rank}.log').read_text(errors='replace') for rank in (0, 1)]
    codes = [process.returncode for process in processes]
    registered = [text.count(f'rank={rank} mode=nonblocking stage=window_register result=PASS')
                  for rank, text in enumerate(texts)]
    reached = ([text.count(f'rank={rank} stage={required_stage} result=PASS')
                for rank, text in enumerate(texts)] if required_stage else [1, 1])
    return dict(returncodes=codes, timed_out=timed_out, registered_ranks=registered,
                required_stage=required_stage, reached_stage=reached, require_window=require_window,
                **{'pass': not timed_out and codes == [0, 0]
                    and registered == ([1, 1] if require_window else [0, 0]) and reached == [1, 1]})


def run_protocol_replay(command, cwd, env, log, timeout=1800):
    """Bound the entire replay tree, not only its Python parent."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return {"returncode": process.wait(timeout=timeout), "timed_out": False}
    except subprocess.TimeoutExpired:
        # Each rank has its own session. Killing the replay parent alone would
        # orphan ranks; ask its normal SIGTERM handler to retire those first.
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        return {"returncode": process.returncode, "timed_out": True}


def main():
    architecture = cuda_build_target(HARDWARE)
    work = Path(tempfile.mkdtemp(prefix="mgbfs-lsa-bfs-", dir="/tmp"))
    logs = Path("/kaggle/working/lsa-bfs-gate")
    logs.mkdir(parents=True, exist_ok=True)
    report = {"source": SOURCE, "status": "INCOMPLETE", "scope":
              ("two physical T4; independent rank-process owner, archive-admission and archive-finish fault propagation"
               if MODE == "process_faults_only" else
               "two physical T4; one-rank archive slot exhaustion before exchange"
               if MODE == "archive_fault_gate" else
               "two physical T4; warmup/CLI admission fault propagation and archive cleanup"
               if MODE == "warmup_admission_gate" else
               "two physical T4; boundary agreement, archive integrity and independent S4 full-state oracle")}
    report["hardware_target"] = HARDWARE
    report["cuda_architecture"] = "sm" + architecture
    report["t4_acceptance_eligible"] = HARDWARE == "T4"
    if HARDWARE != "T4":
        report["scope"] = "explicit " + HARDWARE + " hardware diagnostic; not T4 acceptance"

    def save():
        (logs / "summary.json").write_text(json.dumps(report, indent=2))

    save()
    source = work / "source"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin",
                    "https://github.com/TryDotAtwo/MultiGPUBFS.git"], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", SOURCE], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE:
        raise RuntimeError("SOURCE_COMMIT_MISMATCH")
    spec = importlib.util.spec_from_file_location("gate", source / "kaggle/native-primitives/kernel.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    spec = importlib.util.spec_from_file_location("library", source / "kaggle/library-owner/kernel.py")
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    env = library.isolated_environment(os.environ)
    env["NCCL_CUMEM_ENABLE"] = "1"
    env["MGBFS_EPOCH_WINDOW"] = "3"
    env["PIP_DEFAULT_TIMEOUT"] = "300"
    env["PIP_RETRIES"] = "5"

    def run(command, name, cwd=source, timeout=900):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    compiled_environment_fingerprints = {}

    def fingerprint_dependency(root):
        # Stream large shared libraries; do not load RAPIDS binaries into RAM.
        entries = []
        resolved = {}
        for path in sorted(root.rglob('*')):
            if not path.is_file() or not (path.suffix in ('.h', '.hpp', '.cuh', '.cmake') or
                                         path.name == 'CMakeLists.txt' or '.so' in path.name):
                continue
            identity = path.resolve()
            if identity not in resolved:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                resolved[identity] = digest.hexdigest()
            entries.append((str(path.relative_to(root)), resolved[identity]))
        return dict(files=len(entries), sha256=compiled_cache_key(entries))

    def compiled_target(label, target_source, build_dir, configure, target, library_name):
        if COMPILED_CACHE_MODE not in ('publish', 'required'):
            raise RuntimeError('COMPILED_CACHE_MODE')
        # Conservative dependency closure, independent of Rust/launcher changes.
        # Both roots cover cross-includes; over-invalidation is safe, stale hits are not.
        inputs = sorted(path for root in ('cuda', 'experiments/library_owner')
                        for path in (target_source / root).rglob('*') if path.is_file())
        # Include generated profiler shims too, not only git-tracked files.
        hashes = {str(path.relative_to(target_source)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in inputs}
        if not hashes:
            raise RuntimeError('COMPILED_CACHE_SOURCE_EMPTY')
        replacements = [(str(target_source), '<source>'), (str(work), '<work>')]
        flags = []
        for value in configure:
            for old, new in replacements:
                value = value.replace(old, new)
            flags.append(value)
        if label in ('before', 'before-library'):
            # The frozen control cache was produced in native-build. Restoring
            # its verified ELF into an independent directory does not change
            # source/compiler/dependency identity; retain the real destination
            # outside the frozen build recipe. No other flag is normalized.
            if flags.count('-B') != 1:
                raise RuntimeError('BEFORE_CACHE_BUILD_DIRECTORY_FLAG')
            destination = flags.index('-B') + 1
            expected_destination = '<work>/before-native' if label == 'before' else '<work>/before-library-build'
            frozen_destination = '<work>/native-build' if label == 'before' else '<work>/library-build'
            if flags[destination] != expected_destination:
                raise RuntimeError('BEFORE_CACHE_RESTORE_DIRECTORY')
            report.setdefault('cache_restore_locations', {})[label] = str(build_dir)
            flags[destination] = frozen_destination
        if not compiled_environment_fingerprints:
            compiled_environment_fingerprints['sdk_headers'] = fingerprint_dependency(sdk / 'include')
            # cmake_prefixes includes every site-package plus nested config dirs.
            # Fingerprint the actual RAPIDS/CUDA dependency roots once, not all
            # unrelated notebook packages or the same nested tree repeatedly.
            dependency_roots = ['libcudf', 'librmm', 'cudf', 'rmm', 'librapids_logger',
                                'rapids_logger', 'nvidia']
            compiled_environment_fingerprints['rapids_dependencies'] = {
                name: fingerprint_dependency(site / name) for name in dependency_roots
                if (site / name).is_dir()}
            if not compiled_environment_fingerprints['rapids_dependencies']:
                raise RuntimeError('COMPILED_CACHE_RAPIDS_INPUTS_EMPTY')
        recipe = dict(schema=1, target=target, source_files=hashes, configure=flags,
                      dependency_fingerprints=compiled_environment_fingerprints,
                      cuda_architecture=architecture, docker=KAGGLE_DOCKER_IMAGE,
                      cutlass=gate.CUTLASS_COMMIT, cuco=CUCO,
                      nccl_cache_archive=NCCL_CACHE_ARCHIVE_SHA256,
                      # uv-created environments deliberately need not contain pip.
                      python_packages=json.loads(subprocess.check_output([python, '-c',
                          "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"],
                          env=env, text=True)),
                      compiler=subprocess.check_output(['g++', '--version'], env=env, text=True),
                      libc=subprocess.check_output(['ldd', '--version'], env=env, text=True),
                      nvcc=subprocess.check_output([str(sdk / 'bin/nvcc'), '--version'], env=env, text=True))
        # Retain the exact recipe even on a REQUIRED miss; otherwise diagnosing
        # package or configure drift requires a second failed hardware launch.
        (logs / (label + '-cache-recipe.json')).write_text(json.dumps(recipe, indent=2))
        cache_mode = COMPILED_CACHE_MODE
        # Explicit producer for changed native/public-header closure. Subsequent
        # consumers pin both new archives and use required, never build fallback.
        excluded_native_only_source = None
        if SOURCE in ('ba65ea410b951427e3e8b5cd53b4dcf5466e79e8', '6564cdf1c157e12beaf7c9e1ed854074a05d00ff', '77d12d8ccab6e3114a47c59cc712c616d84874e8', '2684188014ab5acde8565916f158daab454997e5', 'fbf740b3d4e36f8b6061a0a522bed802bf60c3fc', 'a55db99957d2c9c0e1d0953d2edaad2308ee286c'):
            if label == 'native':
                # Only the original changed-native producer may compile. All
                # later same-closure consumers require its checked archive.
                if SOURCE == 'ba65ea410b951427e3e8b5cd53b4dcf5466e79e8':
                    cache_mode = 'publish'
            elif label == 'library':
                reference = '80731e2774af607a7e31d13e788b0abb05b3f9db'
                run(['git', 'fetch', '--depth=1', 'origin', reference], 'weighted-library-closure-reference')
                changed = subprocess.check_output(['git', 'diff', '--name-only', reference, SOURCE,
                    '--', 'cuda', 'experiments/library_owner'], cwd=target_source, text=True).splitlines()
                if changed != ['cuda/state_commit.cu', 'cuda/weighted_state_ring.h']:
                    raise RuntimeError('WEIGHTED_LIBRARY_CLOSURE_CHANGED')
                cmake = (target_source / 'experiments/library_owner/CMakeLists.txt').read_text()
                # state_commit.cu belongs to probe EXECUTABLES, not this shared
                # target. Do not reject merely because it occurs elsewhere.
                import re
                declarations = re.findall(r'(?:add_library|target_sources)\(mgbfs_library_owner\s+([^)]*)\)', cmake)
                if declarations != ['SHARED owner_abi.cpp layout.cu control_transfer.cpp',
                                    'PRIVATE cuco_owner_abi.cu cuco_rank_batch.cu']:
                    raise RuntimeError('WEIGHTED_LIBRARY_TARGET_INPUTS_CHANGED')
                for path in (target_source / 'experiments/library_owner').rglob('*'):
                    if path.is_file() and path.suffix in ('.cpp', '.cu', '.h', '.cuh'):
                        includes = re.findall(r'#\s*include\s*[<"]([^>"]+)', path.read_text())
                        if any(name.endswith(('state_commit.cu', 'weighted_state_ring.h')) for name in includes):
                            raise RuntimeError('WEIGHTED_LIBRARY_NATIVE_ONLY_INCLUDE')
                # Everything else in the conservative source closure is equal
                # to the frozen cache source; shared headers stay unchanged.
                original = subprocess.check_output(['git', 'show', reference + ':cuda/state_commit.cu'], cwd=target_source)
                excluded_native_only_source = dict(source=SOURCE, reference=reference,
                    changed_files=changed, shared_target_sources=declarations,
                    state_commit_sha256=hashes['cuda/state_commit.cu'],
                    weighted_header_sha256=hashes['cuda/weighted_state_ring.h'])
                hashes['cuda/state_commit.cu'] = hashlib.sha256(original).hexdigest()
                del hashes['cuda/weighted_state_ring.h']
                (logs / 'weighted-library-closure-proof.json').write_text(json.dumps(excluded_native_only_source, indent=2))
        if label == 'library' and SOURCE in ('c38235b8140e56201ec153128b97efae4f70a73a', '8205a65fdb1b16292626b274c2c53d91e03393f6') and COMPILED_CACHE_MODE == 'required' and MODE in ('focused_fault_replay', 'old_new_pair', 'paired_measure', 'paired_timeline', 'rank_compare_red', 'rank_compare_green'):
            run(['git', 'fetch', '--depth=1', 'origin', LIBRARY_CACHE_REFERENCE_SOURCE], 'library-native-only-closure-reference')
            changed = subprocess.check_output(['git', 'diff', '--name-only', LIBRARY_CACHE_REFERENCE_SOURCE, SOURCE, '--', 'cuda', 'experiments/library_owner'], cwd=target_source, text=True).splitlines()
            expected_changes = ['cuda/bounded_owner.cu', 'cuda/exchange_pack.cu'] if SOURCE == 'c38235b8140e56201ec153128b97efae4f70a73a' else ['cuda/exchange_pack.cu']
            if changed != expected_changes:
                raise RuntimeError('LIBRARY_CACHE_NATIVE_ONLY_CHANGE_SCOPE')
            # This .cu is not compiled/included by mgbfs_library_owner. Keep all
            # other conservative recipe inputs pinned, including every header.
            cmake = (target_source / 'experiments/library_owner/CMakeLists.txt').read_text()
            if 'exchange_pack' in cmake:
                raise RuntimeError('LIBRARY_CACHE_NATIVE_ONLY_FILE_BECAME_INPUT')
            for path in (target_source / 'experiments/library_owner').rglob('*'):
                if path.is_file() and path.suffix in ('.cpp', '.cu', '.h', '.cuh') and 'exchange_pack' in path.read_text():
                    raise RuntimeError('LIBRARY_CACHE_NATIVE_ONLY_FILE_INCLUDED')
            original = subprocess.check_output(['git', 'show', LIBRARY_CACHE_REFERENCE_SOURCE + ':cuda/exchange_pack.cu'], cwd=target_source)
            excluded_native_only_source = hashes['cuda/exchange_pack.cu']
            hashes['cuda/exchange_pack.cu'] = hashlib.sha256(original).hexdigest()
            if SOURCE == 'c38235b8140e56201ec153128b97efae4f70a73a':
                if 'bounded_owner.cu' in cmake:
                    raise RuntimeError('LIBRARY_CACHE_BOUNDED_OWNER_BECAME_INPUT')
                for path in (target_source / 'experiments/library_owner').rglob('*'):
                    if path.is_file() and path.suffix in ('.cpp', '.cu', '.h', '.cuh') and 'bounded_owner.cu' in path.read_text():
                        raise RuntimeError('LIBRARY_CACHE_BOUNDED_OWNER_INCLUDED')
                original_owner = subprocess.check_output(['git', 'show', LIBRARY_CACHE_REFERENCE_SOURCE + ':cuda/bounded_owner.cu'], cwd=target_source)
                hashes['cuda/bounded_owner.cu'] = hashlib.sha256(original_owner).hexdigest()

        key = compiled_cache_key(recipe)
        filename = 'compiled-' + target + '-' + key + '.tar.gz'
        candidates = sorted(CACHE_INPUT_ROOT.rglob(filename))
        # Distinct immutable producers may contain the same pinned archive.
        # Accept duplicates only after checking EVERY copy against the pin.
        if len(candidates) > 1:
            pinned = COMPILED_CACHE_EXPECTED_ARCHIVES.get(filename)
            if not isinstance(pinned, str) or len(pinned) != 64:
                raise RuntimeError('COMPILED_CACHE_AMBIGUOUS_UNPINNED_' + target)
            if any(file_digest(candidate) != pinned for candidate in candidates):
                raise RuntimeError('COMPILED_CACHE_AMBIGUOUS_CHECKSUM_' + target)
        cache_record = dict(key=key, mode=cache_mode, target=target, hit=bool(candidates), verified_input_copies=len(candidates), excluded_native_only_source_sha256=excluded_native_only_source)
        if candidates:
            if cache_mode == 'required' or filename in COMPILED_CACHE_EXPECTED_ARCHIVES:
                expected = COMPILED_CACHE_EXPECTED_ARCHIVES.get(filename)
                if not isinstance(expected, str) or len(expected) != 64:
                    raise RuntimeError('COMPILED_CACHE_UNPINNED_ARCHIVE_' + target)
                if file_digest(candidates[0]) != expected:
                    raise RuntimeError('COMPILED_CACHE_ARCHIVE_CHECKSUM_' + target)
            payload = read_compiled_cache(candidates[0], recipe, library_name)
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / library_name).write_bytes(payload)
            cache_record['payload_sha256'] = hashlib.sha256(payload).hexdigest()
        else:
            if cache_mode == 'required':
                raise RuntimeError('COMPILED_CACHE_REQUIRED_MISS_' + target + '_' + key)
            run(configure, label + '-configure')
            run(['cmake', '--build', str(build_dir), '--target', target, '-j2'],
                label + '-build', timeout=1800)
        # Resolve dependencies using the current environment, not stale build RPATHs.
        dependency_check = subprocess.run(['ldd', str(build_dir / library_name)],
            env=env, capture_output=True, text=True, timeout=60)
        (logs / (label + '-cache-ldd.log')).write_text(dependency_check.stdout + dependency_check.stderr)
        if dependency_check.returncode or 'not found' in dependency_check.stdout:
            raise RuntimeError('COMPILED_CACHE_LINK_DEPENDENCY_' + target)
        payload = (build_dir / library_name).read_bytes()
        write_compiled_cache(logs / filename, recipe, library_name, payload)
        cache_record['archive_filename'] = filename
        cache_record['archive_sha256'] = file_digest(logs / filename)
        cache_record['payload_sha256'] = hashlib.sha256(payload).hexdigest()
        report.setdefault('compiled_cache', {})[label] = cache_record
        (logs / (label + '-cache-recipe.json')).write_text(json.dumps(recipe, indent=2))
        save()

    def prepare_nsys():
        package_name = "nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb"
        package_sha = "c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b"
        package = work / package_name
        run(["curl", "--fail", "--location", "--max-time", "300",
             "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
             + package_name, "--output", str(package)], "nsys-download")
        with package.open("rb") as downloaded:
            if hashlib.file_digest(downloaded, "sha256").hexdigest() != package_sha:
                raise RuntimeError("NSYS_DIGEST_MISMATCH")
        nsys_root = work / "nsys"
        run(["dpkg-deb", "--extract", str(package), str(nsys_root)], "nsys-extract")
        candidates = list(nsys_root.glob("opt/nvidia/nsight-systems/*/target-linux-x64/nsys"))
        if len(candidates) != 1:
            raise RuntimeError("NSYS_EXECUTABLE_INVENTORY")
        nsys = str(candidates[0])
        run([nsys, "--version"], "nsys-version")
        return nsys

    try:
        report["gpus"] = gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"), hardware=HARDWARE)
        cudart = ctypes.CDLL("libcudart.so.12")
        p2p = []
        for source_gpu, target_gpu in ((0, 1), (1, 0)):
            allowed = ctypes.c_int()
            rc = cudart.cudaDeviceCanAccessPeer(ctypes.byref(allowed), source_gpu, target_gpu)
            p2p.append({"source": source_gpu, "target": target_gpu,
                        "cuda_status": rc, "allowed": allowed.value})
        report["p2p"] = p2p
        if MODE not in ("device_fatal_gate", "boundary_gate", "host_sized_only", "single_rank_production_gate", "old_new_pair", "paired_measure", "paired_timeline", "focused_fault_replay", "typed_rank_gate", "typed_matrix_gate", "rank_compare_red", "rank_compare_green", "macro_capture_gate", "macro_production_dispatch_red", "weighted_driver_gate", "weighted_multi_rank_gate", "weighted_timeline", "owner_graph_gate", "owner_graph_full_gate", "owner_graph_timeline", "weighted_library_gate", "weighted_producer_gate") and any(
                row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
            report["status"] = "UNSUPPORTED_HOST"
            return
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        # Resolve before adding SDK/bin: sanitizer_api includes a launcher there.
        host_sanitizer = shutil.which('compute-sanitizer', path=env.get('PATH', ''))
        profiling_enabled = MODE in ('focused_fault_replay', 'old_new_pair', 'paired_timeline', 'typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate', 'typed_warmup_gate', 'typed_sanitizer_version_gate', 'typed_matrix_gate', 'native_rank_gate', 'timeline', 'timeline_backtrace', 'timeline_analysis')
        nvtx_enabled = profiling_enabled or MODE in ('paired_measure', 'rank_compare_red', 'rank_compare_green', 'macro_capture_gate', 'macro_production_dispatch_red', 'weighted_driver_gate', 'weighted_multi_rank_gate', 'weighted_timeline', 'owner_graph_gate', 'owner_graph_full_gate', 'owner_graph_timeline', 'weighted_library_gate')
        components = list(library.CUDA_COMPONENTS)
        if MODE == 'typed_sanitizer_version_gate':
            # Official redistrib12.9.1, inspected tar includes the actual instrumenter.
            components.append(('cuda_sanitizer_api', '12.9.79',
                'e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf'))
        if nvtx_enabled:
            # NVIDIA redistrib_12.9.1.json; checked archive contains NVTX3 headers.
            components.append(('cuda_nvtx', '12.9.79',
                '819bc39192955e6ba2067de39b85f30e157de462945e54b12bfdeda429d793fb'))
        for component, version, digest in components:
            name = f"{component}-linux-x86_64-{version}-archive"
            archive = work / (name + ".tar.xz")
            url = ("https://developer.download.nvidia.com/compute/cuda/redist/"
                   f"{component}/linux-x86_64/{archive.name}")
            run(["curl", "--fail", "--location", "--silent", "--show-error",
                 "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
                 "--continue-at", "-", "--connect-timeout", "30", "--max-time", "600",
                 url, "--output", str(archive)], component + "-download", timeout=5400)
            with archive.open("rb") as package:
                if hashlib.file_digest(package, "sha256").hexdigest() != digest:
                    raise RuntimeError("CUDA_SDK_CHECKSUM")
            run(["tar", "-xf", str(archive), "-C", str(work)], component + "-extract")
            shutil.copytree(work / name, sdk, dirs_exist_ok=True)
        (sdk / "lib64").symlink_to("lib", target_is_directory=True)
        env["PATH"] = str(sdk / "bin") + ":" + env.get("PATH", "")
        env["CUDACXX"] = str(sdk / "bin/nvcc")
        if MODE in ('nccl_window_processes', 'cuda_posix_import') or profiling_enabled:
            # The compiler is pinned, but the instrumenter comes from the host.
            # Record actual versions; do not infer sanitizer identity from nvcc.
            report['environment_versions'] = {
                'driver': run(['nvidia-smi', '--query-gpu=driver_version',
                    '--format=csv,noheader'], 'driver-version').strip(),
                'cuda_compiler': run([env['CUDACXX'], '--version'], 'nvcc-version').strip(),
                'compute_sanitizer': run([host_sanitizer or 'compute-sanitizer', '--version'],
                    'compute-sanitizer-version').strip(),
            }
            save()
        if MODE == 'cuda_posix_import':
            binary = work / 'cuda-posix-import'
            run([env['CUDACXX'], '-std=c++17', '-arch=sm_' + architecture, '-lineinfo',
                 str(source / 'experiments/cuda_posix_import.cu'), '-lcuda', '-o', str(binary)],
                'vmm-build')
            report['scope'] = 'CUDA VMM local/import diagnostics; no NCCL or BFS acceptance'
            report['vmm_cases'] = []
            for case in vmm_probe_cases():
                command = [str(binary), case['mode']]
                if case['runtime_init']:
                    command.append('--runtime-init')
                if case['symmetric']:
                    command.append('--symmetric')
                if case['tool']:
                    command = ['compute-sanitizer', '--tool', case['tool'],
                               '--error-exitcode', '97'] + command
                row = run_window_process_pair(command, source, env,
                    logs / ('vmm-' + case['label']), timeout=180,
                    required_stage='vmm_' + case['mode'], require_window=False)
                row.update(case)
                row['command'] = command
                report['vmm_cases'].append(row)
                (logs / 'vmm-cases.json').write_text(json.dumps(report['vmm_cases'], indent=2))
            report['status'] = ('COMPLETE' if all(c['pass'] for c in report['vmm_cases'])
                                else 'INCOMPLETE')
            return
        venv = work / "venv"
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        run([sys.executable, "-m", "pip", "--python", python, "install",
             "--only-binary=:all:", "--no-cache-dir", "--require-hashes", "-r",
             str(source / "experiments/library_owner/requirements-linux-x86_64.lock")],
            "dependencies", timeout=1200)
        verifier_dependencies = oracle_dependency_packages(MODE)
        site = subprocess.check_output([python, "-c", "import site; print(site.getsitepackages()[0])"],
                                       text=True, env=env).strip()
        site = Path(site)
        prefixes = library.cmake_prefixes(site)
        libdirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        nccl_target = work / "nccl"
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
             str(nccl_target), "nvidia-nccl-cu12==2.29.7"], "nccl-install")
        nccl = nccl_target / "nvidia/nccl"
        if NCCL_VARIANT in ("minimum_arch_guard", "minimum_arch_guard_posix"):
            upstream = "b91894bd5b190c874d98a017f93f5daa515b65d0"
            patch_file = source / "patches/nccl-2.29.7-minimum-arch.patch"
            patch_digest = hashlib.sha256(patch_file.read_bytes()).hexdigest()
            if patch_digest != "1af3a5df99c2b3b9a4f66ca4c33cf66e513e208f73480e58b4183c8f0e05b26b":
                raise RuntimeError("NCCL_PATCH_DIGEST_MISMATCH")
            vendor = work / "nccl-source"
            run(["git", "clone", "--depth=1", "--branch", "v2.29.7-1",
                 "https://github.com/NVIDIA/nccl.git", str(vendor)], "nccl-source")
            vendor_commit = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
            if vendor_commit != upstream:
                raise RuntimeError("NCCL_SOURCE_COMMIT_MISMATCH")
            run(["git", "apply", "--check", str(patch_file)], "nccl-patch-check", cwd=vendor)
            run(["git", "apply", str(patch_file)], "nccl-patch", cwd=vendor)
            posix_patch_digest = None
            if NCCL_VARIANT == "minimum_arch_guard_posix":
                # Fixed before process startup. Keep VMM/LSA enabled; no probe
                # error suppression, auto fallback, or legacy cudaMalloc.
                env["NCCL_MNNVL_ENABLE"] = "0"
                posix_patch = source / "patches/nccl-2.29.7-explicit-posix.patch"
                posix_patch_digest = hashlib.sha256(posix_patch.read_bytes()).hexdigest()
                if posix_patch_digest != "e73af6f263bb0eebb22904a20251c8b5da0dec2b463fdeb3c5a9c4fbc88b3072":
                    raise RuntimeError("NCCL_POSIX_PATCH_DIGEST_MISMATCH")
                run(["git", "apply", "--check", str(posix_patch)], "nccl-posix-check", cwd=vendor)
                run(["git", "apply", str(posix_patch)], "nccl-posix-patch", cwd=vendor)
                policy_env = dict(env, MGBFS_NCCL_POLICY_SOURCE=str(vendor))
                gate.run([sys.executable, str(source / "scripts/test_nccl_allocator_policy.py")],
                    cwd=source, env=policy_env, logs=logs, name="nccl-allocator-policy", timeout=60)
            local_first_patch = source / "patches/nccl-2.29.7-local-first-map.patch"
            local_first_digest = hashlib.sha256(local_first_patch.read_bytes()).hexdigest()
            if local_first_digest != "95a042db1698504182c0cf0ee34c8c74dbaf5cb376d1aeeae93d464b0378b5b1":
                raise RuntimeError("NCCL_LOCAL_FIRST_PATCH_DIGEST_MISMATCH")
            run(["git", "apply", "--check", str(local_first_patch)], "nccl-local-first-check", cwd=vendor)
            run(["git", "apply", str(local_first_patch)], "nccl-local-first-patch", cwd=vendor)
            # Existing independent-process replay resolves this exact root.
            # Preserve the wheel separately instead of accidentally replaying it.
            shutil.move(str(nccl_target), str(work / "nccl-wheel"))
            nccl = nccl_target / "nvidia/nccl"
            run(["git", "fetch", "--depth=1", "origin", NCCL_CACHE_HELPER_COMMIT],
                "nccl-cache-helper-fetch", cwd=source)
            helper_text = subprocess.check_output(["git", "show",
                NCCL_CACHE_HELPER_COMMIT + ":scripts/nccl_dependency_cache.py"],
                cwd=source, text=True)
            cache_api = {}
            exec(compile(helper_text, "pinned-nccl-cache-helper", "exec"), cache_api)
            recipe = dict(upstream=upstream, architecture="sm" + architecture,
                patches=[patch_digest, posix_patch_digest, local_first_digest], nvtx=1,
                nvcc=subprocess.check_output([str(sdk / "bin/nvcc"), "--version"], text=True),
                compiler=subprocess.check_output(["g++", "--version"], text=True),
                libc=subprocess.check_output(["ldd", "--version"], text=True),
                machine=__import__("platform").machine())
            report['nccl_cache_admission'] = dict(mode=NCCL_CACHE_MODE, recipe=recipe,
                expected_archive_sha256=NCCL_CACHE_ARCHIVE_SHA256,
                requested_kaggle_image=KAGGLE_DOCKER_IMAGE)
            save()
            if NCCL_CACHE_MODE == "build_export":
                run(["make", "-j2", "src.build", "NVTX=1", "CUDA_HOME=" + str(sdk),
                     "NVCC_GENCODE=-gencode=arch=compute_" + architecture + ",code=sm_" + architecture,
                     "BUILDDIR=" + str(nccl)], "nccl-build", cwd=vendor, timeout=5400)
                cache_manifest = cache_api["export_bundle"](nccl,
                    Path("/kaggle/working/nccl-dependency-cache"), recipe)
            elif NCCL_CACHE_MODE == "required":
                manifests = list(CACHE_INPUT_ROOT.rglob("nccl-cache.json"))
                if len(manifests) != 1 or not NCCL_CACHE_ARCHIVE_SHA256:
                    raise RuntimeError("NCCL_CACHE_REQUIRED_IDENTITY_MISSING")
                report['nccl_cache_admission']['manifest_recipe'] = json.loads(manifests[0].read_text())['recipe']
                save()
                cache_manifest = cache_api["restore_bundle"](manifests[0], nccl,
                    recipe, NCCL_CACHE_ARCHIVE_SHA256)
            else:
                raise RuntimeError("NCCL_CACHE_MODE_INVALID")
            report["nccl_cache"] = dict(mode=NCCL_CACHE_MODE,
                helper_commit=NCCL_CACHE_HELPER_COMMIT, manifest=cache_manifest)
            report["nccl_dependency"] = dict(variant=NCCL_VARIANT,
                upstream_commit=upstream, patch_sha256=patch_digest, architecture="sm" + architecture,
                posix_patch_sha256=posix_patch_digest,
                local_first_map_patch_sha256=local_first_digest,
                mnnvl_enable=env.get("NCCL_MNNVL_ENABLE"),
                nvtx=1, experimental=True,
                library_sha256=hashlib.sha256((nccl / "lib/libnccl.so.2.29.7").read_bytes()).hexdigest())
            save()
        elif NCCL_VARIANT != "wheel":
            raise RuntimeError("UNKNOWN_NCCL_VARIANT")
        if not (nccl / "include/nccl_device.h").is_file():
            raise RuntimeError("PINNED_NCCL_DEVICE_HEADER")
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        env["LD_LIBRARY_PATH"] = ":".join([str(nccl / "lib"), str(sdk / "lib"),
                                            *libdirs, env.get("LD_LIBRARY_PATH", "")])
        env["MGBFS_CUDART_LIB_DIR"] = str(sdk / "lib")
        if MODE == "nccl_abort_isolation":
            binary = work / "nccl-nonblocking-abort-isolation"
            run(["g++", "-std=c++17", "-pthread", "-x", "c++",
                 "-I" + str(nccl / "include"), "-I" + str(sdk / "include"),
                 str(source / "experiments/nccl_nonblocking_abort_isolation.cpp"),
                 "-x", "none", str(nccl / "lib/libnccl.so.2"),
                 "-L" + str(sdk / "lib"), "-lcudart",
                 "-Wl,-rpath," + str(nccl / "lib"),
                 "-o", str(binary)], "abort-isolation-build", timeout=600)
            try:
                completed = subprocess.run([str(binary)], cwd=source, env=env,
                                           capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired as error:
                (logs / "abort-isolation.log").write_text(
                    str(error.stdout) + str(error.stderr))
                report["scope"] = "independent NCCL nonblocking abort; no BFS code"
                report["abort_isolation"] = {"status": "TIMEOUT"}
                report["status"] = "DIAGNOSTIC_COMPLETE"
                return
            output = completed.stdout + completed.stderr
            (logs / "abort-isolation.log").write_text(output)
            report["scope"] = ("independent NCCL 2.29.7 nonblocking asymmetric "
                               "abort on two physical T4s; no BFS code")
            report["abort_isolation"] = {
                "returncode": completed.returncode,
                "passed_ranks": output.count("stage=abort result=PASS"),
            }
            if completed.returncode != 0 or report["abort_isolation"]["passed_ranks"] != 2:
                raise RuntimeError("NCCL_NONBLOCKING_ABORT_GATE")
            report["status"] = "COMPLETE"
            return
        if MODE in ("nccl_window_isolation", "nccl_window_nonblocking", "nccl_window_processes"):
            binary = work / "nccl-window-isolation"
            compiler = ([str(sdk / 'bin/nvcc'), '-std=c++17', '-arch=sm_' + architecture, '-lineinfo',
                         '-DMGBFS_WINDOW_DEVICE_PROBE=1', '-Xcompiler=-pthread']
                        if MODE == 'nccl_window_processes' else ['g++', '-std=c++17', '-pthread', '-x', 'c++'])
            linker = (['-Xlinker=-rpath,' + str(nccl / 'lib')] if MODE == 'nccl_window_processes'
                      else ['-x', 'none', '-Wl,-rpath,' + str(nccl / 'lib')])
            nccl_link = (['-Xlinker=' + str(nccl / 'lib/libnccl.so.2')]
                         if MODE == 'nccl_window_processes' else [str(nccl / 'lib/libnccl.so.2')])
            run([*compiler,
                 "-I" + str(nccl / "include"), "-I" + str(sdk / "include"),
                 str(source / "experiments/nccl_window_isolation.cu"),
                 *linker, *nccl_link,
                 "-L" + str(sdk / "lib"), "-lcudart",
                 "-o", str(binary)], "window-isolation-build", timeout=600)
            if MODE == 'nccl_window_processes':
                report['scope'] = 'two independent T4 rank processes; reduced NCCL window probe, not full BFS acceptance'
                report['t4_acceptance_eligible'] = False
                report['window_runs'] = {}
                probe_env = dict(env, NCCL_DEBUG='INFO')
                for stage, argument in (('window', 'nonblocking'),
                        ('device_comm_zero_create', 'device_comm_zero'),
                        ('device_comm_only_create', 'device_comm_only'), ('device_comm_create', 'device_comm')):
                    for tool in ('plain', 'memcheck', 'racecheck', 'initcheck', 'synccheck'):
                        label = stage + '-' + tool
                        command = [str(binary), argument]
                        if tool != 'plain':
                            command = ['compute-sanitizer', '--tool', tool, '--error-exitcode', '97', *command]
                        row = run_window_process_pair(command, source, probe_env, logs / ('window-process-' + label),
                            required_stage=stage if stage != 'window' else None,
                            require_window=argument not in ('device_comm_only', 'device_comm_zero'))
                        row['command'] = command
                        if tool != 'plain':
                            from replay_lsa_cancel_candidate import instrumentation_clean
                            row['instrumentation_clean'] = all(instrumentation_clean(
                                (logs / ('window-process-' + label) / f'rank-{rank}.log').read_text(errors='replace'), tool)
                                for rank in (0, 1))
                            row['pass'] &= row['instrumentation_clean']
                        report['window_runs'][label] = row
                        save()
                report['status'] = 'DIAGNOSTIC_COMPLETE'
                return
            report["scope"] = ("independent NCCL ncclMemAlloc and "
                               "ncclCommWindowRegister on two physical T4s; no BFS code")
            report["window_runs"] = {}
            cases = (("blocking", [str(binary)]),
                     ("nonblocking", [str(binary), "nonblocking"])) if MODE == "nccl_window_nonblocking" else (
                     ("plain", [str(binary)]),
                     ("initcheck", ["compute-sanitizer", "--tool", "initcheck",
                                    "--report-api-errors", "no", "--error-exitcode", "97",
                                    str(binary)]))
            for label, command in cases:
                try:
                    completed = subprocess.run(command, cwd=source, env=env,
                                               capture_output=True, text=True,
                                               timeout=180)
                    output = completed.stdout + completed.stderr
                    (logs / ("window-" + label + ".log")).write_text(output)
                    report["window_runs"][label] = {
                        "returncode": completed.returncode,
                        "registered_ranks": output.count("stage=window_register result=PASS"),
                        "zero_sanitizer_errors": "ERROR SUMMARY: 0 errors" in output,
                    }
                except subprocess.TimeoutExpired as error:
                    (logs / ("window-" + label + ".log")).write_text(
                        str(error.stdout) + str(error.stderr))
                    report["window_runs"][label] = {"status": "TIMEOUT"}
                save()
            report["status"] = ("COMPLETE" if MODE == "nccl_window_nonblocking" and
                                all(row.get("returncode") == 0 and row.get("registered_ranks") == 2
                                    for row in report["window_runs"].values()) else
                                "DIAGNOSTIC_COMPLETE")
            return
        env["CARGO_HOME"] = str(work / "cargo")
        env["RUSTUP_HOME"] = str(work / "rustup")
        installer = work / "rustup-init.sh"
        run(["curl", "--fail", "--location", "--max-time", "180",
             "https://sh.rustup.rs", "-o", str(installer)], "rust-download")
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal",
             "--default-toolchain", gate.RUST_VERSION], "rust-install")
        env["PATH"] = str(work / "cargo/bin") + ":" + env["PATH"]
        cuco = work / "cuco"
        gate.checkout("https://github.com/NVIDIA/cuCollections.git", CUCO,
                      cuco, env, logs, "cuco")
        build = work / "library-build"
        report['compiled_cache_fixtures'] = compiled_cache_fixtures()
        compiled_target('library', source, build,
            ["cmake", "-S", str(source / "experiments/library_owner"), "-B", str(build),
             "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=" + architecture,
             "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUDAToolkit_ROOT=" + str(sdk),
             "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
             "-DCUCO_ROOT=" + str(cuco)], 'mgbfs_library_owner', 'libmgbfs_library_owner.so')
        env["MGBFS_LIBRARY_OWNER_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":" + env["LD_LIBRARY_PATH"]
        cutlass = work / "cutlass"
        gate.checkout("https://github.com/NVIDIA/cutlass.git", gate.CUTLASS_COMMIT,
                      cutlass, env, logs, "cutlass")
        native = work / "native-build"
        compiled_target('native', source, native,
            ["cmake", "-S", str(source / "cuda"), "-B", str(native), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
             *(["-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g1"]
               if MODE == "timeline_backtrace" else []),
             "-DCMAKE_CUDA_ARCHITECTURES=" + architecture, "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUTLASS_ROOT=" + str(cutlass), "-DMGBFS_NCCL_LSA=ON",
             "-DMGBFS_NCCL_ROOT=" + str(nccl),
             *(['-DMGBFS_NVTX=ON', '-DMGBFS_NVTX_INCLUDE_DIR=' + str(sdk / 'include')]
               if nvtx_enabled else [])], 'mgbfs_cuda', 'libmgbfs_cuda.so')
        env["MGBFS_CUDA_LIB_DIR"] = str(native)
        env["LD_LIBRARY_PATH"] = str(native) + ":" + env["LD_LIBRARY_PATH"]
        # Oracle-only Arrow must not alter the native build environment: the
        # preserved cache producer did not compile against its package/prefix.
        # Restore and verify both native libraries before installing the oracle.
        if verifier_dependencies:
            # The full-state oracle reuses the archive reader in the Parquet
            # exporter; its module-level schemas require Arrow at import time.
            run([sys.executable, "-m", "pip", "--python", python, "install",
                 "--only-binary=:all:", "--no-deps", *verifier_dependencies],
                "archive-verifier-dependency", timeout=300)
            run([python, '-c', 'from export_hf_dataset import frames'],
                'archive-verifier-import-preflight', timeout=30,
                cwd=source / 'scripts')
        if MODE in ('rank_compare_red', 'rank_compare_green'):
            env.update(CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER')
            binary=work/'bounded-owner-launch-budget'
            run([str(sdk/'bin/nvcc'),'-std=c++17','-arch=sm_75',
                '-I'+str(source/'cuda'),str(source/'tests/bounded_owner.cu'),
                '-L'+str(native),'-lmgbfs_cuda','-lcudart',
                '-Xlinker','-rpath='+str(native),'-o',str(binary)],
                'rank-compare-test-build',timeout=300)
            result=subprocess.run([str(binary)],env=env,cwd=source,
                capture_output=True,text=True,timeout=120)
            (logs/'rank-compare-launch-budget.log').write_text(result.stdout+result.stderr)
            report['rank_compare_test']={'returncode':result.returncode,'output':result.stdout+result.stderr}
            if MODE == 'rank_compare_red':
                if result.returncode != 1 or 'RANK_COMPARE_KERNEL_NODES=11' not in result.stdout or 'rank compare launch budget grows with scratch groups' not in result.stderr:
                    raise RuntimeError('RED_TEST_DID_NOT_FAIL_FOR_LAUNCH_BUDGET')
                report['status']='EXPECTED_RED';save();return
            if result.returncode != 0 or 'RANK_COMPARE_KERNEL_NODES=8' not in result.stdout or 'BOUNDED_OWNER_PASS' not in result.stdout:
                raise RuntimeError('GREEN_RANK_COMPARE_TEST_FAILED')
            bmma_binary=work/'bounded-owner-bmma'
            run([str(sdk/'bin/nvcc'),'-std=c++17','-arch=sm_75','-DMGBFS_TEST_BMMA',
                '-I'+str(source/'cuda'),str(source/'tests/bounded_owner.cu'),
                '-L'+str(native),'-lmgbfs_cuda','-lcudart',
                '-Xlinker','-rpath='+str(native),'-o',str(bmma_binary)],
                'rank-compare-bmma-test-build',timeout=300)
            output=run([str(bmma_binary)],'rank-compare-bmma-test',timeout=120)
            if 'BOUNDED_OWNER_PASS' not in output:
                raise RuntimeError('BMMA_REGRESSION_TEST_FAILED')
            report['status']='OWNER_UNIT_GREEN';save();return
        if MODE in ('old_new_pair', 'paired_measure', 'paired_timeline'):
            env.update(CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER')
            import statistics, math, difflib
            report['scope']='paired physical 2xT4 S10 old/new HOST_SIZED_NCCL search-only; archive disabled; not LSA acceptance; no performance claim until all repetitions verified'
            oldsha='4ef9ce1d16c8cef62fc610cd6d36e32e673e623b'
            old=work/'old-source';gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git',oldsha,old,env,logs,'old')
            example=old/'crates/mgbfs-runtime/examples/distributed_bench.rs';original=example.read_text();patched=original
            begin=patched.index(' let description=');end=patched.index(' let setup=',begin)
            patched=patched[:begin]+' let pinned=0usize;let disk_bytes=0u64;\n'+patched[end:]
            assert patched.count('bfs.archive_current(&mut archive)?;')==1
            assert patched.count('archive.finish()?;')==1
            patched=patched.replace('bfs.archive_current(&mut archive)?;','').replace('archive.finish()?;','')
            assert patched.count('let args:Vec<_>=std::env::args().collect();')==1
            patched=patched.replace('let args:Vec<_>=std::env::args().collect();','let mut args:Vec<_>=std::env::args().collect();if std::env::var("MGBFS_OLD_PHASE").as_deref()==Ok("warmup"){for index in 3..=5{args[index].push_str(".warmup");}}')
            oldmain='fn main(){if let Err(e)=run(){eprintln!("DISTRIBUTED_BENCH_INCOMPLETE: {e}");std::process::exit(1)}}'
            newmain='fn main(){for phase in ["warmup","measure"]{std::env::set_var("MGBFS_OLD_PHASE",phase);if let Err(e)=run(){eprintln!("DISTRIBUTED_BENCH_INCOMPLETE: {e}");std::process::exit(1)}}}'
            assert patched.count(oldmain)==1;patched=patched.replace(oldmain,newmain)
            # Diagnostic-only CUDA profiler boundaries. The old search algorithm is unchanged.
            patched += '\nextern "C" { #[link_name="cudaProfilerStart"] fn mgbfs_old_profile_start()->i32; #[link_name="cudaProfilerStop"] fn mgbfs_old_profile_stop()->i32; }\n'
            before='let start=Instant::now();let mut layers=Vec::new();'
            after='let profiling=std::env::var("MGBFS_PROFILE_SEARCH").as_deref()==Ok("1") && std::env::var("MGBFS_OLD_PHASE").as_deref()==Ok("measure");if profiling && unsafe{mgbfs_old_profile_start()}!=0{return Err("PROFILE_START".into())}let start=Instant::now();let mut layers=Vec::new();'
            assert patched.count(before)==1;patched=patched.replace(before,after)
            before='let search=start.elapsed().as_secs_f64();'
            after='let search=start.elapsed().as_secs_f64();if profiling && unsafe{mgbfs_old_profile_stop()}!=0{return Err("PROFILE_STOP".into())}'
            assert patched.count(before)==1;patched=patched.replace(before,after)
            example.write_text(patched)
            (logs/'old-search-only-launcher.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='a/distributed_bench.rs',tofile='b/distributed_bench.rs')))
            report['old_source']=oldsha;report['old_launcher_sha256']=hashlib.sha256(patched.encode()).hexdigest();save()
            oldbuild=work/'old-native'
            compiled_target('old', old, oldbuild,
                ['cmake','-S',str(old/'cuda'),'-B',str(oldbuild),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DBUILD_TESTING=OFF','-DCMAKE_CUDA_ARCHITECTURES='+architecture,'-DCMAKE_CUDA_COMPILER='+str(sdk/'bin/nvcc'),'-DCUTLASS_ROOT='+str(cutlass),'-DCMAKE_CUDA_FLAGS=-I'+str(nccl/'include'),'-DCMAKE_SHARED_LINKER_FLAGS=-L'+str(nccl/'lib')], 'mgbfs_cuda', 'libmgbfs_cuda.so')
            oldenv=dict(env,MGBFS_CUDA_LIB_DIR=str(oldbuild),LD_LIBRARY_PATH=str(oldbuild)+':'+env['LD_LIBRARY_PATH'])
            def cargo_at(root,buildenv,label,features):
                p=subprocess.run(['cargo','build','--locked','--release','-p','mgbfs-runtime','--features',features,'--example','distributed_bench'],cwd=root,env=buildenv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
                (logs/(label+'.log')).write_text(p.stdout)
                if p.returncode:raise RuntimeError(label+' failed')
            cargo_at(old,oldenv,'old-rust','cuda');cargo_at(source,env,'new-rust','cuda,library-owner')
            sys.path.insert(0,str(source/'scripts'))
            from distributed_gpu_bench import run_group
            configs=[('old',old/'target/release/examples/distributed_bench',oldenv),('new-cub',source/'target/release/examples/distributed_bench',dict(env,MGBFS_OWNER_BACKEND='CUB_SORT_MERGE',MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL',MGBFS_SHARDS='64')),('new-cuco',source/'target/release/examples/distributed_bench',dict(env,MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_LIBRARY_POOL_BYTES=str(512<<20),MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL',MGBFS_SHARDS='4'))]
            configs.extend([
                ('new-hash-first-cub', source/'target/release/examples/distributed_bench',
                    dict(env, MGBFS_PROFILE='HASH_FIRST', MGBFS_HASH_FIRST_GENERATION='SCALAR',
                        MGBFS_OWNER_BACKEND='CUB_SORT_MERGE', MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64')),
                ('new-hash-first-bmma', source/'target/release/examples/distributed_bench',
                    dict(env, MGBFS_PROFILE='HASH_FIRST', MGBFS_HASH_FIRST_GENERATION='SCALAR',
                        MGBFS_OWNER_BACKEND='BMMA_BUCKET', MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64'))])
            if MODE == 'paired_measure':
                configs.append(('new-dense-bmma', source/'target/release/examples/distributed_bench',
                    dict(env, MGBFS_PROFILE='DENSE', MGBFS_OWNER_BACKEND='BMMA_BUCKET',
                        MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64')))
            configs.append(('new-hash-first-cuco', source/'target/release/examples/distributed_bench',
                dict(env, MGBFS_PROFILE='HASH_FIRST', MGBFS_HASH_FIRST_GENERATION='SCALAR',
                    MGBFS_OWNER_BACKEND='CUCO_RANK', MGBFS_LIBRARY_POOL_BYTES=str(512<<20),
                    MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='4')))
            expected=[1, 3, 6, 12, 24, 47, 87, 161, 297, 528, 927, 1611, 2726, 4492, 7184, 11109, 16751, 24624, 35105, 48718, 66154, 87373, 111996, 140388, 171657, 204213, 236429, 266276, 291271, 308831, 316158, 310824, 290837, 254374, 199563, 129134, 61718, 20467, 5183, 1116, 302, 92, 21, 6, 3, 1]
            if MODE in ('paired_measure', 'paired_timeline'):
                before_sha = BEFORE_SOURCE
                before_source = work / 'before-source'
                gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git', before_sha, before_source, env, logs, 'before')
                # Separate native artifacts: a CUDA candidate must never overwrite
                # the measured candidate's library while preparing its control.
                # Each source retains its independently pinned recipe/digest.
                before_build = work / 'before-native' 
                compiled_target('before', before_source, before_build,
                    ['cmake', '-S', str(before_source/'cuda'), '-B', str(before_build), '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF',
                     '-DCMAKE_CUDA_ARCHITECTURES='+architecture, '-DCMAKE_CUDA_COMPILER='+str(sdk/'bin/nvcc'),
                     '-DCUTLASS_ROOT='+str(cutlass), '-DMGBFS_NCCL_LSA=ON',
                     '-DMGBFS_NCCL_ROOT='+str(nccl), '-DMGBFS_NVTX=ON',
                     '-DMGBFS_NVTX_INCLUDE_DIR='+str(sdk/'include')], 'mgbfs_cuda', 'libmgbfs_cuda.so')
                if before_build.resolve() == native.resolve():
                    raise RuntimeError('PAIR_NATIVE_ARTIFACT_ALIAS')
                report['paired_native_artifacts'] = {
                    'before': report['compiled_cache']['before'],
                    'candidate': report['compiled_cache']['native'],
                    'independent_directories': True,
                }
                before_env = dict(env, MGBFS_CUDA_LIB_DIR=str(before_build), LD_LIBRARY_PATH=str(before_build)+':'+env['LD_LIBRARY_PATH'])
                # The control must load its own source-bound owner ELF as well
                # as its own native ELF. Do not accidentally benchmark before
                # Rust against the candidate's library owner implementation.
                before_library = work / 'before-library-build'
                compiled_target('before-library', before_source, before_library,
                    ['cmake', '-S', str(before_source / 'experiments/library_owner'),
                     '-B', str(before_library), '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=' + architecture,
                     '-DCMAKE_CUDA_COMPILER=' + str(sdk / 'bin/nvcc'),
                     '-DCUDAToolkit_ROOT=' + str(sdk),
                     '-DCMAKE_PREFIX_PATH=' + ';'.join(prefixes),
                     '-DCUCO_ROOT=' + str(cuco)],
                    'mgbfs_library_owner', 'libmgbfs_library_owner.so')
                if before_library.resolve() == build.resolve():
                    raise RuntimeError('PAIR_LIBRARY_ARTIFACT_ALIAS')
                before_env['MGBFS_LIBRARY_OWNER_LIB_DIR'] = str(before_library)
                before_env['LD_LIBRARY_PATH'] = str(before_library) + ':' + before_env['LD_LIBRARY_PATH']
                report['paired_library_artifacts'] = {
                    'before': report['compiled_cache']['before-library'],
                    'candidate': report['compiled_cache']['library'],
                    'independent_directories': True,
                }
                cargo_at(before_source, before_env, 'before-rust', 'cuda,library-owner')
                before_binary = before_source/'target/release/examples/distributed_bench'
                configs.extend([
                    ('before-cub', before_binary, dict(before_env, MGBFS_OWNER_BACKEND='CUB_SORT_MERGE', MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64')),
                    ('before-cuco', before_binary, dict(before_env, MGBFS_OWNER_BACKEND='CUCO_RANK', MGBFS_LIBRARY_POOL_BYTES=str(512<<20), MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='4'))])
                configs.extend([
                    ('before-hash-first-cub', before_binary,
                        dict(before_env, MGBFS_PROFILE='HASH_FIRST', MGBFS_HASH_FIRST_GENERATION='SCALAR',
                            MGBFS_OWNER_BACKEND='CUB_SORT_MERGE', MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64')),
                    ('before-hash-first-bmma', before_binary,
                        dict(before_env, MGBFS_PROFILE='HASH_FIRST', MGBFS_HASH_FIRST_GENERATION='SCALAR',
                            MGBFS_OWNER_BACKEND='BMMA_BUCKET', MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64'))])
                configs.append(('before-hash-first-cuco', before_binary,
                    dict(before_env, MGBFS_PROFILE='HASH_FIRST', MGBFS_HASH_FIRST_GENERATION='SCALAR',
                        MGBFS_OWNER_BACKEND='CUCO_RANK', MGBFS_LIBRARY_POOL_BYTES=str(512<<20),
                        MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='4')))
                if MODE == 'paired_measure':
                    configs.append(('before-dense-bmma', before_binary,
                        dict(before_env, MGBFS_PROFILE='DENSE', MGBFS_OWNER_BACKEND='BMMA_BUCKET',
                            MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL', MGBFS_SHARDS='64')))
                report['before_source'] = before_sha
            if MODE == 'paired_timeline':
                retained = {'old', 'new-cub', 'before-cub',
                    'new-cuco', 'before-cuco',
                    'new-hash-first-cub', 'before-hash-first-cub'}
                configs = [entry for entry in configs if entry[0] in retained]
                if len(configs) != len(retained):
                    raise RuntimeError('PAIRED_TIMELINE_INVENTORY')
            if MODE == 'old_new_pair':
                configs = [entry for entry in configs if entry[0] in {'old', 'new-cub', 'new-cuco'}]
                if len(configs) != 3:
                    raise RuntimeError('OLD_NEW_PAIR_INVENTORY')
            report['expected_layers_provenance']='retained S10 v8 native/CayleyPy paired summary; cycle, inverse cycle, swap(0,1); not adjacent-transposition Mahonian counts'
            def generator_contract(root):
                text=(root/'crates/mgbfs-core/src/matrix.rs').read_text();start=text.index('pub fn symmetric_permutation_matrices');return text[start:text.index('    /// Independent exact oracle:',start)]
            assert generator_contract(old)==generator_contract(source), 'GRAPH_GENERATOR_CONTRACT_CHANGED'
            if MODE in ('paired_measure', 'paired_timeline'):
                assert generator_contract(before_source)==generator_contract(source), 'BEFORE_GRAPH_GENERATOR_CONTRACT_CHANGED'
            assert sum(expected)==math.factorial(10) and len(expected)==46
            report['batch_size']=32768;report['batch_gate']='multiple batches within S10 peak layers; distinguish FinalizeDepth from healthy transitions';report['warmup_protocol']='full untimed BFS inside each rank-process, fresh runtime object for timed pass; both sources; archives disabled';report['rows']=[];report['comparisons']={};save()
            for repetition in range(-1, 5 if MODE in ('paired_measure', 'old_new_pair') else 0):
                order=configs if repetition%2==0 else list(reversed(configs))
                for name,binary,baseenv in order:
                    label=name+'-'+('warmup' if repetition<0 else 'rep'+str(repetition))
                    cfg=dict(baseenv,MGBFS_SEARCH_ONLY='1',MGBFS_BENCH_SKIP_ARCHIVE='1',MGBFS_BENCH_WARMUP='1',MGBFS_BENCH_WORLD_SIZE='2',MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BENCH_CAPACITY=str(math.factorial(10)),MGBFS_FUTURE_CAPACITY=str(math.factorial(10)),MGBFS_PROFILE=baseenv.get('MGBFS_PROFILE', 'DENSE'),MGBFS_PRE_DEDUP='ON',MGBFS_STATE_CODEC='matrix_u8',MGBFS_GENERATION_VARIANT='1',MGBFS_RANK_MAP='0,1',MGBFS_ROUTE_BANKS='3',MGBFS_EPOCH_WINDOW='3')
                    for key in ('MGBFS_TRACE_DEPTHS','MGBFS_TRACE_ROUTE'):cfg.pop(key,None)
                    cmd=['torchrun','--standalone','--nproc-per-node=2','--no-python',str(binary),'s10','32768',str(work/(label+'-bootstrap')),str(work/(label+'-archive')),'{RANK_OUT}']
                    row=run_group(cmd,logs,label,cfg,timeout=180)
                    row.update(configuration=name,repetition=repetition,archive_disabled=True,source=oldsha if name=='old' else before_sha if name.startswith('before-') else SOURCE)
                    report['rows'].append(row);save()
                    if row['status']!='COMPLETE' or row['layer_sizes']!=expected:raise RuntimeError('PAIR_RUN_FAILED '+label)
                    if name!='old' and any(x.get('archive_enabled') is not False for x in row['rank_results']):raise RuntimeError('ARCHIVE_NOT_DISABLED')
                    if name!='old' and any(x.get('frontier_profile') != cfg['MGBFS_PROFILE'] for x in row['rank_results']):raise RuntimeError('PROFILE_PROVENANCE_MISMATCH '+label)
                    row['frontier_profile'] = cfg['MGBFS_PROFILE']
                    row['hash_first_generation'] = cfg.get('MGBFS_HASH_FIRST_GENERATION', 'SCALAR')
            if MODE in ('paired_measure', 'old_new_pair'):
                for name, _, _ in configs:
                    rows = [x for x in report['rows'] if x['configuration'] == name and x['repetition'] >= 0]
                    if len(rows) != 5:
                        raise RuntimeError('PAIR_REPEAT_INVENTORY ' + name)
                    values = [x['search_complete_seconds'] for x in rows]
                    median = statistics.median(values)
                    report['comparisons'][name] = dict(
                        samples_seconds=values, median_seconds=median,
                        mad_seconds=statistics.median(abs(x-median) for x in values),
                        peak_mib_per_rank=[max(x['smi_peak_mib_per_rank'][r] for x in rows) for r in range(2)])
                report['status'] = 'COMPLETE'
                report['scope'] = 'Five unprofiled paired repetitions, full in-process warmup, archive disabled; no profiler timings'
                save()
                if MODE == 'paired_measure':
                    return
            # Separate diagnostic captures: never aggregate these profiler timings.
            nsys=prepare_nsys();report['profile_rows']=[]
            for name,binary,baseenv in configs:
                label='profile-'+name
                cfg=dict(baseenv,MGBFS_SEARCH_ONLY='1',MGBFS_BENCH_SKIP_ARCHIVE='1',MGBFS_BENCH_WARMUP='1',MGBFS_BENCH_WORLD_SIZE='2',MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BENCH_CAPACITY=str(math.factorial(10)),MGBFS_FUTURE_CAPACITY=str(math.factorial(10)),MGBFS_PROFILE=baseenv.get('MGBFS_PROFILE', 'DENSE'),MGBFS_PRE_DEDUP='ON',MGBFS_STATE_CODEC='matrix_u8',MGBFS_GENERATION_VARIANT='1',MGBFS_RANK_MAP='0,1',MGBFS_ROUTE_BANKS='3',MGBFS_EPOCH_WINDOW='3')
                for key in ('MGBFS_TRACE_DEPTHS','MGBFS_TRACE_ROUTE','NCCL_DEBUG','NCCL_DEBUG_SUBSYS'):cfg.pop(key,None)
                cfg['MGBFS_PROFILE_SEARCH']='1'
                if name!='old':cfg['MGBFS_TRACE_RANGES']='1'
                trace=logs/label;trace.mkdir()
                cmd=['torchrun','--standalone','--nproc-per-node=2','--no-python',nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output='+str(trace/'process-%p'),str(binary),'s10','32768',str(work/(label+'-bootstrap')),str(work/(label+'-archive')),'{RANK_OUT}']
                row=run_group(cmd,logs,label,cfg,timeout=300)
                row.update(profiled=True,configuration=name,source=oldsha if name=='old' else before_sha if name.startswith('before-') else SOURCE,scope='CUDA profiler start/stop bounds measured BFS only; warmup and initialization excluded; profiler duration is not A/B timing')
                if row['status']!='COMPLETE' or row['layer_sizes']!=expected:raise RuntimeError('PROFILE_RUN_FAILED '+label)
                if name!='old' and any(x.get('archive_enabled') is not False for x in row['rank_results']):raise RuntimeError('PROFILE_ARCHIVE_ENABLED')
                if name!='old' and any(x.get('frontier_profile') != cfg['MGBFS_PROFILE'] for x in row['rank_results']):raise RuntimeError('TRACE_PROFILE_PROVENANCE_MISMATCH '+label)
                row['frontier_profile'] = cfg['MGBFS_PROFILE']
                reps=sorted(trace.glob('*.nsys-rep'));assert len(reps)==2,'PROFILE_TRACE_INVENTORY'
                row['sqlite']=[]
                for rep in reps:
                    db=rep.with_suffix('.sqlite')
                    run([nsys,'export','--type','sqlite','--force-overwrite=true','--output',str(db),str(rep)],'export-'+rep.stem)
                    row['sqlite'].append(str(db.relative_to(logs)))
                report['profile_rows'].append(row);save()
            report['scope']='Five unprofiled old/CUB/CUCO S10 paired repetitions plus separate measured-search-only CUDA-profiler captures; new batch NVTX enabled; profiler times are not performance samples'
            report['status']='PROFILE_PASS';save();return
            for name,_,_ in configs:
                rows=[x for x in report['rows'] if x['configuration']==name and x['repetition']>=0];values=[x['search_complete_seconds'] for x in rows];median=statistics.median(values)
                report['comparisons'][name]=dict(samples_seconds=values,median_seconds=median,mad_seconds=statistics.median(abs(x-median) for x in values),peak_mib_per_rank=[max(x['smi_peak_mib_per_rank'][r] for x in rows) for r in range(2)])
            report['status']='COMPLETE';save();return
        if MODE == 'single_rank_production_gate':
            report['scope']='one physical T4, RunConfigV1 full-state S4 archives, not multi-rank admission or performance'
            run(['cargo','build','--locked','--release','-p','mgbfs-cli','--features','cuda,library-owner'],'one-rank-cli-build',timeout=1800)
            sys.path.insert(0,str(source/'scripts'))
            from replay_lsa_cancel_candidate import verify_process_archives
            report['single_rank_runs']=[]
            for owner in ('CUB_SORT_MERGE','BMMA_BUCKET','CUCO_RANK'):
                for profile in ('DENSE','HASH_FIRST'):
                    for pre in (False,True):
                        cfg=json.loads((source/'tests/run-s4-two-rank.json').read_text())
                        cfg.update(owner_backend=owner,frontier_profile=profile,local_pre_dedup=pre,completion_epoch_window=3,library_pool_bytes=(96<<20) if owner=='CUCO_RANK' else None)
                        cfg['topology'].update(world_size=1,logical_owner_to_rank=[0]);cfg['capacities'].update(route_slot_count=3)
                        label=f'one-rank-{owner}-{profile}-pre{int(pre)}';case=logs/label;case.mkdir();path=case/'run-config.json';path.write_text(json.dumps(cfg))
                        e=dict(env,CUDA_VISIBLE_DEVICES='0',RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',MGBFS_TRANSPORT_BACKEND='NCCL_LSA',MGBFS_BENCH_WARMUP='0',MGBFS_BENCH_WORLD_SIZE='1',TORCHELASTIC_RUN_ID='one-rank-v109-'+label)
                        row={'label':label,'owner':owner,'profile':profile,'pre_dedup':pre,'pass':False}
                        cli=str(source/'target/release/mgbfs')
                        try:
                            p=subprocess.run([cli,'preflight','--offline',str(path)],env=e,cwd=source,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
                            (case/'preflight.log').write_text(p.stdout);row['preflight_code']=p.returncode
                            if p.returncode==0:
                                digest=json.loads(p.stdout)['config_digest']
                                p=subprocess.run([cli,'run',str(path),str(case/'bootstrap'),str(case/'archive'),str(case/'result')],env=e,cwd=source,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
                                (case/'rank-0.log').write_text(p.stdout);row['exit_code']=p.returncode
                                if p.returncode==0:
                                    row['oracle']=verify_process_archives(case,n=4,world=1,expected_run_contract='RunConfigV1',expected_config_digest=digest,expected_owner_backend=owner);row['pass']=True
                        except subprocess.TimeoutExpired:row['failure']='TIMEOUT'
                        except Exception as err:row['failure_type']=type(err).__name__
                        report['single_rank_runs'].append(row);save()
            report['status']='COMPLETE' if len(report['single_rank_runs'])==12 and all(x['pass'] for x in report['single_rank_runs']) else 'INCOMPLETE';save();return
        if MODE == 'macro_production_dispatch_red':
            report['scope'] = 'RED: actual two-rank RunConfigV1 weighted macro dispatch; not a primitive or performance gate'
            run(['cargo', 'build', '--locked', '--release', '-p', 'mgbfs-cli',
                 '--features', 'cuda,library-owner'], 'macro-production-cli-build', timeout=1800)
            config = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
            config.update(macro_depth=2, completion_epoch_window=3,
                          transport_backend='HOST_SIZED_NCCL')
            # Valid capacity above the complete S4 nonidentity operator ball.
            # V1 admits capacity >= generated rows; unused capacity is not sent.
            config['capacities']['route_slot_records'] = 32
            case = logs / 'macro-production-k2'
            case.mkdir()
            config_path = case / 'run-config.json'
            config_path.write_text(json.dumps(config, indent=2))
            cli = str(source / 'target/release/mgbfs')
            preflight = run([cli, 'preflight', '--offline', str(config_path)],
                            'macro-production-preflight', timeout=60)
            preflight_json = json.loads(preflight)
            (case / 'preflight.json').write_text(json.dumps(preflight_json, indent=2))
            command = ['torchrun', '--standalone', '--nproc-per-node=2',
                '--no-python', '--log-dir', str(case / 'torchrun'),
                '--redirects', '3', '--tee', '3', cli, 'run', str(config_path),
                str(work / 'macro-production-bootstrap'),
                str(case / 'archive'), str(case / 'result')]
            process = subprocess.Popen(command, cwd=source,
                env=dict(env, MGBFS_BENCH_WARMUP='0'),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            started = time.monotonic()
            timed_out = False
            try:
                output, _ = process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate(timeout=15)
            (case / 'launcher.log').write_text(output)
            rank_logs = sorted((case / 'torchrun').rglob('stderr.log'))
            texts = [p.read_text() for p in rank_logs]
            markers = list((case / 'result').rglob('group-complete.json'))
            expected = (not timed_out and process.returncode != 0 and len(texts) == 2
                and all('RUN_MACRO_DISPATCH_UNAVAILABLE' in text for text in texts)
                and not markers)
            report['macro_production_dispatch'] = dict(
                macro_depth=2, world_size=2, preflight_valid=True,
                run_config_digest=preflight_json.get('config_digest'),
                exit_code=process.returncode, seconds=time.monotonic() - started,
                rank_stderr_files=[str(p.relative_to(logs)) for p in rank_logs],
                timed_out=timed_out, group_complete_present=bool(markers),
                observed_missing_runtime=expected, required_result='COMPLETE exact original S4 layers')
            report['status'] = 'EXPECTED_RED' if expected else 'INCOMPLETE'
            save()
            if not expected:
                raise RuntimeError('MACRO_PRODUCTION_RED_FAILURE_NOT_EXPLAINED')
            return
        def check_weighted_sanitizers(expected_tests, skip):
            artifacts = run(['cargo', 'test', '--locked', '-p', 'mgbfs-runtime',
                '--features', 'cuda,library-owner', '--lib', '--no-run', '--message-format=json'],
                'weighted-runtime-artifact', timeout=900)
            executables = []
            for line in artifacts.splitlines():
                try:
                    artifact = json.loads(line)
                except ValueError:
                    continue
                if (artifact.get('reason') == 'compiler-artifact' and artifact.get('executable')
                        and artifact.get('target', {}).get('name') == 'mgbfs-runtime'
                        and artifact.get('profile', {}).get('test') is True
                        and artifact.get('target', {}).get('kind') == ['lib']):
                    executables.append(artifact['executable'])
            if len(set(executables)) != 1:
                raise RuntimeError('WEIGHTED_RUNTIME_TEST_EXECUTABLE_IDENTITY')
            binary = Path(executables[0])
            report['weighted_runtime_test_binary'] = dict(sha256=file_digest(binary),
                bytes=binary.stat().st_size, source=SOURCE, features='cuda,library-owner')
            spec = importlib.util.spec_from_file_location('weighted_sanitizer_checks',
                source / 'scripts/replay_lsa_cancel_candidate.py')
            checks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checks)
            report['weighted_sanitizers'] = []
            for device in (0, 1):
                for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    sanitizer = ['compute-sanitizer', '--tool', tool,
                        '--target-processes', 'all', '--error-exitcode', '97']
                    if tool == 'memcheck':
                        sanitizer += ['--leak-check', 'full']
                    sanitizer += [str(binary), 'distributed_native::weighted_producer_tests',
                        '--nocapture', '--test-threads=1']
                    if skip:
                        sanitizer += ['--skip', skip]
                    output = gate.run(sanitizer, cwd=source,
                        env=dict(env, CUDA_VISIBLE_DEVICES=str(device)), logs=logs,
                        name='weighted-sanitizer-device-' + str(device) + '-' + tool,
                        # Full advance now spans 24 cases, vs 4 in v28 (racecheck 260 s).
                        timeout=3600 if tool == 'racecheck' else 900)
                    passed = (f'test result: ok. {expected_tests} passed; 0 failed' in output
                        and 'MGBFS_WEIGHTED_RING_READER_DAG_CAPTURE launched' in output
                        and checks.instrumentation_clean(output, tool))
                    report['weighted_sanitizers'].append(dict(device=device, tool=tool,
                        command=sanitizer, tests=expected_tests, unfiltered=True, skipped_tests=[skip] if skip else [], status='PASS' if passed else 'FAIL'))
                    save()
                    if not passed:
                        raise RuntimeError('WEIGHTED_SANITIZER_NOT_CLEAN_' + tool)

        if MODE in ('weighted_multi_rank_gate', 'weighted_timeline', 'owner_graph_gate', 'owner_graph_full_gate', 'owner_graph_timeline', 'weighted_library_gate'):
            report['scope'] = ('Actual two-rank original-depth weighted DENSE selected-owner BFS '
                'using shared transport/owner DAG; HOST sizing remains explicit; '
                'LSA requires working P2P, never padded payload fallback')
            env.update(CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER')
            run(['cargo', 'test', '--locked', '-p', 'mgbfs-runtime',
                '--features', 'cuda,library-owner', '--lib', '--', '--test-threads=1'],
                'weighted-shared-runtime-suite', timeout=900)
            if MODE in ('owner_graph_full_gate', 'owner_graph_timeline'):
                env['MGBFS_OWNER_GRAPH_REPLAY'] = '1'
                report['owner_graph_full_gate'] = True
            report['weighted_multi_rank_runs'] = []
            def weighted_replay(cfg, label, extra, timeout=1800):
                path = logs / (label + '-config.json')
                path.write_text(json.dumps(cfg, indent=2))
                command = [python, str(source / 'scripts/replay_lsa_cancel_candidate.py'),
                    str(work), str(logs / label), '--run-config', str(path), *extra]
                with (logs / (label + '.log')).open('w') as output:
                    outcome = run_protocol_replay(command, cwd=source, env=env,
                        log=output, timeout=timeout)
                summary_path = logs / label / 'summary.json'
                detail = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                cases = detail.get('cases', [])
                passed = (outcome['returncode'] == 0 and not outcome['timed_out']
                    and detail.get('base_commit') == SOURCE and bool(cases)
                    and all(row.get('pass') and not row.get('forced_cleanup') for row in cases))
                if passed and MODE == 'owner_graph_full_gate':
                    graph_receipts = []
                    for rank in (0, 1):
                        path = logs / label / 'healthy-None' / ('rank-' + str(rank) + '.log')
                        records = [json.loads(line.split(' ', 1)[1])
                            for line in path.read_text(errors='replace').splitlines()
                            if line.startswith('MGBFS_OWNER_GRAPH_SUMMARY ')]
                        if (len(records) != 1 or records[0].get('rank') != rank
                                or records[0].get('instances', 0) <= 0
                                or records[0].get('launches', 0) <= 0
                                or records[0].get('late_instantiations') != 0):
                            raise RuntimeError('FULL_GATE_GRAPH_NOT_USED:' + label + ':rank=' + str(rank))
                        graph_receipts.append(records[0])
                    detail['owner_graph_receipts'] = graph_receipts
                report['weighted_multi_rank_runs'].append(dict(label=label,
                    config_sha256=file_digest(path), outcome=outcome, detail=detail, passed=passed))
                save()
                if not passed:
                    raise RuntimeError('WEIGHTED_TWO_RANK_GATE_FAILED:' + label)
            transports = ['HOST_SIZED_NCCL']
            if all(row['cuda_status'] == 0 and row['allowed'] == 1 for row in p2p):
                transports.append('NCCL_LSA')
            else:
                report['weighted_lsa_status'] = 'UNSUPPORTED_P2P_HOST_NOT_RETRIED'
            if MODE == 'weighted_library_gate':
                env.pop('MGBFS_OWNER_GRAPH_REPLAY', None)
                for transport in transports:
                    cfg = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
                    cfg.update(macro_depth=2, owner_backend='CUCO_RANK', library_pool_bytes=64 << 20,
                        frontier_profile='HASH_FIRST', local_pre_dedup=True,
                        parent_batch=1, completion_epoch_window=3,
                        transport_backend=transport)
                    cfg['topology'].update(logical_owner_to_rank=[0, 1], buckets_per_shard=32)
                    cfg['capacities'].update(route_slot_records=32, route_slot_count=3,
                        state_ring_records=256, state_extent_descriptors=256,
                        layer_hash_records_per_arena=128, pinned_archive_slots=128)
                    weighted_replay(cfg, 'weighted-hash-first-red-' + transport,
                        ['--healthy-only', '--reference-size', '4', '--batch', '1',
                         '--case-timeout-seconds', '180'], timeout=600)
                report['status'] = 'WEIGHTED_LIBRARY_FULL_STATE_PASS'
                save()
                return
            if MODE == 'owner_graph_gate':
                # Exercises the real production CLI, original-depth full-state
                # oracle, multiple owner packets and modulo target-slot reuse.
                # Rejects silently ignoring the requested production graph backend.
                import re
                env['MGBFS_OWNER_GRAPH_REPLAY'] = '1'
                report['owner_graph_replay'] = []
                for transport in transports:
                    cfg = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
                    cfg.update(macro_depth=2, owner_backend='CUB_SORT_MERGE',
                        frontier_profile='DENSE', local_pre_dedup=True,
                        parent_batch=1, completion_epoch_window=3,
                        transport_backend=transport)
                    cfg['topology'].update(logical_owner_to_rank=[0, 1], buckets_per_shard=32)
                    cfg['capacities'].update(route_slot_records=32, route_slot_count=3,
                        state_ring_records=256, state_extent_descriptors=256,
                        layer_hash_records_per_arena=128, pinned_archive_slots=128)
                    label = 'owner-graph-replay-' + transport
                    weighted_replay(cfg, label, ['--healthy-only', '--reference-size', '4',
                        '--batch', '1', '--case-timeout-seconds', '180'], timeout=600)
                    evidence = []
                    for rank in (0, 1):
                        matches = []
                        for path in (logs / label).rglob('rank-' + str(rank) + '*.log'):
                            for line in path.read_text(errors='replace').splitlines():
                                if line.startswith('MGBFS_OWNER_GRAPH_SUMMARY '):
                                    matches.append(json.loads(line.split(' ', 1)[1]))
                        if len(matches) != 1:
                            raise RuntimeError('PRODUCTION_OWNER_GRAPH_NOT_USED:rank=' + str(rank))
                        row = matches[0]
                        if (row.get('rank') != rank or row.get('instances', 0) < 1
                                or row.get('launches', 0) <= row.get('instances', 0)
                                or row.get('late_instantiations') != 0):
                            raise RuntimeError('PRODUCTION_OWNER_GRAPH_NOT_REUSED:rank=' + str(rank))
                        evidence.append(row)
                    report['owner_graph_replay'].append(dict(transport=transport,
                        rank_evidence=evidence, full_state_oracle=True))
                    save()
                report['status'] = 'PRODUCTION_OWNER_GRAPH_REPLAY_PASS'
                save()
                return
            if MODE == 'weighted_timeline':
                report['paired_owner_runs'] = []
                for transport in transports:
                    for n, depth, batch in ((6, 2, 8), (6, 10, 8), (8, 2, 128)):
                        cfg = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
                        def matrix(mapping):
                            return [int(col == mapping[row]) for row in range(n) for col in range(n)]
                        count = 720 if n == 6 else 40320
                        cfg['graph'].update(rows=n, cols=n, start=matrix(list(range(n))),
                            generators=[matrix([(r + 1) % n for r in range(n)]),
                                matrix([(r - 1) % n for r in range(n)]),
                                matrix([1, 0, *range(2, n)])], inverse_map=[1, 0, 2],
                            expected_max_unique_states=count)
                        cfg.update(macro_depth=depth, owner_backend='CUB_SORT_MERGE',
                            frontier_profile='DENSE', local_pre_dedup=True,
                            parent_batch=batch, completion_epoch_window=3,
                            transport_backend=transport)
                        cfg['topology'].update(logical_owner_to_rank=[0, 1], buckets_per_shard=32)
                        cfg['capacities'].update(route_slot_records=8192, route_slot_count=3,
                            state_ring_records=524288, state_extent_descriptors=65536,
                            layer_hash_records_per_arena=count, next_bucket_capacity_records=count,
                            pinned_archive_slot_bytes=batch * (n*n + 16),
                            pinned_archive_slots=512, disk_extent_bytes_per_rank=64 << 20)
                        for repeat in range(5):
                            for owner in (('CUB_SORT_MERGE', 'CUCO_RANK') if repeat % 2 == 0 else ('CUCO_RANK', 'CUB_SORT_MERGE')):
                                env.pop('MGBFS_TRACE_RANGES', None)
                                env.pop('MGBFS_OWNER_GRAPH_REPLAY', None)
                                cfg.update(owner_backend=owner)
                                if owner == 'CUCO_RANK':
                                    cfg['library_pool_bytes'] = 256 << 20
                                else:
                                    cfg.pop('library_pool_bytes', None)
                                label = f'owner-paired-S{n}-k{depth}-{transport}-r{repeat}-{owner}'
                                weighted_replay(cfg, label, ['--healthy-only', '--reference-size', str(n),
                                    '--batch', str(batch), '--bench-warmup',
                                    '--case-timeout-seconds', '600'], timeout=1500)
                                ranks = []
                                for rank in (0, 1):
                                    case = logs / label / 'healthy-None'
                                    record = json.loads((case / 'result' / f'rank-{rank}.json').read_text())
                                    ranks.append(dict(rank=rank, result=record))
                                report['paired_owner_runs'].append(dict(label=label, repeat=repeat,
                                    owner=owner, transport=transport, n=n, macro_depth=depth,
                                    batch=batch, ranks=ranks))
                                save()
                report['status'] = 'WEIGHTED_OWNER_PAIRED_COMPLETE'
                report['performance_acceptance'] = False
                save()
                return
            if MODE in ('weighted_timeline', 'owner_graph_timeline'):
                nsys = prepare_nsys()
                env['PATH'] = str(Path(nsys).parent) + ':' + env['PATH']
                report['timeline_exports'] = []
                for transport in transports:
                    for n, depth, batch in ((6, 2, 8), (6, 10, 8), (8, 2, 128)):
                        cfg = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
                        def matrix(mapping):
                            return [int(col == mapping[row]) for row in range(n) for col in range(n)]
                        count = 720 if n == 6 else 40320
                        cfg['graph'].update(rows=n, cols=n, start=matrix(list(range(n))),
                            generators=[matrix([(r + 1) % n for r in range(n)]),
                                matrix([(r - 1) % n for r in range(n)]),
                                matrix([1, 0, *range(2, n)])], inverse_map=[1, 0, 2],
                            expected_max_unique_states=count)
                        cfg.update(macro_depth=depth, owner_backend='CUCO_RANK',
                            library_pool_bytes=256 << 20,
                            frontier_profile='DENSE', local_pre_dedup=True,
                            parent_batch=batch, completion_epoch_window=3,
                            transport_backend=transport)
                        cfg['topology'].update(logical_owner_to_rank=[0, 1], buckets_per_shard=32)
                        cfg['capacities'].update(route_slot_records=8192, route_slot_count=3,
                            state_ring_records=524288, state_extent_descriptors=65536,
                            layer_hash_records_per_arena=count, next_bucket_capacity_records=count,
                            pinned_archive_slot_bytes=batch * (n*n + 16),
                            pinned_archive_slots=512, disk_extent_bytes_per_rank=64 << 20)
                        label = f'weighted-scoped-S{n}-k{depth}-{transport}'
                        weighted_replay(cfg, label, ['--healthy-only', '--reference-size', str(n),
                            '--batch', str(batch), '--bench-warmup', '--instrument-processes',
                            'nsys-search', '--case-timeout-seconds', '600'], timeout=1500)
                        detail = report['weighted_multi_rank_runs'][-1]['detail']
                        if detail.get('profile_capture_scope') != 'search_after_full_warmup':
                            raise RuntimeError('WEIGHTED_TIMELINE_CAPTURE_SCOPE')
                        traces = list((logs / label).rglob('rank-*.nsys-rep'))
                        if len(traces) != 2:
                            raise RuntimeError('WEIGHTED_TIMELINE_RANK_COUNT')
                        for trace in traces:
                            db = trace.with_suffix('.sqlite')
                            run([nsys, 'export', '--type=sqlite', '--output=' + str(db), str(trace)],
                                label + '-' + trace.stem + '-export', timeout=600)
                            report['timeline_exports'].append(dict(label=label,
                                nsys_rep=str(trace.relative_to(logs)), sqlite=str(db.relative_to(logs)),
                                nsys_sha256=file_digest(trace), sqlite_sha256=file_digest(db),
                                source=SOURCE, capture_scope='search_after_full_warmup'))
                            save()
                report['status'] = 'WEIGHTED_SCOPED_TIMELINE_COLLECTED'
                report['performance_acceptance'] = False
                save()
                return
            for transport in transports:
                for owner in WEIGHTED_FULL_OWNERS:
                    for depth in (2, 10):
                        for pre in (False, True):
                            for rank_map in ([0, 1], [1, 0]):
                                cfg = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
                                cfg.update(macro_depth=depth, owner_backend=owner,
                                    frontier_profile='DENSE', local_pre_dedup=pre,
                                    completion_epoch_window=3, transport_backend=transport)
                                if owner == 'CUCO_RANK':
                                    cfg['library_pool_bytes'] = 64 << 20
                                cfg['topology'].update(logical_owner_to_rank=rank_map, buckets_per_shard=32)
                                cfg['capacities'].update(route_slot_records=32, route_slot_count=3,
                                    state_ring_records=256, state_extent_descriptors=256,
                                    layer_hash_records_per_arena=128, pinned_archive_slots=128)
                                label = f'weighted-2rank-{transport}-{owner}-k{depth}-pre{int(pre)}-map{rank_map[0]}{rank_map[1]}'
                                weighted_replay(cfg, label, ['--healthy-only',
                                    '--case-timeout-seconds', '120'])
                # S4 can split its peak 3/3, so it cannot guarantee reuse of three
                # source banks. S5 has larger exact layers; test reuse there instead
                # of claiming an uneven S4 hash split is an architectural invariant.
                reuse = copy.deepcopy(cfg)
                n = 5
                def permutation_matrix(mapping):
                    return [int(column == mapping[row]) for row in range(n) for column in range(n)]
                reuse['graph'].update(rows=n, cols=n,
                    start=permutation_matrix(list(range(n))),
                    generators=[permutation_matrix([(row + 1) % n for row in range(n)]),
                        permutation_matrix([(row - 1) % n for row in range(n)]),
                        permutation_matrix([1, 0, 2, 3, 4])],
                    inverse_map=[1, 0, 2], expected_max_unique_states=120)
                reuse.update(owner_backend=WEIGHTED_FULL_OWNERS[0], macro_depth=2)
                reuse['capacities'].update(route_slot_records=128,
                    state_ring_records=2048, state_extent_descriptors=512,
                    layer_hash_records_per_arena=256, next_bucket_capacity_records=128,
                    pinned_archive_slot_bytes=82, pinned_archive_slots=256)
                weighted_replay(reuse, 'weighted-bank-reuse-' + transport,
                    ['--reference-size', '5', '--healthy-only', '--require-bank-reuse',
                        '--case-timeout-seconds', '180'])
                # Same runtime, actual asymmetric startup/owner/capacity and archive
                # finalization failures; preserve both process exits and no-COMPLETE checks.
                fault = copy.deepcopy(cfg)
                fault.update(owner_backend=WEIGHTED_FULL_OWNERS[0], macro_depth=2)
                weighted_replay(fault, 'weighted-faults-' + transport,
                    ['--capacity-faults', '--case-timeout-seconds', '120'], timeout=3600)
                for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    weighted_replay(fault, 'weighted-sanitizer-' + transport + '-' + tool,
                        ['--healthy-only', '--instrument-processes', tool,
                            '--case-timeout-seconds', '600'], timeout=1500)
            report['status'] = 'WEIGHTED_CUCO_TWO_RANK_CORRECTNESS_PASS'
            report['performance_acceptance'] = False
            save()
            return
        if MODE == 'weighted_driver_gate':
            env.update(CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER')
            report['scope'] = ('Integrated world1 DENSE native owner original-depth weighted BFS '
                'on each physical T4; full50 and all4san including public advance; '
                'NOT two-rank weighted transport or performance acceptance')
            command = ['cargo', 'test', '--locked', '-p', 'mgbfs-runtime',
                '--features', 'cuda,library-owner', '--lib',
                '--', '--nocapture', '--test-threads=1']
            report['weighted_driver_devices'] = []
            for device in (0, 1):
                output = gate.run(command, cwd=source,
                    env=dict(env, CUDA_VISIBLE_DEVICES=str(device)), logs=logs,
                    name='weighted-full-driver-device-' + str(device), timeout=900)
                passed = ('test result: ok. 50 passed; 0 failed; 0 ignored' in output
                    and 'MGBFS_WEIGHTED_RING_READER_DAG_CAPTURE launched' in output
                    and 'test distributed_native::weighted_producer_tests::weighted_advance_preserves_original_depth_full_state_layers ...' in output)
                report['weighted_driver_devices'].append(dict(device=device,
                    tests=50, source=SOURCE, status='PASS' if passed else 'FAIL'))
                save()
                if not passed:
                    raise RuntimeError('WEIGHTED_FULL_DRIVER_NOT_GREEN')
            check_weighted_sanitizers(7, None)
            run(['cargo', 'build', '--locked', '--release', '-p', 'mgbfs-cli',
                '--features', 'cuda,library-owner'], 'weighted-production-cli-build', timeout=1800)
            sys.path.insert(0, str(source / 'scripts'))
            from replay_lsa_cancel_candidate import verify_process_archives
            cli = str(source / 'target/release/mgbfs')
            report['weighted_production_runs'] = []
            for device in (0, 1):
                for owner in ('CUB_SORT_MERGE', 'BMMA_BUCKET'):
                    for depth in (2, 10):
                        for pre in (False, True):
                            for seed in (0, 1, 20260828):
                                label = f'weighted-cli-d{device}-{owner}-k{depth}-pre{int(pre)}-s{seed}'
                                case = logs / label
                                case.mkdir()
                                cfg = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
                                cfg.update(macro_depth=depth, owner_backend=owner,
                                    frontier_profile='DENSE', local_pre_dedup=pre,
                                    seed=list(seed.to_bytes(16, 'little')),
                                    completion_epoch_window=3, transport_backend='HOST_SIZED_NCCL')
                                cfg['topology'].update(world_size=1,
                                    logical_owner_to_rank=[0], buckets_per_shard=32)
                                cfg['capacities'].update(route_slot_records=32,
                                    route_slot_count=3, state_ring_records=256,
                                    state_extent_descriptors=256, layer_hash_records_per_arena=128,
                                    pinned_archive_slots=128)
                                path = case / 'run-config.json'
                                path.write_text(json.dumps(cfg, indent=2))
                                local_env = dict(env, CUDA_VISIBLE_DEVICES=str(device),
                                    RANK='0', LOCAL_RANK='0', WORLD_SIZE='1',
                                    MGBFS_BENCH_WORLD_SIZE='1', MGBFS_BENCH_WARMUP='0',
                                    TORCHELASTIC_RUN_ID=label)
                                preflight = gate.run([cli, 'preflight', '--offline', str(path)],
                                    cwd=source, env=local_env, logs=case, name='preflight', timeout=60)
                                digest = json.loads(preflight)['config_digest']
                                gate.run([cli, 'run', str(path), str(case / 'bootstrap'),
                                    str(case / 'archive'), str(case / 'result')],
                                    cwd=source, env=local_env, logs=case, name='rank-0', timeout=180)
                                oracle = verify_process_archives(case, n=4, world=1,
                                    expected_run_contract='RunConfigV1',
                                    expected_config_digest=digest, expected_owner_backend=owner)
                                report['weighted_production_runs'].append(dict(device=device,
                                    owner=owner, macro_depth=depth, pre_dedup=pre, seed=seed,
                                    oracle=oracle, status='PASS'))
                                save()
            if len(report['weighted_production_runs']) != 48:
                raise RuntimeError('WEIGHTED_PRODUCTION_INVENTORY')
            report['status'] = 'WEIGHTED_FULL_DRIVER_PASS'
            save()
            return
        if MODE == 'macro_capture_gate':
            env.update(CUDA_MODULE_LOADING='EAGER', CUDA_MODULE_DATA_LOADING='EAGER')
            report['scope'] = ('full48 runtime suite and five weighted producer/ring oracles plus '
                'all four unfiltered Compute Sanitizer tools on each physical T4; '
                'independent world1 processes, NOT weighted distributed BFS admission')
            command = ['cargo', 'test', '--locked', '-p', 'mgbfs-runtime',
                '--features', 'cuda,library-owner', '--lib',
                '--', '--nocapture', '--test-threads=1']
            report['weighted_producer_devices'] = []
            for device in (0, 1):
                test_env = dict(env, CUDA_VISIBLE_DEVICES=str(device))
                output = gate.run(command, cwd=source, env=test_env, logs=logs,
                    name='weighted-production-producer-device-' + str(device), timeout=900)
                if 'test result: ok. 48 passed; 0 failed' not in output:
                    raise RuntimeError('WEIGHTED_FULL_SUITE_NOT_EXECUTED')
                if 'MGBFS_WEIGHTED_RING_READER_DAG_CAPTURE launched' not in output:
                    raise RuntimeError('WEIGHTED_RING_READER_CAPTURE_NOT_EXECUTED')
                report['weighted_producer_devices'].append(dict(device=device, tests=48,
                    weighted_oracles=5, capture=True, status='PASS'))
                save()
            check_weighted_sanitizers(5, None)
            report['status'] = 'COMPLETE'
            report['weighted_producer'] = 'PASS'
            save()
            return
            report['scope'] = 'single-rank existing macro producer CUDA Graph capture and exact layer oracle; not multi-GPU macro admission'
            report['macro_reference_devices'] = []
            for device in (0, 1):
                macro_env = dict(env, CUDA_VISIBLE_DEVICES=str(device))
                output = gate.run(macro_capture_command(), cwd=source, env=macro_env, logs=logs, name='macro-reference-device-' + str(device), timeout=900)
                if 'test result: ok. 3 passed; 0 failed' not in output:
                    raise RuntimeError('MACRO_REFERENCE_TESTS_NOT_EXECUTED')
                report['macro_reference_devices'].append(dict(device=device, tests=3, full_bfs_cases=112, status='PASS'))
                save()
            build_command = macro_capture_command()[:8] + ['--no-run', '--message-format=json']
            artifacts = run(build_command, 'macro-reference-test-artifact', timeout=900)
            executables = []
            for line in artifacts.splitlines():
                if not line.startswith('{'):
                    continue
                artifact = json.loads(line)
                if artifact.get('reason') == 'compiler-artifact' and artifact.get('target', {}).get('name') == 'mgbfs-runtime' and artifact.get('target', {}).get('kind') == ['lib'] and artifact.get('profile', {}).get('test') and artifact.get('executable'):
                    executables.append(artifact['executable'])
            if len(executables) != 1:
                raise RuntimeError('MACRO_REFERENCE_TEST_ARTIFACT_IDENTITY')
            test_binary = Path(executables[0])
            report['macro_test_binary_sha256'] = hashlib.sha256(test_binary.read_bytes()).hexdigest()
            report['macro_sanitizers'] = []
            sanitizer_spec = importlib.util.spec_from_file_location('macro_sanitizer_contract', source / 'scripts/replay_lsa_cancel_candidate.py')
            sanitizer_contract = importlib.util.module_from_spec(sanitizer_spec)
            sanitizer_spec.loader.exec_module(sanitizer_contract)
            for device in (0, 1):
                macro_env = dict(env, CUDA_VISIBLE_DEVICES=str(device))
                for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    command = [host_sanitizer or 'compute-sanitizer', '--tool', tool,
                               '--error-exitcode', '97', str(test_binary),
                               'macro_native::producer_capture_tests::', '--skip', 'macro_exhaustive_layers_match_cpu_oracle_across_depth_seed_and_prededup', '--nocapture', '--test-threads=1']
                    checked = gate.run(command, cwd=source, env=macro_env, logs=logs,
                                       name='macro-reference-' + tool + '-device-' + str(device), timeout=300)
                    if 'test result: ok. 2 passed; 0 failed' not in checked or not sanitizer_contract.instrumentation_clean(checked, tool):
                        raise RuntimeError('MACRO_REFERENCE_SANITIZER_NOT_GREEN_' + tool)
                    report['macro_sanitizers'].append(dict(device=device, tool=tool, full_bfs_cases=4, producer_capture=True, status='PASS'))
                    save()
            # Exercise the existing real pinned archive, not GPU snapshots.
            archive_build = ['cargo', 'test', '--locked', '-p', 'mgbfs-runtime',
                '--features', 'cuda,library-owner', '--test', 'macro_native',
                '--no-run', '--message-format=json']
            artifacts = run(archive_build, 'macro-archive-test-artifact', timeout=900)
            archive_bins = []
            for line in artifacts.splitlines():
                if line.startswith('{'):
                    artifact = json.loads(line)
                    if (artifact.get('reason') == 'compiler-artifact'
                            and artifact.get('target', {}).get('name') == 'macro_native'
                            and artifact.get('executable')):
                        archive_bins.append(artifact['executable'])
            if len(archive_bins) != 1:
                raise RuntimeError('MACRO_ARCHIVE_TEST_ARTIFACT_IDENTITY')
            archive_binary = archive_bins[0]
            report['macro_archive_test_sha256'] = file_digest(archive_binary)
            report['macro_archive_checks'] = []
            for device in (0, 1):
                for tool in ('plain', 'memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    command = [archive_binary,
                        'native_macro_archive_is_complete_and_verifiable',
                        '--exact', '--nocapture', '--test-threads=1']
                    if tool != 'plain':
                        command = [host_sanitizer or 'compute-sanitizer', '--tool', tool,
                            '--error-exitcode', '97'] + command
                    checked = gate.run(command, cwd=source,
                        env=dict(env, CUDA_VISIBLE_DEVICES=str(device)), logs=logs,
                        name='macro-archive-' + tool + '-device-' + str(device), timeout=300)
                    if ('test result: ok. 46 passed; 0 failed' not in checked
                            or (tool != 'plain' and not sanitizer_contract.instrumentation_clean(checked, tool))):
                        raise RuntimeError('MACRO_ARCHIVE_ROWS_NOT_GREEN_' + tool)
                    report['macro_archive_checks'].append(dict(device=device, tool=tool,
                        matrix_and_compact=True, actual_archived_states_and_hashes=True,
                        snapshot_drain_before_d2h=False, status='PASS'))
                    save()
            report['macro_producer_capture'] = 'PASS'
            report['status'] = 'COMPLETE'
            save()
            return
        if MODE == 'focused_fault_replay':
            report['scope'] = 'Physical two-T4 ' + FOCUSED_TRANSPORT + ' full-state/fault/unfiltered sanitizer validation; not full pipeline/timeline/performance acceptance'
            if FOCUSED_TRANSPORT not in ('HOST_SIZED_NCCL', 'NCCL_LSA'):
                raise RuntimeError('FOCUSED_TRANSPORT_UNKNOWN')
            if FOCUSED_TRANSPORT == 'NCCL_LSA' and any(row['cuda_status'] != 0 or row['allowed'] != 1 for row in p2p):
                report['status'] = 'LSA_UNSUPPORTED_HOST'
                save()
                return
            report['validation_runs'] = []
            validation_plan = focused_validation_plan(FOCUSED_GATE_SUITE)
            report['validation_plan'] = validation_plan
            selection_test = source / 'scripts/validation/test_focused_validation_selection.py'
            run(['git', 'fetch', '--depth=1', 'origin', '1ba7b9bdd08470155dd928cc6b8c63740324b639'], 'selection-test-fetch')
            selection_test.parent.mkdir(parents=True, exist_ok=True)
            selection_test.write_bytes(subprocess.check_output(['git', 'show', '1ba7b9bdd08470155dd928cc6b8c63740324b639:scripts/validation/test_focused_validation_selection.py'], cwd=source))
            run([python, str(selection_test), str(Path(__file__).resolve())], 'focused-selection-contract-green', cwd=source, timeout=60)
            config = json.loads('{"schema":1,"graph":{"schema":1,"rows":4,"cols":4,"modulus":2,"start":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"generators":[[0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0],[0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1]],"inverse_map":[1,0,2],"expected_max_unique_states":24},"seed":[220,39,53,1,0,0,0,0,0,0,0,0,0,0,0,0],"topology":{"world_size":2,"shards_per_rank":4,"buckets_per_shard":8,"logical_owner_to_rank":[0,1]},"frontier_profile":"HASH_FIRST","local_pre_dedup":true,"owner_backend":"BMMA_BUCKET","generation_backend":"CUTLASS_U8_SM75_V1","hash_backend":"GEMM_U8_P32X4_V1","macro_depth":1,"parent_batch":1,"capacities":{"state_ring_records":128,"state_extent_descriptors":128,"layer_hash_records_per_arena":64,"next_bucket_capacity_records":32,"route_slot_records":3,"route_slot_count":3,"pinned_archive_slots":128,"pinned_archive_slot_bytes":96,"disk_extent_bytes_per_rank":8388608,"untouched_vram_reserve_bytes":1073741824},"transport_backend":"HOST_SIZED_NCCL","completion_epoch_window":3,"library_pool_bytes":null}')
            config['transport_backend'] = FOCUSED_TRANSPORT
            run(['git', 'fetch', '--depth=1', 'origin', '820423350fef37ad77b3812a9d3f4baebf473231'], 'replay-helper-fetch')
            for helper_path in ('scripts/replay_lsa_cancel_candidate.py', 'scripts/test_replay_case_selection.py', 'scripts/test_replay_stalled_stacks.py', 'scripts/validation/test_replay_scoped_profile.py'):
                (source / helper_path).write_bytes(subprocess.check_output(['git', 'show', '820423350fef37ad77b3812a9d3f4baebf473231:' + helper_path], cwd=source))
            run([python, '-m', 'unittest', 'test_replay_case_selection', 'test_replay_failure_completion', 'test_replay_deadline', 'test_replay_cleanup', 'test_replay_stalled_stacks'], 'replay-helper-tests', cwd=source / 'scripts', timeout=60)
            run([python, str(source / 'scripts/validation/test_replay_scoped_profile.py'), str(source / 'scripts/replay_lsa_cancel_candidate.py')], 'scoped-profile-command-green', cwd=source, timeout=60)
            # Direct library tests must inherit the loading contract before process constructors.
            env['CUDA_MODULE_LOADING'] = 'EAGER'
            env['CUDA_MODULE_DATA_LOADING'] = 'EAGER'
            run(['cargo', 'test', '--locked', '-p', 'mgbfs-runtime', '--features', 'cuda,library-owner', '--lib', '--', '--test-threads=1'], 'cuda-runtime-unit-tests', cwd=source, timeout=600)
            sys.path.insert(0, str(source / 'scripts'))
            from replay_lsa_cancel_candidate import configure_owner_environment
            def admitted_fixture(cfg):
                cfg = copy.deepcopy(cfg)
                cfg['library_pool_bytes'] = (64 << 20) if cfg['owner_backend'] == 'CUCO_RANK' else None
                configure_owner_environment({}, cfg['owner_backend'], ','.join(map(str, cfg['topology']['logical_owner_to_rank'])), cfg)
                return cfg
            for profile, owner in validation_plan['capture_pairs'] + validation_plan['oracle_pairs']:
                cfg = copy.deepcopy(config)
                cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                admitted_fixture(cfg)
            report['fixture_owner_admission'] = 'PASS'
            save()
            def replay_case(cfg, label, extra, timeout=1800, owner_capture=False, batch_capture=False):
                cfg = admitted_fixture(cfg)
                snapshot = logs / (label + '-config.json')
                snapshot.write_text(json.dumps(cfg, separators=(',', ':')))
                replay_env = dict(env)
                replay_env.pop('MGBFS_TEST_OWNER_DAG_CAPTURE', None)
                replay_env.pop('MGBFS_TEST_BATCH_DAG_CAPTURE', None)
                if batch_capture:
                    replay_env['MGBFS_TEST_BATCH_DAG_CAPTURE'] = '1'
                if owner_capture:
                    replay_env['MGBFS_TEST_OWNER_DAG_CAPTURE'] = '1'
                for key in ('CUDA_MODULE_LOADING', 'CUDA_MODULE_DATA_LOADING', 'MGBFS_FAILURE_STAGE_QUIET', 'LD_PRELOAD', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS'):
                    replay_env.pop(key, None)
                command = [python, str(source / 'scripts/replay_lsa_cancel_candidate.py'), str(work), str(logs / label), '--run-config', str(snapshot), *extra]
                with (logs / (label + '.log')).open('w') as output:
                    outcome = run_protocol_replay(command, cwd=source, env=replay_env, log=output, timeout=timeout)
                detail_path = logs / label / 'summary.json'
                detail = json.loads(detail_path.read_text()) if detail_path.exists() else {}
                cases = detail.get('cases', [])
                if detail.get('owner_dag_capture_requested') is not owner_capture:
                    raise RuntimeError('CAPTURE_REQUEST_PROVENANCE_MISMATCH:' + label)
                if owner_capture:
                    for case in cases:
                        launches = case.get('owner_dag_capture_launches', [])
                        if len(launches) != 2 or not all(count > 0 for count in launches):
                            case['pass'] = False
                row = dict(label=label, config_sha256=file_digest(snapshot), outcome=outcome, detail=detail,
                           passed=outcome['returncode'] == 0 and bool(cases) and all(c.get('pass') and not c.get('forced_cleanup') for c in cases))
                if batch_capture:
                    observed = []
                    for rank in range(2):
                        rank_logs = list((logs / label).rglob('rank-' + str(rank) + '.log'))
                        markers = [line for path in rank_logs for line in path.read_text(errors='replace').splitlines()
                            if line.startswith('MGBFS_BATCH_DAG_CAPTURE scope=')]
                        expected = {'MGBFS_BATCH_DAG_CAPTURE scope=prepared_packet_transport_owner_retirement rank=' + str(rank)
                            + ' depth=3 batch=' + str(batch) + ' archive_submitted_before_capture=true' for batch in range(2)}
                        observed.append(dict(rank=rank, markers=markers, passed=len(markers) == 2 and set(markers) == expected))
                    row['batch_capture_evidence'] = observed
                    row['passed'] = row['passed'] and all(item['passed'] for item in observed)
                report['validation_runs'].append(row)
                save()
                if detail.get('base_commit') != SOURCE:
                    row['provenance_error'] = 'MISSING_REPLAY_SUMMARY' if not detail else 'SOURCE_MISMATCH'
                    save()
                    raise RuntimeError('VALIDATION_REPLAY_FAILED_BEFORE_PROVENANCE:' + label if not detail else 'VALIDATION_SOURCE_MISMATCH')
                print('VALIDATION_PANEL', label, 'PASS' if row['passed'] else 'FAIL',
                    'cases=' + str(len(cases)), 'source=' + SOURCE, flush=True)
                return row
            for profile, owner in validation_plan['oracle_pairs']:
                for prededup in (False, True):
                    for rank_map in ([0, 1], [1, 0]):
                        cfg = copy.deepcopy(config)
                        cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                        cfg['local_pre_dedup'] = prededup
                        cfg['topology']['logical_owner_to_rank'] = rank_map
                        label = 'oracle-' + profile + '-' + owner + '-' + str(int(prededup)) + '-' + ''.join(map(str, rank_map))
                        row = replay_case(cfg, label, ['--healthy-only', '--case-timeout-seconds', '90'])
                        if not row['passed']:
                            raise RuntimeError('PROFILE_ORACLE_GATE_FAILED:' + label)
            for profile, owner in validation_plan['capture_pairs']:
                for prededup in (False, True):
                    for rank_map in ([0, 1], [1, 0]):
                        cfg = copy.deepcopy(config)
                        cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                        cfg['local_pre_dedup'] = prededup
                        cfg['topology']['logical_owner_to_rank'] = rank_map
                        label = 'capture-' + profile + '-' + owner + '-' + str(int(prededup)) + '-' + ''.join(map(str, rank_map))
                        row = replay_case(cfg, label, ['--healthy-only', '--case-timeout-seconds', '90'], owner_capture=True)
                        if not row['passed']:
                            raise RuntimeError('OWNER_CAPTURE_GATE_FAILED:' + label)
            for profile, owner in validation_plan.get('batch_capture_pairs', []):
                for prededup in (False, True):
                    for rank_map in ([0, 1], [1, 0]):
                        cfg = copy.deepcopy(config)
                        cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                        cfg['local_pre_dedup'] = prededup
                        cfg['topology']['logical_owner_to_rank'] = rank_map
                        label = 'batch-capture-' + profile + '-' + owner + '-' + str(int(prededup)) + '-' + ''.join(map(str, rank_map))
                        row = replay_case(cfg, label, ['--healthy-only', '--case-timeout-seconds', '90'], batch_capture=True)
                        if not row['passed']:
                            raise RuntimeError('BATCH_CAPTURE_ORACLE_GATE_FAILED:' + label)
            for profile, owner in validation_plan['fault_pairs']:
                cfg = copy.deepcopy(config)
                cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                row = replay_case(cfg, 'full-fault-' + profile + '-' + owner, ['--capacity-faults', '--case-timeout-seconds', '90'], timeout=2400)
                if not row['passed']:
                    raise RuntimeError('FULL_FAULT_GATE_FAILED:' + profile)
            for profile, owner in validation_plan['sanitizer_pairs']:
                cfg = copy.deepcopy(config)
                cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    replay_case(cfg, 'sanitizer-' + profile + '-' + owner + '-' + tool, ['--healthy-only', '--instrument-processes', tool, '--case-timeout-seconds', '300'], timeout=900, batch_capture=FOCUSED_GATE_SUITE in ('lsa_batch_pipeline', 'lsa_batch_profiles'))
            if validation_plan.get('timeline_pairs'):
                nsys = prepare_nsys()
                env['PATH'] = str(Path(nsys).parent) + ':' + env['PATH']
                n = 8
                identity = [int(row == col) for row in range(n) for col in range(n)]
                cycle = [int(col == (row + 1) % n) for row in range(n) for col in range(n)]
                inverse = [int(col == (row - 1) % n) for row in range(n) for col in range(n)]
                swap = [int(col == (1 if row == 0 else 0 if row == 1 else row)) for row in range(n) for col in range(n)]
                config['graph'].update(rows=n, cols=n, start=identity, generators=[cycle, inverse, swap], expected_max_unique_states=40320)
                config['parent_batch'] = 128
                config['capacities'].update(state_ring_records=80640, state_extent_descriptors=1024,
                    layer_hash_records_per_arena=40320, next_bucket_capacity_records=40320,
                    route_slot_records=384, pinned_archive_slot_bytes=128 * (n*n + 16),
                    disk_extent_bytes_per_rank=64 << 20)
                report['timeline_exports'] = []
                for profile, owner in validation_plan['timeline_pairs']:
                    cfg = copy.deepcopy(config)
                    cfg['frontier_profile'], cfg['owner_backend'] = profile, owner
                    label = 'scoped-S8-' + profile + '-' + owner
                    row = replay_case(cfg, label, ['--healthy-only', '--reference-size', '8', '--batch', '128',
                        '--bench-warmup', '--instrument-processes', 'nsys-search', '--case-timeout-seconds', '300'], timeout=900)
                    if not row['passed'] or row['detail'].get('profile_capture_scope') != 'search_after_full_warmup':
                        raise RuntimeError('SCOPED_LSA_TIMELINE_ORACLE_FAILED:' + label)
                    traces = list((logs / label).rglob('rank-*.nsys-rep'))
                    if len(traces) != 2:
                        raise RuntimeError('SCOPED_LSA_RANK_TRACE_COUNT:' + label)
                    for trace in traces:
                        sqlite = trace.with_suffix('.sqlite')
                        run([nsys, 'export', '--type=sqlite', '--output=' + str(sqlite), str(trace)],
                            label + '-' + trace.stem + '-export', timeout=600)
                        report['timeline_exports'].append(dict(label=label, rank_trace=trace.name,
                            nsys_rep=str(trace.relative_to(logs)), sqlite=str(sqlite.relative_to(logs)),
                            nsys_sha256=file_digest(trace), sqlite_sha256=file_digest(sqlite),
                            source=SOURCE, config_sha256=row['config_sha256'], capture_scope='search_after_full_warmup'))
                        save()
                if len(report['timeline_exports']) != 2 * len(validation_plan['timeline_pairs']):
                    raise RuntimeError('SCOPED_LSA_TOTAL_TRACE_COUNT')
            if len(report['validation_runs']) != validation_plan['expected_panels']:
                raise RuntimeError('VALIDATION_PANEL_COUNT_MISMATCH')
            prefix = 'LSA' if FOCUSED_TRANSPORT == 'NCCL_LSA' else 'HOST'
            report['status'] = ('LSA_SCOPED_TIMELINE_PASS' if validation_plan.get('timeline_pairs') else prefix + '_VALIDATION_PASS') if all(row['passed'] for row in report['validation_runs']) else prefix + '_VALIDATION_INCOMPLETE'
            save()
            return
        if MODE in ('typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate', 'typed_warmup_gate', 'typed_sanitizer_version_gate', 'typed_matrix_gate'):
            report['scope'] = 'typed RunConfigV1; independent two-T4 full-state S4 archives, faults and unfiltered sanitizers'
            report['typed_runs'] = []
            run(['cargo', 'test', '--locked'], 'cpu-workspace-tests', timeout=900)
            run([sys.executable, '-m', 'unittest', 'test_replay_failure_completion'],
                'fault-evidence-fixtures', cwd=source / 'scripts', timeout=60)
            run(['cargo', 'test', '--locked', '-p', 'mgbfs-runtime', '--features',
                'cuda,library-owner', '--lib', 'event_generation::', '--', '--test-threads=1'],
                'route-packet-event-lifetime-tests', timeout=900)
            base = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
            base['transport_backend'] = 'HOST_SIZED_NCCL'
            run(['cargo', 'test', '--locked', '-p', 'mgbfs-core', '--test',
                'production_transport', '--test', 'reference_selection', '--test', 'config'],
                'cuco-hash-first-admission-unit-tests', timeout=900)
            configs = typed_rank_configs(base)
            report['candidate_scope'] = 'Independent HOST size handshake word before receive-slot last-reader wait; DENSE/HASH_FIRST all owners/maps/pre-dedup/two-three banks; exact payloads, single receive slot, bounded faults and all four sanitizers; not CPU-free transport'
            report['typed_unavailable_admission'] = []
            report['typed_positive_admission'] = []
            preflight_cli_ready = False
            def unavailable_admission(config, label):
                nonlocal preflight_cli_ready
                if not (config['transport_backend'] == 'HOST_SIZED_NCCL'
                        and config['frontier_profile'] == 'HASH_FIRST'
                        and config['owner_backend'] == 'CUCO_RANK'):
                    return None
                if not preflight_cli_ready:
                    run(['cargo', 'build', '--locked', '-p', 'mgbfs-cli', '--features', 'cuda,library-owner'],
                        'typed-preflight-cli-build', timeout=1800)
                    preflight_cli_ready = True
                snapshot = logs / (label + '-admission-config.json')
                snapshot.write_text(json.dumps(config, indent=2) + '\n')
                result = subprocess.run([str(source / 'target/debug/mgbfs'), 'preflight',
                    '--offline', str(snapshot)], cwd=source, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
                row = dict(label=label, returncode=result.returncode,
                    run_config_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    pass_positive_admission=result.returncode == 0,
                    bfs_executed=False, scope='offline admission only, not GPU acceptance')
                (logs / (label + '-admission.log')).write_text(result.stdout)
                report['typed_positive_admission'].append(row)
                save()
                if not row['pass_positive_admission']:
                    raise RuntimeError('HASH_FIRST_CUCO_HOST_ADMISSION')
                return None
            def replay_typed(config, label, extra, replay_env=None):
                snapshot = logs / (label + '-config.json')
                snapshot.write_text(json.dumps(config, separators=(',', ':')))
                print('START ' + label, flush=True)
                with (logs / (label + '.log')).open('w') as output:
                    row = run_protocol_replay([python, str(source / 'scripts/replay_lsa_cancel_candidate.py'),
                        str(work), str(logs / label), '--run-config', str(snapshot), *extra],
                        cwd=source, env=env if replay_env is None else replay_env, log=output)
                detail_path = logs / label / 'summary.json'
                detail = json.loads(detail_path.read_text()) if detail_path.exists() else {}
                row.update(label=label, run_config_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    run_contract=detail.get('run_contract'), epoch_window=detail.get('epoch_window'),
                    route_banks=detail.get('route_banks'),
                    replay_status=detail.get('status'))
                row['process_sanitizer'] = detail.get('process_sanitizer')
                row['failed_cases'] = [case for case in detail.get('cases', []) if not case.get('pass', False)]
                row['replay_error'] = detail.get('error')
                row['pass'] = row['returncode'] == 0 and not row['timed_out'] and (
                    detail.get('status') == 'DIAGNOSTIC_CASES_PASS' and detail.get('run_contract') == 'RunConfigV1'
                    and detail.get('epoch_window') == config['completion_epoch_window']
                    and detail.get('route_banks') == config['capacities']['route_slot_count'])
                report['typed_runs'].append(row)
                save()
                print('RESULT ' + json.dumps(row, separators=(',', ':')), flush=True)
                return row
            if MODE == 'typed_matrix_gate':
                report['scope'] = 'production two-rank full-state U4 moduli2..6, profiles/owners/pre-dedup/maps/seeds; not performance or sanitizer acceptance'
                matrix_cases = typed_matrix_cases(base)
                labels = [case['label'] for case in matrix_cases]
                if len(matrix_cases) != 360 or len(set(labels)) != 360:
                    raise RuntimeError('MATRIX_CASE_INVENTORY')
                expected_positive_admissions = sum(
                    case['config']['frontier_profile'] == 'HASH_FIRST'
                    and case['config']['owner_backend'] == 'CUCO_RANK'
                    for case in matrix_cases)
                if expected_positive_admissions != 60:
                    raise RuntimeError('MATRIX_CUCO_HASH_FIRST_INVENTORY')
                for case in matrix_cases:
                    unavailable_admission(case['config'], case['label'])
                    replay_typed(case['config'], case['label'], case['extra'])
                report['matrix_coverage'] = dict(expected_gpu_searches=360,
                    executed_gpu_searches=len(report['typed_runs']),
                    expected_positive_admissions=expected_positive_admissions,
                    checked_positive_admissions=len(report['typed_positive_admission']),
                    requested_modes=['DENSE/CUB', 'DENSE/CUCO', 'DENSE/BMMA',
                        'HASH_FIRST/CUB', 'HASH_FIRST/CUCO', 'HASH_FIRST/BMMA'],
                    all_requested_modes_executed=len(report['typed_runs']) == 360)
                covered = (len(report['typed_runs']) == 360
                    and [row['label'] for row in report['typed_runs']] == labels
                    and len(report['typed_positive_admission']) == expected_positive_admissions
                    and all(row['pass'] for row in report['typed_runs'])
                    and all(row['pass_positive_admission'] for row in report['typed_positive_admission']))
                report['status'] = 'FULL_MATRIX_PASS' if covered else 'INCOMPLETE'
                return
            if MODE == 'typed_sanitizer_version_gate':
                pinned = sdk / 'compute-sanitizer/compute-sanitizer'
                if not host_sanitizer:
                    raise RuntimeError('HOST_SANITIZER_NOT_FOUND')
                report['scope'] = 'full typed BFS, unchanged runtime: host versus CUDA12.9 instrumenter, no suppression'
                report['pinned_sanitizer_version'] = run([str(pinned), '--version'], 'pinned-sanitizer-version').strip()
                report['pinned_sanitizer_executable_sha256'] = hashlib.sha256(pinned.read_bytes()).hexdigest()
                for case in typed_sanitizer_version_cases(base):
                    replay_env = sanitizer_version_environment(env, host_sanitizer, str(pinned), case['version'])
                    row = replay_typed(case['config'], case['label'], case['extra'], replay_env)
                    selected = replay_env['MGBFS_COMPUTE_SANITIZER']
                    row['instrumenter_selection_verified'] = sanitizer_selection_matches(
                        row.get('process_sanitizer'), selected,
                        hashlib.sha256(Path(selected).read_bytes()).hexdigest())
                    row['pass'] = row['pass'] and row['instrumenter_selection_verified']
                    row.update(sanitizer_version=case['version'], tool=case['tool'])
                    save()
                report['status'] = 'TYPED_SANITIZER_VERSION_PASS' if all(row['pass'] for row in report['typed_runs']) else 'INCOMPLETE'
                return
            if MODE == 'typed_warmup_gate':
                report['scope'] = 'typed two-rank production warmup and measured archive oracle; asymmetric failures'
                for case in typed_warmup_cases(base):
                    replay_typed(case['config'], case['label'], case['extra'])
                report['status'] = 'TYPED_WARMUP_PASS' if all(row['pass'] for row in report['typed_runs']) else 'INCOMPLETE'
                return
            if MODE == 'typed_stress_gate':
                report['scope'] = 'full typed U4/F3 state sets; all profile/pre-dedup/rank-map/source-bank combinations'
                for config in typed_stress_configs(base, 3):
                    label = (f"stress-{config['owner_backend']}-{config['frontier_profile']}-pre-{int(config['local_pre_dedup'])}"
                        f"-map-{''.join(map(str,config['topology']['logical_owner_to_rank']))}"
                        f"-banks-{config['capacities']['route_slot_count']}")
                    replay_typed(config, label, ['--healthy-only', '--unitriangular-modulus', '3',
                        '--require-bank-reuse'])
                report['status'] = 'TYPED_STRESS_PASS' if all(row['pass'] for row in report['typed_runs']) else 'INCOMPLETE'
                return
            if MODE == 'typed_followup_gate':
                report['scope'] = 'full typed two-rank BFS: repeated unfiltered initcheck activation and raw owner/transport/retirement timelines'
                nsys = prepare_nsys()
                for case in typed_followup_cases(base):
                    replay_env = dict(env)
                    if case['tool'] == 'initcheck':
                        replay_env.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,ALLOC,REG')
                    else:
                        # Diagnostic logging must not inflate the healthy timeline.
                        replay_env.pop('NCCL_DEBUG', None)
                        replay_env.pop('NCCL_DEBUG_SUBSYS', None)
                        replay_env['PATH'] = str(Path(nsys).parent) + ':' + replay_env['PATH']
                    row = replay_typed(case['config'], case['label'], case['extra'], replay_env)
                    row.update(tool=case['tool'], repeat=case['repeat'])
                    if case['tool'] == 'nsys' and row['pass']:
                        row['rank_sqlite'] = []
                        for rank in (0, 1):
                            prefix = logs / case['label'] / 'healthy-None' / f'rank-{rank}'
                            database = prefix.with_suffix('.sqlite')
                            run([nsys, 'export', '--type', 'sqlite', '--force-overwrite=true',
                                '-o', str(database), str(prefix.with_suffix('.nsys-rep'))],
                                case['label'] + f'-rank{rank}-export', timeout=600)
                            row['rank_sqlite'].append(str(database.relative_to(logs)))
                    save()
                report['status'] = 'TYPED_FOLLOWUP_PASS' if all(row['pass'] for row in report['typed_runs']) else 'INCOMPLETE'
                return
            # Unsupported modes are negative admission checks, never successful
            # runtime coverage. Keep the missing CUCO/HOST/HASH_FIRST scope open.
            supported = []
            for index, config in enumerate(configs):
                row = unavailable_admission(config, 'typed-unavailable-' + str(index))
                if row is not None:
                    if not row['pass_expected_rejection']:
                        report['status'] = 'INCOMPLETE'
                        return
                else:
                    supported.append(config)
            # Prove the producer is actually exercised before the expensive suite.
            for index, config in enumerate(typed_reuse_configs(base)):
                if config['frontier_profile'] != 'HASH_FIRST':
                    continue
                row = replay_typed(config, 'typed-reuse-' + str(index),
                    ['--healthy-only', '--unitriangular-modulus', '2', '--require-bank-reuse'])
                if not row['pass']:
                    report['status'] = 'INCOMPLETE'
                    return
            selected = []
            for index, config in enumerate(supported):
                fault_case = config['completion_epoch_window'] == 3 and config['local_pre_dedup'] and (
                    config['topology']['logical_owner_to_rank'] == [0, 1])
                row = replay_typed(config, 'typed-' + str(index),
                    ['--capacity-faults'] if fault_case else ['--healthy-only'])
                if not row['pass']:
                    report['status'] = 'INCOMPLETE'
                    report['remaining_gates_not_run'] = 'candidate rejected before sanitizer stage'
                    save()
                    return
                if fault_case:
                    selected.append(config)
            for config in selected:
                for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    replay_typed(config, 'typed-' + config['owner_backend'] + '-' + config['frontier_profile'] + '-banks-' +
                        str(config['capacities']['route_slot_count']) + '-' + tool,
                        ['--healthy-only', '--instrument-processes', tool])
            report['status'] = 'TYPED_GATE_PASS' if all(row['pass'] for row in report['typed_runs']) else 'INCOMPLETE'
            return
        if MODE == "native_rank_gate":
            report["scope"] = "native rank owner: separate single-GPU T4 processes; two-rank checks only on verified P2P"
            report["t4_acceptance_eligible"] = False
            report["native_leaf_gates"] = []
            # Exercise the actual host NCCL wrapper's error/progress ordering
            # before GPU gates. This is API-boundary-double coverage only,
            # never a replacement for independent-rank transport faults.
            protocol = work / "native-host-nccl-protocol"
            run(["g++", "-std=c++17", "-I" + str(source / "tests/nccl_stubs"),
                 str(source / "tests/nccl_transport_failure.cpp"), "-o", str(protocol)],
                "native-host-nccl-protocol-build")
            run([str(protocol)], "native-host-nccl-protocol", timeout=30)
            report["native_host_nccl_protocol"] = "PASS_API_BOUNDARY_DOUBLE_NOT_GPU"
            save()
            for backend, defines in (("CUB_SORT_MERGE", []), ("BMMA_BUCKET", ["-DMGBFS_TEST_BMMA=1"])):
                binary = work / ("native-rank-leaf-" + backend.lower())
                run([str(sdk / "bin/nvcc"), "-std=c++17", "-lineinfo", "-arch=sm_75",
                     "-I" + str(source / "cuda"), *defines, str(source / "tests/bounded_owner.cu"),
                     str(source / "cuda/bounded_owner.cu"), str(source / "cuda/bounded_owner_query.cpp"),
                     "-o", str(binary)], "native-rank-leaf-build-" + backend)
                for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                    text = run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "97", str(binary)],
                               "native-rank-leaf-" + backend + "-" + tool, timeout=300)
                    if "BOUNDED_OWNER_PASS" not in text:
                        raise RuntimeError("NATIVE_RANK_LEAF_RESULT")
                    report["native_leaf_gates"].append({"backend": backend, "tool": tool, "status": "PASS"})
                    save()
            run(["cargo", "build", "--locked", "-p", "mgbfs-cli", "--features", "cuda,library-owner"],
                "native-rank-cli-build", timeout=1800)
            run(["cargo", "test", "--locked", "-p", "mgbfs-cli", "--features", "cuda,library-owner",
                 "--test", "native_lsa_capture", "tensor_generation_hardware_admission_and_layer_counts",
                 "--", "--exact", "--nocapture", "--test-threads=1"],
                "tensor-generation-hardware-admission", timeout=300)
            report["tensor_generation_cli_gate"] = "PASS_HARDWARE_CONDITIONAL_COUNTS_ARCHIVE"
            save()
            spec = importlib.util.spec_from_file_location("native_replay", source / "scripts/replay_lsa_cancel_candidate.py")
            replay = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(replay)
            report["native_single_gpu_runs"] = []
            rank_backends = ("CUCO_RANK", "CUB_SORT_MERGE", "BMMA_BUCKET")
            for gpu in (0, 1):
                for backend in rank_backends:
                    for profile in ("DENSE", "HASH_FIRST"):
                        for prededup in ("ON", "OFF"):
                            label = f"native-gpu{gpu}-{backend}-{profile}-{prededup}"
                            case = logs / label
                            case.mkdir()
                            case_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu), RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                                TORCHELASTIC_RUN_ID=label, MGBFS_OWNER_BACKEND=backend, MGBFS_PROFILE=profile,
                                MGBFS_PRE_DEDUP=prededup, MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                                MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2", MGBFS_BUCKET_CAPACITY="32",
                                MGBFS_STATE_CODEC="matrix_u8", MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                                MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0", MGBFS_BENCH_SKIP_ARCHIVE="0",
                                MGBFS_ARCHIVE_STREAM="0", MGBFS_CAPACITY_MODE="max_per_rank", MGBFS_RANK_MAP="0",
                                MGBFS_TRANSPORT_BACKEND="NCCL_LSA", MGBFS_TEST_OWNER_DAG_CAPTURE="1")
                            if backend == "CUCO_RANK":
                                case_env["MGBFS_LIBRARY_POOL_BYTES"] = str(64 << 20)
                            with (case / "rank-0.log").open("w") as output:
                                result = subprocess.run([str(source / "target/debug/mgbfs"), "bench", "--reference", "s4", "7",
                                    str(case / "bootstrap"), str(case / "archive"), str(case / "result")],
                                    cwd=source, env=case_env, stdout=output, stderr=subprocess.STDOUT, timeout=120)
                            if result.returncode != 0 or not (case / "result/group-complete.json").exists():
                                raise RuntimeError("NATIVE_SINGLE_GPU_FAILED: " + label)
                            if "MGBFS_OWNER_DAG_CAPTURE launched" not in (case / "rank-0.log").read_text():
                                raise RuntimeError("NATIVE_OWNER_CAPTURE_NOT_REACHED: " + label)
                            oracle_path = case / 'full-state-oracle.json'
                            run([python, "-c",
                                "import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
                                "from replay_lsa_cancel_candidate import verify_process_archives;"
                                "result=verify_process_archives(Path(sys.argv[2]),n=4,world=1);"
                                "Path(sys.argv[3]).write_text(json.dumps(result));print(json.dumps(result))",
                                str(source / "scripts"), str(case), str(oracle_path)], label + "-oracle")
                            oracle = json.loads(oracle_path.read_text())
                            report["native_single_gpu_runs"].append({"gpu": gpu, "backend": backend,
                                "profile": profile, "prededup": prededup, "oracle": oracle})
                            save()
            if any(row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
                report["status"] = "NATIVE_SINGLE_GPU_PASS_MULTI_GPU_UNSUPPORTED"
                return
            report["native_process_gates"] = []
            for backend in rank_backends:
                for profile in ("DENSE", "HASH_FIRST"):
                    label = "native-process-" + backend + "-" + profile
                    with (logs / (label + ".log")).open("w") as output:
                        result = run_protocol_replay([str(venv / "bin/python"),
                            str(source / "scripts/replay_lsa_cancel_candidate.py"), str(work), str(logs / label),
                            "--owner-backend", backend, "--profile", profile, "--capacity-faults"],
                            cwd=source, env=env, log=output)
                    report["native_process_gates"].append({"backend": backend, "profile": profile, **result})
                    save()
                    for prededup, rank_map in (("ON", "1,0"), ("OFF", "0,1"), ("OFF", "1,0")):
                        label = "native-equivalence-" + backend + "-" + profile + "-" + prededup + "-" + rank_map.replace(',', '')
                        with (logs / (label + ".log")).open("w") as output:
                            result = run_protocol_replay([str(venv / "bin/python"),
                                str(source / "scripts/replay_lsa_cancel_candidate.py"), str(work), str(logs / label),
                                "--owner-backend", backend, "--profile", profile, "--healthy-only",
                                "--pre-dedup", prededup, "--rank-map", rank_map],
                                cwd=source, env=env, log=output)
                        report["native_process_gates"].append({"backend": backend, "profile": profile,
                            "prededup": prededup, "rank_map": rank_map, **result})
                        save()
            report["native_full_bfs_sanitizers"] = []
            sanitizer_env = dict(env, NCCL_DEBUG='INFO')
            for backend in rank_backends:
                for profile in ("DENSE", "HASH_FIRST"):
                    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                        label = "native-full-" + backend + "-" + profile + "-" + tool
                        with (logs / (label + ".log")).open("w") as output:
                            result = run_protocol_replay([str(venv / "bin/python"),
                                str(source / "scripts/replay_lsa_cancel_candidate.py"), str(work), str(logs / label),
                                "--owner-backend", backend, "--profile", profile, "--healthy-only",
                                "--instrument-processes", tool], cwd=source, env=sanitizer_env, log=output)
                        report["native_full_bfs_sanitizers"].append({"backend": backend,
                            "profile": profile, "tool": tool, **result})
                        save()
            report["native_full_bfs_timelines"] = []
            if all(row["returncode"] == 0 and not row["timed_out"]
                   for row in report["native_process_gates"]):
                nsys = prepare_nsys()
                trace_env = dict(env)
                trace_env.pop('MGBFS_TEST_OWNER_DAG_CAPTURE', None)
                trace_env['PATH'] = str(Path(nsys).parent) + ':' + trace_env['PATH']
                for backend in rank_backends:
                    for profile in ("DENSE", "HASH_FIRST"):
                        label = "native-s8-timeline-" + backend + "-" + profile
                        with (logs / (label + ".log")).open("w") as output:
                            result = run_protocol_replay([str(venv / "bin/python"),
                                str(source / "scripts/replay_lsa_cancel_candidate.py"), str(work), str(logs / label),
                                "--owner-backend", backend, "--profile", profile, "--healthy-only",
                                "--reference-size", "8", "--batch", "128", "--instrument-processes", "nsys"],
                                cwd=source, env=trace_env, log=output)
                        row = {"backend": backend, "profile": profile, **result, "rank_reports": []}
                        report["native_full_bfs_timelines"].append(row)
                        save()
                        if result['returncode'] == 0 and not result['timed_out']:
                            for rank in (0, 1):
                                prefix = logs / label / 'healthy-None' / f'rank-{rank}'
                                database = prefix.with_suffix('.sqlite')
                                run([nsys, 'export', '--type', 'sqlite', '--force-overwrite=true',
                                     '-o', str(database), str(prefix.with_suffix('.nsys-rep'))],
                                    label + f'-rank{rank}-export', timeout=600)
                                analysis = prefix.with_suffix('.analysis.json')
                                run([str(venv / 'bin/python'), str(source / 'scripts/nsys_sync_callsites.py'),
                                     str(database), str(analysis)], label + f'-rank{rank}-analysis')
                                row['rank_reports'].append(str(analysis.relative_to(logs)))
                                save()
            report["status"] = ("NATIVE_PROCESS_AND_SANITIZER_GATES_PASS" if all(
                row["returncode"] == 0 and not row["timed_out"] for row in
                report["native_process_gates"] + report["native_full_bfs_sanitizers"])
                else "NATIVE_PROCESS_OR_SANITIZER_GATES_FAILED")
            return
        if MODE == "device_protocol_replay":
            # Reuse the production process replay; keep failed gates visible
            # while collecting independent results from the remaining gates.
            report["protocol_gates"] = []
            replay = source / "scripts/replay_lsa_cancel_candidate.py"
            suites = [
                ("dense", ["--profile", "DENSE", "--oracle", "--capacity-faults"], {}),
                ("hash-first", ["--profile", "HASH_FIRST", "--oracle", "--capacity-faults"], {}),
                ("owner-capture", ["--profile", "HASH_FIRST", "--healthy-only"],
                 {"MGBFS_TEST_OWNER_DAG_CAPTURE": "1"}),
            ]
            suites += [(tool, ["--healthy-only", "--instrument-processes", tool], {})
                       for tool in ("memcheck", "racecheck", "initcheck", "synccheck")]
            for name, arguments, overrides in suites:
                gate_env = dict(env)
                gate_env.update(overrides)
                with (logs / (name + "-replay.log")).open("w") as log:
                    result = run_protocol_replay(
                        [str(venv / "bin/python"), str(replay), str(work),
                         str(logs / name), *arguments], cwd=source, env=gate_env,
                        log=log)
                report["protocol_gates"].append(
                    {"name": name, **result})
                save()
            report["status"] = ("PROTOCOL_GATES_PASS" if all(
                item["returncode"] == 0 and not item["timed_out"]
                for item in report["protocol_gates"])
                else "PROTOCOL_GATES_FAILED")
            return
        if MODE == "warmup_admission_gate":
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "warmup-cli-build", timeout=1800)
            cli = str(source / "target/release/mgbfs")
            env.update(MGBFS_OWNER_BACKEND="CUCO_RANK",
                       MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE="DENSE",
                       MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                       MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2",
                       MGBFS_BUCKET_CAPACITY="32", MGBFS_STATE_CODEC="matrix_u8",
                       MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                       MGBFS_ARCHIVE_SLOTS="128", MGBFS_PRE_DEDUP="ON",
                       MGBFS_BENCH_SKIP_ARCHIVE="0", MGBFS_ARCHIVE_STREAM="0",
                       MGBFS_CAPACITY_MODE="max_per_rank", MGBFS_RANK_MAP="0,1",
                       MGBFS_TRANSPORT_BACKEND="HOST_SIZED_NCCL")
            report["warmup_runs"] = {}

            def launch(name, wrapper=None, warmup="0", expected=None):
                root = work / name
                root.mkdir()
                output = logs / name
                argv = [cli, "bench", "--reference", "s4", "7",
                        str(root / "bootstrap"), str(root / "archive"), str(output)]
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node=2", "--no-python"]
                if wrapper:
                    command += ["/bin/bash", "-c", wrapper, "mgbfs-rank-launch"]
                command += argv
                case_env = dict(env, MGBFS_BENCH_WARMUP=warmup)
                try:
                    completed = subprocess.run(command, cwd=source, env=case_env,
                                               capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired as error:
                    (logs / (name + ".log")).write_text(str(error.stdout) + str(error.stderr))
                    raise RuntimeError("WARMUP_GATE_TIMEOUT: " + name) from error
                output_text = completed.stdout + completed.stderr
                (logs / (name + ".log")).write_text(output_text)
                if expected is not None:
                    if (completed.returncode == 0 or expected not in output_text
                            or "REMOTE_CONFIGURATION_FATAL" not in output_text
                            or (output / "group-complete.json").exists()):
                        raise RuntimeError("WARMUP_GATE_FATAL: " + name)
                    report["warmup_runs"][name] = "PASS_GROUP_FATAL"
                    save()
                    return
                if completed.returncode != 0:
                    raise RuntimeError("WARMUP_GATE_SUCCESS: " + name)
                if not (output / "group-complete.json").is_file():
                    raise RuntimeError("WARMUP_GATE_GROUP_MARKER")
                if (logs / (name + ".warmup/group-complete.json")).exists():
                    raise RuntimeError("WARMUP_GATE_FALSE_MARKER")
                for rank in range(2):
                    if (root / f"archive.warmup-rank-{rank}.mgbfsar1").exists():
                        raise RuntimeError("WARMUP_GATE_ARCHIVE_NOT_RELEASED")
                    row = json.loads((output / f"rank-{rank}.json").read_text())
                    warm_row = json.loads((logs / (name + ".warmup") /
                                           f"rank-{rank}.json").read_text())
                    if (row["status"] != "COMPLETE" or not row["warmup_completed"]
                            or warm_row["archive_commit_scope"] != "warmup_ephemeral"):
                        raise RuntimeError("WARMUP_GATE_RANK_RESULT")
                    run([cli, "verify", str(root / f"archive-rank-{rank}.mgbfsar1")],
                        f"{name}-verify-{rank}", timeout=120)
                report["warmup_runs"][name] = "PASS_MEASURED_GROUP_AND_ARCHIVES"
                save()

            launch("warmup-rank-mismatch",
                   'if [ "$RANK" = 0 ]; then export MGBFS_BENCH_WARMUP=1; fi; exec "$@"',
                   expected="REMOTE_CONFIGURATION_FATAL")
            launch("warmup-invalid",
                   'if [ "$RANK" = 0 ]; then export MGBFS_BENCH_WARMUP=bad; fi; exec "$@"',
                   expected="BENCH_WARMUP_CONFIG")
            launch("macro-invalid",
                   'if [ "$RANK" = 0 ]; then export MGBFS_MACRO_DEPTH=2; fi; exec "$@"',
                   expected="MACRO_MULTI_GPU_UNSUPPORTED")
            launch("archive-invalid",
                   'if [ "$RANK" = 0 ]; then export MGBFS_BENCH_SKIP_ARCHIVE=1; fi; exec "$@"',
                   expected="CLI_BENCH_ARCHIVE_REQUIRED")
            launch("warmup-normal", warmup="1")
            report["status"] = "COMPLETE"
            return
        if MODE == "boundary_gate":
            run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                  "--test", "bootstrap", "--test", "group_commit", "--test", "archive"],
                 "boundary-cpu-tests", timeout=900)
            run(["cargo", "test", "--locked", "-p", "mgbfs-runtime", "--features", "cuda",
                 "--lib", "optional_u32_rejects_invalid_value_without_panicking"],
                "boundary-reference-config-test", timeout=900)
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "boundary-cli-build", timeout=1800)
            if any(row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
                report["boundary_runs"] = {"linux_archive_and_protocol_cpu_tests": "PASS"}
                report["status"] = "UNSUPPORTED_HOST_AFTER_CPU"
                return
            run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                 "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                 "one_rank_constructor_failure_after_nccl_stops_peer",
                 "--", "--ignored", "--nocapture"],
                "boundary-constructor-fault", timeout=75)
            cli = str(source / "target/release/mgbfs")
            env.update(MGBFS_OWNER_BACKEND="CUCO_RANK",
                       MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE="DENSE",
                       MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                       MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2",
                       MGBFS_BUCKET_CAPACITY="32", MGBFS_STATE_CODEC="matrix_u8",
                       MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                       MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0",
                       MGBFS_PRE_DEDUP="ON", MGBFS_BENCH_SKIP_ARCHIVE="0",
                       MGBFS_ARCHIVE_STREAM="0", MGBFS_CAPACITY_MODE="max_per_rank",
                       MGBFS_RANK_MAP="0,1")
            report["boundary_runs"] = {}
            for transport in ("HOST_SIZED_NCCL", "NCCL_LSA"):
                env["MGBFS_TRANSPORT_BACKEND"] = transport
                name = transport.lower()
                root = work / ("boundary-" + name)
                root.mkdir()
                output = logs / ("boundary-" + name)
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                           "s4", "7", str(root / "bootstrap"), str(root / "archive"),
                           str(output)]
                run(command, "boundary-" + name, timeout=300)
                marker = json.loads((output / "group-complete.json").read_text())
                if (marker["status"] != "COMPLETE" or marker["world_size"] != 2
                        or marker["archive_commit_scope"] != "file_fsync"):
                    raise RuntimeError("GROUP_COMMIT_MARKER")
                layers = None
                for rank in range(2):
                    data = (output / f"rank-{rank}.json").read_bytes()
                    row = json.loads(data)
                    if (row["status"] != "COMPLETE" or row["rank"] != rank
                            or row["archive_commit_scope"] != "file_fsync"
                            or list(hashlib.sha256(data).digest()) != marker["rank_sha256"][rank]):
                        raise RuntimeError("GROUP_RANK_RESULT")
                    if layers is None:
                        layers = [0] * len(row["local_layer_sizes"])
                    if len(row["local_layer_sizes"]) != len(layers):
                        raise RuntimeError("GROUP_LAYER_SHAPE")
                    layers = [a + b for a, b in zip(layers, row["local_layer_sizes"])]
                    run([cli, "verify", str(root / f"archive-rank-{rank}.mgbfsar1")],
                        f"boundary-{name}-verify-{rank}", timeout=300)
                if layers != [1, 3, 5, 6, 5, 3, 1]:
                    raise RuntimeError("GROUP_LAYER_ORACLE")
                report["boundary_runs"][name] = "PASS_GROUP_MARKER_AND_ARCHIVES"
                save()
            env["MGBFS_TRANSPORT_BACKEND"] = "NCCL_LSA"
            fault = work / "boundary-archive-fault"
            fault.mkdir()
            (fault / "archive-rank-0.mgbfsar1").write_bytes(b"occupied")
            fault_output = logs / "boundary-archive-fault-results"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                       "s4", "7", str(fault / "bootstrap"), str(fault / "archive"),
                       str(fault_output)]
            failed = subprocess.run(command, cwd=source, env=env, capture_output=True,
                                    text=True, timeout=180)
            output_text = failed.stdout + failed.stderr
            (logs / "boundary-archive-fault.log").write_text(output_text)
            if (failed.returncode == 0 or "ARCHIVE_EXTENT" not in output_text
                    or "REMOTE_ARCHIVE_ADMISSION_FATAL" not in output_text
                    or (fault_output / "group-complete.json").exists()):
                raise RuntimeError("ASYMMETRIC_ARCHIVE_ADMISSION_GATE")
            report["boundary_runs"]["one_rank_archive_admission_failure"] = "PASS_GROUP_FATAL"
            config_fault = work / "boundary-config-fault"
            config_fault.mkdir()
            config_output = logs / "boundary-config-fault-results"
            launcher = 'if [ "$RANK" = 0 ]; then export MGBFS_SHARDS=bad; fi; exec "$@"'
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=2", "--no-python", "/bin/bash", "-c", launcher,
                       "mgbfs-rank-launch", cli, "bench", "--reference", "s4", "7",
                       str(config_fault / "bootstrap"), str(config_fault / "archive"),
                       str(config_output)]
            failed = subprocess.run(command, cwd=source, env=env, capture_output=True,
                                    text=True, timeout=90)
            output_text = failed.stdout + failed.stderr
            (logs / "boundary-config-fault.log").write_text(output_text)
            if (failed.returncode == 0 or "ENV_MGBFS_SHARDS" not in output_text
                    or "REMOTE_CONFIGURATION_FATAL" not in output_text
                    or (config_output / "group-complete.json").exists()):
                raise RuntimeError("ASYMMETRIC_CONFIGURATION_GATE")
            report["boundary_runs"]["one_rank_invalid_configuration"] = "PASS_GROUP_FATAL"
            env.update(MGBFS_BENCH_CAPACITY="6", MGBFS_ARCHIVE_ROWS="1")
            small = work / "boundary-small-layer-capacity"
            small.mkdir()
            small_output = logs / "boundary-small-layer-capacity"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                       "s4", "7", str(small / "bootstrap"), str(small / "archive"),
                       str(small_output)]
            run(command, "boundary-small-layer-capacity", timeout=300)
            if not (small_output / "group-complete.json").is_file():
                raise RuntimeError("ARCHIVE_LAYER_CAPACITY_GROUP_MARKER")
            layer_counts = None
            for rank in range(2):
                row = json.loads((small_output / f"rank-{rank}.json").read_text())
                if (row["status"] != "COMPLETE" or row["rank_capacity_records"] != 6
                        or row["disk_reserved_bytes"] != 6304):
                    raise RuntimeError("ARCHIVE_LAYER_CAPACITY_METADATA")
                if layer_counts is None:
                    layer_counts = [0] * len(row["local_layer_sizes"])
                if len(layer_counts) != len(row["local_layer_sizes"]):
                    raise RuntimeError("ARCHIVE_LAYER_CAPACITY_DEPTHS")
                layer_counts = [a + b for a, b in zip(layer_counts, row["local_layer_sizes"])]
                run([cli, "verify", str(small / f"archive-rank-{rank}.mgbfsar1")],
                    f"boundary-small-layer-capacity-verify-{rank}", timeout=300)
            if layer_counts != [1, 3, 5, 6, 5, 3, 1]:
                raise RuntimeError("ARCHIVE_LAYER_CAPACITY_ORACLE")
            report["boundary_runs"]["archive_total_exceeds_layer_capacity"] = "PASS"
            oracle = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                          "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                          "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle",
                          "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                         "boundary-full-state-oracle", timeout=900)
            if "test result: ok. 46 passed; 0 failed" not in oracle:
                raise RuntimeError("BOUNDARY_FULL_STATE_ORACLE")
            report["boundary_runs"]["independent_full_state_oracle"] = "PASS"
            report["status"] = "COMPLETE"
            return
        if MODE == "paired_cayleypy":
            baseline_commit = "f0f2b8e5ee61173039ab9742f3a7756c9b6365e6"
            baseline = work / "cayleypy-baseline"
            gate.checkout("https://github.com/TryDotAtwo/cayleypy.git",
                          baseline_commit, baseline, env, logs, "cayleypy")
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "paired-cli-build", timeout=1800)
            sys.path.insert(0, str(source / "scripts"))
            from distributed_gpu_bench import run_group, stats
            from library_gpu_screen import run_case
            paired_config = typed_paired_config(json.loads(
                (source / 'tests/run-s4-two-rank.json').read_text()), 10)
            paired_config_path = logs / 'paired-s10-run-config.json'
            paired_config_path.write_text(json.dumps(paired_config, separators=(',', ':')))
            report.update(scope="paired physical 2xT4 S10; production RunConfigV1 native archive mandatory, CayleyPy no archive",
                          baseline_commit=baseline_commit, rows=[])
            save()
            expected = None
            samples = {"native": [], "cayleypy": []}
            for repeat in range(5):
                for backend in (("native", "cayleypy") if repeat % 2 == 0
                                else ("cayleypy", "native")):
                    label = f"paired-s10-{backend}-r{repeat}"
                    if backend == "native":
                        archive_root = logs / (label + '-archives')
                        case_env = dict(env, MGBFS_TRANSPORT_BACKEND="NCCL_LSA",
                                        MGBFS_SHARDS="4", MGBFS_BUCKETS="256",
                                        MGBFS_ARCHIVE_SLOTS="256")
                        result = run_case(str(source / "target/release/mgbfs"),
                                          logs / label, archive_root, "s10", 3_628_800,
                                          2, 32768, 1_000_000, 1_000_000, 96 << 20,
                                          "DENSE", "ON", case_env, owner="CUCO_RANK",
                                          run_config=paired_config_path)
                        row = result["measurement"]
                    else:
                        case_env = dict(env, PYTHONPATH=str(baseline),
                                        CUDA_VISIBLE_DEVICES="0,1",
                                        MGBFS_BENCH_WORLD_SIZE="2")
                        command = [sys.executable, "-m", "torch.distributed.run",
                                   "--standalone", "--nproc-per-node=2",
                                   str(source / "scripts/distributed_gpu_bench.py"),
                                   "baseline-worker", "10", "1048576", "{RANK_OUT}"]
                        row = run_group(command, logs, label, case_env, timeout=1800)
                    if row["status"] != "COMPLETE" or sum(row["layer_sizes"]) != 3_628_800:
                        raise RuntimeError("PAIRED_BFS_INCOMPLETE: " + label)
                    expected = expected or row["layer_sizes"]
                    if row["layer_sizes"] != expected:
                        raise RuntimeError("PAIRED_LAYER_MISMATCH: " + label)
                    row["paired_backend"] = backend
                    row["paired_repeat"] = repeat
                    samples[backend].append(row)
                    report["rows"].append({"label": label, "backend": backend,
                                           "search_seconds": row["search_complete_seconds"],
                                           "durable_seconds": row.get("durable_run_commit_seconds"),
                                           "peak_mib_per_rank": row["smi_peak_mib_per_rank"],
                                           "archive_contract": ("verified file_fsync"
                                                                if backend == "native" else "none")})
                    save()
            report["layers"] = expected
            report["native"] = stats(samples["native"])
            report["cayleypy"] = stats(samples["cayleypy"])
            report["status"] = "COMPLETE"
            return
        if MODE in ("benchmark", "timeline", "timeline_backtrace", "timeline_analysis"):
            report["scope"] = (
                "paired two-T4 S10 CUCO_RANK DENSE; archive-verified; "
                + ("five unprofiled repeats per transport" if MODE == "benchmark" else
                   "one profiled diagnostic run per transport, including startup and archive"))
            report["benchmark_config"] = {
                "group": "s10", "batch": 32768, "capacity_per_rank": 1_000_000,
                "ring_per_rank": 1_000_000, "pool_bytes_per_rank": 96 << 20,
                "shards_per_rank": 4, "buckets": 256, "archive_slots": 256,
            }
            save()
            if MODE == "timeline_backtrace":
                env["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
                env["CARGO_PROFILE_RELEASE_STRIP"] = "none"
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "cli-build", timeout=1800)
            sys.path.insert(0, str(source / "scripts"))
            from distributed_gpu_bench import stats
            from library_gpu_screen import run_case
            cli = str(source / "target/release/mgbfs")
            profiled_cli = cli
            if MODE == "timeline_backtrace":
                wrapper = work / "capture-rank-maps.sh"
                wrapper.write_text(
                    "#!/bin/bash\nset -eu\n"
                    'if [[ -n "${RANK:-}" && -n "${MGBFS_DIAGNOSTIC_MAP_DIR:-}" ]]; then\n'
                    '  mgbfs_target_pid=$$\n'
                    '  ( sleep 0.25; cp "/proc/${mgbfs_target_pid}/maps" '
                    '"${MGBFS_DIAGNOSTIC_MAP_DIR}/rank-${RANK}.maps" ) &\n'
                    "fi\n"
                    f"exec {shlex.quote(cli)} \"$@\"\n"
                )
                wrapper.chmod(0o755)
                profiled_cli = str(wrapper)
            nsys = None
            if MODE in ("timeline", "timeline_backtrace", "timeline_analysis"):
                nsys = prepare_nsys()
            panel = ({"NCCL_LSA": []} if MODE in ("timeline_backtrace", "timeline_analysis") else
                     {"HOST_SIZED_NCCL": [], "NCCL_LSA": []})
            expected_dispatch = {"HOST_SIZED_NCCL": "HostSizedNccl", "NCCL_LSA": "Lsa"}
            for repeat in range(5 if nsys is None else 1):
                order = list(panel) if repeat % 2 == 0 else list(reversed(panel))
                for transport in order:
                    label = f"s10-{transport.lower()}-r{repeat}"
                    case_env = dict(env, MGBFS_TRANSPORT_BACKEND=transport,
                                    MGBFS_SHARDS="4", MGBFS_BUCKETS="256",
                                    MGBFS_ARCHIVE_SLOTS="256")
                    if nsys is not None:
                        case_env["MGBFS_PROFILE_SEARCH"] = "1"
                    if MODE == "timeline_backtrace":
                        case_env["MGBFS_NSYS_CUDA_BACKTRACE"] = "sync,memory"
                        case_env["MGBFS_DIAGNOSTIC_MAP_DIR"] = str(logs / label)
                        case_env["NSYS_CONFIG_DIRECTIVES"] = (
                            f'DbgFileSearchPath="{source / "target/release"}:{native}:{build}"'
                        )
                    result = run_case(profiled_cli, logs / label, work / label, "s10",
                                      3_628_800, 2, 32768, 1_000_000, 1_000_000,
                                      96 << 20, "DENSE", "ON", case_env,
                                      owner="CUCO_RANK", nsys=nsys)
                    if any(rank.get("transport_backend") != expected_dispatch[transport]
                           for rank in result["measurement"]["rank_results"]):
                        raise RuntimeError("TRANSPORT_DISPATCH_MISMATCH")
                    panel[transport].append(result["measurement"])
                    if nsys is not None:
                        run([nsys, "stats", "--report",
                             "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum",
                             "--format", "csv", result["trace"]],
                            label + "-nsys-stats", timeout=600)
                    if MODE == "timeline_analysis":
                        run([nsys, "analyze", "--rule",
                             "cuda_api_sync,gpu_gaps,gpu_time_util", result["trace"]],
                            label + "-nsys-analysis", timeout=600)
                    if MODE == "timeline_backtrace":
                        for rank in range(2):
                            mapping = logs / label / f"rank-{rank}.maps"
                            if not mapping.is_file() or str(source / "target/release/mgbfs") not in mapping.read_text():
                                raise RuntimeError("NSYS_RANK_MAPS_MISSING")
                        database = logs / label / "timeline.sqlite"
                        run([nsys, "export", "--type", "sqlite", "--force-overwrite=true",
                             "--output", str(database), result["trace"]],
                            label + "-nsys-export", timeout=600)
                        run([sys.executable, str(source / "scripts/nsys_sync_callsites.py"),
                             str(database), str(logs / label / "sync-callsites.json")],
                            label + "-sync-callsites", timeout=600)
                        addresses = logs / label / "rank-addresses.json"
                        run([sys.executable, str(source / "scripts/nsys_rank_maps.py"),
                             str(logs / label / "sync-callsites.json"), str(addresses),
                             "--map", f"0:{logs / label / 'rank-0.maps'}",
                             "--map", f"1:{logs / label / 'rank-1.maps'}"],
                            label + "-rank-addresses", timeout=600)
                        address_rows = json.loads(addresses.read_text())["rows"]
                        offsets = sorted({frame["offset"] for row in address_rows
                                          for frame in row["project_frames"]
                                          if frame["module"] == cli},
                                         key=lambda value: int(value, 16))
                        symbols = run(["addr2line", "-f", "-C", "-e", cli, *offsets],
                                      label + "-addr2line", timeout=600).splitlines()
                        if len(symbols) != 2 * len(offsets):
                            raise RuntimeError("NSYS_ADDR2LINE_SHAPE")
                        resolved = {offset: {"function": symbols[2 * index],
                                             "source": symbols[2 * index + 1]}
                                    for index, offset in enumerate(offsets)}
                        (logs / label / "rank-symbols.json").write_text(
                            json.dumps({"source": SOURCE, "executable": cli,
                                        "symbols": resolved}, indent=2))
                        report["symbolized_offsets"] = len(resolved)
                        database.unlink()
                        Path(result["trace"]).unlink()
                    report["runs"] = {key: len(value) for key, value in panel.items()}
                    save()
            if nsys is None:
                report["screen_statistics"] = {key: stats(value) for key, value in panel.items()}
            else:
                report["timeline_scope"] = "timed BFS CUDA profiler range; archive submissions included"
                if MODE == "timeline_backtrace":
                    report["callsite_scope"] = "LSA CUDA sync/copy callchains; profiled diagnostic"
                if MODE == "timeline_analysis":
                    report["analysis_scope"] = "LSA sync API and GPU gap expert rules; profiled diagnostic"
            report["status"] = "COMPLETE"
            return
        run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
             "--features", "cuda,library-owner", "--test", "library_multi_gpu",
             "--no-run"], "bfs-test-build", timeout=1800)
        if MODE in ("host_fault_only", "host_fault_and_process"):
            env["MGBFS_TRACE_ROUTE"] = "1"
            env["MGBFS_TRACE_ROUTE_NO_SYNC"] = "1"
            for name in ("lsa_one_exchange_matches_peer_payload",
                         "cuco_rank_lsa_one_rank_host_owner_error_stops_group"):
                checked = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                               "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                               name, "--", "--ignored", "--exact", "--nocapture",
                               "--test-threads=1"], name, timeout=45)
                if "test result: ok. 46 passed; 0 failed" not in checked:
                    raise RuntimeError("HOST_FAULT_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["host_fault"] = "PASS"
            if MODE == "host_fault_only":
                report["status"] = "COMPLETE"
                return
        if MODE in ("process_host_fault_only", "host_fault_and_process", "process_faults_only"):
            run(["cargo", "build", "--locked", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "process-fault-cli-build", timeout=1800)
            env.update(MGBFS_OWNER_BACKEND="CUCO_RANK",
                       MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE="DENSE",
                       MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                       MGBFS_BUCKETS="8", MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2",
                       MGBFS_BUCKET_CAPACITY="32", MGBFS_STATE_CODEC="matrix_u8",
                       MGBFS_ARCHIVE_CODEC="matrix_u8", MGBFS_ARCHIVE_ROWS="3",
                       MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0",
                       MGBFS_PRE_DEDUP="ON", MGBFS_BENCH_SKIP_ARCHIVE="0",
                       MGBFS_ARCHIVE_STREAM="0", MGBFS_CAPACITY_MODE="max_per_rank",
                       MGBFS_RANK_MAP="0,1", MGBFS_TRANSPORT_BACKEND="NCCL_LSA")
            faults = [("owner", "MGBFS_TEST_OWNER_HOST_FAULT_RANK",
                       "TEST_INJECTED_OWNER_HOST_ERROR")]
            if MODE == "process_faults_only":
                faults += [("archive-admission", "MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK",
                            "TEST_INJECTED_ARCHIVE_ADMISSION_ERROR"),
                           ("archive-finish", "MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK",
                            "TEST_INJECTED_ARCHIVE_FINISH_ERROR")]
            for fault_name, env_key, expected_error in faults:
                for key in ("MGBFS_TEST_OWNER_HOST_FAULT_RANK",
                            "MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK",
                            "MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK"):
                    env.pop(key, None)
                env[env_key] = "0"
                name = "process-" + fault_name + "-fault"
                root = work / name
                root.mkdir()
                output = logs / name
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node=2", "--no-python",
                           str(source / "target/debug/mgbfs"), "bench", "--reference",
                           "s4", "1", str(root / "bootstrap"), str(root / "archive"),
                           str(output)]
                with (logs / (name + ".log")).open("w") as stream:
                    try:
                        completed = subprocess.run(command, cwd=source, env=env, stdout=stream,
                                                   stderr=subprocess.STDOUT, timeout=60)
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError("PROCESS_FAULT_TIMEOUT: " + fault_name) from error
                if completed.returncode == 0 or (output / "group-complete.json").exists():
                    raise RuntimeError("PROCESS_FAULT_FALSE_COMPLETE: " + fault_name)
                checked = (logs / (name + ".log")).read_text(errors="replace")
                if expected_error not in checked:
                    raise RuntimeError("PROCESS_FAULT_NOT_REACHED: " + fault_name)
                report[name] = "PASS_BOUNDED_NO_COMPLETE"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE == "host_sized_only":
            report["scope"] = ("two physical T4; HostSizedNccl only; no LSA or P2P claim; "
                               "full-state oracle and archive fixtures")
            for name in ("library_two_rank_layers_and_archives_match_oracle",
                         "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle"):
                checked = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                               "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                               name, "--", "--exact", "--nocapture", "--test-threads=1"],
                              "host-sized-" + name, timeout=1800)
                if "test result: ok. 46 passed; 0 failed" not in checked:
                    raise RuntimeError("HOST_SIZED_ORACLE: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE in ("lsa_leaf_sanitizer", "lsa_leaf_harness_fault",
                    "lsa_leaf_remaining_sanitizers", "lsa_leaf_initcheck_debug"):
            name = "lsa_one_exchange_matches_peer_payload"
            binary = [path for path in (source / "target/debug/deps").glob("library_multi_gpu-*")
                      if path.is_file() and os.access(path, os.X_OK)]
            if len(binary) != 1:
                raise RuntimeError("LSA_LEAF_BINARY_INVENTORY")
            command = [str(binary[0]), name, "--ignored", "--exact", "--nocapture",
                       "--test-threads=1"]
            plain = run(command, "lsa-leaf-plain", timeout=180)
            if "test result: ok. 46 passed; 0 failed" not in plain:
                raise RuntimeError("LSA_LEAF_PLAIN_RESULT")
            report["plain_leaf"] = "PASS"
            save()
            if MODE == "lsa_leaf_harness_fault":
                fault_env = dict(env, MGBFS_TEST_LSA_BAD_PAYLOAD="1")
                try:
                    failed = subprocess.run(command, cwd=source, env=fault_env,
                                            capture_output=True, text=True, timeout=30)
                except subprocess.TimeoutExpired as error:
                    report["fault_result"] = "TIMEOUT"
                    raise RuntimeError("LSA_LEAF_FAULT_HUNG") from error
                fault_output = failed.stdout + failed.stderr
                (logs / "lsa-leaf-bad-payload.log").write_text(fault_output)
                if failed.returncode == 0 or "assertion" not in fault_output:
                    raise RuntimeError("LSA_LEAF_FAULT_NOT_DETECTED")
                report["fault_result"] = "NONZERO_WITH_ASSERTION"
                report["fault_returncode"] = failed.returncode
                report["status"] = "COMPLETE"
                return
            if MODE in ("lsa_leaf_remaining_sanitizers", "lsa_leaf_initcheck_debug"):
                report["leaf_tools"] = {}
                tools = (("initcheck",) if MODE == "lsa_leaf_initcheck_debug"
                         else ("initcheck", "synccheck"))
                if MODE == "lsa_leaf_initcheck_debug":
                    env["NCCL_DEBUG"] = "INFO"
                for tool in tools:
                    try:
                        checked = subprocess.run(
                            ["compute-sanitizer", "--tool", tool,
                             "--report-api-errors", "no", "--error-exitcode", "97",
                             *command], cwd=source, env=env, capture_output=True,
                            text=True, timeout=300)
                        output = checked.stdout + checked.stderr
                        (logs / ("lsa-leaf-" + tool + ".log")).write_text(output)
                        report["leaf_tools"][tool] = {
                            "returncode": checked.returncode,
                            "test_passed": "test result: ok. 46 passed; 0 failed" in output,
                            "zero_errors": "ERROR SUMMARY: 0 errors" in output,
                        }
                    except subprocess.TimeoutExpired:
                        report["leaf_tools"][tool] = {"status": "TIMEOUT"}
                    save()
                if not all(result.get("returncode") == 0 and
                           result.get("test_passed") and result.get("zero_errors")
                           for result in report["leaf_tools"].values()):
                    raise RuntimeError("LSA_LEAF_REMAINING_SANITIZER_FAILURE")
                report["status"] = "COMPLETE"
                return
            report["leaf_tools"] = {}
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                checked = run(["compute-sanitizer", "--tool", tool,
                               "--report-api-errors", "no", "--error-exitcode", "97",
                               *command], "lsa-leaf-" + tool, timeout=300)
                clean_summary = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                                 if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
                if ("test result: ok. 46 passed; 0 failed" not in checked
                        or clean_summary not in checked):
                    raise RuntimeError("LSA_LEAF_SANITIZER_RESULT: " + tool)
                report["leaf_tools"][tool] = "PASS_API_ERROR_REPORTING_DISABLED"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE == "owner_capacity_gate":
            for name in (
                "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle",
                "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "cuco_rank_lsa_one_rank_host_owner_error_stops_group",
            ):
                result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                              "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                              name, "--", "--ignored", "--exact", "--nocapture",
                              "--test-threads=1"], name, timeout=300)
                if "test result: ok. 46 passed; 0 failed" not in result:
                    raise RuntimeError("OWNER_CAPACITY_GATE_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE == "production_fault_gate":
            report["scope"] = ("two physical P2P T4; owner capacity, archive slot, "
                               "and device retirement fatal propagation")
            save()
            for name in (
                "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "archive_slot_failure_votes_group_fatal_before_lsa_exchange",
                "retirement_fifo_fault_votes_group_fatal_on_two_devices",
            ):
                command = ["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                           "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                           name, "--", "--exact", "--nocapture", "--test-threads=1"]
                if name != "retirement_fifo_fault_votes_group_fatal_on_two_devices":
                    command.insert(-3, "--ignored")
                result = run(command, "production-" + name, timeout=180)
                if "test result: ok. 46 passed; 0 failed" not in result:
                    raise RuntimeError("PRODUCTION_FAULT_GATE_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        if MODE in ("full_bfs_sanitizers", "full_bfs_memcheck_diagnostic"):
            report["scope"] = ("two physical P2P T4; complete S4 CUCO_RANK/LSA BFS "
                               "including archive, " + ("memcheck diagnostic only"
                               if MODE == "full_bfs_memcheck_diagnostic"
                               else "one fixture per sanitizer tool"))
            if MODE == "full_bfs_memcheck_diagnostic":
                env["NCCL_DEBUG"] = "INFO"
                env["MGBFS_TRACE_ROUTE"] = "1"
                env["MGBFS_TRACE_ROUTE_NO_SYNC"] = "1"
                env["MGBFS_TRACE_NCCL_GATE"] = "1"
            binaries = [path for path in (source / "target/debug/deps").glob("library_multi_gpu-*")
                        if path.is_file() and os.access(path, os.X_OK)]
            if len(binaries) != 1:
                raise RuntimeError("FULL_BFS_SANITIZER_BINARY_INVENTORY")
            name = "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle"
            command = [str(binaries[0]), name, "--ignored", "--exact", "--nocapture",
                       "--test-threads=1"]
            report["tools"] = {}
            tools = (("memcheck",) if MODE == "full_bfs_memcheck_diagnostic"
                     else ("memcheck", "racecheck", "initcheck", "synccheck"))
            for tool in tools:
                sanitizer = ["compute-sanitizer", "--tool", tool,
                             "--report-api-errors", "no", "--error-exitcode", "97", *command]
                log = logs / ("full-bfs-" + tool + ".log")
                with log.open("w") as stream:
                    process = subprocess.Popen(sanitizer, cwd=source, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        code = process.wait(timeout=(120 if MODE == "full_bfs_memcheck_diagnostic" else 300))
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                        code = None
                output = log.read_text(errors="replace")
                summary = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                           if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
                report["tools"][tool] = {
                    "status": "TIMEOUT" if code is None else "FINISHED",
                    "returncode": code,
                    "oracle_passed": "test result: ok. 46 passed; 0 failed" in output,
                    "clean_summary": summary in output,
                    "log_bytes": log.stat().st_size,
                }
                save()
            if not all(result.get("returncode") == 0 and result.get("oracle_passed")
                       and result.get("clean_summary")
                       for result in report["tools"].values()):
                raise RuntimeError("FULL_BFS_SANITIZER_GATE_INCOMPLETE")
            report["status"] = "COMPLETE"
            return
        if MODE == "device_fatal_gate":
            result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                          "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                          "retirement_fifo_fault_votes_group_fatal_on_two_devices",
                          "--", "--exact", "--nocapture", "--test-threads=1"],
                         "device-fatal-gate", timeout=180)
            if "test result: ok. 46 passed; 0 failed" not in result:
                raise RuntimeError("DEVICE_FATAL_GATE_RESULT")
            report["device_fatal_gate"] = "PASS"
            report["status"] = "COMPLETE"
            return
        if MODE == "archive_fault_gate":
            for name in ("archive_slot_failure_votes_group_fatal_before_exchange",
                         "archive_slot_failure_votes_group_fatal_before_lsa_exchange"):
                result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                              "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                              name, "--", "--ignored", "--exact", "--nocapture",
                              "--test-threads=1"], name, timeout=180)
                if "test result: ok. 46 passed; 0 failed" not in result:
                    raise RuntimeError("ARCHIVE_FAULT_TEST_RESULT: " + name)
                report[name] = "PASS"
                save()
            report["status"] = "COMPLETE"
            return
        result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                      "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                      "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle",
                      "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                     "lsa-full-bfs", timeout=900)
        if "test result: ok. 46 passed; 0 failed" not in result:
            raise RuntimeError("BFS_TEST_RESULT")
        binaries = [path for path in (source / "target/debug/deps").glob("library_multi_gpu-*")
                    if path.is_file() and os.access(path, os.X_OK)]
        if len(binaries) != 1:
            raise RuntimeError("BFS_TEST_BINARY_INVENTORY")
        report["plain_full_bfs"] = "PASS"
        save()
        if MODE == "rounds_gate":
            for test_name in (
                "library_two_rank_layers_and_archives_match_oracle",
                "cuco_rank_two_gpu_dense_layers_and_archives_match_oracle",
                "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "cuco_rank_lsa_one_rank_host_owner_error_stops_group",
            ):
                is_fault = "one_rank" in test_name
                result = run([str(binaries[0]), test_name,
                              *(["--ignored"] if is_fault else []),
                              "--exact", "--nocapture", "--test-threads=1"],
                             test_name, timeout=180 if is_fault else 1800)
                if "test result: ok. 46 passed; 0 failed" not in result:
                    raise RuntimeError("ROUND_SCHEDULE_TEST_RESULT: " + test_name)
            report["host_sized_profile_gate"] = "PASS"
            report["status"] = "COMPLETE"
            return
        env["NCCL_DEBUG"] = "INFO"
        sanitized = run(["compute-sanitizer", "--tool", "memcheck",
                         "--target-processes", "application-only",
                         "--report-api-errors", "no",
                         "--kernel-name", "kns=lsa_publish_count",
                         "--kernel-name", "kns=lsa_copy_exact",
                         "--kernel-name", "kns=import_transport_fatal",
                         "--error-exitcode", "99",
                         str(binaries[0]),
                         "cuco_rank_lsa_single_fixture_for_sanitizer",
                         "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                        "lsa-single-fixture-filtered-memcheck", timeout=120)
        if "test result: ok. 46 passed; 0 failed" not in sanitized or \
                "ERROR SUMMARY: 0 errors" not in sanitized:
            raise RuntimeError("BFS_MEMCHECK_RESULT")
        report["full_bfs_memcheck"] = (
            "PASS_FILTERED_TRANSPORT_KERNELS_APPLICATION_ONLY_API_ERRORS_DISABLED")
        report["status"] = "COMPLETE"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
