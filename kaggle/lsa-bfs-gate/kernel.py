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

SOURCE = "b47c2707bc3dd346ae0cdc8c2b6df7ff45198443"
CUCO = "532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4"
MODE = "typed_rank_gate"
HARDWARE = "T4"  # A4000 is an explicit diagnostic, never T4 acceptance.
NCCL_VARIANT = "minimum_arch_guard_posix"


def oracle_dependency_packages(mode):
    return ['pyarrow==19.0.1'] if mode in ('device_protocol_replay', 'native_rank_gate',
        'typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate') else []


def typed_rank_configs(base):
    cases = []
    for profile in ('DENSE', 'HASH_FIRST'):
        for credits in (2, 3):
            for prededup in (True, False):
                for mapping in ([0, 1], [1, 0]):
                    for banks in (2, 3, 4):
                        config = copy.deepcopy(base)
                        config.update(frontier_profile=profile, completion_epoch_window=credits,
                            local_pre_dedup=prededup, owner_backend='CUCO_RANK', library_pool_bytes=96 << 20)
                        config['topology']['logical_owner_to_rank'] = mapping.copy()
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
        for owner in ('CUCO_RANK', 'CUB_SORT_MERGE', 'BMMA_BUCKET'):
            selected = copy.deepcopy(candidate)
            selected['owner_backend'] = owner
            selected['library_pool_bytes'] = (96 << 20) if owner == 'CUCO_RANK' else None
            cases.append(selected)
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


def cuda_build_target(hardware):
    """Match the admitted physical GPU; never reuse another major's SASS."""
    targets = {"T4": "75", "RTX2070": "75", "A4000": "86"}
    if hardware not in targets:
        raise ValueError("UNSUPPORTED_CUDA_BUILD_HARDWARE: " + hardware)
    return targets[hardware]


def run_window_process_pair(command, cwd, env, output, timeout=120, required_stage=None,
                            require_window=True):
    """Reduced vendor probe, independent ranks, bounded whole process trees."""
    if not require_window and required_stage not in ('device_comm_only_create', 'device_comm_zero_create'):
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
        if MODE not in ("device_fatal_gate", "boundary_gate", "host_sized_only", "native_rank_gate") and any(
                row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
            report["status"] = "UNSUPPORTED_HOST"
            return
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        profiling_enabled = MODE in ('typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate', 'native_rank_gate', 'timeline', 'timeline_backtrace', 'timeline_analysis')
        components = list(library.CUDA_COMPONENTS)
        if profiling_enabled:
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
        if MODE == 'nccl_window_processes':
            # The compiler is pinned, but the instrumenter comes from the host.
            # Record actual versions; do not infer sanitizer identity from nvcc.
            report['environment_versions'] = {
                'driver': run(['nvidia-smi', '--query-gpu=driver_version',
                    '--format=csv,noheader'], 'driver-version').strip(),
                'cuda_compiler': run([env['CUDACXX'], '--version'], 'nvcc-version').strip(),
                'compute_sanitizer': run(['compute-sanitizer', '--version'],
                    'compute-sanitizer-version').strip(),
            }
            save()
        venv = work / "venv"
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        run([sys.executable, "-m", "pip", "--python", python, "install",
             "--only-binary=:all:", "--no-cache-dir", "--require-hashes", "-r",
             str(source / "experiments/library_owner/requirements-linux-x86_64.lock")],
            "dependencies", timeout=1200)
        verifier_dependencies = oracle_dependency_packages(MODE)
        if verifier_dependencies:
            # The full-state oracle reuses the archive reader in the Parquet
            # exporter; its module-level schemas require Arrow at import time.
            run([sys.executable, "-m", "pip", "--python", python, "install",
                 "--only-binary=:all:", "--no-deps", *verifier_dependencies],
                "archive-verifier-dependency", timeout=300)
            run([python, '-c', 'from export_hf_dataset import frames'],
                'archive-verifier-import-preflight', timeout=30,
                cwd=source / 'scripts')
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
            # Existing independent-process replay resolves this exact root.
            # Preserve the wheel separately instead of accidentally replaying it.
            shutil.move(str(nccl_target), str(work / "nccl-wheel"))
            nccl = nccl_target / "nvidia/nccl"
            run(["make", "-j2", "src.build", "NVTX=1", "CUDA_HOME=" + str(sdk),
                 "NVCC_GENCODE=-gencode=arch=compute_" + architecture + ",code=sm_" + architecture,
                 "BUILDDIR=" + str(nccl)], "nccl-build", cwd=vendor, timeout=5400)
            report["nccl_dependency"] = dict(variant=NCCL_VARIANT,
                upstream_commit=upstream, patch_sha256=patch_digest, architecture="sm" + architecture,
                posix_patch_sha256=posix_patch_digest,
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
        run(["cmake", "-S", str(source / "experiments/library_owner"), "-B", str(build),
             "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=" + architecture,
             "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUDAToolkit_ROOT=" + str(sdk),
             "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
             "-DCUCO_ROOT=" + str(cuco)], "library-configure")
        run(["cmake", "--build", str(build), "--target", "mgbfs_library_owner", "-j2"],
            "library-build", timeout=1800)
        env["MGBFS_LIBRARY_OWNER_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":" + env["LD_LIBRARY_PATH"]
        cutlass = work / "cutlass"
        gate.checkout("https://github.com/NVIDIA/cutlass.git", gate.CUTLASS_COMMIT,
                      cutlass, env, logs, "cutlass")
        native = work / "native-build"
        run(["cmake", "-S", str(source / "cuda"), "-B", str(native), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
             *(["-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g1"]
               if MODE == "timeline_backtrace" else []),
             "-DCMAKE_CUDA_ARCHITECTURES=" + architecture, "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUTLASS_ROOT=" + str(cutlass), "-DMGBFS_NCCL_LSA=ON",
             "-DMGBFS_NCCL_ROOT=" + str(nccl),
             *(['-DMGBFS_NVTX=ON', '-DMGBFS_NVTX_INCLUDE_DIR=' + str(sdk / 'include')]
               if profiling_enabled else [])], "native-configure")
        run(["cmake", "--build", str(native), "--target", "mgbfs_cuda", "-j2"],
            "native-build", timeout=1800)
        env["MGBFS_CUDA_LIB_DIR"] = str(native)
        env["LD_LIBRARY_PATH"] = str(native) + ":" + env["LD_LIBRARY_PATH"]
        if MODE in ('typed_rank_gate', 'typed_followup_gate', 'typed_stress_gate'):
            report['scope'] = 'typed RunConfigV1; independent two-T4 full-state S4 archives, faults and unfiltered sanitizers'
            report['typed_runs'] = []
            base = json.loads((source / 'tests/run-s4-two-rank.json').read_text())
            configs = typed_rank_configs(base)
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
                row['pass'] = row['returncode'] == 0 and not row['timed_out'] and (
                    detail.get('status') == 'DIAGNOSTIC_CASES_PASS' and detail.get('run_contract') == 'RunConfigV1'
                    and detail.get('epoch_window') == config['completion_epoch_window']
                    and detail.get('route_banks') == config['capacities']['route_slot_count'])
                report['typed_runs'].append(row)
                save()
                print('RESULT ' + json.dumps(row, separators=(',', ':')), flush=True)
                return row
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
            selected = []
            for index, config in enumerate(configs):
                fault_case = config['completion_epoch_window'] == 3 and config['local_pre_dedup'] and (
                    config['topology']['logical_owner_to_rank'] == [0, 1])
                replay_typed(config, 'typed-' + str(index),
                    ['--capacity-faults'] if fault_case else ['--healthy-only'])
                if fault_case:
                    selected.append(config)
            for config in selected:
                for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
                    replay_typed(config, 'typed-' + config['frontier_profile'] + '-banks-' +
                        str(config['capacities']['route_slot_count']) + '-' + tool,
                        ['--healthy-only', '--instrument-processes', tool])
            for index, config in enumerate(typed_reuse_configs(base)):
                replay_typed(config, 'typed-reuse-' + str(index),
                    ['--healthy-only', '--unitriangular-modulus', '2', '--require-bank-reuse'])
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
            if "test result: ok. 1 passed; 0 failed" not in oracle:
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
            report.update(scope="paired physical 2xT4 S10; native archive mandatory, CayleyPy no archive",
                          baseline_commit=baseline_commit, rows=[])
            save()
            expected = None
            samples = {"native": [], "cayleypy": []}
            for repeat in range(5):
                for backend in (("native", "cayleypy") if repeat % 2 == 0
                                else ("cayleypy", "native")):
                    label = f"paired-s10-{backend}-r{repeat}"
                    if backend == "native":
                        archive_root = work / label
                        case_env = dict(env, MGBFS_TRANSPORT_BACKEND="NCCL_LSA",
                                        MGBFS_SHARDS="4", MGBFS_BUCKETS="256",
                                        MGBFS_ARCHIVE_SLOTS="256")
                        result = run_case(str(source / "target/release/mgbfs"),
                                          logs / label, archive_root, "s10", 3_628_800,
                                          2, 32768, 1_000_000, 1_000_000, 96 << 20,
                                          "DENSE", "ON", case_env, owner="CUCO_RANK")
                        row = result["measurement"]
                        for rank in range(2):
                            (archive_root / f"archive-rank-{rank}.mgbfsar1").unlink()
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
                if "test result: ok. 1 passed; 0 failed" not in checked:
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
                if "test result: ok. 1 passed; 0 failed" not in checked:
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
            if "test result: ok. 1 passed; 0 failed" not in plain:
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
                            "test_passed": "test result: ok. 1 passed; 0 failed" in output,
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
                if ("test result: ok. 1 passed; 0 failed" not in checked
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
                if "test result: ok. 1 passed; 0 failed" not in result:
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
                if "test result: ok. 1 passed; 0 failed" not in result:
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
                    "oracle_passed": "test result: ok. 1 passed; 0 failed" in output,
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
            if "test result: ok. 1 passed; 0 failed" not in result:
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
                if "test result: ok. 1 passed; 0 failed" not in result:
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
        if "test result: ok. 1 passed; 0 failed" not in result:
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
                if "test result: ok. 1 passed; 0 failed" not in result:
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
        if "test result: ok. 1 passed; 0 failed" not in sanitized or \
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
