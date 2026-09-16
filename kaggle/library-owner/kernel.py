"""Pinned libcudf and full BFS correctness gates; NOT a speed benchmark."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import hashlib
import shutil

SOURCE_COMMIT = "007da906164504c56d1bd5d7ddde14ca63b8e7a8"
FULL_BFS_GATE = True
LOAD_SCREEN = True
SCREEN_REPEATS = 5
SCREEN_OWNERS = ("CUCO_INDEXED",)  # cuDF full correctness still runs below.
SCREEN_WORLDS = (1, 2)
SCREEN_CAPACITY = 1_000_000  # Explicit per-rank capacity, not inferred at runtime.
SCREEN_RING = 1_000_000
SCREEN_POOL_BYTES = 96 << 20  # Explicit admission experiment, never grow/fallback.
SCREEN_ARCHIVE_SLOTS = 256  # Same fixed pinned capacity for every timed backend.
NATIVE_COMPARISON = True
CUCO_PREVIOUS_COMMIT = None  # Optional same-session library-only A/B.
PROFILE_SCREEN = False  # Diagnostic timelines only; never enter speed statistics.
NATIVE_BASELINE_COMMIT = "013ed5c979f4225db273e0015fa9ed72fd230c90"
CUCO_GATE = True
# Recheck pool diagnostics, then compare against the immutable native baseline.
SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
PRIOR_SANITIZER_EVIDENCE = {"kernel_version": 50,
    "source_commit": "05efe4cff5f315e5dbdd2fb7c2435ec4d2638e96"}
CUCO_COMMIT = "532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4"
PACKAGES = ["libcudf-cu12==26.4.0", "librmm-cu12==26.4.0",
            "cmake==3.31.6", "ninja==1.11.1.4"]
# NVIDIA redistrib_12.9.1.json, linux-x86_64. Downloaded on Kaggle only.
CUDA_COMPONENTS = [
    ("cuda_nvcc", "12.9.86", "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35"),
    ("cuda_cudart", "12.9.79", "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99"),
    ("cuda_cccl", "12.9.27", "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa"),
    ("cuda_nvrtc", "12.9.86", "82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af"),
]


def isolated_environment(inherited):
    env = dict(inherited)
    # Kaggle injects a sitecustomize via PYTHONPATH that imports host-only wrapt.
    # It must not load into the isolated library toolchain or pollute path output.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def cmake_prefixes(site):
    # Ubuntu does not necessarily search lib64 under every prefix. Include the
    # actual exported-config directories, including bundled CCCL dependencies.
    paths = {str(p) for p in site.iterdir() if p.is_dir()}
    for pattern in ("*-config.cmake", "*Config.cmake"):
        paths.update(str(p.parent) for p in site.rglob(pattern) if p.is_file())
    return sorted(paths)


def main():
    logs = Path("/kaggle/working/library-owner")
    logs.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mgbfs-library-", dir="/tmp"))
    source = work / "source"
    # Public immutable source; no GitHub/HF secrets are read or forwarded.
    subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                    "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(source)],
                   check=True, timeout=180)
    subprocess.run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=source,
                   check=True, timeout=180)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError("Source commit mismatch")
    spec = importlib.util.spec_from_file_location("gate", source / "kaggle/native-primitives/kernel.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    gpus = gate.validate_gpus(inventory)
    manifest = {"source_commit": actual, "packages_requested": PACKAGES,
                "gpus": gpus, "kind": "library_characterization", "status": "RUNNING",
                "sanitizer_tools": list(SANITIZER_TOOLS),
                "prior_sanitizer_evidence": PRIOR_SANITIZER_EVIDENCE}
    summary_path = logs / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2))
    env = isolated_environment(os.environ)
    def run(command, name, timeout=900, extra_env=None):
        return gate.run(command, cwd=source, env=env if extra_env is None else extra_env,
                        logs=logs, name=name, timeout=timeout)
    try:
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        manifest["cuda_components"] = CUDA_COMPONENTS
        for component, version, sha256 in CUDA_COMPONENTS:
            name = f"{component}-linux-x86_64-{version}-archive"
            archive = work / (name + ".tar.xz")
            url = f"https://developer.download.nvidia.com/compute/cuda/redist/{component}/linux-x86_64/{archive.name}"
            run(["curl", "--fail", "--location", "--max-time", "180", url,
                 "--output", str(archive)], component + "-download")
            with archive.open("rb") as package:
                actual_sha = hashlib.file_digest(package, "sha256").hexdigest()
            if actual_sha != sha256:
                raise RuntimeError(f"CUDA archive checksum mismatch: {component}")
            run(["tar", "-xf", str(archive), "-C", str(work)], component + "-extract")
            shutil.copytree(work / name, sdk, dirs_exist_ok=True)
        # Linux redistributables store libraries in lib; nvcc.profile expects
        # the conventional toolkit lib64 layout. Both names refer to this SDK.
        (sdk / "lib64").symlink_to("lib", target_is_directory=True)
        if not (sdk / "lib64/libcudart_static.a").is_file():
            raise RuntimeError("Pinned CUDA SDK missing static runtime")
        env["PATH"] = str(sdk / "bin") + ":" + env["PATH"]
        env["CUDACXX"] = str(sdk / "bin/nvcc")
        run([str(sdk / "bin/nvcc"), "--version"], "cuda-version")
        venv = work / "venv"
        # Kaggle's ensurepip bootstrap failed in v1. Use the host pip's supported
        # --python entry point; the target environment remains fully isolated.
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        pip = [sys.executable, "-m", "pip", "--python", python]
        requirements = source / "experiments/library_owner/requirements-linux-x86_64.lock"
        manifest["requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        run([*pip, "install", "--only-binary=:all:", "--no-cache-dir", "--require-hashes",
             "--report", str(logs / "pip-install.json"), "-r", str(requirements)], "install", 900)
        run([*pip, "freeze", "--all"], "packages")
        run([*pip, "show", "-f", "libcudf-cu12", "librmm-cu12", "rapids-logger"],
            "sdk-inventory")
        # The normal log helper deliberately merges stdout/stderr. Never parse
        # its result as a path: even a nonfatal sitecustomize diagnostic corrupts
        # that protocol. Preserve diagnostics separately instead of hiding them.
        site_query = subprocess.run(
            [python, "-c", "import site; print(site.getsitepackages()[0])"],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=30)
        (logs / "site-stderr.log").write_text(site_query.stderr)
        (logs / "site-stdout.log").write_text(site_query.stdout)
        site = Path(site_query.stdout.strip())
        prefixes = cmake_prefixes(site)
        lib_dirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        env["LD_LIBRARY_PATH"] = ":".join([str(sdk / "lib"), str(sdk / "lib64")] + lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        build = work / "build"
        cuco_options = []
        if CUCO_GATE:
            cuco = work / "cuco"
            gate.checkout("https://github.com/NVIDIA/cuCollections.git", CUCO_COMMIT,
                          cuco, env, logs, "cuco")
            manifest["cuco_commit"] = CUCO_COMMIT
            cuco_options = ["-DCUCO_ROOT=" + str(cuco)]
        run([str(venv / "bin/cmake"), "-S", str(source / "experiments/library_owner"),
             "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DCUDAToolkit_ROOT=" + str(sdk), "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
             *cuco_options], "configure")
        run([str(venv / "bin/cmake"), "--build", str(build), "-j2"], "build")
        run([str(build / "owner_abi_invalid")], "abi-invalid-handle")
        run([str(venv / "bin/ctest"), "--test-dir", str(build),
             "--output-on-failure", "-R", "^cuco_workspace_lease_"], "workspace-lease-contract")
        # Build the real Rust adapter without pulling the unrelated native BFS
        # library into this ABI gate. All data-plane calls remain native CUDA.
        env["CARGO_HOME"] = str(work / "cargo")
        env["RUSTUP_HOME"] = str(work / "rustup")
        installer = work / "rustup-init.sh"
        run(["curl", "--fail", "--location", "--max-time", "180",
             "https://sh.rustup.rs", "-o", str(installer)], "rust-download")
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal",
             "--default-toolchain", gate.RUST_VERSION], "rust-install")
        env["PATH"] = str(work / "cargo/bin") + ":" + env["PATH"]
        env["MGBFS_LIBRARY_OWNER_LIB_DIR"] = str(build)
        env["MGBFS_CUDART_LIB_DIR"] = str(sdk / "lib")
        env.pop("MGBFS_CUDA_LIB_DIR", None)
        env["LD_LIBRARY_PATH"] = str(build) + ":" + env["LD_LIBRARY_PATH"]
        run(["rustc", "--version", "--verbose"], "rust-version")
        artifacts = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                         "--features", "library-owner", "--test", "library_native_gpu",
                         "--no-run", "--message-format=json"], "rust-build")
        executables = []
        for line in artifacts.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (record.get("reason") == "compiler-artifact"
                    and record.get("target", {}).get("name") == "library_native_gpu"
                    and record.get("executable")):
                executables.append(record["executable"])
        if len(executables) != 1:
            raise RuntimeError("Expected one Rust GPU test executable")
        bfs_executable = None
        multi_executable = None
        if FULL_BFS_GATE:
            cutlass = work / "cutlass"
            gate.checkout("https://github.com/NVIDIA/cutlass.git", gate.CUTLASS_COMMIT,
                          cutlass, env, logs, "cutlass")
            manifest["cutlass_commit"] = gate.CUTLASS_COMMIT
            native_build = work / "native-build"
            run([str(venv / "bin/cmake"), "-S", str(source / "cuda"),
                 "-B", str(native_build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_TESTING=OFF", "-DCMAKE_CUDA_ARCHITECTURES=75",
                 "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
                 "-DCUTLASS_ROOT=" + str(cutlass)], "native-configure")
            run([str(venv / "bin/cmake"), "--build", str(native_build),
                 "--target", "mgbfs_cuda", "-j2"], "native-build")
            env["MGBFS_CUDA_LIB_DIR"] = str(native_build)
            env["LD_LIBRARY_PATH"] = str(native_build) + ":" + env["LD_LIBRARY_PATH"]
            artifacts = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                             "--features", "cuda,library-owner", "--test", "library_bfs_gpu",
                             "--test", "library_multi_gpu",
                             "--no-run", "--message-format=json"], "rust-bfs-build")
            matches = {}
            for line in artifacts.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("reason") == "compiler-artifact"
                        and record.get("target", {}).get("name") in ("library_bfs_gpu", "library_multi_gpu")
                        and record.get("executable")):
                    matches[record["target"]["name"]] = record["executable"]
            if set(matches) != {"library_bfs_gpu", "library_multi_gpu"}:
                raise RuntimeError("Full BFS test executable inventory mismatch")
            bfs_executable = matches["library_bfs_gpu"]
            multi_executable = matches["library_multi_gpu"]
        executable = str(build / "cudf_owner_probe")
        for gpu in gpus:
            device_env = dict(env, CUDA_VISIBLE_DEVICES=gpu["uuid"])
            for tool in ("plain", *SANITIZER_TOOLS):
                transfer_command = [str(build / "control_transfer_contract")]
                if tool != "plain":
                    transfer_command = ["compute-sanitizer", "--tool", tool,
                                        "--error-exitcode", "97", *transfer_command]
                run(transfer_command, f"control-transfer-gpu{gpu['index']}-{tool}", extra_env=device_env)
                gather_command = [str(build / "owner_source_gather")]
                if tool != "plain":
                    gather_command = ["compute-sanitizer", "--tool", tool,
                                      "--error-exitcode", "97", *gather_command]
                run(gather_command, f"source-gather-gpu{gpu['index']}-{tool}", extra_env=device_env)
                if CUCO_GATE:
                    cuco_command = [str(build / "cuco_pool_probe")]
                    if tool != "plain":
                        cuco_command = ["compute-sanitizer", "--tool", tool,
                                        "--error-exitcode", "97", *cuco_command]
                    run(cuco_command, f"cuco-gpu{gpu['index']}-{tool}", extra_env=device_env)
                    owner_command = [str(build / "cuco_owner_probe")]
                    if tool != "plain":
                        owner_command = ["compute-sanitizer", "--tool", tool,
                                         "--error-exitcode", "97", *owner_command]
                    run(owner_command, f"cuco-owner-gpu{gpu['index']}-{tool}", extra_env=device_env)
                    abi_command = [str(build / "cuco_owner_abi_contract")]
                    if tool != "plain":
                        abi_command = ["compute-sanitizer", "--tool", tool,
                                       "--error-exitcode", "97", *abi_command]
                    run(abi_command, f"cuco-abi-gpu{gpu['index']}-{tool}", extra_env=device_env)
                command = [executable] if tool == "plain" else [
                    "compute-sanitizer", "--tool", tool, "--error-exitcode", "97", executable]
                run(command, f"gpu{gpu['index']}-{tool}", extra_env=device_env)
                rust_command = [executables[0], "--test-threads=1"]
                if tool != "plain":
                    rust_command = ["compute-sanitizer", "--tool", tool,
                                    "--error-exitcode", "97", *rust_command]
                run(rust_command, f"rust-gpu{gpu['index']}-{tool}", extra_env=device_env)
                if bfs_executable is not None:
                    command = [bfs_executable, "--test-threads=1"]
                    if tool != "plain":
                        command = ["compute-sanitizer", "--tool", tool,
                                   "--error-exitcode", "97", *command]
                    run(command, f"bfs-gpu{gpu['index']}-{tool}", extra_env=device_env)
            manifest["full_bfs_gate"] = bfs_executable is not None
        if multi_executable is not None:
            device_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(gpu["uuid"] for gpu in gpus))
            for tool in ("plain", *SANITIZER_TOOLS):
                command = [multi_executable, "--test-threads=1"]
                if tool != "plain":
                    command = ["compute-sanitizer", "--tool", tool,
                               "--error-exitcode", "97", *command]
                run(command, f"bfs-two-gpu-{tool}", extra_env=device_env)
            manifest["two_gpu_gate"] = "two device threads with NCCL; not torchrun processes"
            # Actual process-launch contract, with mandatory file archives and a
            # separate warmup pass. Tiny S4 is a correctness gate, not a speed result.
            run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
                 "--features", "library-owner"], "cli-build")
            cli = str(source / "target/release/mgbfs")
            scenarios = [
                    ("s4", 24, "DENSE", "SCALAR"),
                    ("s4", 24, "HASH_FIRST", "INT_MMA_SM75"),
                    ("u4m2", 64, "DENSE", "SCALAR")]
            for owner, group, expected_count, profile, generation in [
                    (owner, *scenario) for owner in ("CUDF_RELATIONAL", "CUCO_INDEXED")
                    for scenario in scenarios]:
                label = owner.lower() + "-" + group + "-" + profile.lower()
                run_root = work / ("cli-" + label)
                run_root.mkdir()
                output_dir = logs / ("cli-" + label)
                process_env = dict(device_env, MGBFS_OWNER_BACKEND=owner,
                    MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE=profile,
                    MGBFS_HASH_FIRST_GENERATION=generation, MGBFS_BENCH_CAPACITY="64",
                    MGBFS_FUTURE_CAPACITY="128", MGBFS_BUCKETS="8", MGBFS_SHARDS="4",
                    MGBFS_JOB_BUCKETS="2", MGBFS_BUCKET_CAPACITY="32",
                    MGBFS_STATE_CODEC="matrix_u8", MGBFS_ARCHIVE_CODEC="matrix_u8",
                    MGBFS_ARCHIVE_ROWS="3", MGBFS_ARCHIVE_SLOTS="128",
                    MGBFS_BENCH_WARMUP="1", MGBFS_PRE_DEDUP="ON",
                    MGBFS_BENCH_SKIP_ARCHIVE="0", MGBFS_ARCHIVE_STREAM="0",
                    MGBFS_CAPACITY_MODE="max_per_rank", MGBFS_RANK_MAP="0,1")
                run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                     "--nproc-per-node=2", "--no-python", cli, "bench", "--reference",
                     group, "7", str(run_root / "bootstrap"), str(run_root / "archive"),
                     str(output_dir)], "cli-" + label, extra_env=process_env)
                total = 0
                for rank in range(2):
                    result = json.loads((output_dir / f"rank-{rank}.json").read_text())
                    if (result["status"] != "COMPLETE" or result["owner_backend"] != owner
                            or result["library_pool_reserved_bytes"] != (64 << 20)
                            or result["group"] != group or not result["backend"].startswith("library_")
                            or not result["archive_enabled"] or not result["warmup_completed"]):
                        raise RuntimeError("CLI library dispatch/archive contract mismatch")
                    total += sum(result["local_layer_sizes"])
                    run([cli, "verify", str(run_root / f"archive-rank-{rank}.mgbfsar1")],
                        f"cli-{label}-verify-rank{rank}", extra_env=process_env)
                if total != expected_count:
                    raise RuntimeError(f"CLI {group} count mismatch: {total}")
            manifest["cli_gate"] = "torchrun two-process DENSE and HASH_FIRST Tensor; S4/U4m2 correctness only"
            if LOAD_SCREEN:
                manifest["screen_archive_slots"] = SCREEN_ARCHIVE_SLOTS
                sys.path.insert(0, str(source / "scripts"))
                from library_gpu_screen import run_case
                from distributed_gpu_bench import stats
                nsys = None
                if PROFILE_SCREEN:
                    # Official Ubuntu package; unpack into this job's private /tmp tree.
                    package_name = "nsight-systems-2025.3.2_2025.3.2.474-1_amd64.deb"
                    package_sha = "c7cfe27e2250eb91e1a67e7feb5f2c490c7f598e3b3a3d047aff000bc49f9d6b"
                    package = work / package_name
                    run(["curl", "--fail", "--location", "--max-time", "300",
                         "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/" + package_name,
                         "--output", str(package)], "nsys-download")
                    with package.open("rb") as downloaded:
                        digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
                    if digest != package_sha:
                        raise RuntimeError("NSYS_DIGEST_MISMATCH")
                    nsys_root = work / "nsys"
                    run(["dpkg-deb", "--extract", str(package), str(nsys_root)], "nsys-extract")
                    candidates = list(nsys_root.glob("opt/nvidia/nsight-systems/*/target-linux-x64/nsys"))
                    if len(candidates) != 1:
                        raise RuntimeError("NSYS_EXECUTABLE_INVENTORY")
                    nsys = str(candidates[0])
                    run([nsys, "--version"], "nsys-version")
                    manifest["profiler"] = dict(package=package_name, sha256=digest,
                        scope="diagnostic only; includes process startup, warmup, BFS and archive")
                if NATIVE_COMPARISON:
                    preserved = work / "preserved-native"
                    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git",
                                  NATIVE_BASELINE_COMMIT, preserved, env, logs, "preserved-native")
                    preserved_build = work / "preserved-native-build"
                    preserved_env = dict(device_env, MGBFS_CUDA_LIB_DIR=str(preserved_build),
                        LD_LIBRARY_PATH=str(preserved_build) + ":" + device_env["LD_LIBRARY_PATH"])
                    preserved_env.pop("MGBFS_LIBRARY_OWNER_LIB_DIR", None)
                    def preserved_run(command, name):
                        return gate.run(command, cwd=preserved, env=preserved_env,
                                        logs=logs, name=name, timeout=1800)
                    preserved_run([str(venv / "bin/cmake"), "-S", "cuda", "-B", str(preserved_build),
                        "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
                        "-DCMAKE_CUDA_ARCHITECTURES=75",
                        "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
                        "-DCUTLASS_ROOT=" + str(cutlass)], "preserved-configure")
                    preserved_run([str(venv / "bin/cmake"), "--build", str(preserved_build),
                                   "--target", "mgbfs_cuda", "-j2"], "preserved-cuda-build")
                    preserved_run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime",
                                   "--features", "cuda", "--example", "distributed_bench"], "preserved-rust-build")
                    preserved_run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli"],
                                  "preserved-cli-build")
                    manifest["native_baseline_commit"] = NATIVE_BASELINE_COMMIT
                panel = {}
                if CUCO_PREVIOUS_COMMIT:
                    previous_source = work / "previous-owner"
                    previous_build = work / "previous-owner-build"
                    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git",
                                  CUCO_PREVIOUS_COMMIT, previous_source, env, logs, "previous-owner")
                    run([str(venv / "bin/cmake"), "-S", str(previous_source / "experiments/library_owner"),
                         "-B", str(previous_build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                         "-DCUDAToolkit_ROOT=" + str(sdk), "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
                         "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
                         *cuco_options], "previous-owner-configure")
                    run([str(venv / "bin/cmake"), "--build", str(previous_build),
                         "--target", "mgbfs_library_owner", "-j2"], "previous-owner-build")
                    previous_env = dict(device_env,
                        LD_LIBRARY_PATH=str(previous_build) + ":" + device_env["LD_LIBRARY_PATH"])
                    linkage = subprocess.check_output(["ldd", cli], env=previous_env, text=True)
                    (logs / "previous-owner-linkage.log").write_text(linkage)
                    if str(previous_build / "libmgbfs_library_owner.so") not in linkage:
                        raise RuntimeError("PREVIOUS_OWNER_LINKAGE_MISMATCH")
                    manifest["previous_owner_commit"] = CUCO_PREVIOUS_COMMIT
                for world in SCREEN_WORLDS:
                    for repeat in range(SCREEN_REPEATS):
                        owners = ("CUCO_INDEXED",) if PROFILE_SCREEN else SCREEN_OWNERS
                        if CUCO_PREVIOUS_COMMIT:
                            owners += ("CUCO_PREVIOUS",)
                        if NATIVE_COMPARISON:
                            owners += ("CUB_SORT_MERGE",)
                        if repeat % 2:
                            owners = owners[::-1]
                        for owner in owners:
                            label = f"screen-s10-dense-{owner.lower()}-w{world}-r{repeat}"
                            native = owner == "CUB_SORT_MERGE"
                            case_cli = str(preserved / "target/release/mgbfs") if native else cli
                            extra = {"native_example": str(preserved / "target/release/examples/distributed_bench")} if native else {}
                            if PROFILE_SCREEN:
                                extra["nsys"] = nsys
                            selected_env = previous_env if owner == "CUCO_PREVIOUS" else device_env
                            case_env = dict(preserved_env if native else selected_env,
                                            MGBFS_ARCHIVE_SLOTS=str(SCREEN_ARCHIVE_SLOTS))
                            result = run_case(case_cli, logs / label, work / label,
                                "s10", 3628800, world, 32768, SCREEN_CAPACITY, SCREEN_RING,
                                0 if native else SCREEN_POOL_BYTES, "DENSE", "ON",
                                case_env, owner="CUCO_INDEXED" if owner == "CUCO_PREVIOUS" else owner, **extra)
                            panel.setdefault(f"{owner}-w{world}", []).append(result["measurement"])
                            if PROFILE_SCREEN:
                                run([nsys, "stats", "--report", "cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum",
                                     "--format", "csv", result["trace"]], label + "-nsys-stats", timeout=600)
                if PROFILE_SCREEN:
                    manifest["profile_runs"] = {key: len(rows) for key, rows in panel.items()}
                    manifest["load_screen"] = "S10 DENSE diagnostic timelines; NOT speed or VRAM acceptance evidence"
                else:
                    manifest["screen_statistics"] = {key: stats(rows) for key, rows in panel.items()}
                    manifest["load_screen"] = "S10 DENSE owners and preserved native; matched matrix settings; tuned Pareto acceptance pending"
        manifest["status"] = "PASS"
    except Exception as error:
        manifest.update(status="FAILED", error=str(error))
        raise
    finally:
        summary_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
