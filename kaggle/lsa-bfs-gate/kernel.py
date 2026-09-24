"""Exact-source, two-T4 CUCO_RANK/LSA full-state BFS correctness gate."""
import hashlib
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE = "d8df7db1e31662072e3609407efd88fb5b22ccbd"
CUCO = "532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4"
MODE = "nccl_window_isolation"


def main():
    work = Path(tempfile.mkdtemp(prefix="mgbfs-lsa-bfs-", dir="/tmp"))
    logs = Path("/kaggle/working/lsa-bfs-gate")
    logs.mkdir(parents=True, exist_ok=True)
    report = {"source": SOURCE, "status": "INCOMPLETE", "scope":
              ("two physical T4; one-rank archive slot exhaustion before exchange"
               if MODE == "archive_fault_gate" else
               "two physical T4; NCCL LSA; CUCO_RANK DENSE full layer sets and archives")}

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
    env["PIP_DEFAULT_TIMEOUT"] = "300"
    env["PIP_RETRIES"] = "5"

    def run(command, name, cwd=source, timeout=900):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    try:
        report["gpus"] = gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"))
        cudart = ctypes.CDLL("libcudart.so.12")
        p2p = []
        for source_gpu, target_gpu in ((0, 1), (1, 0)):
            allowed = ctypes.c_int()
            rc = cudart.cudaDeviceCanAccessPeer(ctypes.byref(allowed), source_gpu, target_gpu)
            p2p.append({"source": source_gpu, "target": target_gpu,
                        "cuda_status": rc, "allowed": allowed.value})
        report["p2p"] = p2p
        if MODE != "device_fatal_gate" and any(
                row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
            report["status"] = "UNSUPPORTED_HOST"
            return
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        for component, version, digest in library.CUDA_COMPONENTS:
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
        venv = work / "venv"
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        run([sys.executable, "-m", "pip", "--python", python, "install",
             "--only-binary=:all:", "--no-cache-dir", "--require-hashes", "-r",
             str(source / "experiments/library_owner/requirements-linux-x86_64.lock")],
            "dependencies", timeout=1200)
        site = subprocess.check_output([python, "-c", "import site; print(site.getsitepackages()[0])"],
                                       text=True, env=env).strip()
        site = Path(site)
        prefixes = library.cmake_prefixes(site)
        libdirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        nccl_target = work / "nccl"
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
             str(nccl_target), "nvidia-nccl-cu12==2.29.7"], "nccl-install")
        nccl = nccl_target / "nvidia/nccl"
        if not (nccl / "include/nccl_device.h").is_file():
            raise RuntimeError("PINNED_NCCL_DEVICE_HEADER")
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        env["LD_LIBRARY_PATH"] = ":".join([str(nccl / "lib"), str(sdk / "lib"),
                                            *libdirs, env.get("LD_LIBRARY_PATH", "")])
        env["MGBFS_CUDART_LIB_DIR"] = str(sdk / "lib")
        if MODE == "nccl_window_isolation":
            binary = work / "nccl-window-isolation"
            run([str(sdk / "bin/nvcc"), "-std=c++17", "-arch=sm_75",
                 "-I" + str(nccl / "include"), "-L" + str(nccl / "lib"),
                 str(source / "experiments/nccl_window_isolation.cu"),
                 "-Wl,-rpath," + str(nccl / "lib"), "-lnccl", "-lcudart",
                 "-o", str(binary)], "window-isolation-build", timeout=600)
            report["scope"] = ("independent NCCL ncclMemAlloc and "
                               "ncclCommWindowRegister on two physical T4s; no BFS code")
            report["window_runs"] = {}
            for label, command in (
                ("plain", [str(binary)]),
                ("initcheck", ["compute-sanitizer", "--tool", "initcheck",
                               "--report-api-errors", "no", "--error-exitcode", "97",
                               str(binary)]),
            ):
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
            report["status"] = "DIAGNOSTIC_COMPLETE"
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
             "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
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
             "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUTLASS_ROOT=" + str(cutlass), "-DMGBFS_NCCL_LSA=ON",
             "-DMGBFS_NCCL_ROOT=" + str(nccl)], "native-configure")
        run(["cmake", "--build", str(native), "--target", "mgbfs_cuda", "-j2"],
            "native-build", timeout=1800)
        env["MGBFS_CUDA_LIB_DIR"] = str(native)
        env["LD_LIBRARY_PATH"] = str(native) + ":" + env["LD_LIBRARY_PATH"]
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
            nsys = None
            if MODE in ("timeline", "timeline_backtrace", "timeline_analysis"):
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
                    result = run_case(cli, logs / label, work / label, "s10",
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
                        database = logs / label / "timeline.sqlite"
                        run([nsys, "export", "--type", "sqlite", "--force-overwrite=true",
                             "--output", str(database), result["trace"]],
                            label + "-nsys-export", timeout=600)
                        run([sys.executable, str(source / "scripts/nsys_sync_callsites.py"),
                             str(database), str(logs / label / "sync-callsites.json")],
                            label + "-sync-callsites", timeout=600)
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
            ):
                result = run([str(binaries[0]), test_name,
                              "--exact", "--nocapture", "--test-threads=1"],
                             test_name, timeout=1800)
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
