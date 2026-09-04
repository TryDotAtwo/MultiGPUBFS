"""GitHub-pinned T4 primitive gate; not a multi-rank BFS benchmark."""
import csv
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request

# Immutable source and toolchain configuration. No token is used.
SOURCE_COMMIT = "3dba577632f34554489e2cd52c632ef87ac90559"
CUTLASS_COMMIT = "ffa119a1255d78998536107466cc7097ecefa393"
RUST_VERSION = "1.75.0"
SANITIZERS = ("memcheck", "racecheck", "initcheck", "synccheck")

def run_gpu_suites(gpus, worker):
    # Child processes have disjoint CUDA_VISIBLE_DEVICES. This is NOT NCCL or
    # a multi-rank run. The executor drains both workers before returning/error.
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(worker, gpus))

def ping_pong_selection(tool):
    if tool == "plain":
        return (), "all"
    if tool == "racecheck":
        # All four generation variants still run through the small full-depth
        # fixture. The m2..m6 variant sweep has its own plain/other-tool coverage.
        return ("full_u4_pipelined_sweep", "generation_variants_preserve_full_layers"), "variants-m2-m3-plus-slot-reuse-and-capacity-failure"
    if tool in SANITIZERS:
        return ("full_u4_pipelined_sweep",), "variants-m2-m6-plus-small-feedback-slot-reuse-and-capacity-failure"
    raise ValueError("Unknown sanitizer")

def validate_gpus(csv_text):
    rows = []
    for row in csv.reader(io.StringIO(csv_text)):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError("GPU inventory schema")
        index, name, uuid, total, free = (item.strip() for item in row)
        rows.append(dict(index=int(index), name=name, uuid=uuid, total_mib=int(total), free_mib=int(free)))
    if len(rows) != 2 or sorted(r["index"] for r in rows) != [0, 1]:
        raise ValueError("Gate requires exactly two physical GPUs")
    if len({r["uuid"] for r in rows}) != 2:
        raise ValueError("GPU UUIDs must differ")
    if any(r["name"] not in ("Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4") or r["free_mib"] < 1024 for r in rows):
        raise ValueError("Gate requires two T4s with at least 1 GiB free each")
    return sorted(rows, key=lambda r: r["index"])

def validate_commit(commit):
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Source must be an immutable full commit")
    return commit

def run(command, *, cwd, env, logs, name, timeout=900):
    """Timeout is external to BFS; these are bounded build/primitive tests."""
    print(f"START {name}", flush=True)
    started = time.monotonic()
    # File redirection avoids a full pipe deadlocking a compiler. A heartbeat
    # distinguishes a live long compile from an unresponsive notebook worker.
    with (logs / (name + ".log")).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                print(f"RUNNING {name} {time.monotonic()-started:.0f}s", flush=True)
                if time.monotonic()-started > timeout:
                    process.kill()
                    process.wait()
                    raise RuntimeError(f"TIMEOUT: {name}")
    output = (logs / (name + ".log")).read_text(encoding="utf-8", errors="replace")
    print(output[-8000:], flush=True)
    if code:
        raise RuntimeError(f"FAILED: {name}, exit={code}")
    return output

def checkout(repo, commit, destination, env, logs, label):
    validate_commit(commit)
    destination.mkdir()
    run(["git", "init", "-q"], cwd=destination, env=env, logs=logs, name=label+"-init")
    run(["git", "remote", "add", "origin", repo], cwd=destination, env=env, logs=logs, name=label+"-remote")
    run(["git", "fetch", "--depth=1", "origin", commit], cwd=destination, env=env, logs=logs, name=label+"-fetch")
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination, env=env, logs=logs, name=label+"-checkout")
    actual = run(["git", "rev-parse", "HEAD"], cwd=destination, env=env, logs=logs, name=label+"-sha").strip()
    if actual != commit:
        raise RuntimeError("Checkout commit mismatch")

def execute_gpu_suite(gpu, executables, source, env, logs, record):
    # Bind the physical UUID, not an assumed CUDA ordinal/NVML-index mapping.
    device_env = dict(env, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    for name, executable in sorted(executables.items(), key=lambda item: (item[0] == "dense_device", item[0])):
        for tool in ("plain",) + SANITIZERS:
            label = f"gpu{gpu['index']}-{name}-{tool}"
            command = [executable, "--test-threads=1", "--nocapture"]
            fixture = "all"
            if name == "dense_device":
                fixture = "m2-m3-full-depth" if tool == "racecheck" else "m2-m6-full-depth"
                command += ["--skip", "gpu_feedback_exhausts_exact_layers_without_cpu_supplied_frontiers" if tool == "racecheck" else "gpu_feedback_small_full_depth_sanitizer_fixture"]
            if name == "ping_pong":
                skips, fixture = ping_pong_selection(tool)
                for skip in skips:
                    command += ["--skip", skip]
            if tool != "plain":
                command = ["compute-sanitizer", "--error-exitcode", "99", "--tool", tool] + command
            run(command, cwd=source, env=device_env, logs=logs, name=label, timeout=1800 if name in ("dense_device", "macro_native") else (900 if name in ("ping_pong", "generate") else 180))
            record(dict(gpu=gpu["uuid"], test=name, tool=tool, fixture=fixture, status="PASS"))

def main():
    logs = Path("/kaggle/working/native-primitive-gate")
    logs.mkdir(exist_ok=True)
    summary = dict(schema=1, scope="GPU primitives and single-bucket full-depth feedback; NOT production archived or multi-rank BFS", status="INCOMPLETE", source_commit=SOURCE_COMMIT, results=[])
    try:
        validate_commit(SOURCE_COMMIT)
        root = Path(tempfile.mkdtemp(prefix="mgbfs-gate-", dir="/tmp"))
        env = os.environ.copy()
        inventory = run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"], cwd=root, env=env, logs=logs, name="gpu-inventory")
        summary["gpus"] = validate_gpus(inventory)
        summary["disk_free_bytes"] = shutil.disk_usage(root).free
        if summary["disk_free_bytes"] < 8 * 1024**3:
            raise RuntimeError("BUILD_DISK_PREFLIGHT: less than 8 GiB free")
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
        for tool in ("git", "cmake", "ninja", "nvcc", "compute-sanitizer"):
            if not shutil.which(tool, path=env["PATH"]):
                raise RuntimeError("MISSING_TOOL: " + tool)
        run(["nvcc", "--version"], cwd=root, env=env, logs=logs, name="cuda-version")
        source, cutlass = root/"source", root/"cutlass"
        checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE_COMMIT, source, env, logs, "source")
        checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS_COMMIT, cutlass, env, logs, "cutlass")
        env["CARGO_HOME"] = str(root/"cargo")
        env["RUSTUP_HOME"] = str(root/"rustup")
        installer = root/"rustup-init.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", RUST_VERSION], cwd=root, env=env, logs=logs, name="rust-install")
        env["PATH"] = str(root/"cargo/bin") + ":" + env["PATH"]
        run(["rustc", "--version"], cwd=source, env=env, logs=logs, name="rust-version")
        run(["cargo", "test", "--locked"], cwd=source, env=env, logs=logs, name="cpu-contracts")
        build = source/"build/native-cuda"
        env["MGBFS_CUDA_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUTLASS_ROOT="+str(cutlass)], cwd=source, env=env, logs=logs, name="cmake-configure")
        run(["cmake", "--build", str(build), "--parallel", "2"], cwd=source, env=env, logs=logs, name="cuda-build")
        run(["ctest", "--test-dir", str(build), "-R", "^(allocation-query|route-query)$", "--output-on-failure"], cwd=source, env=env, logs=logs, name="allocation-queries")
        artifacts = run(["cargo", "test", "--locked", "-p", "mgbfs-cuda", "--features", "cuda", "--no-run", "--message-format=json"], cwd=source, env=env, logs=logs, name="gpu-test-build")
        artifacts += "\n" + run(["cargo", "test", "--locked", "-p", "mgbfs-runtime", "--features", "cuda", "--test", "dense_device", "--test", "ping_pong", "--test", "macro_native", "--no-run", "--message-format=json"], cwd=source, env=env, logs=logs, name="gpu-stepper-build")
        executables = {}
        for line in artifacts.splitlines():
            if not line.startswith("{"):
                continue
            entry = json.loads(line)
            if entry.get("reason") == "compiler-artifact" and entry.get("executable") and "test" in entry["target"]["kind"] and entry["target"]["name"] != "allocation_report":
                executables[entry["target"]["name"]] = entry["executable"]
        if set(executables) != {"generate", "hash", "route", "owner", "pipeline", "materialize", "future_merge", "macro_settle", "dense_device", "ping_pong", "macro_native"}:
            raise RuntimeError("GPU_TEST_INVENTORY_MISMATCH")
        inventory = run([executables["dense_device"], "--list"], cwd=source, env=env, logs=logs, name="dense-device-test-inventory")
        for fixture in ("gpu_feedback_small_full_depth_sanitizer_fixture", "gpu_feedback_exhausts_exact_layers_without_cpu_supplied_frontiers"):
            if fixture + ": test" not in inventory:
                raise RuntimeError("DENSE_DEVICE_TEST_INVENTORY_MISMATCH")
        summary["execution"] = "two-independent-concurrent-gpu-suites"
        results_lock = threading.Lock()
        def record(result):
            with results_lock:
                summary["results"].append(result)
                (logs/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run_gpu_suites(summary["gpus"], lambda gpu: execute_gpu_suite(gpu, executables, source, env, logs, record))
        summary["status"] = "PASS_PRIMITIVE_GATE"
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        (logs/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
