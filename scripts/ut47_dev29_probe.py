"""Bounded dev29 UT(4,47) eight-GPU reproduction; no library modifications."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile

WHEEL = "https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261017-v29/multigpubfs-0.2.0.dev29-py3-none-any.whl#sha256=ab5d6452dd1588bf4c1a003ad911b4cea03d3e6e61e7a6ea3cc18d2d40bf7a51"
GRAPH_URL = "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/e5001ed173fbd28c0422515efc77959f99442c92/docs/acceptance/cuda999-ut47-graph.json"
GRAPH_SHA = "3d32098284aa9129e94a76f81c37900b68f08884482cf67795c9f25654da49c9"
GRAPH_DIGEST = "ff71582e6018295a7b1d23605f350623df9059f4e3d595c25ec29a75f9666e7e"
NATIVE_SHA = "10e423aee53f450d5a14f064db0cb1c7784e8080e7b94045adbad378ad304fce"
EXPECTED_LAYERS = [1, 6, 26, 110, 414]
PROBE_TIMEOUT = 300

def write_json(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

def phase(root, name):
    write_json(root, "supervisor-phase.json", {"phase": name, "at_unix": time.time(), "pid": os.getpid()})
    print("PHASE", name, flush=True)

def capture(root, name, command, env=None):
    with (root / (name + ".log")).open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=10)
            log.write("\nEXIT_CODE " + str(result.returncode) + "\n")
        except Exception as error:
            log.write("\n" + type(error).__name__ + ": " + str(error) + "\n")

def worker(root):
    phase(root, "worker-start")
    write_json(root, "environment.json", {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
        "LD_LIBRARY_PATH", "LD_PRELOAD", "MGBFS_EXECUTABLE",
        "MGBFS_NATIVE_CACHE", "MGBFS_NATIVE_OFFLINE", "WORLD_SIZE")})
    os.environ.pop("MGBFS_EXECUTABLE", None)
    for key in list(os.environ):
        if key.startswith("MGBFS_TEST_"):
            del os.environ[key]
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MGBFS_NATIVE_OFFLINE"] = "0"
    live = root / "session-live"
    live.mkdir()
    os.environ["MGBFS_GENERIC_SESSION_LOG_ROOT"] = str(live)
    phase(root, "package-import")
    import multigpubfs
    from multigpubfs import GraphDefinition, run_graph, persistent_graph_workers
    from multigpubfs.native_distribution import native_runtime, runtime_identity
    version = importlib.metadata.version("multigpubfs")
    write_json(root, "package.json", {
        "version": version, "path": multigpubfs.__file__, "python": sys.version,
        "executable": sys.executable, "cpu_affinity": sorted(os.sched_getaffinity(0))})
    if version != "0.2.0.dev29":
        raise RuntimeError("Expected multigpubfs 0.2.0.dev29")
    phase(root, "graph-download-and-validation")
    with urllib.request.urlopen(GRAPH_URL, timeout=30) as response:
        raw = response.read(1 << 20)
    if hashlib.sha256(raw).hexdigest() != GRAPH_SHA:
        raise RuntimeError("Graph checksum mismatch")
    (root / "graph.json").write_bytes(raw)
    # The published payload IS the schema-2 definition; it has no graph wrapper.
    graph = GraphDefinition.from_dict(json.loads(raw))
    if graph.digest() != GRAPH_DIGEST:
        raise RuntimeError("UT(4,47) semantic digest mismatch")
    write_json(root, "graph-validation.json", {
        "status": "PASS", "digest": graph.digest(), "expected_layers": EXPECTED_LAYERS,
        "devices": list(range(8)), "original_candidate_geometry": True})
    phase(root, "hardware-evidence")
    capture(root, "gpu-info", ["nvidia-smi", "-q"])
    capture(root, "topology", ["nvidia-smi", "topo", "-m"])
    phase(root, "native-preparation")
    native, env = native_runtime()
    identity = runtime_identity(native, env)
    write_json(root, "runtime.json", {"native": native, "identity": identity})
    if identity["native_sha256"] != NATIVE_SHA:
        raise RuntimeError("Unexpected native executable")
    os.environ["MGBFS_NATIVE_OFFLINE"] = "1"
    env.update(
        MGBFS_NATIVE_OFFLINE="1", MGBFS_GENERIC_HISTORY="hash",
        MGBFS_GENERIC_GENERATOR="cuda", MGBFS_GENERIC_TRANSPORT="full",
        MGBFS_GENERIC_SORT="none", MGBFS_TRACE_NATIVE_CALLS="1",
        NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="ALL",
        NCCL_DEBUG_FILE=str(root / "nccl-%h-%p.log"))
    write_json(root, "effective-settings.json", {key: env.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING",
        "LD_LIBRARY_PATH", "LD_PRELOAD", "MGBFS_NATIVE_CACHE",
        "MGBFS_GENERIC_HISTORY", "MGBFS_GENERIC_GENERATOR", "MGBFS_GENERIC_TRANSPORT",
        "MGBFS_GENERIC_SORT", "MGBFS_TRACE_NATIVE_CALLS", "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL", "NCCL_CUMEM_ENABLE", "NCCL_CUMEM_HOST_ENABLE")})
    capture(root, "dependencies", ["ldd", native], env)
    checks = []
    pids = None
    phase(root, "persistent-pool")
    with persistent_graph_workers() as pool:
        try:
            for repeat in range(2):
                phase(root, "ut47-eight-rank-probe-" + str(repeat))
                started = time.monotonic()
                report = run_graph(
                    graph, root / ("run-" + str(repeat)),
                    devices=list(range(8)), capacity=3080192, max_seconds=120,
                    autotune=False, backend="generic", transport="full",
                    candidate_order="none", shards=1, _batch=174762,
                    _profile_layers=4, _native_env=env)
                current = report.get("rank_worker_pids") or []
                reference = {"world": 8, "capacity": 3080192, "batch": 174762,
                             "table_slots": 33554432, "shards": 1, "history_layers": 3}
                if (report["layer_sizes"] != EXPECTED_LAYERS
                    or report["status"] != "INCOMPLETE"
                    or report["reason"] != "PROFILE_LAYER_LIMIT"
                    or any(report["plan"].get(key) != value for key, value in reference.items())
                    or len(current) != 8 or len(set(current)) != 8
                    or (pids is not None and current != pids)):
                    raise RuntimeError("Probe validation failed; inspect run report")
                checks.append({
                    "repeat": repeat, "status": report["status"], "reason": report["reason"],
                    "layers": report["layer_sizes"], "rank_worker_pids": current,
                    "bfs_seconds": report.get("bfs_seconds"),
                    "wall_seconds": time.monotonic() - started, "plan": report["plan"]})
                write_json(root, "checks.json", checks)
                pids = current
        finally:
            # Preserve partial evidence before cleanup, and final logs after shutdown.
            shutil.copytree(pool.root, root / "session-evidence",
                            ignore=shutil.ignore_patterns("job-*"), dirs_exist_ok=True)
            pool.close_workers()
            shutil.copytree(pool.root, root / "session-evidence",
                            ignore=shutil.ignore_patterns("job-*"), dirs_exist_ok=True)
    phase(root, "probe-finished")
    write_json(root, "probe-result.json", {
        "status": "PASS_SHORT_PREFIX", "gpu_count": 8, "repeats": 2,
        "layers": EXPECTED_LAYERS, "same_eight_worker_pids": pids,
        "full_graph_computed": False})

def stop_group(child):
    if child is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass

def snapshot(root, child):
    if child is None:
        return
    try:
        from multigpubfs.deep_diagnostics import process_snapshot
        process_snapshot(child.pid, root / "process-snapshot.json", process_group=child.pid)
    except Exception as error:
        write_json(root, "snapshot-error.json", {"error": str(error)})

def bundle(root):
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and path.suffix in (".json", ".log", ".txt", ".py")
                   and not path.name.startswith("job-")
                   and "state" not in path.name.lower())
    manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}
    write_json(root, "manifest.json", manifest)
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in files + [root / "manifest.json"]:
            output.write(path, root.name + "/" + str(path.relative_to(root)))
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for relative, expected in manifest.items():
            if hashlib.sha256(check.read(root.name + "/" + relative)).hexdigest() != expected:
                raise RuntimeError("Archive SHA verification failed: " + relative)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(digest + "  " + archive.name + "\n")
    print("ARCHIVE:", archive, flush=True)
    print("ARCHIVE_SHA256:", digest, flush=True)
    return archive

def main():
    root = Path(tempfile.mkdtemp(
        prefix="ut47-dev29-" + time.strftime("%Y%m%d-%H%M%S") + "-", dir=".")).resolve()
    shutil.copyfile(Path(__file__).resolve(), root / "probe-source.py")
    child = None
    code = 1
    outcome = "FAIL"
    try:
        print("OUTPUT:", root, flush=True)
        phase(root, "install-dev29")
        with (root / "install.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                 "--no-deps", "--no-input", "--disable-pip-version-check", WHEEL],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        print("START: UT(4,47), 8 GPU together, two four-layer persistent probes", flush=True)
        with (root / "probe.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=PROBE_TIMEOUT)
            except subprocess.TimeoutExpired:
                code = 124
                outcome = "TIMEOUT"
                snapshot(root, child)
        if code == 0:
            result = json.loads((root / "probe-result.json").read_text())
            if result.get("status") != "PASS_SHORT_PREFIX":
                raise RuntimeError("Missing validated probe receipt")
            outcome = "PASS_SHORT_PREFIX"
        elif outcome != "TIMEOUT":
            snapshot(root, child)
    except KeyboardInterrupt:
        code, outcome = 130, "INTERRUPTED"
        snapshot(root, child)
        (root / "launcher-error.log").write_text(traceback.format_exc())
    except BaseException:
        code, outcome = 1, "FAIL"
        snapshot(root, child)
        (root / "launcher-error.log").write_text(traceback.format_exc())
    finally:
        stop_group(child)
        write_json(root, "summary.json", {
            "status": outcome, "exit_code": code, "gpu_count_requested": 8,
            "repeats_requested": 2, "max_depth": 4, "full_graph_computed": False,
            "scope": "exact original first-candidate pilot; no autotuning catalog or CUDA999 root-cause claim"})
        print("RESULT:", outcome, flush=True)
        if code:
            for name in ("probe.log", "launcher-error.log", "install.log"):
                path = root / name
                if path.is_file():
                    print("\n" + name + ":\n" + "\n".join(
                        path.read_text(errors="replace").splitlines()[-25:]), flush=True)
        bundle(root)
    return 0 if code == 0 else 1

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]).resolve())
    else:
        sys.exit(main())
