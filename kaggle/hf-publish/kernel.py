"""Live S8 FIFO -> bounded Parquet -> HF staging -> atomic main promotion gate."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from kaggle_secrets import UserSecretsClient

SOURCE = "3d5d287ee0b7292bef170ed3405ca3566f0d6132"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
REPO_ID = "TryDotAtwo/multigpubfs-bfs-results"
RUST = "1.75.0"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-hf-stream-", dir="/tmp"))
    logs = Path("/kaggle/working/hf-stream")
    logs.mkdir()
    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_SECRET_HF_TOKEN_EMPTY")
    helper = root / "gate.py"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
        "c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py",
        helper,
    )
    gate = load(helper, "gate")
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["HF_TOKEN"] = token

    def run(command, name, cwd=root, timeout=1800):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    inventory = run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        "inventory",
    )
    gpus = gate.validate_gpus(inventory)
    if shutil.disk_usage(root).free < 4 * 1024**3:
        raise RuntimeError("DISK_PREFLIGHT_LT_4_GIB")
    source, cutlass = root / "source", root / "cutlass"
    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
    gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS, cutlass, env, logs, "cutlass")
    env["CARGO_HOME"], env["RUSTUP_HOME"] = str(root / "cargo"), str(root / "rustup")
    installer = root / "rustup.sh"
    urllib.request.urlretrieve("https://sh.rustup.rs", installer)
    run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", RUST], "rust-install")
    env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
    env["CUDA_VISIBLE_DEVICES"] = "0"
    build = source / "build/hf-stream"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run([
        "cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUTLASS_ROOT=" + str(cutlass),
    ], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run([
        "cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda",
        "--example", "macro_bench",
    ], "rust-build", source)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=0.34"], "install-hf")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"s8-native-stream-{stamp}"
    branch = f"staging-{run_id}"
    fifo = root / "rank-0.mgbfsar1.pipe"
    os.mkfifo(fifo)
    staging = root / "rank-0-slots"
    stream_stdout = logs / "streamer.stdout.log"
    stream_stderr = logs / "streamer.stderr.log"
    stream_command = [
        sys.executable, "scripts/stream_hf_archive.py", "--run-id", run_id, "--group-id", "s8",
        "--rank", "0", "--input", str(fifo), "--staging-dir", str(staging), "--repo-id", REPO_ID,
        "--branch", branch, "--rows-per-shard", "10000", "--slot-count", "8",
        "--max-slot-bytes", str(16 * 1024**2), "--create-branch",
    ]
    with stream_stdout.open("wb") as stdout, stream_stderr.open("wb") as stderr:
        streamer = subprocess.Popen(stream_command, cwd=source, env=env, stdout=stdout, stderr=stderr)
        time.sleep(1)
        if streamer.poll() is not None:
            raise RuntimeError("STREAMER_EARLY_EXIT")
        env.update(
            MGBFS_BENCH_CAPACITY="40320",
            MGBFS_FUTURE_CAPACITY="40320",
            MGBFS_ARCHIVE_ROWS="4096",
            # S8 completes in a sub-second burst.  Keep the no-backpressure
            # contract by preallocating enough pinned slots for that entire
            # acceptance burst while the first Parquet upload is in flight.
            MGBFS_ARCHIVE_SLOTS="16",
            MGBFS_ARCHIVE_STREAM="1",
        )
        try:
            raw_text = run([
                str(source / "target/release/examples/macro_bench"), "s8", "32768", "1", "1",
                "verify", str(fifo),
            ], "native-s8-stream", source)
        except Exception:
            streamer.terminate()
            streamer.wait(timeout=30)
            raise
        try:
            stream_code = streamer.wait(timeout=900)
        except subprocess.TimeoutExpired:
            streamer.kill()
            streamer.wait()
            raise RuntimeError("STREAMER_TIMEOUT")
    if stream_code != 0:
        raise RuntimeError(f"STREAMER_FAILED_{stream_code}: {stream_stderr.read_text(errors='replace')[-4000:]}")
    raw = next(json.loads(line) for line in reversed(raw_text.splitlines()) if line.startswith("{"))
    commit_path = staging / "rank-00000-stream-commit.json"
    rank_commit = json.loads(commit_path.read_text())
    if raw.get("total_unique_states") != 40320 or rank_commit.get("total_unique_states") != 40320:
        raise RuntimeError("S8_CARDINALITY_GATE")
    promotion_text = run([
        sys.executable, "scripts/promote_hf_stream.py", "--repo-id", REPO_ID, "--world-size", "1",
        str(commit_path),
    ], "promote", source)
    promotion = json.loads(promotion_text.splitlines()[-1])
    if promotion.get("status") != "COMPLETE" or not promotion.get("commit_url"):
        raise RuntimeError("PROMOTION_GATE")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    expected = {
        f"runs/{run_id}.json",
        f"layers/{run_id}.parquet",
        f"verification/{run_id}.json",
    }
    files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    if not expected.issubset(files):
        raise RuntimeError("PUBLISHED_METADATA_MISSING")
    final_states = {
        f"states/{run_id}-{Path(item['path']).name}" for item in rank_commit["files"]
    }
    if not final_states.issubset(files):
        raise RuntimeError("PUBLISHED_STATE_SHARDS_MISSING")
    api.delete_branch(repo_id=REPO_ID, repo_type="dataset", branch=branch)
    result = {
        "schema": "MGBFS_KAGGLE_HF_STREAM_GATE_V1",
        "status": "PASS",
        "source": SOURCE,
        "dataset": REPO_ID,
        "run_id": run_id,
        "commit_url": promotion["commit_url"],
        "total_unique_states": 40320,
        "state_shards": len(final_states),
        "peak_live_slots": rank_commit["peak_live_slots"],
        "slot_count": rank_commit["slot_count"],
        "search_complete_seconds": raw["search_complete_seconds"],
        "durable_run_commit_seconds": raw["durable_run_commit_seconds"],
        "gpu": gpus[0],
    }
    (logs / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
