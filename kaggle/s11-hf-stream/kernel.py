"""Complete 2xT4 S13 BFS with direct bounded Parquet/HF archive streaming."""
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

SOURCE = "48b1e12807e180f392e3d08b2279ad23fa8930d8"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
REPO_ID = "TryDotAtwo/multigpubfs-bfs-results"
GROUP = "s13"
CARDINALITY = 6_227_020_800
PER_RANK_CAPACITY = 160_000_000
BATCH = 262_144
ARCHIVE_ROWS = 262_144
ARCHIVE_SLOTS = 96


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-s11-hf-", dir="/tmp"))
    logs = Path("/kaggle/working/s11-hf-stream")
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

    def run(command, name, cwd=root, timeout=7200):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    inventory = run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
         "--format=csv,noheader,nounits"], "inventory"
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
    run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal",
         "--default-toolchain", "1.75.0"], "rust-install")
    env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
    build = source / "build/s11-hf"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime",
         "--features", "cuda", "--example", "distributed_bench"], "rust-build", source)
    try:
        import huggingface_hub
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=0.34"], "install-hf")
        import huggingface_hub

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{GROUP}-native-2xt4-{stamp}"
    branches = [f"staging-{run_id}-rank-{rank}" for rank in range(2)]
    api = huggingface_hub.HfApi(token=token)
    for branch in branches:
        api.create_branch(repo_id=REPO_ID, repo_type="dataset", branch=branch)

    archive_prefix = root / "archive"
    fifos = [Path(f"{archive_prefix}-rank-{rank}.mgbfsar1") for rank in range(2)]
    for fifo in fifos:
        os.mkfifo(fifo)
    streamers = []
    handles = []
    try:
        for rank, fifo in enumerate(fifos):
            stdout = (logs / f"streamer-rank-{rank}.stdout.log").open("wb")
            stderr = (logs / f"streamer-rank-{rank}.stderr.log").open("wb")
            handles.extend((stdout, stderr))
            command = [
                sys.executable, "scripts/stream_hf_archive.py", "--run-id", run_id,
                "--group-id", GROUP, "--rank", str(rank), "--input", str(fifo),
                "--staging-dir", str(root / f"rank-{rank}-slots"), "--repo-id", REPO_ID,
                "--branch", branches[rank], "--rows-per-shard", "1000000", "--slot-count", "8",
                "--max-slot-bytes", str(128 * 1024**2),
            ]
            streamers.append(subprocess.Popen(command, cwd=source, env=env, stdout=stdout, stderr=stderr))
        time.sleep(1)
        if any(process.poll() is not None for process in streamers):
            raise RuntimeError("STREAMER_EARLY_EXIT")

        sys.path.insert(0, str(source / "scripts"))
        bench = load(source / "scripts/distributed_gpu_bench.py", "bench")
        run_env = dict(
            env,
            MGBFS_CAPACITY_MODE="max_per_rank",
            MGBFS_BENCH_CAPACITY=str(PER_RANK_CAPACITY),
            MGBFS_FUTURE_CAPACITY=str(PER_RANK_CAPACITY),
            MGBFS_ARCHIVE_ROWS=str(ARCHIVE_ROWS),
            MGBFS_ARCHIVE_SLOTS=str(ARCHIVE_SLOTS),
            MGBFS_ARCHIVE_STREAM="1",
            MGBFS_ARCHIVE_CODEC="permutation_u8",
            MGBFS_STATE_CODEC="permutation_u8",
        )
        command = [
            "torchrun", "--standalone", "--nproc-per-node=2", "--no-python",
            str(source / "target/release/examples/distributed_bench"), GROUP, str(BATCH),
            str(root / "bootstrap"), str(archive_prefix), "{RANK_OUT}",
        ]
        wall_start = time.perf_counter()
        row = bench.run_group(command, logs, "s11-native-hf", run_env, timeout=7200)
        codes = [process.wait(timeout=3600) for process in streamers]
        wall_seconds = time.perf_counter() - wall_start
        if codes != [0, 0]:
            raise RuntimeError(f"STREAMERS_FAILED_{codes}")
        if row["status"] != "COMPLETE" or sum(row["layer_sizes"]) != CARDINALITY:
            raise RuntimeError("S11_SEARCH_GATE")

        commits = [root / f"rank-{rank}-slots/rank-{rank:05d}-stream-commit.json" for rank in range(2)]
        promotion_text = run([
            sys.executable, "scripts/promote_hf_stream.py", "--repo-id", REPO_ID,
            "--world-size", "2", *(str(path) for path in commits),
        ], "promote", source)
        promotion = json.loads(promotion_text.splitlines()[-1])
        if promotion.get("status") != "COMPLETE" or not promotion.get("commit_url"):
            raise RuntimeError("PROMOTION_GATE")
        rank_commits = [json.loads(path.read_text()) for path in commits]
        if sum(item["total_unique_states"] for item in rank_commits) != CARDINALITY:
            raise RuntimeError("S11_ARCHIVE_CARDINALITY_GATE")
        summary = {
            "schema": "MGBFS_S11_HF_STREAM_V2", "status": "PASS", "source": SOURCE,
            "dataset": REPO_ID, "run_id": run_id, "commit_url": promotion["commit_url"],
            "total_unique_states": CARDINALITY, "state_shards": sum(len(x["files"]) for x in rank_commits),
            "peak_live_slots": [x["peak_live_slots"] for x in rank_commits],
            "search_complete_seconds": row["search_complete_seconds"],
            "durable_run_commit_seconds": row["durable_run_commit_seconds"],
            "wall_seconds": wall_seconds, "peak_vram_mib": row["smi_peak_mib_per_rank"],
            "gpus": gpus,
        }
        (logs / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
        for branch in branches:
            api.delete_branch(repo_id=REPO_ID, repo_type="dataset", branch=branch)
    except Exception:
        for process in streamers:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
