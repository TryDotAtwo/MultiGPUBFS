"""Physical 2xT4 S11 native capacity and complete-layer probe."""
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "3f66af2a8e5de3ecbac208f3c09c676b03be47f2"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
CARDINALITY = 39_916_800
PER_RANK_CAPACITY = 8_000_000
BATCH = 262_144
ARCHIVE_ROWS = 16_384
ARCHIVE_SLOTS = 1_400


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-s11-probe-", dir="/tmp"))
    logs = Path("/kaggle/working/s11-distributed-probe")
    logs.mkdir()
    helper = root / "gate.py"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
        "c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py",
        helper,
    )
    gate = load(helper, "gate")
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    def run(command, name, cwd=root, timeout=3600):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    inventory = run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
         "--format=csv,noheader,nounits"], "inventory"
    )
    gpus = gate.validate_gpus(inventory)
    source, cutlass = root / "source", root / "cutlass"
    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
    gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS, cutlass, env, logs, "cutlass")
    env["CARGO_HOME"], env["RUSTUP_HOME"] = str(root / "cargo"), str(root / "rustup")
    installer = root / "rustup.sh"
    urllib.request.urlretrieve("https://sh.rustup.rs", installer)
    run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal",
         "--default-toolchain", "1.75.0"], "rust-install")
    env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
    build = source / "build/s11-probe"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime",
         "--features", "cuda", "--example", "distributed_bench"], "rust-build", source)
    bench = load(source / "scripts/distributed_gpu_bench.py", "bench")
    bootstrap = root / "bootstrap"
    archive_prefix = root / "archive"
    run_env = dict(
        env,
        MGBFS_BENCH_CAPACITY=str(PER_RANK_CAPACITY),
        MGBFS_FUTURE_CAPACITY=str(PER_RANK_CAPACITY),
        MGBFS_ARCHIVE_ROWS=str(ARCHIVE_ROWS),
        MGBFS_ARCHIVE_SLOTS=str(ARCHIVE_SLOTS),
    )
    command = [
        "torchrun", "--standalone", "--nproc-per-node=2", "--no-python",
        str(source / "target/release/examples/distributed_bench"), "s11", str(BATCH),
        str(bootstrap), str(archive_prefix), "{RANK_OUT}",
    ]
    row = bench.run_group(command, logs, "s11-native-capacity", run_env, timeout=7200)
    if row["status"] == "COMPLETE":
        if sum(row["layer_sizes"]) != CARDINALITY:
            raise RuntimeError("S11_CARDINALITY_MISMATCH")
        row["unique_states"] = CARDINALITY
    summary = {
        "schema": 1,
        "status": row["status"],
        "source": SOURCE,
        "gpus": gpus,
        "group": "s11",
        "per_rank_capacity": PER_RANK_CAPACITY,
        "batch": BATCH,
        "archive_slots_per_rank": ARCHIVE_SLOTS,
        "result": row,
    }
    (logs / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
