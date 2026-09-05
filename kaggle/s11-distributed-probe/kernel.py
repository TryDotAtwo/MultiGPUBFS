"""Physical 2xT4 S11 native capacity and complete-layer probe."""
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "9f440a1ceff379f1c9df57fe97baa6ff98bd21c0"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
CARDINALITY = 39_916_800
PER_RANK_CAPACITY = 8_000_000
BATCH = 262_144
ARCHIVE_ROWS = 262_144
ARCHIVE_SLOTS = 96


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
    run(["cargo", "test", "--locked", "--release", "-p", "mgbfs-cuda",
         "--features", "cuda", "--test", "generate", "compact_permutation_generation_matches_gather"], "compact-generation", source)
    executables = [p for p in (source / "target/release/deps").glob("generate-*")
                   if p.is_file() and os.access(p, os.X_OK) and p.suffix == ""]
    if len(executables) != 1:
        raise RuntimeError("AMBIGUOUS_GENERATION_TEST_BINARY")
    for gpu in range(2):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                 str(executables[0]), "compact_permutation_generation_matches_gather"],
                f"compact-gpu{gpu}-{tool}", source)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    sys.path.insert(0, str(source / "scripts"))
    bench = load(source / "scripts/distributed_gpu_bench.py", "bench")
    bootstrap = root / "bootstrap"
    archive_prefix = root / "archive"
    run_env = dict(
        env,
        MGBFS_CAPACITY_MODE="max_per_rank",
        MGBFS_BENCH_CAPACITY=str(PER_RANK_CAPACITY),
        MGBFS_FUTURE_CAPACITY=str(PER_RANK_CAPACITY),
        MGBFS_ARCHIVE_ROWS=str(ARCHIVE_ROWS),
        MGBFS_ARCHIVE_SLOTS=str(ARCHIVE_SLOTS),
        MGBFS_ARCHIVE_CODEC="permutation_u8",
        MGBFS_STATE_CODEC="permutation_u8",
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
