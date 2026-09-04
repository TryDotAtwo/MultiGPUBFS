"""Physical 2xT4 S11 A/B for equal-global and max-per-rank allocation modes."""
import importlib.util
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "3378317bfce34ed38c06a886d692c9f8f6a91769"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
CARDINALITY = 39_916_800
DECLARED_CAPACITY = 8_000_000
BATCH = 262_144


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-capacity-modes-", dir="/tmp"))
    logs = Path("/kaggle/working/capacity-modes")
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

    def run(command, name, cwd=root, timeout=1800):
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
    build = source / "build/capacity-modes"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime",
         "--features", "cuda", "--example", "distributed_bench"], "rust-build", source)
    sys.path.insert(0, str(source / "scripts"))
    bench = load(source / "scripts/distributed_gpu_bench.py", "bench")
    rows = []
    for mode in ("equal_global", "max_per_rank"):
        bootstrap = root / f"bootstrap-{mode}"
        archive_prefix = root / f"archive-{mode}"
        run_env = dict(
            env,
            MGBFS_CAPACITY_MODE=mode,
            MGBFS_BENCH_CAPACITY=str(DECLARED_CAPACITY),
            MGBFS_FUTURE_CAPACITY=str(DECLARED_CAPACITY),
            MGBFS_ARCHIVE_ROWS="16384",
            MGBFS_ARCHIVE_SLOTS="1400",
        )
        command = [
            "torchrun", "--standalone", "--nproc-per-node=2", "--no-python",
            str(source / "target/release/examples/distributed_bench"), "s11", str(BATCH),
            str(bootstrap), str(archive_prefix), "{RANK_OUT}",
        ]
        try:
            row = bench.run_group(command, logs, f"s11-{mode}", run_env, timeout=3600)
        finally:
            for rank in range(2):
                Path(f"{archive_prefix}-rank-{rank}.mgbfsar1").unlink(missing_ok=True)
            bootstrap.unlink(missing_ok=True)
        if row["status"] != "COMPLETE" or sum(row["layer_sizes"]) != CARDINALITY:
            raise RuntimeError(f"{mode.upper()}_CORRECTNESS_GATE")
        expected_rank_capacity = DECLARED_CAPACITY // (2 if mode == "equal_global" else 1)
        for rank_result in row["rank_results"]:
            if rank_result["rank_capacity_records"] != expected_rank_capacity:
                raise RuntimeError(f"{mode.upper()}_CAPACITY_GATE")
        rows.append(row)
    if rows[0]["layer_sizes"] != rows[1]["layer_sizes"]:
        raise RuntimeError("MODE_LAYER_MISMATCH")
    result = {
        "schema": "MGBFS_CAPACITY_MODES_T4_V1",
        "status": "PASS",
        "source": SOURCE,
        "gpus": gpus,
        "group": "s11",
        "declared_capacity_records": DECLARED_CAPACITY,
        "rows": rows,
    }
    (logs / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
