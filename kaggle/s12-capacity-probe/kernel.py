"""Find the reproducible 2xT4 S12 capacity boundary without materializing output."""
import importlib.util
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "2cccb5ccbaec31c9028b2f08b5a0ef5f58ef8b3a"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
CARDINALITY = 479_001_600
BATCH = 262_144
CAPACITIES_PER_RANK = [32_000_000]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-s12-capacity-", dir="/tmp"))
    logs = Path("/kaggle/working/s12-capacity-probe")
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

    def run(command, name, cwd=root, timeout=7200):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    inventory = run([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], "inventory")
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
    build = source / "build/s12-capacity"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime",
         "--features", "cuda", "--example", "distributed_bench"], "rust-build", source)
    import sys
    sys.path.insert(0, str(source / "scripts"))
    bench = load(source / "scripts/distributed_gpu_bench.py", "bench")

    results = []
    for capacity in CAPACITIES_PER_RANK:
        label = f"s12-capacity-{capacity}"
        run_env = dict(env, MGBFS_CAPACITY_MODE="max_per_rank",
                       MGBFS_BENCH_CAPACITY=str(capacity),
                       MGBFS_FUTURE_CAPACITY=str(capacity),
                       MGBFS_ARCHIVE_ROWS="1", MGBFS_ARCHIVE_SLOTS="2",
                       MGBFS_BENCH_SKIP_ARCHIVE="1", MGBFS_TRACE_DEPTHS="1")
        command = [
            "torchrun", "--standalone", "--nproc-per-node=2", "--no-python",
            str(source / "target/release/examples/distributed_bench"), "s12", str(BATCH),
            str(root / f"bootstrap-{capacity}"), str(root / "unused-archive"), "{RANK_OUT}",
        ]
        row = bench.run_group(command, logs, label, run_env, timeout=1800)
        row["capacity_per_rank"] = capacity
        results.append(row)
        if row["status"] == "COMPLETE":
            if sum(row["layer_sizes"]) != CARDINALITY:
                raise RuntimeError("S12_CARDINALITY")
            break

    summary = {
        "schema": "MGBFS_S12_CAPACITY_PROBE_V1", "status": "COMPLETE",
        "source": SOURCE, "gpus": gpus, "cardinality": CARDINALITY,
        "results": results,
    }
    (logs / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
