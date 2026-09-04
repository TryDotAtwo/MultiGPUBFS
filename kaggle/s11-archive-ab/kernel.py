"""Exact 2xT4 S11 A/B: identical allocations, archive submission on versus off."""
import importlib.util
import json
import os
import shutil
import statistics
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "50770121362d889d31d6e540c383376824a66451"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
CARDINALITY = 39_916_800
BATCH = 262_144


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-s11-archive-ab-", dir="/tmp"))
    logs = Path("/kaggle/working/s11-archive-ab")
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
    build = source / "build/s11-archive-ab"
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

    # One BFS-sized archive transfer per slot keeps the same bounded pinned
    # capacity while replacing thousands of small D2H/event submissions with
    # roughly eighty large transactions per rank.
    base_env = dict(env, MGBFS_CAPACITY_MODE="equal_global",
                    MGBFS_BENCH_CAPACITY="8000000", MGBFS_FUTURE_CAPACITY="8000000",
                    MGBFS_ARCHIVE_ROWS=str(BATCH), MGBFS_ARCHIVE_SLOTS="128")
    rows = []
    # Alternation includes a warm-up pair and prevents all measurements of one
    # mode from occupying a different thermal/runtime phase.
    schedule = [(False, "warmup"), (True, "warmup")]
    schedule += [(mode, f"measure-{i}") for i in range(5) for mode in (False, True)]
    for archive_enabled, phase in schedule:
        archive_prefix = root / "archive"
        label = f"s11-{'archive' if archive_enabled else 'no-archive'}-{phase}"
        run_env = dict(base_env)
        if not archive_enabled:
            run_env["MGBFS_BENCH_SKIP_ARCHIVE"] = "1"
        command = [
            "torchrun", "--standalone", "--nproc-per-node=2", "--no-python",
            str(source / "target/release/examples/distributed_bench"), "s11", str(BATCH),
            str(root / f"bootstrap-{label}"), str(archive_prefix), "{RANK_OUT}",
        ]
        row = bench.run_group(command, logs, label, run_env, timeout=7200)
        if row["status"] != "COMPLETE" or sum(row["layer_sizes"]) != CARDINALITY:
            raise RuntimeError(f"A_B_GATE_{label}")
        row["archive_enabled"] = archive_enabled
        row["phase"] = phase
        rows.append(row)
        for rank in range(2):
            Path(f"{archive_prefix}-rank-{rank}.mgbfsar1").unlink(missing_ok=True)

    measured = [row for row in rows if row["phase"].startswith("measure")]
    grouped = {}
    for enabled in (False, True):
        samples = [row["search_complete_seconds"] for row in measured
                   if row["archive_enabled"] == enabled]
        grouped[str(enabled).lower()] = {
            "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "mad_seconds": statistics.median(abs(x - statistics.median(samples)) for x in samples),
        }
    overhead = grouped["true"]["median_seconds"] - grouped["false"]["median_seconds"]
    summary = {
        "schema": "MGBFS_S11_ARCHIVE_AB_V1", "status": "PASS", "source": SOURCE,
        "gpus": gpus, "unique_states": CARDINALITY, "runs": rows, "search": grouped,
        "archive_overhead_seconds": overhead,
        "archive_overhead_percent": 100 * overhead / grouped["false"]["median_seconds"],
    }
    (logs / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}), flush=True)
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
