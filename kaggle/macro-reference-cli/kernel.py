"""Physical single-T4 gate for the public K>1 reference CLI and archive."""
import importlib.util
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "2d7e480f770c472b5e8774a81a55373b6fe13e3f"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
SEED = "0000000000000000000000000000002a"


def load(path):
    spec = importlib.util.spec_from_file_location("gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-macro-cli-", dir="/tmp"))
    logs = Path("/kaggle/working/macro-reference-cli")
    logs.mkdir()
    helper = root / "gate.py"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
        "c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/"
        "kaggle/native-primitives/kernel.py", helper,
    )
    gate = load(helper)
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    def run(command, name, cwd=root, timeout=1200, config=None):
        return gate.run(command, cwd=cwd, env=config or env, logs=logs,
                        name=name, timeout=timeout)

    gpus = gate.validate_gpus(run([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits"], "inventory"))
    env["CUDA_VISIBLE_DEVICES"] = gpus[0]["uuid"]
    source, cutlass = root / "source", root / "cutlass"
    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE,
                  source, env, logs, "source")
    gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS,
                  cutlass, env, logs, "cutlass")
    env["CARGO_HOME"] = str(root / "cargo")
    env["RUSTUP_HOME"] = str(root / "rustup")
    installer = root / "rustup.sh"
    urllib.request.urlretrieve("https://sh.rustup.rs", installer)
    run(["sh", str(installer), "-y", "--no-modify-path", "--profile",
         "minimal", "--default-toolchain", "1.75.0"], "rust-install")
    env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
    build = source / "build/macro-reference-cli"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli",
         "--features", "cuda"], "rust-build", source)
    binary = source / "target/release/mgbfs"
    report = {"schema": 1, "status": "INCOMPLETE", "source": SOURCE,
              "cutlass": CUTLASS, "gpu": gpus[0], "rows": []}

    def save():
        (logs / "summary.json").write_text(json.dumps(report, indent=2))

    try:
        for group, order, codec in [("u4m2", 64, "matrix_u8"),
                                    ("s5", math.factorial(5), "permutation_u8")]:
            expected_layers = None
            for depth in [1, 2, 3]:
                label = f"{group}-k{depth}"
                out = logs / label
                out.mkdir()
                archive_prefix = root / label
                archive = root / f"{label}-rank-0.mgbfsar1"
                config = dict(env, MGBFS_MACRO_DEPTH=str(depth),
                              MGBFS_BENCH_CAPACITY="256",
                              MGBFS_FUTURE_CAPACITY="4096",
                              MGBFS_STATE_CODEC=codec,
                              MGBFS_ARCHIVE_CODEC=codec,
                              MGBFS_ARCHIVE_ROWS="7",
                              MGBFS_ARCHIVE_SLOTS="32",
                              MGBFS_HASH_SEED_HEX=SEED,
                              MGBFS_PRE_DEDUP="ON")
                run(["torchrun", "--standalone", "--nproc-per-node=1",
                     "--no-python", str(binary), "bench", "--reference",
                     group, "7", str(root / f"{label}.bootstrap"),
                     str(archive_prefix), str(out)], label, source,
                    timeout=600, config=config)
                run([str(binary), "verify", str(archive)],
                    label + "-verify", source, config=config)
                result = json.loads((out / "rank-0.json").read_text())
                layers = result["local_layer_sizes"]
                if (result["status"] != "COMPLETE" or sum(layers) != order
                        or result["macro_depth"] != depth
                        or result["hash_seed_hex"] != SEED
                        or result["output_contract"] != "archive_and_layer_counts"):
                    raise RuntimeError(f"{label}: incorrect result")
                if expected_layers is not None and layers != expected_layers:
                    raise RuntimeError(f"{label}: K changes exact layer sizes")
                expected_layers = layers
                report["rows"].append({"group": group, "macro_depth": depth,
                                       "layers": layers, "search_seconds": result["search_complete_seconds"],
                                       "durable_seconds": result["durable_run_commit_seconds"],
                                       "archive_verified": True})
                save()
                archive.unlink()
        report["status"] = "COMPLETE"
    finally:
        save()


if __name__ == "__main__":
    main()
