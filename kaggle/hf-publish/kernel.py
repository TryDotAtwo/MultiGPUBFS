"""Regenerate, verify, and publish the complete S10 BFS dataset from Kaggle."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from kaggle_secrets import UserSecretsClient

SOURCE = "e209340050a23f470a845f931e8a493ee59695cb"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
REPO_ID = "TryDotAtwo/multigpubfs-bfs-results"
RUST = "1.75.0"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-hf-publish-", dir="/tmp"))
    logs = Path("/kaggle/working/hf-publish")
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
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        "inventory",
    )
    gpus = gate.validate_gpus(inventory)
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError("DISK_PREFLIGHT_LT_12_GIB")

    source, cutlass = root / "source", root / "cutlass"
    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
    gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS, cutlass, env, logs, "cutlass")
    env["CARGO_HOME"] = str(root / "cargo")
    env["RUSTUP_HOME"] = str(root / "rustup")
    installer = root / "rustup.sh"
    urllib.request.urlretrieve("https://sh.rustup.rs", installer)
    run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", RUST], "rust-install")
    env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
    env["CUDA_VISIBLE_DEVICES"] = "0"
    build = source / "build/hf-publish"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(
        ["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUTLASS_ROOT=" + str(cutlass)],
        "cmake", source,
    )
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda", "--example", "macro_bench"], "rust-build", source)

    archive = root / "s10.mgbfsar1"
    env.update(MGBFS_BENCH_CAPACITY="3628800", MGBFS_FUTURE_CAPACITY="3628800",
               MGBFS_ARCHIVE_ROWS="16384", MGBFS_ARCHIVE_SLOTS="286")
    raw_text = run(
        [str(source / "target/release/examples/macro_bench"), "s10", "262144", "1", "1", "verify", str(archive)],
        "native-s10", source,
    )
    raw = next(json.loads(line) for line in reversed(raw_text.splitlines()) if line.startswith("{"))
    raw_path = root / "run-summary-raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    summary = root / "run-summary.json"
    run(
        [sys.executable, "scripts/prepare_hf_run_summary.py", str(raw_path), str(summary),
         "--source-commit", SOURCE, "--hardware", gpus[0]["name"]],
        "prepare-summary", source,
    )
    parquet = root / "parquet"
    run(
        [sys.executable, "scripts/export_hf_dataset.py", "--run-id", "s10-native-k1-seed-20260828",
         "--summary", str(summary), "--archive", "0=" + str(archive), "--output", str(parquet),
         "--rows-per-shard", "100000"],
        "export-parquet", source,
    )
    run([sys.executable, "scripts/verify_hf_dataset.py", str(parquet), "--sort-memory-records", "500000"],
        "verify-parquet", source, timeout=3600)
    verification = json.loads((parquet / "verification.json").read_text())
    if verification.get("status") != "PASS" or verification.get("unique_states") != 3628800 or not verification.get("hash_state_pairs_verified"):
        raise RuntimeError("VERIFICATION_GATE")

    try:
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as error:
        raise RuntimeError("KAGGLE_SECRET_HF_TOKEN_UNAVAILABLE") from error
    if not token:
        raise RuntimeError("KAGGLE_SECRET_HF_TOKEN_EMPTY")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=0.34"], "install-hf")
        from huggingface_hub import HfApi
    commit = HfApi(token=token).upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(parquet),
        path_in_repo="",
        commit_message="Publish Kaggle-regenerated verified S10 Parquet dataset",
        allow_patterns=["manifest.json", "verification.json", "layers/*.parquet", "runs/*.parquet", "states/*.parquet"],
    )
    result = {
        "schema": 1,
        "status": "COMPLETE",
        "source": SOURCE,
        "dataset": REPO_ID,
        "commit_url": str(commit),
        "verification": verification,
        "parquet_bytes": sum(path.stat().st_size for path in parquet.rglob("*") if path.is_file()),
        "gpu": gpus[0],
    }
    (logs / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
