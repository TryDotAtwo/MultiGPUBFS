"""Build, verify and append S8-S10 exhaustive BFS archives to the HF catalog."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from kaggle_secrets import UserSecretsClient

SOURCE = "88c3215280df1757f4dcc34c52951caee78045f2"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
REPO_ID = "TryDotAtwo/multigpubfs-bfs-results"
RUST = "1.75.0"
GROUPS = [("s8", 40_320), ("s9", 362_880), ("s10", 3_628_800)]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-symmetric-catalog-", dir="/tmp"))
    logs = Path("/kaggle/working/symmetric-catalog")
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

    def run(command, name, cwd=root, timeout=3600):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    inventory = run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        "inventory",
    )
    gpus = gate.validate_gpus(inventory)
    source, cutlass = root / "source", root / "cutlass"
    gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
    gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS, cutlass, env, logs, "cutlass")
    env["CARGO_HOME"], env["RUSTUP_HOME"] = str(root / "cargo"), str(root / "rustup")
    installer = root / "rustup.sh"
    urllib.request.urlretrieve("https://sh.rustup.rs", installer)
    run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", RUST], "rust-install")
    env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
    env["CUDA_VISIBLE_DEVICES"] = "0"
    build = source / "build/symmetric-catalog"
    env["MGBFS_CUDA_LIB_DIR"] = str(build)
    env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
    run(["cmake", "--build", str(build), "--parallel", "2"], "cuda-build", source)
    run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda",
         "--example", "macro_bench"], "rust-build", source)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub>=0.34"], "install-hf")
        from huggingface_hub import HfApi
    api = HfApi(token=token)
    results = []
    for group, cardinality in GROUPS:
        work = root / group
        work.mkdir()
        archive = work / f"{group}.mgbfsar1"
        env.update(MGBFS_BENCH_CAPACITY=str(cardinality), MGBFS_FUTURE_CAPACITY=str(cardinality),
                   MGBFS_ARCHIVE_ROWS="16384", MGBFS_ARCHIVE_SLOTS="286")
        raw_text = run([str(source / "target/release/examples/macro_bench"), group, "262144", "1", "1",
                        "verify", str(archive)], f"native-{group}", source, timeout=3600)
        raw = next(json.loads(line) for line in reversed(raw_text.splitlines()) if line.startswith("{"))
        raw_path = work / "raw.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        run_id = f"{group}-native-k1-seed-20260828"
        summary = work / "summary.json"
        run([sys.executable, "scripts/prepare_hf_run_summary.py", str(raw_path), str(summary),
             "--source-commit", SOURCE, "--hardware", gpus[0]["name"], "--run-id", run_id],
            f"summary-{group}", source)
        package = work / "parquet"
        run([sys.executable, "scripts/export_hf_dataset.py", "--run-id", run_id, "--summary", str(summary),
             "--archive", "0=" + str(archive), "--output", str(package), "--rows-per-shard", "100000"],
            f"export-{group}", source)
        run([sys.executable, "scripts/verify_hf_dataset.py", str(package), "--sort-memory-records", "500000"],
            f"verify-{group}", source, timeout=3600)
        verification = json.loads((package / "verification.json").read_text())
        if verification.get("status") != "PASS" or verification.get("unique_states") != cardinality:
            raise RuntimeError(f"VERIFICATION_GATE_{group}")
        staging = work / "catalog-upload"
        run([sys.executable, "scripts/prepare_hf_catalog_upload.py", str(package), str(staging)],
            f"catalog-{group}", source)
        commit = api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(staging), path_in_repo="",
                                   commit_message=f"Add verified exhaustive {group.upper()} BFS graph")
        results.append({"group": group, "cardinality": cardinality, "run_id": run_id,
                        "search_seconds": raw["search_complete_seconds"], "durable_seconds": raw["durable_run_commit_seconds"],
                        "dataset_commit": str(commit), "verification": verification})
        shutil.rmtree(work)
    result = {"schema": 1, "status": "COMPLETE", "source": SOURCE, "dataset": REPO_ID,
              "gpu": gpus[0], "runs": results}
    (logs / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
