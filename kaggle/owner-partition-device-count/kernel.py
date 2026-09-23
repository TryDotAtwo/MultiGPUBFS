"""Physical T4 gate for device-count owner packing, not a complete BFS."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE = "064185ae2784e8e627bde930c8acddbeb35ac350"
OUT = Path("/kaggle/working/device-count-gate")


def run(name, argv, cwd, env=None):
    with (OUT / (name + ".log")).open("w") as log:
        result = subprocess.run(argv, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise RuntimeError(f"{name} failed: {result.returncode}")
    return (OUT / (name + ".log")).read_text(errors="replace")


def main():
    OUT.mkdir()
    report = {"schema": 1, "status": "INCOMPLETE", "source": SOURCE,
              "scope": "single-GPU owner pack primitive on each of two T4s",
              "results": []}
    try:
        inventory = run("inventory", ["nvidia-smi", "--query-gpu=index,name,uuid",
                                      "--format=csv,noheader"], OUT)
        devices = [x.strip().split(",") for x in inventory.splitlines() if x.strip()]
        if len(devices) != 2 or any("T4" not in x[1] for x in devices):
            raise RuntimeError("REQUIRES_2XT4")
        report["gpus"] = devices
        source = Path(tempfile.mkdtemp(prefix="mgbfs-device-count-", dir="/tmp"))
        run("init", ["git", "init", "-q"], source)
        run("remote", ["git", "remote", "add", "origin",
                       "https://github.com/TryDotAtwo/MultiGPUBFS.git"], source)
        run("fetch", ["git", "fetch", "--depth=1", "origin", SOURCE], source)
        run("checkout", ["git", "checkout", "--detach", "FETCH_HEAD"], source)
        exe = source / "owner-partition-test"
        run("build", ["nvcc", "-std=c++17", "-arch=sm_75", "-lineinfo",
                      str(source / "tests/owner_partition_gpu.cu"),
                      str(source / "cuda/exchange_pack.cu"),
                      str(source / "cuda/directories.cu"), "-o", str(exe)], source)
        for gpu in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[gpu][2].strip())
            for mode in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
                name = f"gpu{gpu}-{mode}"
                argv = [str(exe)] if mode == "plain" else [
                    "compute-sanitizer", "--tool", mode, "--error-exitcode", "99", str(exe)]
                output = run(name, argv, source, env)
                if "OWNER_PARTITION_GPU_PASS" not in output:
                    raise RuntimeError(f"MISSING_PASS:{name}")
                report["results"].append(name)
        report["status"] = "COMPLETE"
    finally:
        (OUT / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
