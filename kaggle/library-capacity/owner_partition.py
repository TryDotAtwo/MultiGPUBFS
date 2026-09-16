"""Isolated logical-owner primitives on two physical T4s; NOT 8-rank BFS."""
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCE_COMMIT = "069612bee18d1c048b161a89b1382044859edeff"
OUT = Path("/kaggle/working/owner-partition")
SRC = Path("/tmp/mgbfs-owner-partition")


def run(name, args, env=None):
    print(name, flush=True)
    with (OUT / (name + ".log")).open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=600)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"source_commit": SOURCE_COMMIT, "status": "INCOMPLETE",
               "scope": "1/2/4/8 logical-owner primitives on each T4; no NCCL",
               "passed": []}
    try:
        summary["gpus"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
            text=True).splitlines()
        if len(summary["gpus"]) != 2 or not all("T4" in g for g in summary["gpus"]):
            raise RuntimeError("REQUIRES_TWO_PHYSICAL_T4")
        nvcc, sanitizer = shutil.which("nvcc"), shutil.which("compute-sanitizer")
        if not nvcc or not sanitizer:
            raise RuntimeError("CUDA_TOOLCHAIN_MISSING")
        run("toolchain", [nvcc, "--version"])
        run("clone", ["git", "clone", "--no-checkout",
                      "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(SRC)])
        run("checkout", ["git", "-C", str(SRC), "checkout", "--detach", SOURCE_COMMIT])
        exe = str(SRC / "owner-partition-test")
        run("build", [nvcc, "-std=c++17", "-arch=sm_75", "-lineinfo",
                      str(SRC / "tests/owner_partition_gpu.cu"),
                      str(SRC / "cuda/exchange_pack.cu"),
                      str(SRC / "cuda/directories.cu"), "-o", exe])
        for gpu in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            for tool in (None, "memcheck", "racecheck", "initcheck", "synccheck"):
                name = f"gpu{gpu}-{tool or 'plain'}"
                args = [exe] if tool is None else [sanitizer, "--tool", tool,
                                                   "--error-exitcode", "99", exe]
                run(name, args, env)
                if "OWNER_PARTITION_GPU_PASS" not in (OUT / (name + ".log")).read_text():
                    raise RuntimeError("MISSING_TEST_COMPLETION:" + name)
                summary["passed"].append(name)
        summary["status"] = "PASS"
    finally:
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
