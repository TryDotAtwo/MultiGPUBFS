"""Focused state-commit gate on two independent T4s, NOT a multi-rank BFS."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import threading

SOURCE_COMMIT = "0a1124ad8c441d464ce3f223f9547334c339618c"

def main():
    output = Path("/kaggle/working/state-commit-gate")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"source": SOURCE_COMMIT, "status": "INCOMPLETE", "checks": []}
    try:
        source = Path(tempfile.mkdtemp(prefix="mgbfs-owner-", dir="/tmp"))
        for cmd in (["git", "init", "-q"],
                    ["git", "remote", "add", "origin", "https://github.com/TryDotAtwo/MultiGPUBFS.git"],
                    ["git", "fetch", "--depth=1", "origin", SOURCE_COMMIT],
                    ["git", "checkout", "--detach", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=source, check=True)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if actual != SOURCE_COMMIT:
            raise RuntimeError("Source mismatch")
        helpers = runpy.run_path(str(source / "kaggle/native-primitives/kernel.py"))
        env = dict(os.environ)
        def run(cmd, name, child_env=env):
            return helpers["run"](cmd, cwd=source, env=child_env, logs=output, name=name, timeout=600)
        inventory = run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"], "inventory")
        gpus = helpers["validate_gpus"](inventory)
        summary["gpus"] = gpus
        run(["nvcc", "--version"], "nvcc")
        build = source / "owner-build"
        run(["cmake", "-S", str(source / "cuda"), "-B", str(build), "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=RelWithDebInfo"], "configure")
        run(["cmake", "--build", str(build), "--target", "mgbfs-state-commit-test", "-j2"], "build")
        lock = threading.Lock()
        def worker(gpu):
            child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu["uuid"])
            for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
                cmd = [str(build / "mgbfs-state-commit-test")]
                if tool != "plain":
                    cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99"] + cmd
                name = f"gpu{gpu['index']}-{tool}"
                log = run(cmd, name, child_env)
                if "STATE_COMMIT_PASS" not in log:
                    raise RuntimeError("Missing test completion")
                if tool == "racecheck":
                    if "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" not in log:
                        raise RuntimeError("Racecheck incomplete or nonzero")
                elif tool != "plain" and "ERROR SUMMARY: 0 errors" not in log:
                    raise RuntimeError("Sanitizer incomplete or nonzero")
                with lock:
                    summary["checks"].append({"gpu": gpu["index"], "uuid": gpu["uuid"], "tool": tool, "log": name + ".log"})
        helpers["run_gpu_suites"](gpus, worker)
        if len(summary["checks"]) != 10:
            raise RuntimeError("Incomplete matrix")
        summary["status"] = "COMPLETE"
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
