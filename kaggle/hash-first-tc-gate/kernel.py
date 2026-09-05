"""Hash-only integer MMA leaf: explicit experimental backend, no graph datasets."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

SOURCE = "b3929cc3c8a2dbc4b1a0d84c1df39a2567cc1374"


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-hash-tc-", dir="/tmp"))
    logs = Path("/kaggle/working/hash-first-tc")
    logs.mkdir()
    helper = root / "helper.py"
    urllib.request.urlretrieve("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py", helper)
    spec = importlib.util.spec_from_file_location("gate", helper)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    env = dict(os.environ)
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    def run(cmd, name):
        return gate.run(cmd, cwd=root, env=env, logs=logs, name=name, timeout=1200)
    report = dict(source=SOURCE, status="INCOMPLETE", scope="single T4 hash-only leaf, not full BFS or performance")
    try:
        report["gpus"] = gate.validate_gpus(run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"], "inventory"))
        source = root / "source"
        gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
        binary = root / "hash-first-tc-test"
        run(["nvcc", "-std=c++17", "-arch=sm_75", "-lineinfo", "-DMGBFS_TEST_HASH_TC=1", str(source / "tests/hash_first_generate.cu"), str(source / "cuda/hash_first_generate.cu"), "-o", str(binary)], "build")
        sass = run(["cuobjdump", "--dump-sass", str(binary)], "sass")
        report["imma"] = [line.strip() for line in sass.splitlines() if re.search(r"\bIMMA\.", line)]
        if not report["imma"]:
            raise RuntimeError("IMMA_INSTRUCTION_MISSING")
        for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
            cmd = [str(binary)]
            if tool != "plain":
                cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99"] + cmd
            output = run(cmd, tool)
            if "HASH_FIRST_GENERATE_PASS" not in output:
                raise RuntimeError("HASH_FIRST_FIXTURE_NOT_PASSED")
            if tool == "racecheck":
                summaries = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", output)
                if not summaries or any(row != ("0", "0", "0") for row in summaries):
                    raise RuntimeError("RACECHECK_NOT_CLEAN")
            elif tool != "plain":
                summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", output)
                if not summaries or any(row != "0" for row in summaries):
                    raise RuntimeError("SANITIZER_NOT_CLEAN")
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (logs / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
