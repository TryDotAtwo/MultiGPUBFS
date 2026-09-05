"""Explicit BMMA owner contract gate. No graph data, no HF credentials."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import urllib.request

SOURCE = "cdb0fd460e6ff607fcc81fb8121cdc064e5592ae"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-bmma-", dir="/tmp"))
    logs = Path("/kaggle/working/bmma-owner")
    logs.mkdir()
    helper = root / "helper.py"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
        "c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py", helper)
    spec = importlib.util.spec_from_file_location("helper", helper)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    env = dict(os.environ)
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    def run(cmd, name, cwd=root):
        return gate.run(cmd, cwd=cwd, env=env, logs=logs, name=name, timeout=1200)
    report = {"source": SOURCE, "status": "INCOMPLETE", "scope": "BMMA owner leaf; not full BFS"}
    try:
        report["gpus"] = gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"))
        source, cutlass = root / "source", root / "cutlass"
        gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
        gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS, cutlass, env, logs, "cutlass")
        build = root / "build"
        run(["cmake", "-S", str(source / "cuda"), "-B", str(build), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DCMAKE_CUDA_ARCHITECTURES=75",
             "-DCUTLASS_ROOT=" + str(cutlass)], "cmake")
        run(["cmake", "--build", str(build), "--target", "mgbfs-bmma-owner-test", "--parallel", "2"], "build")
        output = run([str(build / "mgbfs-bmma-owner-test")], "plain")
        if "BOUNDED_OWNER_PASS" not in output:
            raise RuntimeError("OWNER_FIXTURE_NOT_PASSED")
        report["status"] = "PLAIN_PASS"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (logs / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
