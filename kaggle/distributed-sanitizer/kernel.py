"""Two-device matrix and compact-permutation archive correctness gates."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

SOURCE = "8f04c4642470c8c48954ac705dda941e06b08625"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-sanitize-", dir="/tmp"))
    logs = Path("/kaggle/working/distributed-sanitizer")
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
    def run(cmd, name, cwd=root, timeout=1200):
        return gate.run(cmd, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)
    report = {"status": "INCOMPLETE", "source": SOURCE, "tests": [],
              "scope": "unitriangular(3,3) generation1 and S4 compact generation5, two devices, NCCL, archive"}
    def save():
        (logs / "summary.json").write_text(json.dumps(report, indent=2))
    save()
    try:
        report["gpus"] = gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"))
        source, cutlass = root / "source", root / "cutlass"
        gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git", SOURCE, source, env, logs, "source")
        gate.checkout("https://github.com/NVIDIA/cutlass.git", CUTLASS, cutlass, env, logs, "cutlass")
        env["CARGO_HOME"], env["RUSTUP_HOME"] = str(root / "cargo"), str(root / "rustup")
        installer = root / "rustup.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", "1.75.0"], "rust-install")
        env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
        build = source / "build/sanitizer"
        env["MGBFS_CUDA_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
             "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
        run(["cmake", "--build", str(build), "--target", "mgbfs_cuda", "--parallel", "2"], "cuda-build", source)
        output = run(["cargo", "test", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda",
                      "--test", "distributed_archive", "--no-run", "--message-format=json"], "test-build", source)
        binaries = [json.loads(line)["executable"] for line in output.splitlines()
                    if line.startswith("{") and json.loads(line).get("executable")]
        if len(binaries) != 1:
            raise RuntimeError("AMBIGUOUS_TEST_BINARY")
        for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
            cmd = [binaries[0], "--test-threads=1", "--nocapture"]
            if tool != "plain":
                cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99"] + cmd
            output = run(cmd, tool, source)
            if "2 passed; 0 failed" not in output:
                raise RuntimeError("FIXTURE_NOT_PASSED")
            if tool == "racecheck":
                matches = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", output)
                if not matches or any(x != ("0", "0", "0") for x in matches):
                    raise RuntimeError("RACECHECK_NOT_CLEAN")
            elif tool != "plain":
                matches = re.findall(r"ERROR SUMMARY: (\d+) errors", output)
                if not matches or any(x != "0" for x in matches):
                    raise RuntimeError("SANITIZER_NOT_CLEAN")
            report["tests"].append({"tool": tool, "status": "PASS"})
            save()
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
