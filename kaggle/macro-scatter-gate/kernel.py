"""Private physical 2xT4 weighted DENSE frame/NCCL transport gate."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

SOURCE = "a7a9d5f6669388aacf9a8e3e69b8ab5cd058ff9b"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"
FIXTURE = "admitted_adapter_native_scatter_and_depth_rollover"


def main():
    root = Path(tempfile.mkdtemp(prefix="mgbfs-macro-scatter-", dir="/tmp"))
    logs = Path("/kaggle/working/macro-scatter-gate")
    logs.mkdir()
    helper = root / "helper.py"
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
        "c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/"
        "kaggle/native-primitives/kernel.py", helper,
    )
    spec = importlib.util.spec_from_file_location("helper", helper)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    def run(command, name, cwd=root, timeout=1200):
        return gate.run(command, cwd=cwd, env=env, logs=logs,
                        name=name, timeout=timeout)

    report = {"schema": 1, "status": "INCOMPLETE", "source": SOURCE,
              "cutlass": CUTLASS, "fixture": FIXTURE, "tests": []}
    def save():
        (logs / "summary.json").write_text(json.dumps(report, indent=2))
    save()
    try:
        report["gpus"] = gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"))
        save()
        source, cutlass = root / "source", root / "cutlass"
        gate.checkout("https://github.com/TryDotAtwo/MultiGPUBFS.git",
                      SOURCE, source, env, logs, "source")
        gate.checkout("https://github.com/NVIDIA/cutlass.git",
                      CUTLASS, cutlass, env, logs, "cutlass")
        env["CARGO_HOME"] = str(root / "cargo")
        env["RUSTUP_HOME"] = str(root / "rustup")
        installer = root / "rustup.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile",
             "minimal", "--default-toolchain", "1.75.0"], "rust-install")
        env["PATH"] = str(root / "cargo/bin") + ":" + env["PATH"]
        build = source / "build/macro-scatter-gate"
        env["MGBFS_CUDA_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        run(["cmake", "-S", "cuda", "-B", str(build), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DCMAKE_CUDA_ARCHITECTURES=75",
             "-DCUTLASS_ROOT=" + str(cutlass)], "cmake", source)
        run(["cmake", "--build", str(build), "--target", "mgbfs_cuda",
             "--parallel", "2"], "cuda-build", source)
        run(["cmake", "--build", str(build), "--target",
             "mgbfs-macro-future-checked-test", "--parallel", "2"],
            "future-build", source)
        output = run(["cargo", "test", "--locked", "--release", "-p",
                      "mgbfs-runtime", "--features", "cuda", "--test",
                      "native_scatter", "--no-run", "--message-format=json"],
                     "scatter-build", source)
        binaries = [item["executable"] for line in output.splitlines()
                    if line.startswith("{")
                    for item in [json.loads(line)]
                    if item.get("reason") == "compiler-artifact"
                    and item.get("executable")
                    and item.get("target", {}).get("name") == "native_scatter"]
        if len(binaries) != 1:
            raise RuntimeError("SCATTER_BINARY_AMBIGUOUS")
        for tool in ["plain", "memcheck", "racecheck", "initcheck", "synccheck"]:
            command = [binaries[0], FIXTURE, "--exact", "--test-threads=1", "--nocapture"]
            if tool != "plain":
                command = ["compute-sanitizer", "--error-exitcode", "99",
                           "--tool", tool] + command
            result = run(command, "scatter-" + tool, source, timeout=600)
            if not re.search(r"test result: ok\. 1 passed; 0 failed", result):
                raise RuntimeError("SCATTER_RESULT_MISMATCH")
            if tool == "racecheck":
                if not re.search(r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)", result):
                    raise RuntimeError("RACECHECK_NOT_CLEAN")
            elif tool != "plain" and not re.search(r"ERROR SUMMARY: 0 errors", result):
                raise RuntimeError("SANITIZER_NOT_CLEAN")
            report["tests"].append({"tool": tool, "status": "PASS"})
            save()
        for tool in ["plain", "memcheck", "racecheck", "initcheck", "synccheck"]:
            command = [str(build / "mgbfs-macro-future-checked-test")]
            if tool != "plain":
                command = ["compute-sanitizer", "--error-exitcode", "99",
                           "--tool", tool] + command
            result = run(command, "future-" + tool, source, timeout=600)
            if "MACRO_FUTURE_CHECKED_PASS" not in result:
                raise RuntimeError("FUTURE_RESULT_MISMATCH")
            if tool == "racecheck":
                if not re.search(r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)", result):
                    raise RuntimeError("FUTURE_RACECHECK_NOT_CLEAN")
            elif tool != "plain" and not re.search(r"ERROR SUMMARY: 0 errors", result):
                raise RuntimeError("FUTURE_SANITIZER_NOT_CLEAN")
            report["tests"].append({"tool": "future-" + tool, "status": "PASS"})
            save()
        report["status"] = "COMPLETE"
    finally:
        save()


if __name__ == "__main__":
    main()
