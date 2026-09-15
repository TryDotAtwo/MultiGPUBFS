"""Pinned libcudf boundary probe on T4; NOT an end-to-end BFS benchmark."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE_COMMIT = "67adf6bf46220f60bae7c59bbbc2076171dd7e42"
PACKAGES = ["libcudf-cu12==26.4.0", "librmm-cu12==26.4.0",
            "cmake==3.31.6", "ninja==1.11.1.4"]


def isolated_environment(inherited):
    env = dict(inherited)
    # Kaggle injects a sitecustomize via PYTHONPATH that imports host-only wrapt.
    # It must not load into the isolated library toolchain or pollute path output.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def main():
    logs = Path("/kaggle/working/library-owner")
    logs.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mgbfs-library-", dir="/tmp"))
    source = work / "source"
    # Public immutable source; no GitHub/HF secrets are read or forwarded.
    subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                    "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(source)],
                   check=True, timeout=180)
    subprocess.run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=source,
                   check=True, timeout=180)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError("Source commit mismatch")
    spec = importlib.util.spec_from_file_location("gate", source / "kaggle/native-primitives/kernel.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    gpus = gate.validate_gpus(inventory)
    manifest = {"source_commit": actual, "packages_requested": PACKAGES,
                "gpus": gpus, "kind": "library_characterization", "status": "RUNNING"}
    summary_path = logs / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2))
    env = isolated_environment(os.environ)
    def run(command, name, timeout=900, extra_env=None):
        return gate.run(command, cwd=source, env=env if extra_env is None else extra_env,
                        logs=logs, name=name, timeout=timeout)
    try:
        venv = work / "venv"
        # Kaggle's ensurepip bootstrap failed in v1. Use the host pip's supported
        # --python entry point; the target environment remains fully isolated.
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        pip = [sys.executable, "-m", "pip", "--python", python]
        run([*pip, "install", "--only-binary=:all:", "--no-cache-dir",
             "--report", str(logs / "pip-install.json"), *PACKAGES], "install", 900)
        run([*pip, "freeze", "--all"], "packages")
        run([*pip, "show", "-f", "libcudf-cu12", "librmm-cu12", "rapids-logger"],
            "sdk-inventory")
        # The normal log helper deliberately merges stdout/stderr. Never parse
        # its result as a path: even a nonfatal sitecustomize diagnostic corrupts
        # that protocol. Preserve diagnostics separately instead of hiding them.
        site_query = subprocess.run(
            [python, "-c", "import site; print(site.getsitepackages()[0])"],
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=30)
        (logs / "site-stderr.log").write_text(site_query.stderr)
        (logs / "site-stdout.log").write_text(site_query.stdout)
        site = Path(site_query.stdout.strip())
        prefixes = sorted({str(p.parent.parent) for p in site.rglob("libcudf") if p.is_dir()})
        # Wheels install their exported CMake config under individual package roots.
        prefixes = [str(p) for p in site.iterdir() if p.is_dir()] + prefixes
        lib_dirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        build = work / "build"
        run([str(venv / "bin/cmake"), "-S", str(source / "experiments/library_owner"),
             "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes)], "configure")
        run([str(venv / "bin/cmake"), "--build", str(build), "-j2"], "build")
        executable = str(build / "cudf_owner_probe")
        for gpu in gpus:
            device_env = dict(env, CUDA_VISIBLE_DEVICES=gpu["uuid"])
            for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
                command = [executable] if tool == "plain" else [
                    "compute-sanitizer", "--tool", tool, "--error-exitcode", "97", executable]
                run(command, f"gpu{gpu['index']}-{tool}", extra_env=device_env)
        manifest["status"] = "PASS"
    except Exception as error:
        manifest.update(status="FAILED", error=str(error))
        raise
    finally:
        summary_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
