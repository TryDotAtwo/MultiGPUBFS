"""Pinned libcudf boundary probe on T4; NOT an end-to-end BFS benchmark."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import hashlib
import shutil

SOURCE_COMMIT = "be1f57bfe0b246a3e1407b105710a0d038dfa4fe"
PACKAGES = ["libcudf-cu12==26.4.0", "librmm-cu12==26.4.0",
            "cmake==3.31.6", "ninja==1.11.1.4"]
# NVIDIA redistrib_12.9.1.json, linux-x86_64. Downloaded on Kaggle only.
CUDA_COMPONENTS = [
    ("cuda_nvcc", "12.9.86", "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35"),
    ("cuda_cudart", "12.9.79", "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99"),
    ("cuda_cccl", "12.9.27", "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa"),
    ("cuda_nvrtc", "12.9.86", "82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af"),
]


def isolated_environment(inherited):
    env = dict(inherited)
    # Kaggle injects a sitecustomize via PYTHONPATH that imports host-only wrapt.
    # It must not load into the isolated library toolchain or pollute path output.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def cmake_prefixes(site):
    # Ubuntu does not necessarily search lib64 under every prefix. Include the
    # actual exported-config directories, including bundled CCCL dependencies.
    paths = {str(p) for p in site.iterdir() if p.is_dir()}
    for pattern in ("*-config.cmake", "*Config.cmake"):
        paths.update(str(p.parent) for p in site.rglob(pattern) if p.is_file())
    return sorted(paths)


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
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        manifest["cuda_components"] = CUDA_COMPONENTS
        for component, version, sha256 in CUDA_COMPONENTS:
            name = f"{component}-linux-x86_64-{version}-archive"
            archive = work / (name + ".tar.xz")
            url = f"https://developer.download.nvidia.com/compute/cuda/redist/{component}/linux-x86_64/{archive.name}"
            run(["curl", "--fail", "--location", "--max-time", "180", url,
                 "--output", str(archive)], component + "-download")
            with archive.open("rb") as package:
                actual_sha = hashlib.file_digest(package, "sha256").hexdigest()
            if actual_sha != sha256:
                raise RuntimeError(f"CUDA archive checksum mismatch: {component}")
            run(["tar", "-xf", str(archive), "-C", str(work)], component + "-extract")
            shutil.copytree(work / name, sdk, dirs_exist_ok=True)
        # Linux redistributables store libraries in lib; nvcc.profile expects
        # the conventional toolkit lib64 layout. Both names refer to this SDK.
        (sdk / "lib64").symlink_to("lib", target_is_directory=True)
        if not (sdk / "lib64/libcudart_static.a").is_file():
            raise RuntimeError("Pinned CUDA SDK missing static runtime")
        env["PATH"] = str(sdk / "bin") + ":" + env["PATH"]
        env["CUDACXX"] = str(sdk / "bin/nvcc")
        run([str(sdk / "bin/nvcc"), "--version"], "cuda-version")
        venv = work / "venv"
        # Kaggle's ensurepip bootstrap failed in v1. Use the host pip's supported
        # --python entry point; the target environment remains fully isolated.
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        pip = [sys.executable, "-m", "pip", "--python", python]
        requirements = source / "experiments/library_owner/requirements-linux-x86_64.lock"
        manifest["requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        run([*pip, "install", "--only-binary=:all:", "--no-cache-dir", "--require-hashes",
             "--report", str(logs / "pip-install.json"), "-r", str(requirements)], "install", 900)
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
        prefixes = cmake_prefixes(site)
        lib_dirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        env["LD_LIBRARY_PATH"] = ":".join([str(sdk / "lib"), str(sdk / "lib64")] + lib_dirs + [env.get("LD_LIBRARY_PATH", "")])
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        build = work / "build"
        run([str(venv / "bin/cmake"), "-S", str(source / "experiments/library_owner"),
             "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             "-DCUDAToolkit_ROOT=" + str(sdk), "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
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
