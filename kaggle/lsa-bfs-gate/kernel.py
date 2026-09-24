"""Exact-source, two-T4 CUCO_RANK/LSA full-state BFS correctness gate."""
import hashlib
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE = "d95ef218d321cd35b601c7db6440ec22038e966d"
CUCO = "532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4"


def main():
    work = Path(tempfile.mkdtemp(prefix="mgbfs-lsa-bfs-", dir="/tmp"))
    logs = Path("/kaggle/working/lsa-bfs-gate")
    logs.mkdir(parents=True, exist_ok=True)
    report = {"source": SOURCE, "status": "INCOMPLETE", "scope":
              "two physical T4; NCCL LSA; CUCO_RANK DENSE full layer sets and archives"}

    def save():
        (logs / "summary.json").write_text(json.dumps(report, indent=2))

    save()
    source = work / "source"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin",
                    "https://github.com/TryDotAtwo/MultiGPUBFS.git"], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", SOURCE], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE:
        raise RuntimeError("SOURCE_COMMIT_MISMATCH")
    spec = importlib.util.spec_from_file_location("gate", source / "kaggle/native-primitives/kernel.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    spec = importlib.util.spec_from_file_location("library", source / "kaggle/library-owner/kernel.py")
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    env = library.isolated_environment(os.environ)
    env["NCCL_CUMEM_ENABLE"] = "1"

    def run(command, name, cwd=source, timeout=900):
        return gate.run(command, cwd=cwd, env=env, logs=logs, name=name, timeout=timeout)

    try:
        report["gpus"] = gate.validate_gpus(run([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits"], "inventory"))
        cudart = ctypes.CDLL("libcudart.so.12")
        p2p = []
        for source_gpu, target_gpu in ((0, 1), (1, 0)):
            allowed = ctypes.c_int()
            rc = cudart.cudaDeviceCanAccessPeer(ctypes.byref(allowed), source_gpu, target_gpu)
            p2p.append({"source": source_gpu, "target": target_gpu,
                        "cuda_status": rc, "allowed": allowed.value})
        report["p2p"] = p2p
        if any(row["cuda_status"] != 0 or row["allowed"] != 1 for row in p2p):
            report["status"] = "UNSUPPORTED_HOST"
            return
        sdk = work / "cuda-12.9"
        sdk.mkdir()
        for component, version, digest in library.CUDA_COMPONENTS:
            name = f"{component}-linux-x86_64-{version}-archive"
            archive = work / (name + ".tar.xz")
            url = ("https://developer.download.nvidia.com/compute/cuda/redist/"
                   f"{component}/linux-x86_64/{archive.name}")
            run(["curl", "--fail", "--location", "--max-time", "180", url,
                 "--output", str(archive)], component + "-download")
            with archive.open("rb") as package:
                if hashlib.file_digest(package, "sha256").hexdigest() != digest:
                    raise RuntimeError("CUDA_SDK_CHECKSUM")
            run(["tar", "-xf", str(archive), "-C", str(work)], component + "-extract")
            shutil.copytree(work / name, sdk, dirs_exist_ok=True)
        (sdk / "lib64").symlink_to("lib", target_is_directory=True)
        env["PATH"] = str(sdk / "bin") + ":" + env.get("PATH", "")
        env["CUDACXX"] = str(sdk / "bin/nvcc")
        venv = work / "venv"
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], "venv")
        python = str(venv / "bin/python")
        run([sys.executable, "-m", "pip", "--python", python, "install",
             "--only-binary=:all:", "--no-cache-dir", "--require-hashes", "-r",
             str(source / "experiments/library_owner/requirements-linux-x86_64.lock")],
            "dependencies", timeout=1200)
        site = subprocess.check_output([python, "-c", "import site; print(site.getsitepackages()[0])"],
                                       text=True, env=env).strip()
        site = Path(site)
        prefixes = library.cmake_prefixes(site)
        libdirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
        nccl_target = work / "nccl"
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--target",
             str(nccl_target), "nvidia-nccl-cu12==2.29.7"], "nccl-install")
        nccl = nccl_target / "nvidia/nccl"
        if not (nccl / "include/nccl_device.h").is_file():
            raise RuntimeError("PINNED_NCCL_DEVICE_HEADER")
        env["PATH"] = str(venv / "bin") + ":" + env["PATH"]
        env["LD_LIBRARY_PATH"] = ":".join([str(nccl / "lib"), str(sdk / "lib"),
                                            *libdirs, env.get("LD_LIBRARY_PATH", "")])
        env["MGBFS_CUDART_LIB_DIR"] = str(sdk / "lib")
        env["CARGO_HOME"] = str(work / "cargo")
        env["RUSTUP_HOME"] = str(work / "rustup")
        installer = work / "rustup-init.sh"
        run(["curl", "--fail", "--location", "--max-time", "180",
             "https://sh.rustup.rs", "-o", str(installer)], "rust-download")
        run(["sh", str(installer), "-y", "--no-modify-path", "--profile", "minimal",
             "--default-toolchain", gate.RUST_VERSION], "rust-install")
        env["PATH"] = str(work / "cargo/bin") + ":" + env["PATH"]
        cuco = work / "cuco"
        gate.checkout("https://github.com/NVIDIA/cuCollections.git", CUCO,
                      cuco, env, logs, "cuco")
        build = work / "library-build"
        run(["cmake", "-S", str(source / "experiments/library_owner"), "-B", str(build),
             "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
             "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUDAToolkit_ROOT=" + str(sdk),
             "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
             "-DCUCO_ROOT=" + str(cuco)], "library-configure")
        run(["cmake", "--build", str(build), "--target", "mgbfs_library_owner", "-j2"],
            "library-build", timeout=1800)
        env["MGBFS_LIBRARY_OWNER_LIB_DIR"] = str(build)
        env["LD_LIBRARY_PATH"] = str(build) + ":" + env["LD_LIBRARY_PATH"]
        cutlass = work / "cutlass"
        gate.checkout("https://github.com/NVIDIA/cutlass.git", gate.CUTLASS_COMMIT,
                      cutlass, env, logs, "cutlass")
        native = work / "native-build"
        run(["cmake", "-S", str(source / "cuda"), "-B", str(native), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
             "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_CUDA_COMPILER=" + str(sdk / "bin/nvcc"),
             "-DCUTLASS_ROOT=" + str(cutlass), "-DMGBFS_NCCL_LSA=ON",
             "-DMGBFS_NCCL_ROOT=" + str(nccl)], "native-configure")
        run(["cmake", "--build", str(native), "--target", "mgbfs_cuda", "-j2"],
            "native-build", timeout=1800)
        env["MGBFS_CUDA_LIB_DIR"] = str(native)
        env["LD_LIBRARY_PATH"] = str(native) + ":" + env["LD_LIBRARY_PATH"]
        run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
             "--features", "cuda,library-owner", "--test", "library_multi_gpu",
             "--no-run"], "bfs-test-build", timeout=1800)
        result = run(["cargo", "test", "--locked", "-p", "mgbfs-runtime",
                      "--features", "cuda,library-owner", "--test", "library_multi_gpu",
                      "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle",
                      "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                     "lsa-full-bfs", timeout=900)
        if "test result: ok. 1 passed; 0 failed" not in result:
            raise RuntimeError("BFS_TEST_RESULT")
        binaries = [path for path in (source / "target/debug/deps").glob("library_multi_gpu-*")
                    if path.is_file() and os.access(path, os.X_OK)]
        if len(binaries) != 1:
            raise RuntimeError("BFS_TEST_BINARY_INVENTORY")
        report["plain_full_bfs"] = "PASS"
        save()
        sanitized = run(["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
                         str(binaries[0]),
                         "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle",
                         "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                        "lsa-full-bfs-memcheck", timeout=360)
        if "test result: ok. 1 passed; 0 failed" not in sanitized or \
                "ERROR SUMMARY: 0 errors" not in sanitized:
            raise RuntimeError("BFS_MEMCHECK_RESULT")
        report["full_bfs_memcheck"] = "PASS_UNFILTERED"
        report["status"] = "COMPLETE"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        save()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
