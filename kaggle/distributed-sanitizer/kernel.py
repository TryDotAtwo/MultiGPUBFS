"""Two-device matrix and compact-permutation archive correctness gates."""
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

SOURCE = "e3fd4870d32aa3aae7117fa915eb44c77e8fc6a3"
CUTLASS = "ffa119a1255d78998536107466cc7097ecefa393"


def require_clean(tool, output):
    if tool == "racecheck":
        matches = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", output)
        if not matches or any(x != ("0", "0", "0") for x in matches):
            raise RuntimeError("RACECHECK_NOT_CLEAN")
    elif tool != "plain":
        matches = re.findall(r"ERROR SUMMARY: (\d+) errors", output)
        if not matches or any(x != "0" for x in matches):
            raise RuntimeError("SANITIZER_NOT_CLEAN")


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
              "scope": "unitriangular(3,3) DENSE/HASH_FIRST scalar reference across 3 seeds, 2 owner maps, pre-dedup ON/OFF; S4 DENSE generation5; two T4, NCCL, full archive layers"}
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
        run(["cmake", "--build", str(build), "--target", "mgbfs-owner-memory-query-test", "--parallel", "2"], "owner-query-build", source)
        query_output = run([str(build / "mgbfs-owner-memory-query-test")], "owner-query", source)
        if "BOUNDED_OWNER_QUERY_PASS" not in query_output:
            raise RuntimeError("OWNER_MEMORY_QUERY_FAILED")
        run(["cmake", "--build", str(build), "--target", "mgbfs-regenerate-test", "--parallel", "2"], "regenerate-build", source)
        run([str(build / "mgbfs-regenerate-test")], "regenerate-plain", source)
        run(["cmake", "--build", str(build), "--target", "mgbfs-hash-first-generate-test", "--parallel", "2"], "hash-first-build", source)
        run([str(build / "mgbfs-hash-first-generate-test")], "hash-first-plain", source)
        run(["cmake", "--build", str(build), "--target", "mgbfs-materialize-requests-test", "--parallel", "2"], "requests-build", source)
        run([str(build / "mgbfs-materialize-requests-test")], "requests-plain", source)
        run(["cmake", "--build", str(build), "--target", "mgbfs-sort-origins-test", "--parallel", "2"], "sort-origins-build", source)
        run([str(build / "mgbfs-sort-origins-test")], "sort-origins-plain", source)
        run(["cmake", "--build", str(build), "--target", "mgbfs-apply-responses-test", "--parallel", "2"], "apply-responses-build", source)
        run([str(build / "mgbfs-apply-responses-test")], "apply-responses-plain", source)
        output = run(["cargo", "test", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda",
                      "--test", "distributed_archive", "--no-run", "--message-format=json"], "test-build", source)
        binaries = [json.loads(line)["executable"] for line in output.splitlines()
                    if line.startswith("{") and json.loads(line).get("executable")]
        if len(binaries) != 1:
            raise RuntimeError("AMBIGUOUS_TEST_BINARY")
        macro_output = run(["cargo", "test", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda",
                            "--test", "macro_native", "--no-run", "--message-format=json"], "macro-build", source)
        macro_binaries = [json.loads(line)["executable"] for line in macro_output.splitlines()
                          if line.startswith("{") and json.loads(line).get("executable")]
        if len(macro_binaries) != 1:
            raise RuntimeError("AMBIGUOUS_MACRO_TEST_BINARY")
        report["macro_scope"] = "single T4 nonidentity UT(3,3) source; K=1,2,3,10; pre-dedup OFF/ON; full original layers"
        for tool in ("plain", "memcheck", "racecheck", "initcheck", "synccheck"):
            if tool != "plain":
                leaf = run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                            str(build / "mgbfs-regenerate-test")], "regenerate-" + tool, source)
                if "REGENERATE_PASS" not in leaf:
                    raise RuntimeError("REGENERATE_NOT_PASSED")
                require_clean(tool, leaf)
                hashed = run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                              str(build / "mgbfs-hash-first-generate-test")], "hash-first-" + tool, source)
                if "HASH_FIRST_GENERATE_PASS" not in hashed:
                    raise RuntimeError("HASH_FIRST_GENERATE_NOT_PASSED")
                require_clean(tool, hashed)
                requests = run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                                str(build / "mgbfs-materialize-requests-test")], "requests-" + tool, source)
                if "MATERIALIZE_REQUESTS_PASS" not in requests:
                    raise RuntimeError("MATERIALIZE_REQUESTS_NOT_PASSED")
                require_clean(tool, requests)
                sorted_origins = run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                                      str(build / "mgbfs-sort-origins-test")], "sort-origins-" + tool, source)
                if "SORT_ORIGINS_PASS" not in sorted_origins:
                    raise RuntimeError("SORT_ORIGINS_NOT_PASSED")
                require_clean(tool, sorted_origins)
                applied = run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                               str(build / "mgbfs-apply-responses-test")], "apply-responses-" + tool, source)
                if "APPLY_RESPONSES_PASS" not in applied:
                    raise RuntimeError("APPLY_RESPONSES_NOT_PASSED")
                require_clean(tool, applied)
            cmd = [binaries[0], "--test-threads=1", "--nocapture"]
            if tool != "plain":
                cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99"] + cmd
            output = run(cmd, tool, source)
            if "10 passed; 0 failed" not in output:
                raise RuntimeError("FIXTURE_NOT_PASSED")
            require_clean(tool, output)
            macro_cmd = [macro_binaries[0], "native_macro_nonidentity_source_preserves_original_layers",
                         "--exact", "--test-threads=1", "--nocapture"]
            if tool != "plain":
                macro_cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99"] + macro_cmd
            macro_output = run(macro_cmd, "macro-" + tool, source)
            if "1 passed; 0 failed" not in macro_output:
                raise RuntimeError("MACRO_SOURCE_FIXTURE_NOT_PASSED")
            require_clean(tool, macro_output)
            archive_cmd = [macro_binaries[0], "native_macro_archive_is_complete_and_verifiable",
                           "--exact", "--test-threads=1", "--nocapture"]
            if tool != "plain":
                archive_cmd = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99"] + archive_cmd
            archive_output = run(archive_cmd, "macro-archive-" + tool, source)
            if "1 passed; 0 failed" not in archive_output:
                raise RuntimeError("MACRO_ARCHIVE_FIXTURE_NOT_PASSED")
            require_clean(tool, archive_output)
            report["tests"].append({"tool": tool, "status": "PASS"})
            save()
        # Exercise the actual process-launcher/benchmark selection, not just
        # direct constructors. These tiny runs are correctness, not timing data.
        run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-runtime", "--features", "cuda",
             "--example", "distributed_bench"], "bench-build", source)
        run(["cargo", "build", "--locked", "--release", "-p", "mgbfs-cli"], "cli-build", source)
        report["reference_profile_smoke"] = []
        expected_layers = None
        for profile, generation in (("DENSE", "SCALAR"), ("HASH_FIRST", "SCALAR"),
                                    ("HASH_FIRST", "INT_MMA_SM75")):
            for owner in ("CUB_SORT_MERGE", "BMMA_BUCKET"):
                for pre in ("OFF", "ON"):
                    label = f"smoke-{profile}-{generation}-{owner}-{pre}"
                    output_dir = logs / label
                    output_dir.mkdir()
                    env.update(MGBFS_PROFILE=profile, MGBFS_OWNER_BACKEND=owner, MGBFS_PRE_DEDUP=pre,
                               MGBFS_HASH_FIRST_GENERATION=generation,
                               MGBFS_BENCH_WARMUP="1",
                               MGBFS_STATE_CODEC="matrix_u8", MGBFS_ARCHIVE_CODEC="matrix_u8",
                               MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128",
                               MGBFS_MATERIALIZATION_CAPACITY="42", MGBFS_BMMA_TILE_LIMIT="8",
                               MGBFS_BENCH_SKIP_ARCHIVE="0", MGBFS_ARCHIVE_STREAM="0")
                    prefix = root / (label + "-archive")
                    run(["torchrun", "--standalone", "--nproc-per-node=2", "--no-python",
                         str(source / "target/release/examples/distributed_bench"), "s4", "7",
                         str(root / (label + "-bootstrap")), str(prefix), str(output_dir)], label, source)
                    rows = [json.loads((output_dir / f"rank-{rank}.json").read_text()) for rank in range(2)]
                    for rank, row in enumerate(rows):
                        if (row["status"], row["frontier_profile"], row["owner_backend"], row["pre_dedup"], row["archive_enabled"]) != ("COMPLETE", profile, owner, pre, True):
                            raise RuntimeError("PROFILE_SELECTION_MISMATCH")
                        if row["hash_first_generation"] != generation:
                            raise RuntimeError("GENERATION_SELECTION_MISMATCH")
                        if row["warmup_completed"] is not True:
                            raise RuntimeError("WARMUP_NOT_COMPLETED")
                        warm_row = json.loads(Path(str(output_dir) + ".warmup", f"rank-{rank}.json").read_text())
                        if warm_row["local_layer_sizes"] != row["local_layer_sizes"]:
                            raise RuntimeError("WARMUP_LAYER_MISMATCH")
                        if Path(str(prefix) + f".warmup-rank-{rank}.mgbfsar1").exists():
                            raise RuntimeError("WARMUP_ARCHIVE_NOT_RELEASED")
                        run([str(source / "target/release/mgbfs"), "verify", str(prefix) + f"-rank-{rank}.mgbfsar1"], label + f"-verify-{rank}", source)
                    if len(rows[0]["local_layer_sizes"]) != len(rows[1]["local_layer_sizes"]):
                        raise RuntimeError("PROFILE_DEPTH_MISMATCH")
                    layers = [a + b for a, b in zip(rows[0]["local_layer_sizes"], rows[1]["local_layer_sizes"])]
                    if sum(layers) != 24 or (expected_layers is not None and layers != expected_layers):
                        raise RuntimeError("PROFILE_LAYER_MISMATCH")
                    expected_layers = layers
                    report["reference_profile_smoke"].append(dict(profile=profile, generation=generation, owner=owner, pre_dedup=pre, layers=layers, status="PASS"))
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
