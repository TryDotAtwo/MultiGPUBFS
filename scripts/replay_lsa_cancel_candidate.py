"""Incremental Linux replay on an already provisioned diagnostic work tree.

No provisioning, rental, download, publication or hardware-gate substitution.
Records the dirty source digest: this is a candidate test, not a commit replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import re
import signal


def instrument_rank_command(binary, arguments, tool, report_prefix):
    target = [str(binary), *arguments]
    if tool is None:
        return target
    if tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck'):
        return ['/usr/local/cuda/bin/compute-sanitizer', '--tool', tool,
                '--error-exitcode', '97', *target]
    if tool == 'nsys':
        return ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--sample=process-tree',
                '--cudabacktrace=memory:0,sync:0,other:0', '--cpuctxsw=none',
                '-o', str(report_prefix), *target]
    raise ValueError('UNKNOWN_PROCESS_INSTRUMENTATION')


def instrumentation_clean(text, tool):
    if tool == 'racecheck':
        rows = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', text)
        return bool(rows) and not any(int(n) for row in rows for n in row)
    rows = re.findall(r'ERROR SUMMARY: (\d+) errors', text)
    return bool(rows) and not any(int(n) for n in rows)


def s4_reference_layers():
    return s_reference_layers(4)


def s_reference_layers(n):
    """Independent full-state oracle: row permutations, no GPU hash/dedup code."""
    if not 2 <= n <= 8:
        raise ValueError('BOUNDED_REFERENCE_SIZE')
    frontier = {tuple(range(n))}
    visited, layers = set(frontier), []
    while frontier:
        layers.append({bytes(int(column == state[row])
                             for row in range(n) for column in range(n))
                       for state in frontier})
        children = set()
        for state in frontier:
            children.update((state[1:] + state[:1], state[-1:] + state[:-1],
                             (state[1], state[0], *state[2:])))
        frontier = children - visited
        visited.update(frontier)
    return layers


def verify_process_archives(case, frame_reader=None, n=4):
    """Reuse the checksummed archive reader, then compare every state/depth."""
    if frame_reader is None:
        from export_hf_dataset import frames
        frame_reader = frames
    expected = s_reference_layers(n)
    actual = [set() for _ in expected]
    visited, digest = set(), None
    for rank in (0, 1):
        local = [0] * len(expected)
        for depth, width, count, payload, config in frame_reader(
                case / f'archive-rank-{rank}.mgbfsar1'):
            if width != n * n or not 0 <= depth < len(expected):
                raise ValueError('PROCESS_ORACLE_SHAPE')
            if digest is not None and config != digest:
                raise ValueError('PROCESS_ORACLE_CONFIG')
            digest = config
            for index in range(count):
                state = payload[index * width:(index + 1) * width]
                if state in visited:
                    raise ValueError('PROCESS_ORACLE_DUPLICATE')
                visited.add(state)
                actual[depth].add(state)
                local[depth] += 1
        result = json.loads((case / f'result/rank-{rank}.json').read_text())
        if result.get('status') != 'COMPLETE' or result.get('local_layer_sizes') != local:
            raise ValueError('PROCESS_ORACLE_RANK_COUNTS')
    if actual != expected:
        raise ValueError('PROCESS_ORACLE_LAYER')
    return dict(unique_states=len(visited), layer_sizes=list(map(len, actual)),
                scope='two independent rank-process archives; full canonical states at every depth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--sanitizers", action="store_true")
    parser.add_argument('--instrument-processes', choices=(
        'memcheck', 'racecheck', 'initcheck', 'synccheck', 'nsys'))
    parser.add_argument('--healthy-only', action='store_true')
    parser.add_argument('--capacity-faults', action='store_true',
                        help='also exhaust actual owner capacity on either independent rank')
    parser.add_argument('--reference-size', type=int, choices=range(2, 9), default=4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--profile', choices=('DENSE', 'HASH_FIRST'), default='DENSE')
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    work, output = args.work.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = work / "source"
    env = dict(os.environ)
    site = next((work / "venv/lib").glob("python*/site-packages"))
    libdirs = sorted({str(p.parent) for p in site.rglob("*.so*") if p.is_file()})
    env.update(CARGO_HOME=str(work / "cargo"), RUSTUP_HOME=str(work / "rustup"),
               MGBFS_CUDA_LIB_DIR=str(work / "native-build"),
               MGBFS_LIBRARY_OWNER_LIB_DIR=str(work / "library-build"),
               MGBFS_CUDART_LIB_DIR=str(work / "cuda-12.9/lib"))
    env["LD_LIBRARY_PATH"] = ":".join([str(work / "native-build"),
        str(work / "library-build"), str(work / "nccl/nvidia/nccl/lib"),
        str(work / "cuda-12.9/lib"), *libdirs, env.get("LD_LIBRARY_PATH", "")])
    env["PATH"] = str(work / "cargo/bin") + ":" + env.get("PATH", "")
    report = {"scope": "dirty candidate; hardware diagnostic, not T4 acceptance",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=source, text=True).strip(), "cases": []}
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=source)
    (output / "source.patch").write_bytes(diff)
    report["source_patch_sha256"] = hashlib.sha256(diff).hexdigest()
    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    save()
    with (output / "build.log").open("w") as log:
        subprocess.run([str(work / "cargo/bin/cargo"), "build", "--locked",
            "-p", "mgbfs-cli", "--features", "cuda,library-owner"],
            cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=600, check=True)
    env.update(MGBFS_OWNER_BACKEND="CUCO_RANK", MGBFS_TRACE_FAILURE_TEARDOWN="1",
        MGBFS_LIBRARY_POOL_BYTES=str(64 << 20), MGBFS_PROFILE=args.profile,
        MGBFS_BENCH_CAPACITY="64", MGBFS_FUTURE_CAPACITY="128", MGBFS_BUCKETS="8",
        MGBFS_SHARDS="4", MGBFS_JOB_BUCKETS="2", MGBFS_BUCKET_CAPACITY="32",
        MGBFS_STATE_CODEC="matrix_u8", MGBFS_ARCHIVE_CODEC="matrix_u8",
        MGBFS_ARCHIVE_ROWS="3", MGBFS_ARCHIVE_SLOTS="128", MGBFS_BENCH_WARMUP="0",
        MGBFS_PRE_DEDUP="ON", MGBFS_BENCH_SKIP_ARCHIVE="0", MGBFS_ARCHIVE_STREAM="0",
        MGBFS_CAPACITY_MODE="max_per_rank", MGBFS_RANK_MAP="0,1",
        MGBFS_TRANSPORT_BACKEND="NCCL_LSA", NCCL_CUMEM_ENABLE="1")
    if args.reference_size != 4:
        # Capacity is deliberately conservative for this bounded full-state
        # oracle, not a prediction of unknown production frontiers.
        import math
        capacity = max(64, math.factorial(args.reference_size))
        env.update(MGBFS_BENCH_CAPACITY=str(capacity),
                   MGBFS_FUTURE_CAPACITY=str(capacity * 2),
                   MGBFS_BUCKET_CAPACITY=str(capacity),
                   MGBFS_ARCHIVE_ROWS='512', MGBFS_ARCHIVE_SLOTS='128')
    report['reference_size'] = args.reference_size
    report['batch'] = args.batch
    report['profile'] = args.profile
    report['owner_dag_capture_requested'] = 'MGBFS_TEST_OWNER_DAG_CAPTURE' in env
    faults = [("startup", "MGBFS_TEST_NCCL_STARTUP_FAULT_RANK"),
              ("owner", "MGBFS_TEST_OWNER_HOST_FAULT_RANK"),
              ("admission", "MGBFS_TEST_ARCHIVE_ADMISSION_FAULT_RANK"),
              ("finish", "MGBFS_TEST_ARCHIVE_FINISH_FAULT_RANK")]
    cases = [("healthy", None, None)] + [(name, key, rank)
        for name, key in faults for rank in (0, 1)]
    if args.capacity_faults:
        cases += [('capacity', 'MGBFS_TEST_OWNER_CAPACITY_RANK', rank) for rank in (0, 1)]
    if args.healthy_only:
        cases = cases[:1]
    report['process_instrumentation'] = args.instrument_processes
    for name, key, fault_rank in cases:
        case = output / f"{name}-{fault_rank}"
        case.mkdir()
        case_env = dict(env)
        case_env.pop('MGBFS_TEST_OWNER_CAPACITY_RANK', None)
        for _, fault in faults:
            case_env.pop(fault, None)
        if key:
            case_env[key] = str(fault_rank)
        case_env["TORCHELASTIC_RUN_ID"] = "lsa-cancel-" + uuid.uuid4().hex
        processes, streams = [], []
        started = time.monotonic()
        try:
            for rank in (0, 1):
                stream = (case / f"rank-{rank}.log").open("w")
                streams.append(stream)
                rank_env = dict(case_env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
                command = instrument_rank_command(source / 'target/debug/mgbfs', [
                    "bench", "--reference", f"s{args.reference_size}", str(args.batch), str(case / "bootstrap"),
                    str(case / "archive"), str(case / "result")],
                    args.instrument_processes, case / f'rank-{rank}')
                processes.append(subprocess.Popen(command, cwd=source,
                    env=rank_env, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True))
            deadline = started + (120 if args.instrument_processes else 45)
            while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
                time.sleep(.05)
            forced = any(p.poll() is None for p in processes)
            row = {"name": name, "fault_rank": fault_rank, "forced_cleanup": forced,
                   "returncodes": [p.poll() for p in processes],
                   "seconds": time.monotonic() - started,
                   "group_complete": (case / "result/group-complete.json").exists()}
            row["pass"] = (not forced and (all(c == 0 for c in row["returncodes"])
                if key is None else all(c not in (None, 0) for c in row["returncodes"])
                and not row["group_complete"]))
            if key is None:
                row["pass"] &= row["group_complete"]
                if row["pass"]:
                    row["full_state_oracle"] = verify_process_archives(case, n=args.reference_size)
            for stream in streams:
                stream.flush()
            text = "\n".join((case / f"rank-{rank}.log").read_text(errors="replace")
                             for rank in (0, 1))
            if report['owner_dag_capture_requested'] and name == 'healthy':
                row['owner_dag_capture_launches'] = [
                    (case / f'rank-{rank}.log').read_text(errors='replace')
                    .count('MGBFS_OWNER_DAG_CAPTURE launched') for rank in (0, 1)]
                row['pass'] &= all(row['owner_dag_capture_launches'])
            if args.instrument_processes and args.instrument_processes != 'nsys':
                row['instrumentation_clean'] = all(instrumentation_clean(
                    (case / f'rank-{rank}.log').read_text(errors='replace'),
                    args.instrument_processes) for rank in (0, 1))
                row['pass'] &= row['instrumentation_clean']
            expected = {"startup": "TEST_INJECTED_NCCL_STARTUP_ERROR",
                        "owner": "TEST_INJECTED_OWNER_HOST_ERROR",
                        "admission": "TEST_INJECTED_ARCHIVE_ADMISSION_ERROR",
                        "finish": "TEST_INJECTED_ARCHIVE_FINISH_ERROR"}
            if key:
                if name == 'capacity':
                    row['fault_reached'] = any(marker in text for marker in (
                        'LIBRARY_RANK_DEPTH_FATAL', 'GROUP_OWNER_OR_PRE_OWNER_FATAL'))
                    row['capacity_profile'] = args.profile
                else:
                    row["fault_reached"] = expected[name] in text
                row["pass"] &= row["fault_reached"]
            report["cases"].append(row)
            save()
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            for stream in streams:
                stream.close()
        if not row["pass"]:
            raise RuntimeError("CANDIDATE_CASE_FAILED: " + name)
    if args.oracle:
        with (output / "full-state-oracle.log").open("w") as log:
            checked = subprocess.run([str(work / "cargo/bin/cargo"), "test", "--locked",
                "-p", "mgbfs-runtime", "--features", "cuda,library-owner", "--test",
                "library_multi_gpu", ("cuco_rank_lsa_hash_first_layers_and_archives_match_oracle"
                    if args.profile == 'HASH_FIRST' else
                    "cuco_rank_lsa_two_gpu_dense_layers_and_archives_match_oracle"),
                "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        report["oracle_returncode"] = checked.returncode
        report["oracle_scope"] = "two physical GPUs, threads in one process; full state/archive; U/S, maps, pre-dedup"
        save()
        if checked.returncode or "test result: ok. 1 passed; 0 failed" not in (
            output / "full-state-oracle.log").read_text():
            raise RuntimeError("FULL_STATE_ORACLE_FAILED")
        with (output / "capacity-fault.log").open("w") as log:
            checked = subprocess.run([str(work / "cargo/bin/cargo"), "test", "--locked",
                "-p", "mgbfs-runtime", "--features", "cuda,library-owner", "--test",
                "library_multi_gpu", "cuco_rank_lsa_one_rank_owner_capacity_failure_stops_group",
                "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=45)
        report["capacity_fault_returncode"] = checked.returncode
        report["capacity_fault_profile"] = "DENSE"
        save()
        if checked.returncode:
            raise RuntimeError("CAPACITY_FAULT_FAILED")
    if args.sanitizers:
        compiled = subprocess.check_output([str(work / "cargo/bin/cargo"), "test",
            "--locked", "-p", "mgbfs-runtime", "--features", "cuda,library-owner",
            "--test", "library_multi_gpu", "--no-run", "--message-format=json"],
            cwd=source, env=env, text=True, timeout=300)
        artifacts = [json.loads(line) for line in compiled.splitlines() if line.startswith("{")]
        binaries = [a["executable"] for a in artifacts if a.get("reason") == "compiler-artifact"
                    and a.get("executable") and a["target"]["name"] == "library_multi_gpu"]
        if len(binaries) != 1:
            raise RuntimeError("TEST_BINARY_INVENTORY")
        report["sanitizers"] = []
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
            log_path = output / (tool + ".log")
            command = ["/usr/local/cuda/bin/compute-sanitizer", "--tool", tool,
                "--error-exitcode", "97", binaries[0],
                "cuco_rank_lsa_single_fixture_for_sanitizer", "--ignored", "--exact",
                "--nocapture", "--test-threads=1"]
            try:
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=source, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        returncode = process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                        raise
                text = log_path.read_text(errors="replace")
                summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", text)
                races = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", text)
                clean = (bool(summaries) and not any(int(n) for n in summaries)) or (
                    tool == "racecheck" and bool(races) and not any(int(n) for row in races for n in row))
                report["sanitizers"].append({"tool": tool, "returncode": returncode,
                    "pass": returncode == 0 and clean and
                    "test result: ok. 1 passed; 0 failed" in text})
            except subprocess.TimeoutExpired:
                report["sanitizers"].append({"tool": tool, "pass": False, "status": "TIMEOUT"})
                save()
                raise
            save()
            if not report["sanitizers"][-1]["pass"]:
                raise RuntimeError("SANITIZER_FAILED: " + tool)
    report["status"] = "DIAGNOSTIC_CASES_PASS"
    save()


if __name__ == "__main__":
    main()
