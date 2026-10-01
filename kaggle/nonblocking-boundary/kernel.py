"""Exact-source two-T4 independent-process correctness and sanitizer replay."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = "46e73bac4f3d0b53d5b17730f588de88fc058261"
os.environ.update(MGBFS_GATE_SOURCE=SOURCE, MGBFS_GATE_MODE="provision_only",
                  MGBFS_DIAGNOSTIC_HARDWARE="T4")
repo = Path("/tmp/mgbfs-nonblocking-boundary")
subprocess.run(["git", "clone", "-q", "https://github.com/TryDotAtwo/MultiGPUBFS.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", SOURCE], check=True)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
assert actual == SOURCE
binary = repo / "nccl-wrapper-test"
subprocess.run(["g++", "-std=c++17", "-Itests/nccl_stubs", "tests/nccl_transport_failure.cpp",
                "-o", str(binary)], cwd=repo, check=True)
subprocess.run([str(binary)], cwd=repo, check=True)
spec = importlib.util.spec_from_file_location("gate", repo / "kaggle/host-sized-nccl-gate/kernel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.main()
logs = Path("/kaggle/working/lsa-bfs-gate")
provision = json.loads((logs / "summary.json").read_text())
if provision.get("status") != "PROVISIONED_NOT_VALIDATED":
    raise RuntimeError("T4_PROVISION_NOT_READY: " + provision.get("status", "MISSING"))
work = Path(provision["work"])
python = str(work / "venv/bin/python")
subprocess.run([sys.executable, "-m", "pip", "--python", python, "install",
                "--no-cache-dir", "pyarrow==19.0.1"], check=True, timeout=180)
spec = importlib.util.spec_from_file_location("primitives", repo / "kaggle/native-primitives/kernel.py")
primitives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primitives)
report = dict(source=SOURCE, hardware="2xT4", status="INCOMPLETE", stages=[])
summary = logs / "independent-process-gates.json"
for tool in (None, "memcheck", "racecheck", "initcheck", "synccheck"):
    name = tool or "full-state-and-failures"
    output = logs / ("process-" + name)
    command = [python, str(work / "source/scripts/replay_lsa_cancel_candidate.py"),
               str(work), str(output)]
    command += ["--healthy-only", "--instrument-processes", tool] if tool else ["--oracle"]
    stage = dict(name=name, status="RUNNING")
    report["stages"].append(stage)
    summary.write_text(json.dumps(report, indent=2))
    try:
        primitives.run(command, cwd=repo, env=dict(os.environ), logs=logs,
                       name="process-" + name, timeout=900 if tool is None else 180)
        stage["status"] = "PASS"
    except Exception as error:
        stage.update(status="FAIL", error=str(error))
        summary.write_text(json.dumps(report, indent=2))
        if tool is None:
            raise
    if (output / "summary.json").exists():
        stage["replay"] = json.loads((output / "summary.json").read_text())
    summary.write_text(json.dumps(report, indent=2))
report["status"] = "GATES_PASS" if all(s["status"] == "PASS" for s in report["stages"]) else "GATES_OPEN"
summary.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
