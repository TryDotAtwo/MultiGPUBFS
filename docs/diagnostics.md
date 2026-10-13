# Diagnose a remote MultiGPUBFS installation

After installing the current native GitHub wheel, run one command:

```bash
multigpubfs doctor
```

Equivalent Python-module command: `python -m multigpubfs.diagnostics`.
If the server accepts Python submissions only, submit:

```python
from multigpubfs.diagnostics import run_diagnostics
report = run_diagnostics()
print(report["status"], report.get("failed_components", []))
```

The caller prints live START/PHASE/PASS/FAIL/TIMEOUT messages. A new `mgbfs-diagnostics-<UTC>-<PID>` directory contains `report.txt`, structured `report.json`, component logs, `manifest.json` with SHA256 checksums and `diagnostics.zip`. Return this directory or ZIP through the server's normal job-output interface. Nothing is uploaded automatically. If the service discards both standard output and files, its administrator must enable one of those output channels.

## Components

| Stage | Actual check |
|---|---|
| system-info | Bounded nvidia-smi query; absence is a warning, not proof CUDA failed |
| filesystem | Atomic file write/read in the output and worker-temp locations |
| native-runtime | Packaged hashes, executable permission, native key-info and resolved library paths |
| cuda-discovery | Device count from the actual native binary |
| gpu-N-context | CUDA runtime load and context initialization on each selected GPU |
| gpu-N-memory | Allocate, copy and verify 4 KiB on that GPU |
| gpu-N-kernel | Execute the real native successor CUDA kernel and check its exact output |
| gpu-N-persistent | Two inventory requests through the filesystem protocol to the same resident worker |
| gpu-N-bfs | Native single-GPU S3 BFS with exact layer oracle `[1,2,2,1]` |
| nccl-init | One rank process per selected GPU; create and close the real communicator |
| nccl-exchange | Native NCCL all-reduce with a known result |
| multi-gpu-bfs | Distributed native BFS checked against the same exact S3 oracle |

Dependent checks are explicitly SKIPPED when an earlier required stage fails. Independent checks continue. A one-GPU installation skips multi-GPU stages and can still PASS. The filesystem/persistent stages make subprocess or temporary-file restrictions visible without needing SSH.

## Bounded execution

```bash
multigpubfs doctor --devices 0,1 --stage-timeout 60 --output my-new-report
```

The timeout is per component, not for the entire diagnostic. The output directory must be new. Each component runs in an isolated process group on Linux. On timeout or process failure, the runner cleans up only its own diagnostic group, including native/rank children. It never resets GPUs, retries a failed stage, restarts production BFS or terminates an existing persistent pool. Interrupting the runner saves a failed partial report. Stage status is written before the child starts, so an externally killed diagnostic still leaves the latest partial JSON on a surviving filesystem.

Exit codes: 0 = required checks passed, 1 = component failure/timeout, 2 = invalid options or output setup failure. A successful smoke test is component acceptance on the selected local GPUs, not throughput, maximum-memory, all-graph or multi-node acceptance. Running diagnostics while another GPU job is active can change latency and available memory; run it in the service's ordinary isolated GPU job allocation.

The ZIP contains diagnostic metadata and logs, not BFS state payloads, raw worker-job environment files or authentication tokens collected from the environment. Environment collection is a small explicit whitelist. Library paths, error logs, configured network-interface names and host/platform details may still identify the server: inspect the bundle before publishing it publicly. The tiny test's raw local output is retained on the server but is excluded from the diagnostic bundle.

`NATIVE_STARTUP_QUERY_TIMEOUT` means an unanswered native service query. The doctor separates runtime discovery, GPU initialization, worker protocol and NCCL so the last printed phase and failed component narrow the cause. It cannot diagnose a service that forbids executing the diagnostic itself or hides all its output.
