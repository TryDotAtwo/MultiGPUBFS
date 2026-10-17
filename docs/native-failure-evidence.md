# Native CUDA/NCCL failure evidence

Starting with 0.2.0.dev23, the generic native engine labels its checked CUDA/NCCL FFI calls with the rank, device, target BFS layer, last completed layer, call sequence and Rust source location. The existing `GENERIC_CUDA_<code>` prefix is retained. This is diagnostic instrumentation; CUDA kernels, precompiled SM libraries, buffer geometry and BFS mathematics are unchanged.

## Changes in 0.2.0.dev24

The first native error remains in `native_error`; later cleanup errors are separate. Preserved tuning errors now name the durable evidence directory after temporary cleanup. Captured CUDA 999 in an optional probe has an explicit conservative fallback with fresh rank processes; timeouts and production BFS failures still propagate. See [tuning failure recovery](tuning-failure-recovery.md) for the policy and limits.

## Update once, then reproduce the calculation

```bash
python -m pip install --upgrade "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261017-v24/multigpubfs-0.2.0.dev24-py3-none-any.whl"
```

Start a **new Python process** after the update. For the next failing calculation, add this before `enable_native()` and the calculation:

```python
import os
os.environ["MGBFS_TRACE_NATIVE_CALLS"] = "1"

import multigpubfs
multigpubfs.enable_native()
# Keep the existing CayleyPy graph definitions and BFS options.
with multigpubfs.persistent_graph_workers():
    # Existing graph.bfs() calls go here.
    pass
```

Equivalently, launch the existing script with `MGBFS_TRACE_NATIVE_CALLS=1 python your_calculation.py`. Set the variable in the actual compute job, before creating persistent workers. Both approaches keep the existing GPU allocation and API.

An update replaces the native CLI as well as Python code. Its precompiled common component is downloaded and verified once; existing per-SM CUDA component hashes are unchanged and their cache entries remain reusable. When compute jobs must be offline, run `python -m multigpubfs prefetch --sm 120` once beforehand with the same persistent `MGBFS_NATIVE_CACHE`; substitute the known target SM or use `--all`. Leave `MGBFS_NATIVE_OFFLINE=1` for the calculation only after the new component has been prefetched. Installing the small Python wheel alone does not populate an offline native cache.

## Interpreting the evidence

Every checked nonzero native return emits `MGBFS_NATIVE_ERROR` with structured JSON and an error such as:

```text
GENERIC_CUDA_999 call=cudaMemcpyAsync rank=1 device=1 layer=7 last_completed_layer=6 source=crates/mgbfs-runtime/src/generic_distributed_native.rs:155 call_sequence=123
```

The line and sequence above are illustrative, not a measurement. The starting state is layer 0. `layer=7` means the engine was trying to compute distance layer 7, while `last_completed_layer=6` is the last completed layer. Null/unknown layer means failure occurred before layer processing. A call that reports an asynchronous CUDA error may follow the operation that originally caused it; the reporting call is **not automatically the faulty kernel**. A wrapper name beginning with `mgbfs_` identifies the FFI entry point, not every internal CUDA call inside that wrapper.

On a distributed rank failure or completion timeout, the launcher captures evidence before stopping owned workers and again after stopping them:

```text
<failed-output>/failure-evidence/summary.json
<failed-output>/failure-evidence/rank-<rank>/phase.json
<failed-output>/failure-evidence/rank-<rank>/response.json
<failed-output>/failure-evidence/rank-<rank>/worker.log
```

`response.json` is present when the worker produced a response. An interrupted rank may have phase/log evidence without a response; absence is not success. The summary records both captures, rank/device/PID/job sequence and per-file checksums, offsets and truncation. Each copied file is bounded to 2 MiB; the worker log contains the current job slice or its tail. Existing early evidence is retained when a later capture cannot read an ephemeral source file. The exception includes the saved evidence directory.

Persistent worker scratch directories may then be deleted normally: `failure-evidence` remains inside the calculation output, including a failed autotuning `pilot-*` directory. The doctor ZIP explicitly includes these phase, response and summary files in addition to logs. It excludes raw job environment files and BFS state payloads. Archive the **failed output directory including `failure-evidence`**, not only `rank-*.log`.

Without `MGBFS_TRACE_NATIVE_CALLS=1`, error details and layer checkpoints remain available. The opt-in trace also records entry/return around each checked FFI call and updates the phase file, making a blocked call distinguishable from a returned call. It adds logging and filesystem work: use it for a bounded reproducer, then disable it for speed measurements. No new GPU synchronization or automatic BFS restart is introduced.

This update does not establish or fix the root cause of a previously observed production CUDA 999 error. GPU acceptance and deliberately injected failures are reported separately from any reproduction on the affected eight-GPU server.
