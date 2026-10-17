# Optional autotuning failures and native evidence (0.2.0.dev24)

## What changed

A captured CUDA 999 in an optional distributed tuning probe no longer automatically aborts the caller's main BFS. The library preserves the probe's evidence, detects a recorded native failure before waiting for process exit, closes every process in the failed resident rank group, discards its pending admission ticket, and performs fresh memory admission for the original graph and selected devices. It issues a RuntimeWarning and returns an explicitly **unmeasured** baseline profile with status `TUNING_PROBE_FAILED_CONSERVATIVE_PROFILE`. The main BFS then starts in a new rank group. A failed probe is not retried.

This handles failure of the optional selector. It does **not** establish or repair the underlying cause of an observed CUDA 999 on a particular eight-GPU server. A reporting `cudaMemcpyAsync` can expose an earlier asynchronous failure. No claim is made that a hardware/driver problem or a faulty kernel has been eliminated.

The baseline retains normal VRAM admission, all selected GPUs and explicit user configuration. Without overrides it uses the generic CUDA/HASH path and one shard. It is not labeled optimal or measured. Comparative and online tuning are disabled for this fallback. In a persistent context, later calls with the same runtime, selected devices and environment reuse the conservative policy rather than repeating the failed probe; each graph still receives its own memory admission. A new context can attempt tuning again.

| Situation | Behavior |
|---|---|
| Healthy tuning | Existing measured/cached selection |
| Optional probe with captured primary CUDA 999 | Warn, preserve evidence, discard failed group, fresh unmeasured baseline admission |
| Missing/corrupt evidence or another unexplained rank failure | Propagate the error |
| Query or completion timeout | Propagate the timeout; no automatic retry |
| Fresh fallback admission fails | Propagate that failure; no retry |
| Main BFS fails | Preserve evidence and propagate; no automatic BFS restart |
| Main BFS reaches a resource limit | Keep its honest INCOMPLETE result and completed-layer evidence |

## Update and run

```bash
python -m pip install --upgrade "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261017-v24/multigpubfs-0.2.0.dev24-py3-none-any.whl"
```

Use a new Python process after updating. The existing CayleyPy API remains:

```python
import multigpubfs
from multigpubfs import NativeOptions
from cayleypy import CayleyGraph, PermutationGroups

multigpubfs.enable_native(NativeOptions(devices=tuple(range(8)), autotune=True))
with multigpubfs.persistent_graph_workers():
    for n in (5, 6, 7):
        graph = CayleyGraph(PermutationGroups.coxeter(n), device="cpu")
        result = graph.bfs()
        print(n, result.diameter())
```

Select the device count actually allocated to the job. `device="cpu"` constructs the CayleyPy graph definition; native activation dispatches BFS to the selected GPUs. Existing `autotune=False` usage remains supported.

For offline jobs, installation of the small wheel does not refresh the native cache. Prefetch once, without GPU access, into the same persistent `MGBFS_NATIVE_CACHE`:

```bash
MGBFS_NATIVE_OFFLINE=0 python -m multigpubfs prefetch --all
```

Or select the known architectures with `--sm 100,120`. Existing per-SM CUDA components are unchanged and remain reusable. The rebuilt common component is downloaded and checksum-verified once. Compute jobs may then keep `MGBFS_NATIVE_OFFLINE=1`.

## Failed communicator teardown

After a fatal generic native error, the runtime aborts its NCCL communicator before draining streams and never parks the failed communicator for reuse. The supervisor observes both resident and one-shot phase files; a recorded nonzero native error is a rank failure even while the process is unwinding. Healthy-run completion polling remains prompt, while phase-file reads are bounded to one check per 50 ms. A timeout without recorded native failure remains a timeout and does not trigger the optional-probe fallback.

## Evidence and interpretation

- The exception's `evidence=` path and copied summary metadata point to the durable preserved directory after temporary-workspace cleanup.
- `failure-context.json` retains both the rebased error and `original_error` for provenance.
- `phase.json.native_error` is the **first** error of that native job.
- `native_last_error` and `native_secondary_error_count` separately retain later failures, including teardown.
- `tuning_failure.source_graph_digest` identifies the graph whose optional probe failed. On later graphs in the same persistent session it remains the source of the conservative policy.
- The fallback report sets `measured=False`; a successful main graph is still distinguished from an INCOMPLETE graph.
- Opt-in `MGBFS_TRACE_NATIVE_CALLS=1` continues to add per-call entry/return evidence. Disable the trace for performance measurements.

The changed Rust CLI preserves native context without changing CUDA kernels, per-SM libraries, CUDA runtime, NCCL or BFS mathematics. Installed physical-GPU validation is recorded separately from CPU regressions and compile success.
