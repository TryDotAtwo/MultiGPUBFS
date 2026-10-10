# Precompiled four-B300 production

The image contains real SM103 cubins for B300 and real SM120 cubins for the
development RTX5070 validation. The core shared library has no PTX fallback.
Cross compilation is not physical B300 execution; every production launch
still requires the CPU-oracle startup gate on its actual four devices.

Use the published immutable image digest, mount a token file and persistent
results, and pass these arguments to its entrypoint:

```sh
--root /results/lrx-production --expected-gpus 4 \
--repo-id TryDotAtwo/multigpubfs-bfs-results \
--token-file /run/secrets/hf_token --deadline-unix ABSOLUTE_UNIX_WORK_DEADLINE
```

No compilation, package installation or source fetching occurs at launch.
The work deadline excludes the external lease's publication/deletion reserve.
An external exact-instance budget guard remains required: the image does not
contain Vast credentials and cannot impose a provider billing cap.

Startup verifies CUDA and exact GEMM across the visible GPUs, then checks the
automatic backend against the independent n9r1 CPU oracle. The sweep uses
the current run_graph hardware-derived admission, automatic profile cache,
and measured small/large-frontier profiles. The profile reports preserve
tuning time separately from BFS time; profiles are reused only when their
hardware/configuration/dependency identities pass the existing cache gates.

The default grid is n=2..128, r=1..n: 8255 pairs. Two value-relabelled starts
check common completed layer counts; complete pairs also compare canonical
terminal states. Only compact final states (up to1000) and the previous layer
when below1000 are retained. Closed pairs are grouped into sequential tar.gz
cohorts by the background publisher. Every cohort is SHA256-verified on HF
readback; the final ledger, manifest and report are pinned by HF revisions.

Only source-confirmed generic table/arena/future capacity stops may prune
larger n at fixed r. This is an explicitly labelled heuristic exclusion,
never a computed graph or a COMPLETE result. Other failures stop the run.
Pending pairs, one-seed partial cases and resource stops remain distinct.
Existing output roots fail rather than silently restarting an old sweep.

The flag --allow-development-hardware is only for cheap-host acceptance.
It produces a DEVELOPMENT receipt and does not validate B300 execution.
--preflight-only performs startup checks and publishes their evidence without
running the pair grid.

CUDA 13.2 native package requires NVIDIA driver branch R580 or newer. Startup checks the branch before launching BFS; actual GPU capability tests remain mandatory. https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
