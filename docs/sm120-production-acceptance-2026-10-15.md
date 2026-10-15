# Shared SM120 production package acceptance — 2026-10-15

Package `0.2.0.dev16`, [GitHub Release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261015-v16).
Wheel SHA256: `65941655242e7a48fa3f27989992fe5f8d5d898c716d481ce0c92cdd9d4a5a6b`. Source/build and finite GPU results are in
[the machine-readable receipt](acceptance/sm120-production-20261015.json).

The normal production library now uses shared CUDA runtime linkage and native
SM75, SM86 and SM120 cubins. All 24 CUDA compilation modules contain all three
architectures. The native Rust executable and BFS algorithms remain unchanged;
the source identities are explicit in the receipt. Native SM120 images are
compiled evidence, not an assertion that a T4 executed SM120 instructions.

Remote CPU acceptance: 672 project tests, including 89 diagnostic regressions,
passed. Clean Linux installation directly from the public GitHub URL, explicit
CayleyPy activation, private library resolution and `pip check` passed.

Private Kaggle `trydotatwo/multigpubfs-sm120-production-acceptance`, version 2, ran the exact published
wheel on physical two T4 GPUs. Ordinary doctor passed. Full deep execution took
1021.091 seconds, counts `{"PASS": 191, "SKIPPED": 6}`.
All four build variants passed their direct API, resident and exact S3 BFS
checks on both GPUs. The production library also passed normal single/multi-GPU
CayleyPy BFS, distributed generic/matrix and specialized HASH/SORT_MERGE probes.
Unsupported GEMM on SM75, uncompiled LSA and unavailable tracing tools remain
explicit skips. The original target server still needs its own fresh doctor
report; no original-server SM120 BFS acceptance is claimed.

The [source notebook](../examples/acceptance_shared_sm120_t4.ipynb) reproduces
the installed two-T4 acceptance. The normal CayleyPy API used automatic profiles and a persistent worker pool.
Every Coxeter layer count was checked against an independent inversion-count
generating polynomial, as well as total n! and diameter n(n-1)/2.

| n | GPUs | States | Diameter | End-to-end seconds |
|---:|---:|---:|---:|---:|
| 3 | 1 | 6 | 3 | 3.336 |
| 5 | 1 | 120 | 10 | 0.071 |
| 7 | 1 | 5,040 | 21 | 0.065 |
| 3 | 2 | 6 | 3 | 2.655 |
| 5 | 2 | 120 | 10 | 0.075 |
| 7 | 2 | 5,040 | 21 | 0.081 |
| 11 | 2 | 39,916,800 | 55 | 1.964 |
| 12 | 2 | 479,001,600 | 66 | 25.936 |

These are finite correctness checks with observed end-to-end times, not a
controlled benchmark or a claim of optimal device utilization.

The owned-process regression first reproduced immediate group inspection after
leader exit and duplicate cleanup. Normal and deep doctor now wait through a
bounded TERM/KILL group deadline, record cleanup once, preserve post-cleanup
evidence on failure, reject a false PASS and stop further probes if a child
really remains. Production verification and broad coverage run before legacy
JIT/build contrasts. The ordinary doctor was rechecked after the deep suite;
the one-second global-budget case correctly returned INCOMPLETE and an archive.

Diagnostic ZIP manifests were read back and verified. No BFS state payloads,
collaborator diagnostic archives or full raw job environments are published.

Update inside the assigned GPU job, in a fresh Python process:

```sh
python -m pip install --upgrade --force-reinstall --no-deps "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261015-v16/multigpubfs-0.2.0.dev16-py3-none-linux_x86_64.whl#sha256=65941655242e7a48fa3f27989992fe5f8d5d898c716d481ce0c92cdd9d4a5a6b" && python -m multigpubfs.diagnostics
```

For ordinary calculation and Python-only submissions see
[the server quickstart](server-quickstart-ru.md).
