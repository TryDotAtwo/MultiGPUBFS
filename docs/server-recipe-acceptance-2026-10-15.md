# Verified server recipe acceptance — 2026-10-15

Package `0.2.0.dev17`, SHA-256 `979ea02ab7f2d271bdbbac4f6b93588e4d7fb2382e99de8e6dc89ca2802105d7`; Python source commit `7e5eec96f5ad36f159d05fdea5b9ac0a13343483`. Native executable and all four CUDA build payloads are byte-identical to v16; only Python diagnostics, explicit launcher, packaging provenance and documentation changed.

Full CPU regression suite: **689 passed**. Real installed-wheel acceptance: private Kaggle `trydotatwo/multigpubfs-verified-server-recipe-acceptance`, physical **2×T4**.

Validation provenance: version 1 passed ordinary selection, the printed command and S5/S11/S12, then the test harness incorrectly expected a LAZY candidate to pass despite the native EAGER policy. The candidate was correctly rejected. Version 2 retained those exact-wheel scientific results, rebuilt its own server-bound profile and completed the remaining checks with an EAGER alternate. No old profile was carried into a new GPU job, and no unchanged S11/S12 benchmark was repeated. The public JSON preserves phase provenance and the first receipt SHA.

| Check | Result |
|---|---|
| Ordinary doctor selects both GPUs and prints a tested command | PASS |
| Printed command from another current directory, including spaces | PASS |
| Non-default static-SM120, fresh processes, EAGER, NCCL socket | PASS |
| Explicit single-GPU fresh configuration | PASS |
| Changed settings, native hash or GPU assignment rejected before BFS | PASS |
| Full deep diagnostic | PASS: {"PASS": 193, "SKIPPED": 6} |
| Recipe and proof files included in checksummed ZIP; state payloads excluded | PASS |
| One-second diagnostic budget produces no unchecked ready command | PASS |

The alternative and single-GPU test candidates were deliberately restricted by the validation harness to exercise these branches. They are actual GPU BFS/command checks, not evidence that the normal two-GPU production setup failed.

| Graph launched through the issued configuration | States | Eccentricity | BFS seconds | Observed cold-command wall seconds |
|---|---:|---:|---:|---:|
| S5 | 120 | 10 | 0.197528 | 15.702740 |
| S11 | 39916800 | 55 | 2.699858 | 20.674765 |
| S12 | 479001600 | 66 | 36.029146 | 52.791626 |

Every layer count was independently checked against the Mahonian generating polynomial; state counts equal n!, eccentricities n(n−1)/2. These timings are one installed acceptance run, not a matched speedup benchmark. T4 executed SM75 images; no execution on the original SM120 server is claimed.

Sanitized receipt: [JSON](acceptance/server-recipe-20261015.json). User instructions: [Russian quickstart](server-quickstart-ru.md).
