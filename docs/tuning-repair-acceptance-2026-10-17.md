# Autotuning failure containment and diagnostics acceptance — 2026-10-17

Release: **0.2.0.dev24**. Package source: `cf973ddf72d4b47dcc1b8d2fbcff9d9f75367259`. Native CLI source: `7200e28377dad7269cb7e93ebb234698ca70235c`.
Wheel SHA-256: `693917bbede1fe2f4808481f32335c3df0d4bda77e7f0eafe09e48537e36980d`.
Native CLI SHA-256: `e41928d64cd1c4f291bf2872144a1bd1713cdb10f5dbab6a04686724a7292223`.
Measured GPU receipt SHA-256: `9e63f84fcbd2a73eb3b5b14d930bb22686d0356a67dff3d53ecbf771ef667d1d`.

## Result and limits

The optional-probe failure boundary, durable evidence paths, primary native error retention and the documented top-level persistent API passed installed-package acceptance on **two physical Tesla T4 GPUs (SM75)**. [Machine-readable aggregate receipt](acceptance/tuning-fault-recovery-20261017.json).

Acceptance spans private Kaggle versions 4 and 6 of `trydotatwo/multigpubfs-tuning-repair-acceptance`, both installing the exact wheel above. Version 4 accepted six cases, including persistent optional-failure recovery. Its overall result is FAIL because the test-only one-shot injector did not activate for that filename format; the subsequent assertion correctly rejected the missing injection. After the injector correction, version 5 confirmed optional-failure containment and successful main BFS, then rejected a test expectation that a second cleanup fault must always occur: the fail-fast supervisor can terminate a one-shot process before cleanup. Version 6 ran only the two pending cases with the corrected expectation, carrying the six accepted cases with their original receipt hash and hardware provenance. The [unaltered partial version-4 receipt](acceptance/tuning-fault-recovery-partial-v4-20261017.json) has SHA-256 `ccbcb4270eb9412667ae9d0752d8718908fc908e6ce3175d728a52995484d627`. No shipped code changed between these runs. The public acceptance notebook below runs all eight cases with the corrected test harness.

This does **not** reproduce or establish the underlying cause of the production eight-GPU CUDA 999. Healthy old dev23 also completed the synthetic matrix workload on these two T4 GPUs. No eight-GPU acceptance or throughput improvement is claimed. The new behavior deliberately exposes an **unmeasured conservative baseline** after a captured optional-probe failure; it does not pretend the failed profile was measured or optimal.

## Verified behavior

| Check | Evidence |
|---|---|
| CPU Python regression | 790 tests; GitHub Actions 39035507929 |
| Native workspace regression | 461 tests, 97 binaries, zero ignored; Actions 39035062147 |
| Diagnostic red/green | Stale path and overwritten first error fail before fix; Actions 39031993013 / 39032115229 |
| Optional-probe red/green | Previously aborted the operation; Actions 39032558779 / 39032681804 |
| Early native failure supervision | Failure was hidden behind teardown/completion timeout; Actions 39034632989 / 39035060345 |
| Public API red/green | Installed GPU run exposed missing top-level persistent context; export fixed and regression added before final wheel |
| Clean installation | Wheel and pinned GitHub source install without GPU/network initialization on import |
| Normal native BFS | Permutation, repeated-symbol and matrix fixtures, two persistent calculations each |
| CayleyPy public API | Persistent S3/S4/S5; exact diameters 3/6/10 |
| Healthy tuning | Native matrix16/i64 run and repeat complete with exact CPU layers |
| Injected optional failure | CUDA 999 at layer 2 on rank 1; persistent and one-shot cases |
| Main BFS after optional failure | Completes exact graph in new rank PIDs; failed PIDs are not reused |
| Next persistent call | Reuses healthy ranks and conservative policy rather than retrying poisoned probe |
| Diagnostic cleanup | Persistent proof retains first 999/cudaMemcpyAsync separately from 21 later cleanup faults; early one-shot termination need not reach cleanup |
| Durable path | Updated exception and summary identify existing preserved evidence; original temporary path has been cleaned |
| Main calculation error | Propagates; no automatic production BFS restart |

The matrix fixture uses six 4×4 permutation matrices modulo 257 and a 256-scaled identity start: 16 int64 state elements, 24 reachable states, layers `[1,6,11,6]`. It exercises the ABI shape of a matrix16/i64 probe but is **not** the original user's graph.

In the deliberately failing `main-failure` case, the receipt's `states` and `layers` fields describe this fixture's CPU oracle, not a completed native BFS. The accepted outcome for that case is the recorded native exception and the absence of an automatic restart. The aggregate `wall_seconds` measures only the version-6 remainder, not both acceptance sessions or any throughput benchmark.

CUDA 999/901 in the failure cases is produced by a **test-only LD_PRELOAD interposer**. It is absent from the shipped wheel and native library. Real synchronization still drains GPU work before the interposer reports the later cleanup failure.

Earlier installed candidates were rejected after tests found the missing public export and stalled NCCL teardown hiding a native fault behind a completion timeout. Abort-before-stream-drain and early phase supervision were added before the final package. The final wheel is bound to the receipt above. CUDA kernels, per-SM libraries, runtime and NCCL are unchanged.

[Recovery policy and server instructions](tuning-failure-recovery.md).
