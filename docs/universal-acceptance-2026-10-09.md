# Unified BFS acceptance audit, 2026-10-09

Implementation artifact: 73cdef732ba116c4540491b405d659a103ee91e4. CUDA13.2 SM86+PTX, two actual RTX3060 12GB on one host. Current wheel and immutable installation URL are in install-native-package.md. Source-only delivery preserves historical commits on the sole integrated GitHub branch.

| Requirement | Evidence inspected | Result and coverage |
|---|---|---|
| One public graph entry point | clean installed run_graph; real-cayley-installed-gpu/verification.json | Actual CayleyGraph object and CayleyGraphDef; permutation and matrix actions; one/all visible GPUs and reversed explicit placement |
| Preserve all CayleyPy action kinds | actual cayleypy0.2.0 GeneratorType enum, MatrixGenerator apply, real-cayleypy-adapter.log, final-test_graph_definition.py.log | Enum contains PERMUTATION and MATRIX. Integer vector identity, mixed modulus, rectangular matrices, repeated values and signed-int64 wrapping preserved. Invalid/lossy scalars rejected |
| Exact GPU BFS | error-load-gate-status.json; rolling-full-graph-gate/verification.json; error-load-retained and error-load-rolling | Full small graph CPU oracles, every completed layer/state, byte/int64/wide storage, forced hash collisions and resource boundaries. CPU successor functions are verification references; production generation/dedup are CUDA |
| Automatic history | final-clean-public/verification.json | Proved inverse-closed actions use previous/current/future VRAM banks. Directed actions and unsafe modular inverse proofs retain all visited keys |
| Compact retention | final-clean-public/verification.json | Last completed layer sample <=1000 globally; previous only if globally<1000. A 3476-state layer was sampled. Deadline, actual SIGTERM and resource stop preserve previous/current and reject partial future |
| Automatic memory/buffers | graph-info free-VRAM admission, joint memory planner, rolling-cpu-admission.log, clean public launches | Byte-exact shared table, arena/banks, route/receive/control/readout budget with headroom; weighted per-rank capacities. Logical1/2/8/128 geometry is CPU evidence |
| Bounded performance selection | final-clean-network/verification.json; heavy-installed-autotune/verification.json | Four same-graph GPU profiles, identical completed layer counts, conservative5% switch threshold, native/graph/driver/topology cache and repeated free-VRAM admission |
| External ranks | final-clean-network/verification.json | Real TCP control and NCCL with independent directories, explicit local/global placement, parameter mismatch rejection, directed/inverse/resource oracles and collective tuning on two GPUs |
| Clean package without checkout | final-wheel-status.json; final-clean-extended-status.json | Installed outside source directory, environment overrides removed; one/twoGPU oracle and terminal/network checks |
| Public package integrity | final-public-release-receipt.json | Wheel and manifest anonymously downloaded and SHA256 checked |
| Source/proof delivery and cleanup | checkpoint-7 source-only.bundle and proof archive plus publication receipt; separate exact-rental teardown receipt | Delivery and teardown must be checked after publication; a build/test receipt is not proof of either |

The heavy tuning fixture is a complete 3,628,800-state inverse-closed10-element graph. GPU layer counts and canonical terminal states matched the pre-existing full CPU enumeration by its immutable reference hash. At common depth36, scores for1shard/full batch,4/full,16/full and4/quarter were0.652231349,1.347878108,1.422294506,0.445070589s. The tuner selected4/quarter. The final BFS took0.554407533s of layer time; first-time tuning took12.473829937s and was cached. These are different intervals; a31.8% pilot improvement does not mean a31.8% end-to-end first-run speedup or a global optimum.

The isolated atomic-load change produced no acceleration in three paired runs: median old0.509893420s versus new0.516095143s. Native exactness/failure gates passed. Keep these matched results separate from former hardware/workload baselines.

## Explicit remaining boundaries

- Physical8/128GPUs, separate hosts, heterogeneous GPU models, B200/B300 and throughput scaling remain unverified. Logical geometry and two local ranks cannot prove those results.
- Calibration measures four admitted shard/batch profiles, not every possible bank/hash/sort algorithm or frontier size. It cannot guarantee maximum throughput for every graph.
- The generic byte path packs complete candidate states directly into routing records for width<=16. Wider byte/int64 candidates still use full-state transport before dedup; key-first survivor-only transport is not implemented in that path.
- Specialized SHARD_AB remains separately available; the unified generic launcher does not automatically select every specialized kernel, LSA or HOST fallback capability.
- NCCL chooses its available communication transport. The accepted network fixture used local TCP/NCCL, not physical inter-node networking.
- Pure GraphDefinition use has no CayleyPy dependency. Optional catalog integration is pinned and tested at CayleyPy0.2.0; future third-party APIs are not covered automatically.
- A development artifact with external CUDA/NCCL dependencies is not a universal precompiled binary for every architecture.

This audit proves the listed bounded implemented behavior. It does not establish the broader claim that all graphs on arbitrary hardware are already maximally optimized; further transport/algorithm selection work and any later hardware acceptance must retain these boundaries.

## Cyclic action admission follow-up

For generators consisting of one permutation and its inverse/identity/duplicate aliases, startup admission computes the least common multiple of label periods on that permutation's disjoint cycles. This is the exact start-orbit cardinality, capped at the arena index limit; unrelated generator sets keep the conservative multiset bound. The calculation visits definition entries, not BFS states, and does not enter the hot path.576 independently enumerated small cyclic fixtures matched the metadata result. The existing logical1/2/8/128 allocation contracts also passed.

Actual one/twoRTX3060 automatic launches passed17-state wide inverse rotation,2-state100-element periodic directed rotation,15-state mixed-cycle aliases and noncyclic720-state fallback. Capacities were no larger than the verified orbit cardinalities and planned buffers stayed below1MB. The single-device retained-history report now exposes its own actual memory plan and device list, preserving existing fields. `cyclic-bound-gpu-v2/verification.json` is the GPU receipt. New package publication and source delivery must be verified separately.


## Recovery acceptance

Source 2cdfdc30b88dba7fb69c9458393131157bf71ca7 was reconstructed remotely after lease teardown. Fresh Rust CLI SHA256 exactly matches the previously tested cyclic implementation. Native CUDA library is unchanged and reused from its verified public wheel. Fresh clean installed cyclic fixtures and public/network suites passed on one/two RTX3060. Proof receipt: recovery-clean-status.json. Generic wide states still transmit candidates before dedup; survivor-only transport and broader algorithm-profile tuning remain unfinished. Physical8/128 GPUs, separate nodes and Blackwell are not accepted by these tests.
