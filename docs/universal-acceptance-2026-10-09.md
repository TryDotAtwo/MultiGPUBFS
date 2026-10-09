# Unified BFS acceptance audit, 2026-10-09

Current runtime artifact: fde169f6e5ae9ba11d6176c14fe7a17448133665. Earlier receipts below are historical evidence for their corresponding source revisions. CUDA13.2 SM86+PTX, two actual RTX3060 12GB on one host. Current wheel and immutable installation URL are in install-native-package.md. Source-only delivery preserves historical commits on the sole integrated GitHub branch.

| Requirement | Evidence inspected | Result and coverage |
|---|---|---|
| One public graph entry point | clean installed run_graph; real-cayley-installed-gpu/verification.json | Actual CayleyGraph object and CayleyGraphDef; permutation and matrix actions; one/all visible GPUs and reversed explicit placement |
| Preserve all CayleyPy action kinds | actual cayleypy0.2.0 GeneratorType enum, MatrixGenerator apply, real-cayleypy-adapter.log, final-test_graph_definition.py.log | Enum contains PERMUTATION and MATRIX. Integer vector identity, mixed modulus, rectangular matrices, repeated values and signed-int64 wrapping preserved. Invalid/lossy scalars rejected |
| Exact GPU BFS | error-load-gate-status.json; rolling-full-graph-gate/verification.json; error-load-retained and error-load-rolling | Full small graph CPU oracles, every completed layer/state, byte/int64/wide storage, forced hash collisions and resource boundaries. CPU successor functions are verification references; production generation/dedup are CUDA |
| Automatic history | final-clean-public/verification.json | Proved inverse-closed actions use previous/current/future VRAM banks. Directed actions and unsafe modular inverse proofs retain all visited keys |
| Compact retention | final-clean-public/verification.json | Last completed layer sample <=1000 globally; previous only if globally<1000. A 3476-state layer was sampled. Deadline, actual SIGTERM and resource stop preserve previous/current and reject partial future |
| Automatic memory/buffers | graph-info free-VRAM admission, joint memory planner, rolling-cpu-admission.log, clean public launches | Byte-exact shared table, arena/banks, route/receive/control/readout budget with headroom; weighted per-rank capacities. Logical1/2/8/128 geometry is CPU evidence |
| Bounded performance selection | final-clean-network/verification.json; heavy-installed-autotune/verification.json | Five packed/seven wide same-graph GPU profiles, identical completed layer counts, conservative5% switch threshold, native/graph/driver/topology cache and repeated free-VRAM admission |
| External ranks | final-clean-network/verification.json | Real TCP control and NCCL with independent directories, explicit local/global placement, parameter mismatch rejection, directed/inverse/resource oracles and collective tuning on two GPUs |
| Clean package without checkout | final-wheel-status.json; final-clean-extended-status.json | Installed outside source directory, environment overrides removed; one/twoGPU oracle and terminal/network checks |
| Public package integrity | final-public-release-receipt.json | Wheel and manifest anonymously downloaded and SHA256 checked |
| Source/proof delivery and cleanup | checkpoint-7 source-only.bundle and proof archive plus publication receipt; separate exact-rental teardown receipt | Delivery and teardown must be checked after publication; a build/test receipt is not proof of either |

The heavy tuning fixture is a complete 3,628,800-state inverse-closed10-element graph. GPU layer counts and canonical terminal states matched the pre-existing full CPU enumeration by its immutable reference hash. At common depth36, scores for1shard/full batch,4/full,16/full and4/quarter were0.652231349,1.347878108,1.422294506,0.445070589s. The tuner selected4/quarter. The final BFS took0.554407533s of layer time; first-time tuning took12.473829937s and was cached. These are different intervals; a31.8% pilot improvement does not mean a31.8% end-to-end first-run speedup or a global optimum.

The isolated atomic-load change produced no acceleration in three paired runs: median old0.509893420s versus new0.516095143s. Native exactness/failure gates passed. Keep these matched results separate from former hardware/workload baselines.

## Explicit remaining boundaries

- Physical8/128GPUs, separate hosts, heterogeneous GPU models, B200/B300 and throughput scaling remain unverified. Logical geometry and two local ranks cannot prove those results.
- Calibration measures five packed or seven wide admitted shard/batch/transport/order profiles, not every possible kernel, history algorithm or frontier size. It cannot guarantee maximum throughput for every graph.
- Packed byte states<=24 use exact routing records. Wider byte/int64 states have a measured parent-origin alternative that removes full candidate materialization/transmission; full transport remains a reproducible fallback. Destination regeneration and parent replication can lose on different actions/topologies.
- Specialized SHARD_AB remains separately available; the unified generic launcher does not automatically select every specialized kernel, LSA or HOST fallback capability.
- NCCL chooses its available communication transport. The accepted network fixture used local TCP/NCCL, not physical inter-node networking.
- Pure GraphDefinition use has no CayleyPy dependency. Optional catalog integration is pinned and tested at CayleyPy0.2.0; future third-party APIs are not covered automatically.
- A development artifact with external CUDA/NCCL dependencies is not a universal precompiled binary for every architecture.

This audit proves the listed bounded implemented behavior. It does not establish the broader claim that all graphs on arbitrary hardware are already maximally optimized; further specialized/history algorithm selection work and any later hardware acceptance must retain these boundaries.

## Cyclic action admission follow-up

For generators consisting of one permutation and its inverse/identity/duplicate aliases, startup admission computes the least common multiple of label periods on that permutation's disjoint cycles. This is the exact start-orbit cardinality, capped at the arena index limit; unrelated generator sets keep the conservative multiset bound. The calculation visits definition entries, not BFS states, and does not enter the hot path.576 independently enumerated small cyclic fixtures matched the metadata result. The existing logical1/2/8/128 allocation contracts also passed.

Actual one/twoRTX3060 automatic launches passed17-state wide inverse rotation,2-state100-element periodic directed rotation,15-state mixed-cycle aliases and noncyclic720-state fallback. Capacities were no larger than the verified orbit cardinalities and planned buffers stayed below1MB. The single-device retained-history report now exposes its own actual memory plan and device list, preserving existing fields. `cyclic-bound-gpu-v2/verification.json` is the GPU receipt. New package publication and source delivery must be verified separately.


## Recovery acceptance (historical)

Source 2cdfdc30b88dba7fb69c9458393131157bf71ca7 was reconstructed remotely after lease teardown. Fresh Rust CLI SHA256 exactly matches the previously tested cyclic implementation. Native CUDA library is unchanged and reused from its verified public wheel. Fresh clean installed cyclic fixtures and public/network suites passed on one/two RTX3060. Proof receipt: recovery-clean-status.json. Generic wide states still transmit candidates before dedup; survivor-only transport and broader algorithm-profile tuning remain unfinished. Physical8/128 GPUs, separate nodes and Blackwell are not accepted by these tests.


## Packed24 route extension (historical checkpoint)

Route records remain32 bytes:8 bytes hash plus24 exact state bytes. Permutation states17..24 bytes no longer allocate/generate/exchange a separate full candidate plane. Wider states and int64 states retain the existing exact transport. Owner tests used forced full hash collisions with states differing only in the final bytes, malformed counts and cross-shard pending origins on both RTX3060 GPUs. Logical1/2/3/8/128 source layouts are synthetic, not rank-scale acceptance. Public automatic launches matched independent full-state CPU oracles at widths16/17/23/24/25, including the fallback boundary.

Three alternating matched full runs of the3,628,800-state10-element permutation graph embedded in17 coordinates with7 fixed values produced layer medians0.737101221s old versus0.482075518s new, ratio1.529016. Every layer count and canonical retained terminal state matched. Both used2RTX3060,4shards, batch65536, capacity3628800. This is a measured improvement for this fixture; setup/wall timing remains separate. General survivor-only transport above24 bytes and broader algorithm selection remain unfinished.

Clean installed Packed24 package passed `VERIFIED_INSTALLED_PACKED24_PUBLIC_NETWORK_TERMINAL`; immutable public wheel and manifest both passed anonymous SHA256 readback. Runtime source is `a7198174dd778e59dcf234c04c4837e51d6e30b3`.


## General wide parent-origin path (historical checkpoint)

Full wide candidates remain an explicit fallback; the new parent-origin path eliminates their materialization and transmission. It preserves full-state equality, including forced full-hash collisions, via GPU reconstruction from immutable origin-indexed parent caches. CPU1/2/8/128 allocation contracts include `(world+2)*batch*elements*state_bytes + world*12` extra parent/cursor/count bytes, and remove padded candidate payloads. GPU owner collision and stale-lease checks passed on bothRTX3060, with synthetic1/2/3/8/128-source layouts. Full independent layer/state oracles passed one/twoGPU directed and inverse-closed permutations25/31/300elements, byte/int64 storage, mixed-modulus matrices, signed actions and overflow fallback. Deadline/cancellation/resource snapshots passed on the wide path. Actual two-rank TCP/NCCL tests and six-profile collective tuning passed; physical multi-host and8/128 hardware remain unverified.

Three alternating full same-configuration runs of a25-coordinate embedding of the3,628,800-state permutation graph produced median layer time0.858009269s full-child versus0.560284978s parent-origin, ratio1.531380106. Every completed layer count and canonical terminal retained state matched. These are layer times on2RTX3060,4shards, batch65536, capacity3628800; setup/wall time and first-time tuning are separate. Broader hash/sort algorithm selection, actual untested hardware and final new package acceptance remain pending.


## Radix origin ordering and component memory bound

GPU correctness passed parent/full transport, packed4/17/24byte and wide25/31/300coordinate cases, int64 and matrix actions, directed retained history and inverse-closed rolling history on one/two RTX3060. Synthetic source layouts1/2/3/8/128 exercised forced full-hash collisions and stale leases; max uint64 hash/padding and invalid counts passed on both physical GPUs. All exact completed layers and retained terminal states matched independent CPU oracles. Local seven-profile wide tuning/cache and collective two-rank TCP/NCCL tuning passed with automatically bounded40320-state capacity for a25coordinate graph with17fixed coordinates. Component-bound CPU randomized exact-oracle checks passed; no new CPU production state generation was added.

On the same complete3628800-state25coordinate embedding, three alternating paired runs measured median layer time0.561821422s unsorted versus0.641451771s radix, ratio0.87585918. Radix was about14.17percent slower, so this is evidence against forcing radix on that workload. All layers and canonical terminal states matched. Measurements cover twoRTX3060, fourshards,batch65536,capacity3628800; startup/wall/tuning costs remain separate. This path sorts candidate indices and retains hash-table history; it does not implement a globally sorted merge history. Physical8/128GPUs, multi-host and Blackwell remain untested.

Clean installed parent/radix package passed VERIFIED_INSTALLED_PARENT_SORT_PUBLIC_TERMINAL_NETWORK_AUTOTUNE; public wheel and manifest passed anonymous SHA256 readback. Runtime source `fde169f6e5ae9ba11d6176c14fe7a17448133665`.


## Current delivery and unfinished integration

Clean installed current runtime, public wheel and anonymous integrity checks are complete. GitHub delivery is one integrated branch. Private immutable proof revisions: parent checkpoint `ad03c53b84e8d6d31cb39b70b6936605f08b8558`, radix/component checkpoint `5b29ba983df2394d372d0f91a4eb33ab33f61cfb`, clean-package proof `2338a96ff06ff869a697fa418d9245d021049028`. Source-only transport does not download state/proof archives to the user's computer.

Remaining engineering scope includes automatic selection of specialized SHARD_AB/LSA/HOST capabilities where contracts permit, and any additional persistent sorted-history algorithm/profile integration. Optional radix ordering currently feeds exact hash-table membership, not sorted-history merge. These points prevent declaring the broader optimization goal complete. Physical multi-node/8/128GPU/Blackwell acceptance remains explicitly outside the current two3060 evidence. Current rental token removal and APIABSENT teardown are pending while authorized work continues under fixed independent guards.
