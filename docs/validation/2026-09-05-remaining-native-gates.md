# Remaining implementation gates (not a completion claim)

Updated through `b5f7a1e`; v17 hardware gate is running. The published S13 dataset does not prove the
whole architecture is implemented.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| DENSE compact, 2 T4, archive | Complete S13, verified remote objects | Repeated paired performance runs and broader adversarial correctness |
| HASH_FIRST native | Scalar CUDA hash-only generation, NCCL requests/responses, selected regeneration and owner materialization connected in b5f7a1e | Full archive fixture v17 pending; seed/map/pre-dedup matrix next; Tensor Core generation/hash path and overlap still required |
| BMMA_BUCKET native | Config enum and architecture contract | Actual SM75 backend, backend selection, equality/collision fixtures and end-to-end comparison |
| Continuous overlap | Distributed reference uses repeated stream synchronizations and synchronous count readback | Bounded route slots, sequenced independent streams, timeline evidence; current code is not the final pipeline |
| 1 vs 2 ranks | DistributedConfig is two-rank-specific, separate single-rank executor exists | Same algorithm/output-contract benchmark; do not infer scaling from different executors |
| CLI | Root legacy executable and benchmark examples | Agreed standalone production CLI and config-to-runtime wiring |
| Macro expansion | CPU oracle and native macro components/tests | Audit multi-rank production wiring and all configured depths |
| Sanitizers | v16 matrix/compact distributed archive and native HASH_FIRST stages pass four tools | v17 integrated HASH_FIRST pending; further lifetime/capacity/slot cases; primitive success is not universal safety |

`cargo test --locked -p mgbfs-core -p mgbfs-runtime` completed with exit 0
on Windows after the compact fixture addition. CUDA-feature tests are not
executed by that command; their authoritative execution is on Kaggle.

Implementation dependency for HASH_FIRST: remote owner must preserve source
origin through hash-only routing, commit winners once, return dense sorted
materialization requests, retain source state until responses complete, and
release the batch only when both local and remote obligations terminate.
The existing single-rank materializer alone does not satisfy this protocol.

## HASH_FIRST dependency audit (historical findings at 44ba7a3)

Canonical sections 7/9.2 require hash + OriginRef without persistent child
states, followed by regeneration only for committed survivors. Current
`cuda/materialize.cu` copies rows from a previously materialized source slot;
it cannot regenerate from parent + move and must not be relabeled HASH_FIRST.
`distributed_native.rs` currently exchanges complete packed states before
owner commit. Its `commit_owner_batch` constructs identity row references,
so an OriginRef would be lost if only the packet format were changed.

Next native leaf must regenerate selected parent/move requests into dense
response order, validate source lifetime/ranges before stores, and preserve
the request-to-target mapping. Test this against independent matrix successor
states before connecting owner commit and NCCL requests. Then replace the
candidate generator output with hash-only generation (not a full child buffer
followed by discarding states). Both changes are required for the profile's
memory contract. Existing CPU receipts and transport are verification oracles,
not production native implementations.

## Integration status at b5f7a1e

The historical missing-path findings above now have an implementation, not yet
a full-runtime hardware proof. HASH_FIRST avoids the DENSE child-state buffers
and DENSE generation/hash plans. It preallocates OriginRef packets, per-source
request/target arrays, two response buffers and shared materialization scratch.
Requests are capacity-checked before owner commit. Source parents remain live
through local regeneration, remote response completion and the archive event.
Responses are sorted by target and written as contiguous reserved ring extents.
The current reference deliberately synchronizes stages and reads counts on the
host; it does not satisfy the final continuous-overlap performance contract.
V17 pins b5f7a1e and adds a sixth full archived BFS oracle fixture. An expanded
seed/rank-map/pre-dedup fixture is prepared separately and is not part of v17.
