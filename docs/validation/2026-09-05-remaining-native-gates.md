# Remaining implementation gates (not a completion claim)

Source inspected at `44ba7a3`. The published S13 dataset does not prove the
whole architecture is implemented.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| DENSE compact, 2 T4, archive | Complete S13, verified remote objects | Repeated paired performance runs and broader adversarial correctness |
| HASH_FIRST native | CPU leases, receipts and scheduling simulations | Native owner request/response, origin lifetime, post-commit materialization; full-state equality against DENSE |
| BMMA_BUCKET native | Config enum and architecture contract | Actual SM75 backend, backend selection, equality/collision fixtures and end-to-end comparison |
| Continuous overlap | Distributed reference uses repeated stream synchronizations and synchronous count readback | Bounded route slots, sequenced independent streams, timeline evidence; current code is not the final pipeline |
| 1 vs 2 ranks | DistributedConfig is two-rank-specific, separate single-rank executor exists | Same algorithm/output-contract benchmark; do not infer scaling from different executors |
| CLI | Root legacy executable and benchmark examples | Agreed standalone production CLI and config-to-runtime wiring |
| Macro expansion | CPU oracle and native macro components/tests | Audit multi-rank production wiring and all configured depths |
| Sanitizers | v1 matrix distributed archive fixture passes four tools | v2 compact results, further lifetime/capacity/slot cases; primitive success is not universal safety |

`cargo test --locked -p mgbfs-core -p mgbfs-runtime` completed with exit 0
on Windows after the compact fixture addition. CUDA-feature tests are not
executed by that command; their authoritative execution is on Kaggle.

Implementation dependency for HASH_FIRST: remote owner must preserve source
origin through hash-only routing, commit winners once, return dense sorted
materialization requests, retain source state until responses complete, and
release the batch only when both local and remote obligations terminate.
The existing single-rank materializer alone does not satisfy this protocol.
