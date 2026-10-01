# HASH_FIRST device-driven owner/transport/retirement candidate

Hardware: two RTX A4000, Vast 53673926, 16376 MiB each. Diagnostic only: not the required 2xT4 gate. Rental $0.18444444444444444/hour; unchanged watchdog deadline 2026-10-01 13:55:48 UTC.

The candidate integrates device counts, two committed extent/control descriptors, sorted OriginRef requests and full-state responses into the existing LSA window. Payload widths are exact, not maximum-buffer padding. Last-reader events follow materialization; parents retire afterwards. A distinct LSA fatal-vote ABI preserves the legacy NCCL collective gate. A host cancellation mirror alone previously failed intermittent owner-fault replays; those failures are not erased.

Explicit-ABI candidate patch SHA256 f7e6fd4423c7aa3fa5fc1cec36810591887cc7ec70d67640af6fc88989515309 passed four nine-case independent-process HASH_FIRST replays (healthy, either-rank startup/owner/archive-admission/archive-finish faults) and DENSE regression. All fault cases terminated without forced cleanup or false group COMPLETE. U/S full-state/archive fixtures cover rank maps and pre-dedup variants. Separate capacity fixture remains DENSE-only.

The first marked S8 Nsight capture exposed a remaining generation stream drain and host fatal readback in each HASH_FIRST batch. The runtime was changed to import generation fatal and vote on-device for the LSA path. The new candidate passed the nine-case replay, full-state fixtures and S8 full archive oracle: 40320 states, 29 depths.

The corrected marked timeline contains 178 mgbfs.batch intervals per rank, with zero host Synchronize and zero non-Async memcpy calls on the submitting thread inside those intervals. This is narrower than proof of complete overlap or absence of all CPU dependencies. Async D2H remains: rank0 352 copies/1621680 bytes, rank1 346 copies/1603920 bytes. Sizes follow state/hash archive pairs; callchain extraction returned empty stacks, so callsite attribution is not independently established by those stacks.

Corrected candidate patch SHA256: a2615b78a3acd6c02c2e02046634aae19b0dea1ec782955deb2bcb35e00feee1. Current sanitizer scope is the full two-process healthy S4 HASH_FIRST path, unfiltered. Racecheck and synccheck passed. Memcheck failed with 13 CUDA API errors per rank (209 kernel image availability and 800 cuMemCreate permission). Initcheck failed at application startup: both processes returned 1 after 11.14 seconds, no forced cleanup, no group COMPLETE; its zero tool error summary does not mean successful application execution. These do not close the required four gates, T4 acceptance, asymmetric HASH_FIRST capacity tests, full owner DAG capture, overlap quantification or paired A/B.

Raw artifacts reside under test_results/vast_a4000_cancel_20261001 and remote /tmp/mgbfs-hf-generation-device-*; preserve failed and passed evidence. Shared unrelated worktree changes remain outside this candidate.
