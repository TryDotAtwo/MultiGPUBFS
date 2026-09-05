# Coordinated VRAM admission: v28 hardware gate

Kaggle distributed-sanitizer v28, source
`664327848f74130c87a0628342b88d2edfa88d8c`, two real T4 devices, COMPLETE.
Retained summary and runtime logs: `test_results/distributed-sanitizer-v28/`.

Verified eleven runtime fixtures in plain and all four sanitizers. The
asymmetric VRAM fixture passes in every mode: only rank 1 requests an impossible
reserve, yet both ranks return the group preflight error before large runtime
allocation. Three runtime ERROR summaries are zero; runtime racecheck reports
zero hazards, errors and warnings.

The pinned launcher's COMPLETE summary includes all twelve two-process profile
smokes. It checks positive explicit device allocation payload/aligned counts,
1 GiB reserve, full warmup, matching layers and archive verification.

This validates the explicit-device ledger/admission slice, not global startup
failure handling, OS/platform pinned or disk admission, or no later NCCL
internal allocations. The subsequent archive startup reordering and one-rank
runtime support are assigned to v29 and not proved by this gate.
