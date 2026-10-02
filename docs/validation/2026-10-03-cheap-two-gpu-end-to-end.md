# Cheap two GPU end-to-end acceptance

Verified on 2026-10-02 UTC / 2026-10-03 Moscow, using two RTX 3060 12 GiB
on the isolated Vast instance 53913282 (machine 21769). The instance was
destroyed after all gates; provider deletion success, list absence and the
independent watchdog's ABSENT receipt were checked. The private HF credential
was removed from the GPU host before deletion. No GPU states were downloaded
to the user's computer.

Frozen final report:
[faa00319](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/faa00319ddc6cf89bb2897d6721b9c5177c02816/evidence/20261002-e2e/final-report.json).
The report includes exact configurations, source archives and verification
receipts. Validated program commit: `bf2c96de9a0c732881e1525c992f4854647fe4fb`.

## Automatic coverage

The full single-command sweep at `8f71f87af0c594b5d870a5a885ac281cd081effd`
used no n/r range or capacity arguments. It classified all 527 pairs with
2 <= n <= 32 and 1 <= r <= n: 245 attempted, 210 COMPLETE, 35 INCOMPLETE,
138 independently pruned at fixed r, and 144 excluded by packing/u64 limits.
There were no pending pairs. All 8,250 payloads / 27,089,579,320 bytes and
245 manifests passed streamed HF readback at `491252c15c374d8b09ea960c7241611e8dab6d64`.
Thirteen background publication generations were observed, including publication
before computation finished. The 27 GB cohort required two bounded payload
commits before metadata. Resume performed no native relaunch.

All table metadata and retention checks passed: exact start and L/R/X,
8/16-byte nibble packing, every completed layer's counts and whole-layer time,
per-GPU observed peaks at 50 ms intervals, whole COMPLETE tails >=3 layers
and >=10 decimal GB or graph start, INCOMPLETE suffixes <=1 GB, file depths,
counts, full/partial flags, sizes and SHA-256. Short layers without samples
retain null rather than an invented VRAM peak. Independent complete-word CPU
BFS checks passed for 147 small COMPLETE graphs.

Ten archive-related stops were rerun after correcting cgroup clean-file-cache
accounting and using the existing batch-sized archive scratch. There were no
archive failures: (17,10), (24,18), (25,19), (26,20) completed; six remaining
pairs returned source-confirmed capacity codes. The combined coverage is
214 COMPLETE and 31 INCOMPLETE among 245 attempted pairs. This is the original
full sweep plus explicitly identified targeted reruns, not a second full sweep.
All 708 rerun/cancellation/benchmark payloads / 13,777,931,144 bytes and
14 manifests passed readback at `e2c60edb6fb1de39f6b32fb8cb188fab65b75289`.

Pruning accepts confirmed device allocation failures and native capacity codes
11/12/16, never generic archive/transport/timeout failures. The earlier v2
ledger and its 149 untrusted prunes remain unchanged with explicit errata.
Automatic memory sizing is conservative; maximum hardware capacity is not proven.

## Storage, cancellation and timing

The separate physical synthetic fixture generated 13 GB, published and read back
four whole layers / 10.4 GB and a 1 GB partial suffix. These are storage-test
payloads, not a completed native graph of that size.

55 Linux CPU tests passed, including publication-error summaries, snapshot
preservation and controlled-stop eligibility. The native numeric fatal-word GPU
test passed. A real SIGTERM after committed native layers drained and published
six complete layers / 624 bytes as INCOMPLETE. A separate actual automatic CLI
SIGTERM stopped after two attempted cases, published and checked both manifests
and two payloads / 16 bytes, and preserved 381 eligible pending pairs with no
resource pruning. Its frozen ledger/readback is
`506776a100a50c6bcb4d346493d82a1286523767`.

On the same 29-layer (15,4) prefix, three repeated CUCO_RANK runs gave median
whole-layer totals of 1.074159 s without archive and 1.443065 s with archive
(32768 archive rows). The original Rank/Indexed comparison also matched this
prefix: no-archive medians 1.076023 / 1.189822 s, archived 1.388074 / 1.510326 s.
These are matched-prefix measurements, not a completed (15,4) graph or a
universal fastest-backend claim.
