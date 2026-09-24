# Isolated LSA peer exchange under unfiltered memcheck, 2×T4

Private Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v44 ran pinned
source `48bc324c771128e327015c32f4ea4a8f8fab2b3f` on two physical
P2P-capable T4s. V43 selected a non-P2P host and returned
`UNSUPPORTED_HOST` before build; it supplies no test result.

The new ignored `lsa_one_exchange_matches_peer_payload` fixture initializes
two ranks, allocates/activates one NCCL LSA receive window per rank, exchanges
one Hash128 and one 16-byte state each way, and checks received count, fatal
word and exact peer payload. It bypasses BFS generation, routing, owner,
archive and retirement.

- Plain fixture: **passed** in 1.32 s.
- Unfiltered `compute-sanitizer --tool memcheck`: the same fixture printed
  `1 passed; 0 failed` in 13.18 s, but sanitizer exited 97 with
  `ERROR SUMMARY: 26 errors`. The 26 reports are 24 NCCL initialization
  `cudaErrorNoKernelImageForDevice` (209) API reports and two NCCL
  `ncclMemAlloc`/`cuMemCreate` permission (800) reports. No invalid memory
  access appears in this leaf log. This is **not** a sanitizer pass.

The exact NCCL API report family appears in the full BFS v62 timeout, yet
the isolated exchange *completes* under memcheck here. Therefore those API
reports alone do not explain v62's lack of progress at depth 0. The next
controlled probe disables API-error reporting while retaining memcheck
instrumentation, then checks all four tools on the same leaf. A leaf pass
will not close the full-BFS gate.

Raw evidence: `test_results/kaggle_lsa_leaf_v44/lsa-bfs-gate/summary.json`,
`lsa-leaf-plain.log`, `lsa-leaf-memcheck.log`.

## Follow-up v45

At the same source on another P2P-capable two-T4 host, plain passed again.
With `--report-api-errors no`, the isolated exchange passed memcheck with
`ERROR SUMMARY: 0 errors` and racecheck with
`RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`.
The notebook classified racecheck as failed solely because it expected the
memcheck-style summary string. Therefore initcheck and synccheck did not run
in v45. The parser was corrected for the next run; v46 selected a non-P2P
host and stopped before build. This is a **two-tool leaf result with narrowed
API reporting**, not a full BFS sanitizer pass.

Raw follow-up: `test_results/kaggle_lsa_leaf_v45/lsa-bfs-gate/summary.json`,
`lsa-leaf-memcheck.log`, `lsa-leaf-racecheck.log`.
