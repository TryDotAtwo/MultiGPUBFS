# Native memory-query completion gate

Source: `ab2b3abc` on `codex/bfs-tail-archive`.
Hardware: two RTX A4000, machine 149741, CUDA 12.9, NCCL 2.30.7.

The query intentionally returns `MEMORY_QUERY_DONE` after publishing warmed
free-memory measurements and completing an all-rank rendezvous. The constructor
failure guard previously treated that intentional return as a failure and could
cancel a peer while it was finishing the rendezvous. Atomic diagnostic writes
alone did not fix this: an eight-query run still failed with one
`REMOTE_SEARCH_CANCELLED` report.

The constructor now disarms its failure guard only for the exact successful
query sentinel. The outer search sideband reports success and retirement for
that sentinel. All other constructor/search failures retain cancellation.
The Python parser still requires every rank's measurement and exact terminal
sentinel, and rejects foreign failures.

Twelve consecutive real two-rank queries passed after the fix. Release build
passed on the GPU host. Eight local memory-query/admission tests passed.
An independent CPU full-state/layer oracle also verified all 17 COMPLETE cases
from the previous automatic sweep. That previous sweep attempted 19 cases,
including two empty INCOMPLETE startup failures; it did not exhaust the grid.

Evidence: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/dcfd8ba5a37769b6b897f12aee4b879a2a1560b9/evidence/20261003-query-guard/report.json

This validates the repeated startup query and the stated complete-case oracles.
It does not establish full-grid completion or a universal performance bound.
A fresh 900-second automatic supported-grid/HF gate was launched afterwards;
its result is not claimed here.
