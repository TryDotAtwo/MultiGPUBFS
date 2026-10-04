# Typed runtime v100: sanitizer acceptance still incomplete

Worker COMPLETE, report INCOMPLETE. Source
`b47c2707bc3dd346ae0cdc8c2b6df7ff45198443`, two physical T4s, P2P enabled
both directions. Root summary retained at
`build/kaggle-v100-summary/lsa-bfs-gate/summary.json`, SHA256
`886af00f44fa527e4d0fe43356c32bdbacba99bfab3298c42eebec800d104214`.

Root report: 78 typed replays, 72 pass. Six failures are precisely
DENSE/HASH_FIRST × route banks 2/3/4 under initcheck; no outer timeout is
reported. Passing groups: 48 S4 configurations, six U4(F2) reuse fixtures,
and six each for memcheck, racecheck and synccheck. This does not close the
four-tool sanitizer gate. Root status alone is not independent verification
of every nested asymmetric fault result.

All small JSON/log outputs are being retained under
`build/kaggle-v100-observation` before replacing this notebook. Download
completion and detailed fault/first-failing-API verification remain pending.
The second notebook v5 remains live; it has not been overwritten.

Retention completed: all 78 nested summaries and small JSON/rank logs saved
on D. Detailed summaries confirm 108/108 asymmetric fault cases pass,
zero forced cleanup and zero false group COMPLETE. Maximum observed
process time is 2.160 seconds rounded upward. Example DENSE/banks2
initcheck fails before BFS in `window_register`; rank1 returns unhandled
CUDA error and rank0 reports cancellation/timeout without terminal result.
No sanitizer waiver follows from these successful failure-protocol checks.

Follow-up prepared: direct CUDA POSIX import diagnostic, source
`1a00b1da7df33870cd1127c19c11b316bd3ff86c`. Actual nvcc sm75 build passes
with Wall/Wextra. Local executable invocation cannot run without
`libcuda.so.1` in this CPU-only Docker session; no local execution pass is
claimed. All script regression tests: 59 tests, OK, three skipped.
The follow-up is diagnostic only, not a substitute for full BFS acceptance.
