# Joint diagnostics for an allocated GPU set

Since 0.2.0.dev21, both `python -m multigpubfs doctor` and `doctor --deep`
test the complete selected GPU allocation. With eight visible GPUs, BFS and
NCCL are actual eight-rank runs. No independent one-GPU BFS sweep or automatic
fallback to a single GPU is performed. `--devices` can explicitly select a
different logical allocation; the scheduler visibility mask is never changed.

Direct CUDA API/kernel probes start one child per selected GPU concurrently;
they have one combined result plus per-rank logs and results. These independent
primitives are not presented as NCCL tests. Actual distributed BFS/NCCL stages
verify the whole allocation separately. A failing rank makes its joint test fail.

Deep mode runs the catalog as one group: each rank keeps one Python process and
CUDA context while checking all finite constructor representatives. It records
each family/result/error in `deep-catalog-all-gpus/rank-N/probe.json`, continuing
to the next family after an ordinary family error. This checks exact successors,
not a complete BFS of potentially enormous puzzles. No arbitrary user graph or
production work is restarted. The loading matrix has 14 group configurations,
not 14 times the GPU count separate sequential configurations.

A case is not started when the remaining global budget cannot fund its full
configured stage limit. It is then `SKIPPED: GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED`;
an otherwise successful report is `INCOMPLETE`, not a manufactured short-stage
failure. A real timeout or rank failure remains a failure. Cleanup and report
finalization require a small additional bounded allowance.

Run inside the allocated GPU job, in a fresh process after upgrading:

```sh
python -m multigpubfs doctor
# Optional complete synthetic matrix, total execution budget 30 minutes:
python -m multigpubfs doctor --deep --total-timeout 1800
```

Prefetch/cache format and pinned native binaries are unchanged. Existing cached
SM builds are reused without re-downloading. This changes diagnostic scheduling
only, not CUDA kernels, the production BFS algorithm or its tuning defaults.
