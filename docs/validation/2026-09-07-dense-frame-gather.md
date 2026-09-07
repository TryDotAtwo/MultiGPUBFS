# Direct DENSE payload gather: v42 hardware verification

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` v42 completed on source
`9c59e73a8eeb40a451ad1226a99906fdc8387ed7`, package `c5da123`.
Two Tesla T4 devices, each 15360 MiB, were recorded:

- `GPU-253be21d-3491-aaed-2b85-eef36d836884`
- `GPU-91036100-a301-0fe1-a5d4-98cdf19dca1b`

The single-device direct-gather test checks exact schema2 hash, ordinal and state
planes against hand-derived bytes, including zero padding and a nonzero sorted
input offset. Invalid references set a device fatal instead of reading outside
the child array; insufficient capacity and invalid ranges are rejected. Empty
payload enqueue is also covered. The kernel writes directly from generated
children into the frame, without an intermediate gathered-state array.

All three native-scatter-file tests passed under plain execution, memcheck,
racecheck, initcheck and synccheck. Raw reconciliation also covers 40 tool logs,
36 measured rank records, 36 warmup rank records and 36 archive verifiers. All
24 reference profile selections retain layer counts `[1,3,5,6,5,3,1]`.

```powershell
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v42/distributed-sanitizer test_results/distributed-sanitizer-v42/distributed-sanitizer/summary.json 9c59e73a8eeb40a451ad1226a99906fdc8387ed7 3
```

Result: `RAW_GATE_RECONCILED`. Only logs and JSON metadata were downloaded.
A network download failure was resumed without rerunning the GPU computation.
The installed Kaggle CLI parses a `/version` suffix but does not use it in its
output-list request; the next notebook was therefore not pushed until the v42
download and source reconciliation finished.

## Scope boundary

This gate does not verify the later 256-byte transport prefix, DenseFrames rank
plan, or the combined framed NCCL path. Those changes are in source `a7378e3` and
private gate package `9712d87`, launched as v43; its hardware result is pending.
The production BFS still uses its older
reference exchange loop. No BFS speedup or full asynchronous-runtime completion
is claimed from this component gate; no S13 rerun or republication occurred.
