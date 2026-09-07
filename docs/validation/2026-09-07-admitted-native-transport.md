# Admitted native transport integration

Source: `45899cff53c4595e9c485d9ef2f3be1a2098f750`.
Kaggle package: `190c86f`, private `trydotatwo/mgbfs-distributed-sanitizer`
version 41. Kaggle execution completed and the full downloaded raw regression
artifacts were reconciled with the pinned source. Only logs and JSON metadata
were downloaded; state archives were not copied to this workstation.

The adapter now enforces globally ordered, once-only submission before transfer
completion. Submission failure poisons the adapter and closes its control
connections. The native bridge enqueues NCCL scatter and records a CUDA event
without a host synchronization. The caller still owns device allocations,
event/consumer lifetime, and communicator abort on failure.

Scatter does not copy the source's self payload into receive storage.
`payload_view` resolves that payload to the original send allocation plus the
sum of earlier ranks' byte counts; other ranks read their receive allocation.
The same consumer lease protects either range. A view is not a readiness proof.

## Confirmed checks

- Local full `cargo test --locked -p mgbfs-runtime` passed.
- Nine admitted-buffer CPU tests include three successive depth publications,
  out-of-order and duplicate submission rejection, failure poisoning, empty
  receiver lifetime, and actual TCP admission.
- GitHub run `34106908360` completed successfully, including Linux Rust 1.75
  CUDA-feature type checking. It does not link or execute CUDA.

## Hardware transport result

The added `admitted_adapter_native_scatter_and_depth_rollover` test uses two
actual devices, both source ranks, nonempty and empty payloads, and two depths.
It checks exact received bytes and the nonzero source self-view prefix.
The v41 package requires both native scatter tests to pass under plain execution,
memcheck, racecheck, initcheck and synccheck, in addition to existing BFS archive
regressions. Both transport tests passed in all five modes (10 executions).
The downloaded raw scatter logs pass the stricter exact test-count parser and
sanitizer error/hazard checks, not just the notebook's summary flag.

The summary identifies two distinct Tesla T4 devices, each 15360 MiB:

- `GPU-0df51b1f-5ae5-f52d-cead-721f987d6e61`
- `GPU-75ee57f7-0817-1ff2-263a-21ea6da7b425`

All five overall modes are PASS, with raw reconciliation covering 40 tool logs,
36 measured rank records, 36 warmup rank records and 36 archive verifiers. The
source-sha log matches the pinned source above. All 24 reference profile smoke
selections preserve the expected complete layer counts `[1,3,5,6,5,3,1]`.

Reconciliation command (the final argument requires two scatter tests):

```powershell
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v41/distributed-sanitizer test_results/distributed-sanitizer-v41-summary/distributed-sanitizer/summary.json 45899cff53c4595e9c485d9ef2f3be1a2098f750 2
```

Result: `RAW_GATE_RECONCILED`. The new adapter test serially exercises
its payloads; concurrency remains covered only by the older lower-level test.
Neither test establishes end-to-end production BFS overlap.

## Production integration boundary

`distributed_native.rs` still uses the synchronous reference exchange path.
Its route output contains separate sorted hashes and gathered state/origin
storage. Native DENSE schema2 transport instead requires destination-contiguous
hash, ordinal and state planes with aligned padding. Packing directly into the
admitted source slots must replace, not duplicate, the existing gather staging.
Generation/route scratch, native events, owner readers and archive/parent leases
must then be independently owned per active slot before concurrent batches can
reuse those allocations. Replacing only the NCCL function would not establish
that ownership or produce the requested overlapped BFS pipeline.

This change establishes a native submission boundary, not completed asynchronous
BFS, a performance improvement, or a new graph-catalog result. S13 is unchanged.
