# Admitted native transport integration

Source: `45899cff53c4595e9c485d9ef2f3be1a2098f750`.
Kaggle package: `190c86f`, private `trydotatwo/mgbfs-distributed-sanitizer`
version 41. Hardware result is pending; RUNNING is not a correctness result.

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

## Hardware test under execution

The added `admitted_adapter_native_scatter_and_depth_rollover` test uses two
actual devices, both source ranks, nonempty and empty payloads, and two depths.
It checks exact received bytes and the nonzero source self-view prefix.
The v41 package requires both native scatter tests to pass under plain execution,
memcheck, racecheck, initcheck and synccheck, in addition to existing BFS archive
regressions. No result has yet been claimed for v41.

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
