# Two-rank archive hang: device binding and bounded merge fix

Physical 2xT4 diagnostics exposed two independent defects in the first S10
benchmark harness.

1. `PinnedArchive` was constructed before `cudaSetDevice(local_rank)`.  Rank 1
   therefore created its CUDA events on device 0 and later attempted to record
   them on a device-1 archive stream.  The rank owning the initial state stopped
   in the first archive submission while its peer waited in the first NCCL
   epoch.  The fix binds the CUDA device before any archive allocation and the
   archive now rejects a later device mismatch explicitly.
2. Distributed incremental merge launched grids sized to configured graph
   capacity rather than the synchronized live run bounds.  A new bounded ABI
   keeps device counts authoritative but limits merge/select/gather work to
   caller-proven old and incoming upper bounds.  Its old-run-wins semantics and
   sticky bound failure have a dedicated CUDA test.

The archive path also stopped recomputing Hash128 with a second GEMM.  It now
copies the already-resident current-layer hash run paired with the current
states.

Kaggle diagnostic version 7, source
`132296cdbb4992b6739b26194c836f09f217a59e`, exhausted S8 on both physical T4s:

- exact global layer counts matched the known 40,320-state traversal;
- search completion: 0.091704 s;
- durable archive completion: 0.250959 s;
- observed external VRAM: 213 MiB per rank, 426 MiB total;
- both rank archives completed and reported independent local layer counts.

The earlier version 5 reproduces the pre-fix first-depth timeout, making the
failure and the correction independently auditable.
