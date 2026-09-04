# HASH_FIRST source-lifetime contract

The source-side lifetime rule is now represented by one `HashFirstLease`
object instead of separate receipt bookkeeping and manual `StateRing` lease
release calls.

For one emitted parent batch it:

- acquires an origin lease only when at least one owner receives a candidate;
- accepts exactly one terminal receipt from every non-empty owner range;
- counts a materialization response only at response-send completion;
- releases the parent extent exactly once, after receipts and all accepted
  responses agree;
- remains poisoned and retains the parent on any mismatch.

Both sequential and readiness-reordered concurrent HASH_FIRST CPU protocol
simulators use this contract. This closes the CPU lifetime/protocol slice; it
does not claim that the production CUDA/NCCL HASH_FIRST data plane exists.

## Evidence

The new test was observed failing before implementation because
`HashFirstLease` did not exist. After implementation:

```text
cargo test -p mgbfs-runtime --test hash_first_lease
3 passed; 0 failed

cargo test -p mgbfs-runtime --quiet
all runtime suites passed; 0 failed
```

The full workspace command still requires the configured CUDA native library
and cannot be used as a Windows CPU-only gate without
`MULTIGPUBFS_CUDA_LIB_DIR`. The physical CUDA/NCCL gate remains a later Kaggle
2xT4 run.
