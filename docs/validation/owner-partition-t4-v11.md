# Owner partition gate, Kaggle capacity v11

Source: `069612bee18d1c048b161a89b1382044859edeff`.
Kernel: `trydotatwo/mgbfs-library-capacity-t4`, version 11, COMPLETE.
Runner: `kaggle/library-capacity/owner_partition.py`.

Two distinct Tesla T4 devices executed the standalone fixture separately.
Both plain runs and memcheck/racecheck/initcheck/synccheck runs printed
`OWNER_PARTITION_GPU_PASS`. All eight sanitizer logs report zero errors;
racecheck also reports zero warnings. No GPU timing claim is made.

Fixture covers 1/2/4/8 logical-owner splits, zero-row owners, prefix boundaries,
gathered state payloads, count-tail canary, local bucket directories, zero input,
invalid source references and foreign-owner rejection. This is NOT an eight-GPU
NCCL test or full BFS validation.

Evidence: `test_results/owner-partition-v11/owner-partition/summary.json`
and the ten `gpu*-*.log` files in that directory (local small artifacts).

## Next integration contract

For power-of-two worlds, symmetric peer round `r` pairs physical rank `p`
with `p XOR r`. Round zero is local; remaining rounds cover each peer once.
Every rank issues every round, including zero-payload rounds. Sorted payloads
remain in logical-owner order; only fixed-size offset/count metadata is mapped
to physical ranks. One bounded receive slot can be reused only after its owner
consumer completes. This avoids an eightfold incoming payload arena.

CPU tests cover reciprocal pairing, complete pair coverage, invalid topology,
permuted owner maps, empty intervals and capacity errors. Both new behaviors
failed against initial stubs, then passed after implementation. Existing
two-rank routing uses the new peer and interval helpers. CUDA/library-owner
Rust paths pass `cargo check` (type-check only, not linked GPU execution).

Still required: generalize runtime receive/commit loops, HASH_FIRST source
lifetimes, configuration, capacity planning and fatal convergence; then test
full 1/2/8-GPU BFS. No Vast rental has been started by this gate.
