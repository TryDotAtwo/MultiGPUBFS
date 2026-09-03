# U4(28) single-T4 capacity probe

Private Kaggle `trydotatwo/mgbfs-native-capacity-t4` version 1 completed.
Source `4fea92d2228c19be83ca2d16464daf5623aa21ba`, baseline
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`. This predates asynchronous D2H.
Same source as the 10/10 native-runtime v3 correctness/sanitizer gate.

One native trial, not a median: 481,890,304 states, 43 nonempty layers,
largest frontier 34,670,406; F=48,000,000, batch=524280, pre-dedup ON.
Full-workload warmup excluded, GPU0 Tesla T4
`GPU-ace42b1a-f229-d84f-e76b-fdefe6fbcbb5`.

- Search: 67.461538413 s.
- Durable archive commit from search start: 161.701365145 s.
- Requested CUDA allocations: 5,779,303,491 bytes.
- cudaMemGetInfo used peak: 5,935,857,664 bytes.
- nvidia-smi whole-device sampled peak: 5661 MiB.
- Pinned RAM: 17,582,254,080 bytes.
- Preallocated disk: 15,487,598,592 bytes.

All three unchanged CayleyPy processes failed with explicit
`torch.OutOfMemoryError: CUDA out of memory`: batches 1048576, 262144, 65536.
No completed baseline timing exists here; therefore no speedup ratio is claimed.
These failures do not prove every possible baseline configuration fails.

Raw summary, native result, baseline tracebacks, environment and sample CSVs:
`test_results/native-capacity-v1/native-capacity/`. Large archives remain private
Kaggle outputs, not included in local JSON/log/CSV downloads. Native result has
the expected total cardinality, not an independently compared m28 full-state set.
