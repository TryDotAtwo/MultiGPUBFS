# Follow-up: bitmap timing, VRAM and published wheel (2026-11-20)

This report supplements the two existing acceptance notes. Results below are
specific to the adjacent-transposition Coxeter graph, not a benchmark of all
graphs or B200/B300 acceptance. The code under test is source commit
`2b97b094af3f94c250bd099004024a520c5238f4`.

## Two RTX 3060: whole-backend timing

The following runs used two RTX 3060 12 GiB GPUs, SM86, P2P disabled, exact
exchange, eight shards and lanes, and 1,048,576 rows per segment. Hash used
compact 32-bit slot positions and three state banks; bitmap used two state
banks. All complete layer sizes matched an independently computed Mahonian
polynomial. Times are BFS seconds, not setup time.

| Graph / profile | Segmented hash | Permutation bitmap | Retries |
| --- | --- | --- | --- |
| S11, initial ceiling 2048, two runs each | 7.6300, 7.6321 | 5.9705, 6.1117 | none for either backend |
| S12, initial ceiling 2048, two runs each | 91.3641, 90.6875 | 114.1447, 110.1553 | hash none; bitmap one per rank and reduced ceiling to 1024 |
| S12, active size profile 1024, capacity 5632, one run each | 148.6225 | 122.2744 | none for either backend |

The last line fixes the active batch to 1024 through a size profile while
retaining a queue capacity of 5632. The plan's ceiling is 2048; the active
profile is 1024. This is one paired observation, not a tuned result or a
confidence interval. In the second line, the two backends actually operated
with different batch ceilings after retry; it is not a controlled measurement
of bitmap-intrinsic overhead. Timings can differ with graph, capacity, queue
geometry, hardware and topology.

For S11, reported state-segment allocations per rank were 914,358,464 bytes
for hash versus 184,549,568 bytes for bitmap, plus the bitmap's 2,494,800 bytes.
These are allocation counters for selected structures, not total peak VRAM.

## Qualification of the earlier eight-GPU S13 comparison

The previous 8×RTX 3090 S13 paired comparison reported 393.32 s for hash and
593.55 s for bitmap and both completed all layers. Those runs started with the
same initial profile but did **not** maintain the same effective source batch:
the bitmap run reduced its ceiling from 8192 to 2048 after two retries while
the hash run retained 8192. Thus the 1.51× elapsed-time ratio is a comparison of
the observed end-to-end configurations, not an isolated bitmap-kernel penalty.
No target-hardware counters were available to assign a precise share of time
to rank computation, atomics, communication or synchronization.

## Published dev39 wheel GPU acceptance

The GitHub prerelease `native-pip-20261120-v39` wheel and SM86 component were
installed and prefetched, then tested in offline mode on 2×RTX 3060. The
installed native executable identifies the same source commit above. For both
`segmented24` and `permutation_bitmap`, and both `balanced` and
`max_capacity` presets:

* The public CayleyPy API with persistent workers produced the independent S5
  layer oracle and verified retained samples.
* Forced INCOMPLETE preserved its samples and the same worker processes
  successfully executed a following job.
* S9 completed with the oracle's 362,880 states; the initial call selected
  measured equal-prefix GPU pilots, and the repeat hit the profile cache.
  All four combinations reused the same pair of persistent worker PIDs.

This is a bounded two-GPU verification, not proof of performance or
correctness on every supported GPU, topology or graph. The two-bank bitmap
preset allocates state banks in addition to the visited bitset.

## Limitations and retained evidence

The separate existing legacy archive suite had eight passes and six failures.
One `EPOCH_CREDIT_TIMEOUT` failure was also reproduced on the dev38 baseline;
two `NATIVE_PLAN_CREATE_FAILED status=3` causes remain unestablished, and
the LSA-disabled and unsupported-Tensor cases are not acceptance of those paths.
Neither B200 hardware nor more than 2^32 simultaneously live rows on one rank
were tested. In particular, do not infer full-runtime acceptance from a
64-bit arithmetic regression or a small GPU test.

The bounded synthetic VRAM-admission tests and earlier GPU receipts are in
[segment-vram evidence](segment-vram-20261120-evidence.tar.gz), SHA-256
`6065f60005d49289454be5c7fd77723e5ef4324b48bac4b50e8409967e766826`.
The timing, benchmark inputs, rank receipts and published-wheel checks above
are in [bitmap queue evidence](bitmap-queue-20261120-evidence.tar.gz), SHA-256
`d84fdf1fc54c673c66fd447c5febc569757f9e1073d4e9e9e15af49fe91930b9`.
The archives omit raw state files. Measured peak GPU usage, where present,
uses 1-second sampling and can miss short peaks.
