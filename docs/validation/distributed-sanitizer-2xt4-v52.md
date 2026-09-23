# Distributed native regression on two T4s: v52

Private Kaggle notebook `trydotatwo/mgbfs-distributed-sanitizer`, version 52,
completed at source `63377c4aa236c314f2c7caf647a858f5b7f55aa7`.
The downloaded summary and source-SHA log agree. Both enumerated devices were
physical Tesla T4s (15,360 MiB each).

The notebook reports plain, memcheck, racecheck, initcheck and synccheck PASS.
The main memcheck log ends with `ERROR SUMMARY: 0 errors`; racecheck ends with
`0 hazards displayed (0 errors, 0 warnings)`. All 24 reference-profile smokes
passed across one/two ranks, DENSE/HASH_FIRST, scalar/SM75 generation,
CUB/BMMA owner and pre-dedup ON/OFF. Every smoke reports S4 global layers
`[1,3,5,6,5,3,1]` with verified archives.

Raw summary and logs: `test_results/kaggle_distributed_sanitizer_v52/`.
This is a regression of the existing runtime with the new rank-batch reserve
leaf present but unused. It does not prove a device-driven owner, LSA transport,
CPU-free retirement or large-graph performance.
