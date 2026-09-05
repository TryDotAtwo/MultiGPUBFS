# Compact macro archive hardware gate

Kaggle `trydotatwo/mgbfs-distributed-sanitizer`, version 23, completed on two
distinct Tesla T4 devices. Tested source:
`a906232e949d672204ad8fe5137aafa9d317eefb`.

Downloaded evidence: `test_results/distributed-sanitizer-v23/distributed-sanitizer/`.
The summary and individual logs were inspected, not just the terminal status.

- Nine distributed archive fixtures passed normally and under memcheck,
  racecheck, initcheck and synccheck. Each sanitizer reported zero errors;
  racecheck also reported zero warnings and hazards.
- `native_macro_archive_is_complete_and_verifiable` passed normally and under
  all four sanitizers. This fixture covers matrix UT(4,2) and compact S4
  generation5, with full-state layer oracle comparison and archive verification.
  This macro executor test is single-device, not distributed macro execution.
- Nonidentity-source macro tests also passed all five executions.
- Eight actual two-process torchrun smoke runs covered DENSE/HASH_FIRST,
  CUB_SORT_MERGE/BMMA_BUCKET and pre-dedup OFF/ON. All produced S4 layer counts
  `[1,3,5,6,5,3,1]`, totaling 24, and both rank archives passed CLI verification.

These are correctness gates, not performance measurements. The smoke archive
verifier checks committed checksums and counts; full-state equality is covered
by the separate runtime fixtures. Neither the new integer-MMA HASH_FIRST
integration nor a multi-rank weighted macro scheduler is proven by this run.

Next gate pins `1b44a0e0a265c580bff55f73b80ae746d536f708`, adds the tenth
distributed fixture for integer-MMA HASH_FIRST with both owner backends, and
builds the benchmark's explicit Tensor-generation selector on Linux.
