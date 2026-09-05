# Twelve-profile process gate, Kaggle v25

Kernel `trydotatwo/mgbfs-distributed-sanitizer`, version 25,
source `b08d519bdcd1758cbc4f58a443396f8e5dca16ea`.

Authoritative retained evidence:
`test_results/distributed-sanitizer-v25/mgbfs-distributed-sanitizer.log`.
The complete JSON event log includes the final source-pinned COMPLETE summary
and all subprocess outputs. The individual output-file download is incomplete;
it must not be treated as a complete inventory.

Verified from that full event log:

- Five occurrences of `10 passed; 0 failed`, for plain execution and the four
  sanitizer modes of the distributed archive fixtures.
- 24 ERROR SUMMARY entries, all zero, and eight RACECHECK SUMMARY entries,
  all zero hazards/errors/warnings (including supporting leaf/macro tests).
- Twelve successful native two-process S4 smoke configurations:
  DENSE, scalar HASH_FIRST and integer-MMA HASH_FIRST, each with CUB/BMMA and
  pre-dedup OFF/ON. Every configuration reported layers `[1,3,5,6,5,3,1]`.
- The launcher verifies each rank archive with the CLI after each smoke.
  This is committed checksum/count verification, not an independent full-state
  oracle for every process smoke. Separate runtime fixtures compare full states.

## Retrieval caveat

The installed Kaggle CLI's `kernels_output` parses `/version` but does not pass
that version to `ApiListKernelSessionOutputRequest`. Starting v26 while v25's
paginated download was still active caused later pages to follow the new
session. CLI exit zero therefore did not prove a complete download. The first
response's full kernel event log still identifies v25's source and complete
result, allowing verification without rerunning v25.

Do not start a new version of the same notebook until its output inventory has
been downloaded and checked, or use a separately verified immutable download
mechanism. Merely appending `/25` to this installed CLI is not sufficient.

Version 26 is a different gate: it tests the newly introduced full warmup pass,
warmup/measured layer equality and rank-local warmup archive release. Those
behaviors are not established by v25.
