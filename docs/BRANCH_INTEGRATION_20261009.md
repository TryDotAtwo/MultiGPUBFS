# BFS branch integration, 2026-10-09

The compact SHARD_AB and RAM-first implementation from 3ce7428 is combined with the library/T4 line ending at 9bfa5a5. The integration retains both execution strategies rather than assigning full-state route banks to the compact hash-first path.

## Runtime contracts

- Compact SHARD_AB keeps one physical hash/request packet and selected-row regeneration. Its memory admission adds one aligned 256-byte generation control block, with no duplicate full-state payload planes.
- Ordinary and weighted library execution keeps two to four physical source banks, last-reader events, bounded epoch credits, rank ownership and original-depth weighted settlement.
- RAM-first archive queues, preallocated reserves, large sequential transfers, writer backpressure and resident session buffer/communicator reuse are retained.
- Compact COMPLETE retains whole small layers and the last complete layer. INCOMPLETE retains up to 1000 unprocessed current states; the incomplete future is not presented as complete.
- Weighted HASH_FIRST remains explicitly unsupported by the T4 source's WEIGHTED_HASH_FIRST_NOT_READY gate. It has not been silently redirected to DENSE.

## Repairs made during integration

Separate compact one-bank admission from the public multibank contract; restore compact direct-key generation, archive selection, query-only unwind and session reuse; retain weighted production dispatch and native failure annotation. The HOST build now accepts an explicit NCCL package root. An unused conditional-node device kernel is removed: it poisoned EAGER loading on driver 535 despite having no callers. The route-query CTest now requires EAGER module/data loading to detect this regression.

## Verification on 2 x RTX 3060 12 GiB

427 CPU tests, 20 Python archive/packing tests, CUDA build/check and library contracts passed. Independent complete-state GPU oracles passed for HASH and SORT_MERGE on (4,1), (8,1), (18,17), (32,31), (64,63), (128,127). The baseline separately passed both (8,1) full-state oracles. Four resource/cancel cases retained 4000 total samples with independent CPU-layer membership and uniqueness checks. Selected COMPLETE and resident query/reuse/shape-change tests passed. Production weighted CUB/CUCO at macro depths 2 and 3 retained all 24 S4 states at the correct original depths. Opt-in weighted owner graphs instantiated 8 graphs per rank and launched 80 times with no late instantiation.

## Matched heavy timing

Workload (14,1), compact retention, two ranks, same geometry, three alternating pairs for each algorithm. Both variants completed depths 0 through 32: 104,814,443 total states in the prefix, 30,759,714 states in the last completed layer. Both then stopped on a resource limit; this is not the complete graph's time.

| Mode | Base median BFS seconds | Integrated median BFS seconds | Base / integrated |
| --- | ---: | ---: | ---: |
| HASH | 4.482360428 | 4.464207324 | 1.0041 |
| SORT_MERGE | 10.433513401 | 10.340188494 | 1.0090 |

These measurements show retained performance, not a demonstrated material speedup. Layer counts match in all twelve runs; each archive's checksum was verified. INCOMPLETE samples are unordered frontier subsets and differ between runs. The compact benchmark therefore does not establish equality of every state in the large prefix; complete-state correctness is independently checked on the bounded oracle workloads. The baseline has only the same unused-kernel compatibility removal as the candidate, recorded as a separate patch; its hot computation is unchanged.

## Coverage limits

Hardware is Vast 54975893, machine 20571, 2 x RTX 3060, actual driver 535.161.07. HOST/NCCL is GPU-verified. The LSA-enabled SM86 native library is compiled, but this host reports no P2P in either direction; it is not LSA GPU acceptance. No B200/B300, eight-rank, or complete (14,1) acceptance is claimed. Optional LSA and weighted HASH_FIRST limitations are preserved explicitly.
