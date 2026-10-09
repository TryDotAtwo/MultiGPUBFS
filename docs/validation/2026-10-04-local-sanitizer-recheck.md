# Current local sanitizer and fixture-resource guard

Hardware: one RTX 3070 Laptop GPU, sm86, driver 572.70, 8192 MiB.
Runtime source at main `99aadc3`; CUDA runtime implementation remains
`75494be`. Test binary SHA256 before guard:
`f610f1a828631faee3c259c475546fcb925c2df3634a4ec65c51b915246895f4`.
Native lib SHA256:
`89a25dd5961964f4e4898bf6603ea3b13764aea79e819f60ca0d4ab0f819a851`.

Compute Sanitizer ran sequentially with `--error-exitcode 99` on
`cuco_lsa_route_banks_preserve_full_states_and_reuse_across_depths --exact
--test-threads=1`. No suppressions or kernel filters:

| Tool | Result | Fixture seconds |
|---|---|---:|
| memcheck | 0 errors | 22.74 |
| racecheck | 0 hazards, 0 errors, 0 warnings | 369.90 |
| initcheck | 0 errors | 17.65 |
| synccheck | 0 errors | 9.97 |

Whole command exit zero. Fixture covers six U4(F2) full-state/hash/archive
cases, DENSE and scalar HASH_FIRST, physical banks 2/3/4, K=3, batch=1,
actual within-depth reuse. Memory-backed archive, world=1. This does not
close two-process T4/NCCL registration initcheck, asymmetric failure gates,
Nsight timeline, disk/HF durability or A/B performance.

The library GPU test binary previously failed when two test threads created
process-global RMM pools concurrently. A test-only mutex now scopes all five
fixtures' resource lifetime. No runtime synchronization was added. Poison is
recovered so the original failing fixture remains visible without cascaded
lock failures. Multi-rank acceptance still uses independent rank processes.

Guard compiled into separate `/linux-build/target-fixture-guard`, leaving the
live sanitizer binary unchanged. After all sanitizer gates exited zero, the
two bank fixtures ran with `--test-threads=2`: 2 passed, 0 failed, 10.53 s.
They cover six healthy/archive and six capacity-failure cases. This is the
GREEN rerun of the previously observed concurrent pool-creation failure,
not a pass claim for the unrelated sm75 Tensor generation fixture on sm86.
