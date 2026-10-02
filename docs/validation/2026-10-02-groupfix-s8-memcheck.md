# Current group-progress runtime: full S8 memcheck

Runtime source9d2908b, one physical RTX3070 Laptop sm86. Same diagnostic
minimum-arch NCCL library as preceding local panels. Direct unfiltered
Compute Sanitizer2025.1 memcheck, error-exitcode97, external deadline240s.

CUCO_RANK+NCCL_LSA, DENSE and HASH_FIRST scalar, S8/batch128, pre-dedupON,
matrix_u8 state/archive, seed20260828, capacity40320/ring80640,
buckets8/shards4/job2/bucket capacity40320, pool64MiB,
archive rows512/slots128, warmupOFF/archiveON/macro1, owner-DAG captureON.

Both direct processes returned0, reached capture launch, and reported zero
memcheck errors. Production archive verification passed. Independent CPU
full-byte oracle confirmed all40320unique states and all29exact layers for
each profile. This exercises repeated batch slots on the updated binary,
not only the small S4 fixture.

Native SHA25625e87d75e6d1d8c7f78278ebfcc04ba31bc32c8c1f82490a7ee3260da6632e4d.
Artifacts:test_results/cuco-s8-groupfix-9d2908b/.
Not evidence of two-rank cancellation, T4 registration initcheck, Tensor,
BMMA or paired speedup. Prior four-tool panels remain pinned to their older
native hash; only these two memcheck results apply to this rebuilt runtime.
