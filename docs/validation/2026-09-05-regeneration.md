# Selected-parent regeneration validation

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` v4 completed at source
`28cb1a74f8dfaba485684204ae37d2a086f47d3a` on two physical Tesla T4 devices.
The selected-parent matrix regeneration leaf passed plain execution,
memcheck, racecheck, initcheck and synccheck (zero errors; racecheck zero
warnings/hazards). Both distributed archive fixtures also passed all tools.
Evidence: `test_results/distributed-sanitizer-v4/distributed-sanitizer/`.

The preceding v3 build failed at link time on the missing regeneration symbol,
before implementation was added. The independent expected matrices validate
left multiplication, request order and zero padding; invalid-parent rejection
must leave the entire output untouched.

This is an allocation-free scalar CUDA reference leaf, not a Tensor Core
backend or a completed HASH_FIRST executor. Source-parent lifetime, owner
request/response routing and hash-only generation still require integration.

V5 adds zero-count, count-overflow, malformed-origin and sticky-fatal cases.
Its terminal Kaggle status is COMPLETE; detailed outputs must be checked before
claiming those additional gates passed. The next fixture consumes core wire
OriginRef bytes through the Rust C ABI on each device; local cargo check proves
type-checking only, not CUDA linkage or execution.
