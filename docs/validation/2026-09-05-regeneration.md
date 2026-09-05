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
Its downloaded outputs were checked: all four tools passed for both the leaf
and the distributed fixtures, with zero errors and zero race warnings/hazards.
Evidence: `test_results/distributed-sanitizer-v5/distributed-sanitizer/`.
The next fixture consumes core wire
OriginRef bytes through the Rust C ABI on each device; local cargo check proves
type-checking only, not CUDA linkage or execution.

V6 at `caf1728e62f051f07d1a7a202cdc9525118624c1` completed on Kaggle:
all three Rust tests passed plain and under all four sanitizers. The wire
OriginRef fixture executed on device 0 and device 1; all summaries were clean.
Downloaded evidence: `test_results/distributed-sanitizer-v6/distributed-sanitizer/`.

V7 is the deliberate RED gate for `mgbfs_generate_hash_only`, at source
`12c4950af523bcb09ab2370169977d8e75dec6d3`. No implementation exists yet.
The fixture supplies no child-state output, checks 514 parent-major candidates,
four affine residues including reduction near p, 64-bit parent references,
and device-count capacity rejection without output writes. Expected next
failure is the missing symbol at link time, not a runtime regression.
