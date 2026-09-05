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

V7 RED was confirmed: CUDA fixture compiled, linking failed specifically on
`mgbfs_generate_hash_only` at both calls. Log: `test_results/distributed-sanitizer-v7/distributed-sanitizer/hash-first-build.log`.
Implementation `b607e983e29779ac96771344edc8fe8ac78f73e6` adds a warp-per-child
CUDA reference, affine F_p projections with bounded 64-bit reductions, and
device-side capacity validation before output writes. No child-state global
buffer exists. This is not the requested Tensor Core implementation; it is
the correctness reference for that backend and native HASH_FIRST integration.
GPU execution remains unverified until the following gate completes.

V8 completed: hash-only fixture passed plain and all four sanitizer tools
(zero errors, racecheck zero warnings/hazards). The regeneration leaf and
three Rust fixtures also passed. Evidence:
`test_results/distributed-sanitizer-v8/distributed-sanitizer/summary.json`
and `hash-first-{memcheck,racecheck,initcheck,synccheck}.log`.

V9 source `b3cb1666b80637ba9a8eb0cda85762626fb06bd9` adds a real-seed Rust
oracle chain: hash-only device output origins/count feed selected regeneration
on a nonblocking stream without host count readback between kernels. Full
states and hashes are compared against CPU successors and GemmHash for seeds
0, 1, 20260828 and matrix/modulus pairs (7,2), (7,5), (3,256), on each T4.
This tests widths crossing the 32-lane boundary and byte modulus 256; it does
not yet test remote owner commit or claim an end-to-end HASH_FIRST executor.

V9 completed successfully: four Rust fixtures passed plain, memcheck,
racecheck, initcheck and synccheck; all sanitizer summaries clean. The seeded
hash-only-to-regeneration chain executed on each physical T4. Evidence:
`test_results/distributed-sanitizer-v9/distributed-sanitizer/`.

V10 is the next RED gate at `80197900aabc2f6bbffe6f28e2ae1fce4a5d3a49`:
post-commit request extraction must compact only selected origins, preserve
their final absolute StateRefs, reject malformed indices before any output
write, and leave Extent.ready unset while states remain unmaterialized.

V10 RED confirmed: fixture compiled, linker reported only missing
`mgbfs_state_build_requests` at the two calls. Downloaded evidence:
`test_results/distributed-sanitizer-v10/distributed-sanitizer/requests-build.log`.
Implementation `804642c38ff07ef052ff8039fd9b08ae469be5c3` reuses owner-stage,
reserved-extent and selection validation before compacting requests. It emits
absolute sequence StateRefs (not physical ring indices), leaves ready unset,
and returns zero requests after fatal. V11 validates it under all four tools.

V11 completed; request extraction passed plain and all four sanitizer tools,
with zero errors and race warnings/hazards. Evidence:
`test_results/distributed-sanitizer-v11/distributed-sanitizer/requests-*.log`.

V12 RED source `813845bf1bc0913d3017b210a6f65742a983e71b` requires stable
per-source parent sorting while preserving the paired final target StateRef.
The new operation will reuse the existing MaterializePlan CUB buffers rather
than allocate another scratch arena. Mixed source ranks must reject the whole
job before writes. Sorting is needed before regeneration; pairing is needed
to restore final dense target order after request/response routing.
