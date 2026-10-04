# Runtime-selected LRX layouts through n=128

The automatic grid is n=2..128, r=1..n (8255 pairs). No alphabet/u64-orbit
exclusion is applied inside this domain. Resource pruning still follows each
fixed-r branch independently. Existing archived runs remain readable.

Symbol slots use 4,8,16,... bits according to alphabet size. The state occupies
ceil(n*bits/64) uint64 words, padded with zeros and aligned to 8 bytes. Within
the requested domain alphabet<=128, so only the 4/8-bit paths are used.
Examples: n17/alphabet17 ->24 bytes; n65/alphabet2 ->40 bytes;
n128/alphabet16 ->64 bytes; n128/alphabet128 ->128 bytes.
Both the batch writer and manifest consumers select the layout at runtime.
Always pass manifest packing.bits_per_symbol to the analytical state decoder.

Orbit size n!/r! is exact startup metadata, computed into a variable-length
little-endian array of uint64 words. Python launch configuration also records
its decimal value. The legacy native allocation ABI still receives a saturated
u64 upper bound for oversized orbits, explicitly labelled in the native report.
This bound is not substituted for the exact mathematical size. Actual resident
row counters and layer ordinals remain machine-sized: GPU capacity is reached
far earlier than their limit. No allocation proportional to the whole orbit.

CUCO_RANK means rank-owner deduplication, not a Lehmer-number state encoding.
The device continues to hash and compare complete runtime-width words. Thus
wide orbit arithmetic belongs to startup, not every GPU batch. No JIT kernel
generation or separate compilation for individual n/r pairs was introduced.

Verified on CPU: seven Rust multiset tests including exact wide orbit arithmetic
and complete n128r127 oracle; four dynamic layout/Parquet tests;70 tail tests;
15 automatic-launch tests;12 sweep tests. Byte order and legacy4-bit cases
remain covered. The wide Parquet test covers n17r1,n65r64,n128r1 with synthetic
single-layer payloads, not full GPU enumerations.

Hardware expansion gate completed on 2026-10-04: two RTX3060 and two RTXA4000,
native CUDA12.9/NCCL2.30.7, commit3fd7730. Seven wide cases per model were
checked against independent CPU layers and published/read back from HF.
See validation/2026-10-04-wide-two-gpu.md for measured scope and timings.
The streaming archive's whole-orbit extent overflow was fixed in this gate.
Normal startup selected HOST_SIZED_NCCL on both rentals: 3060 lacks usable LSA;
A4000's LSA correctness gate failed with22_22 on this branch. This does not
validate LSA/Graph32 or B300, nor diagnose newer main-branch code.
