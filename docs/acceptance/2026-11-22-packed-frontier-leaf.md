# Experimental four-bit permutation frontier: isolated leaves

This branch is experimental. The codec, converter, rank accessor and memory
arithmetic below are **not connected to the production BFS, wheel, public API,
or automatic profile selection**. There is no packed-frontier BFS, NCCL, spill,
archive, persistent-session or S14/S15 capacity result here.

## Exactness and boundaries

The dictionary maps a sorted signed alphabet of at most 16 symbols to four-bit
ordinals; full canonical signed values remain the external representation.
The codec validates padding and symbol ordinals; permutation mappings must be
bijective. Tests include repeated symbols and the sixteenth nibble.

On two physical RTX 3060 devices (SM86, 12 GiB each, driver 580.178.04,
CUDA toolkit 12.8.93), independently constructed CPU oracles verified:

- Every S8 permutation and all 282,240 adjacent-swap transitions through the
  CUDA codec, with signed symbols. Each GPU handled half the parents.
- A packed action accessor used with the current exact bitmap-rank function:
  the same S8 edges and all transitions of a 30-state repeated-symbol domain.
  Error inputs left the result unwritten.
- Separate SoA conversion in both directions for signed and byte alphabets,
  repeated values, invalid rows, padding, and 16 elements: eight GPU cases
  with 257 rows each across the two devices.

Compute Sanitizer memcheck, initcheck and synccheck passed on reduced codec and
action samples. For the converter, memcheck passed on the complete eight
cases, and initcheck/synccheck passed on the device code before two additional
host-only shape assertions were added. These checks do not establish absence
of logical races in an integrated distributed BFS.

Rust 1.75 tests: two new layout/descriptor tests, three segmented-history
tests and all 46 runtime-library tests passed. The checked layout uses
8 bytes per packed row per bank plus 24 counter bytes. Its virtual arithmetic
tests permit two separately admitted segments of 2^32 rows each; they do not
claim a physical allocation or traversal of 2^32 rows. Production admission
and existing segment layouts remain unchanged.

## Reproduction and evidence

From this branch on a CUDA 12.8 Linux machine with two SM86 devices:

    nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 tests/bitmap_pack4_gpu.cu -o /tmp/pack4-gpu
    /tmp/pack4-gpu
    nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 tests/bitmap_pack4_action_gpu.cu -o /tmp/pack4-action
    /tmp/pack4-action
    nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 tests/bitmap_pack4_convert_gpu.cu cuda/bitmap_pack4_convert.cu -o /tmp/pack4-convert
    /tmp/pack4-convert
    cargo test --locked -p mgbfs-runtime --lib
    compute-sanitizer --tool memcheck --error-exitcode 86 /tmp/pack4-action --sample
    compute-sanitizer --tool initcheck --error-exitcode 86 /tmp/pack4-action --sample
    compute-sanitizer --tool synccheck --error-exitcode 86 /tmp/pack4-action --sample

The source, logs and original baseline build checksums were preserved in
mgbfs-pack4-stage2-evidence.tgz, SHA-256
177b1167bda3c3c49cca2a58ea04a47dcd5a517af42a3200e26c2fbb161a6d48.
The archive was copied to D: and the checksum verified. The matching
unmodified production CUDA library and executable were built on this host, but
the isolated leaf tests do not exercise that library. See the architecture and
implementation plan for remaining integration and acceptance gates.
