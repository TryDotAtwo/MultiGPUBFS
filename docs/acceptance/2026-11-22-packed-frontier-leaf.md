# Experimental four-bit permutation frontier: isolated prototype

This branch is experimental. The codec, converter, rank accessor and memory
arithmetic below are **not connected to the production BFS, wheel, public API,
or automatic profile selection**. An isolated two-GPU test contour now exercises
NCCL and spills, but it copies frontier data through the host and is not the
production runtime. There is no packed-frontier archive, persistent-session,
8-GPU, S14/S15 capacity or performance result here.

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


## Isolated owner, route and two-GPU BFS contour

The experimental packed owner validates canonical inputs, uses the current
exact bitmap identity, and stores an admitted frontier row in eight bytes. It
handles duplicate inputs, a full segment and replay, including shared
rank-capacity contention between shards. A separate test-only kernel barrier
forces overlap between two shard streams when they update different bits of
one bitmap word. The barrier macro is for the test build, not release code.

The packed route emits existing route records and supports both per-record and
warp-grouped queue reservation. A CPU oracle compared outputs in 64
configurations: two physical GPUs, signed/byte payloads, repeated/distinct
symbols, identity/reversed rank mapping and cyclic/page-8 ownership. A forced
overflow is discarded and replayed with a larger queue; only successful
outputs are compared.

A separate test driver combines the experimental route and owner with actual
NCCL peer transfers on two RTX 3060 GPUs. Its host controls each layer and
copies states to the CPU for verification and back; it is a correctness fixture,
not an integration or performance benchmark. It verifies the full multiset of
canonical states for every completed layer against an independently enumerated
CPU BFS, including duplicates from separate origins, one-row segments, forced
queue overflow and retry. Four unlimited cases and two sufficient-cap cases
completed, and two one-below-cap cases stopped with exact completed prefixes:

| Graph | Rank cap | Last complete depth | First unfinished depth |
|---|---:|---:|---:|
| S5 adjacent swaps | 10 | 3 | 4 |
| Repeated-symbol example | 6 | 4 | 5 |

All eight cases passed. The S5 cases verified 120 states across 11 layers,
with 221 forced spills and 160 records sent remotely; the unlimited runs used
20 discarded overflow waves each. The repeated-symbol cases verified 60
states across 10 layers. These numbers describe this fixture, not throughput.

A CUDA virtual-memory boundary test maps one 2 MiB physical page and writes
local row 2^32-1 in a virtually reserved segment on each GPU. The following
candidate spills without publishing its bit. It does not physically allocate
or process 2^32 rows, and uses synthetic IDs to isolate addressing.

Independent owner, seed, route and boundary CUDA fixtures passed
Compute Sanitizer memcheck, initcheck and synccheck, and the two-stream overlap
test passed synccheck. For the combined NCCL fixture, initcheck and synccheck
passed normally; memcheck found no memory-access errors when run with
--report-api-errors no. With API-error reporting enabled, memcheck reports
160 CUDA error 209 events while NCCL probes kernel variants; that unrestricted
memcheck is therefore not represented as a clean run. The initial route
initcheck found that the test copied uninitialised empty queue slots; the
fixture was fixed to initialise them and the checks were repeated.

The NCCL fixture can be built with:

    nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 \
      -I$NCCL_ROOT/include tests/bitmap_pack4_nccl_bfs_gpu.cu \
      cuda/bitmap_pack4_route.cu cuda/bitmap_pack4_owner.cu \
      -L$NCCL_ROOT/lib -Xlinker -rpath=$NCCL_ROOT/lib -lnccl -o /tmp/pack4-nccl-bfs
    /tmp/pack4-nccl-bfs

Production integration still needs versioned ABI and capability checks,
collective readiness and matching codec metadata, admission and real memory
accounting, all relevant strategies, failure/rollback handling, archive and
persistent-worker lifetimes, public CayleyPy invocation, and layer-exact
multi-rank tests. S14/S15 and B200 remain unverified.

Stage 3 source and test evidence archive SHA-256: 
5794eeafb43c148f51e59e5ac8adba20e211eefd491356c20a1ce6eb4f8cbe52.
