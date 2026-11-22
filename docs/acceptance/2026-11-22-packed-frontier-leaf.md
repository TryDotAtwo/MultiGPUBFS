# Experimental four-bit permutation frontier: isolated prototype

This branch is experimental. It includes an opt-in integration of the packed
bitmap frontier into the native distributed runtime and the CayleyPy adapter
when an explicitly compiled experimental library is supplied. It is not in the
published wheel and is not selected by the normal profile. Only bitmap/atomic
is implemented; requests for the other strategies or backends fail explicitly.
The tests below report their exact hardware and verification scope. Eight-GPU
acceptance, full archive export, S14/S15 capacity and B200 measurements remain
open.

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
claim a physical allocation or traversal of 2^32 rows. The isolated stage did not change admission; the explicit runtime
integration described below adds its own packed admission path.

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

The isolated driver is followed below by runtime integration checks. It does
not by itself prove the production runtime, all strategies, full archive
export, eight-rank behavior, or S14/S15 capacity.

Stage 3 source and test evidence archive SHA-256: 
5794eeafb43c148f51e59e5ac8adba20e211eefd491356c20a1ce6eb4f8cbe52.

## Opt-in native-runtime integration on two GPUs

The library is built with MGBFS_EXPERIMENTAL_PACK4=ON. A job can request
MGBFS_SEGMENT_PACK4=1 with permutation_bitmap and atomic; it owns an
immutable dictionary of at most 16 signed symbols. The native executable and
library check a versioned capability and structure sizes, and every rank
agrees on the dictionary and mode. Unsupported backend/strategy combinations,
a missing/old library, and disagreement across ranks fail explicitly. This
flag is experimental and is not a supported installation instruction.

The two-GPU native-runtime tests used two RTX 3060 (SM86, 12 GiB), driver
580.178.04, CUDA toolkit 12.8.93 and NCCL 2.30.7. Six small graphs, including
S4, signed/repeated symbols, directed transitions and coupled position
orbits, were compared to an independent CPU BFS using full multisets at
**every** completed depth. Exact and padded exchange, two/three-bank presets,
page/cyclic ownership and a resource cap one below sufficiency were exercised.
The latter returned INCOMPLETE with an exact completed prefix. Switching
packed -> ordinary bitmap -> segmented hash -> packed reused the same native
workers and preserved the checked results.

The public CayleyPy CayleyGraph.bfs() path was exercised on two GPUs using a
source checkout of CayleyPy at 2d35a8917ee1d123d6ec8c771bc08002c974a57f
and the experimental local runtime, not an installed release wheel. The tests
covered distinct and repeated states, both presets, a cap-induced
RESOURCE_10, codec switching and worker PID reuse. All layer counts and
full terminal state multisets were checked; the native-runtime fixture above
provides full per-layer comparisons.

For the **non-packed** defaults in the same binary, atomic/warp on both
backends and batch with exact/padded exchange passed small per-layer state-set
checks, including empty peer queues, repeated symbols, spills and persistent
reuse. These are not a substitute for eight-rank fault-injection acceptance.
The CPU-only Rust library tests passed 46/46. With CUDA enabled and
CUDA_MODULE_LOADING=EAGER, 89 tests passed, one GPU-sizing test was ignored,
and two LSA-specific tests failed because this test library was configured
MGBFS_NCCL_LSA=OFF (the disabled entry point returns status 7). Re-running
the other tests with those two filtered gave 89 passed, one ignored. A first
run without EAGER also failed the runtime's loading-policy checks. Neither
failure is counted as a passing full CUDA suite.

### S12 same-machine experiment

Two complete repetitions per mode, interleaved ordinary/packed/packed/ordinary,
used one binary and two RTX 3060. Both modes used two frontier banks,
four shards, segment_rows=1,048,576, 4 owner lanes, queue capacity 8192,
batch ceiling 4096 and exact exchange. These are explicit experimental
settings, not claims about default preset geometry. All runs completed all
479,001,600 states across 67 layers. Every layer size matched an independently
calculated Mahonian distribution; full terminal and penultimate state sets
matched, not every large layer's full state set.

| Resident frontier | BFS seconds | Depths 27–38 seconds | Frontier bytes per GPU | Sampled max MiB per GPU |
|---|---:|---:|---:|---:|
| ordinary SoA | 101.2866, 101.3311 | 56.9041, 56.9742 | 402,653,184 | 723 |
| packed u64 | 102.9840, 103.4047 | 57.9944, 58.0766 | 268,435,456 | 595 |

Bitmap visited storage was 29,937,600 bytes/GPU and queues 6,291,552
bytes/GPU in both modes; reserve was 2 GiB per GPU. Exact state-allocation
receipts exclude the rest of device use. NVML sampling was every 0.5 s and can
miss short peaks. Both modes reported two source retries and 25,598,186 peak
live rows on each rank. The experiment changes both payload and the
implementation of routing/acceptance; it does not isolate one operation.
At S10 the packed variant was faster in two short observations, but that
does not imply an S12 or eight-GPU speedup.

Build artifact SHA-256 used for these runs:
mgbfs 9431576ff47f834be3c649cefcd9de9f5959379c0facb839317361e269bc6f87;
libmgbfs_cuda.so e6d51a55218cea1b4685b16d0be31e2ce7ca6532388da9761b2ab749e3840ee1.
The work remains opt-in: full packed archive, warp/batch implementations,
failure injection on eight GPUs, practical capacity pressure sweeps and B200
runtime verification are not established by these measurements.

### Balanced with three banks

Under the same GPU pair, geometry and four-run alternating protocol as the
two-bank test above, the balanced variant completed S12 twice in each mode.
Ordinary SoA took 100.5546/101.5104 s (window 56.5319/57.0970 s); packed
took 103.1903/103.0101 s (window 57.8443/57.8312 s). Frontier allocations
were 603,979,776 versus 402,653,184 bytes per GPU. The sampled NVML maxima
were 915 versus 723 MiB per GPU; both reported two source retries and
38,179,213 peak live rows per rank. As above, layer counts and only the two
terminal full state sets were validated at this scale.

The CayleyPy adapter was also tested with autotune enabled and a tight
per-rank state cap: both balanced and max_capacity correctly returned
RESOURCE_10 with validated resource-limited pilots and exact completed
prefixes, followed by a healthy job in the same worker processes. This does
not establish a measured packed tuning winner on a large graph.

The saved measurement snapshot includes the original source diff and binary;
a later edit only corrected outdated comments in segmented_history.rs and did
not change executable logic. The recorded hashes identify the actual binaries
used in the benchmark.

Timing correction: the original benchmark summary used a one-layer-offset time slice. Window values above have been recomputed from raw layer_seconds: index d-1 times creation of depth d. Full BFS times and correctness results are unchanged; the original summaries are retained in the evidence archive.
