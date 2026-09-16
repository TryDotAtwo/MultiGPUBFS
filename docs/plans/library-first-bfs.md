# Library-first BFS implementation ledger

Accepted 2026-09-15. Baseline: `04c7a70`; measured lookahead panels:
`013ed5c979f4225db273e0015fa9ed72fd230c90`. CayleyPy remains pinned to
`f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.

## Acceptance

Minimize per-device **peak total VRAM**, not average or live pool bytes.
Search and durable completion may each regress at most 20% against the best
existing native for the same workload/rank count. Reserve at least 1 GiB/T4.
Pool suballocation is allowed; physical pool growth, managed-memory spill,
CPU data-plane fallback, and runtime backend switching are not.

## Work sequence

- [ ] Fixed-pool admission and shared owner ABI; immutable dependency pins.
- [ ] Isolated CCCL_BULK, CUCO_INDEXED, CUDF_RELATIONAL owner slices.
- [ ] Sirius and HeavyDB closed-loop feasibility probes; reject with evidence
      if bounded GPU-resident execution cannot be provided by a small adapter.
- [ ] DENSE/HASH_FIRST integration, rank exchange and macro settlement.
- [ ] Taskflow/CUDA Graphs versus existing event scheduler.
- [ ] Arrow CPU versus libcudf Parquet writer; bounded KvikIO/nvCOMP experiments.
- [ ] Full-state oracles and four sanitizer tools on real T4.
- [ ] Five-repeat S10/S11/S12 and U4(m=12,16,24) panels on 1/2 GPUs.
- [ ] S13 capacity gate for winners, S14 preflight; no duplicate bulk local data.
- [ ] Pareto report, code-size comparison, verified HF artifacts.

## Interfaces and lifetime rules

Library nodes receive explicit counts, GPU pointers and caller stream. Hash128
identity and archive encoding do not change. Results carry survivor source
indices; no mandatory full-state gather inside dedup. Per-shard writers are
serialized; different shards may overlap. Old hash tables and referenced key
arrays outlive all probes. Never publish an index into a recyclable transient
slot. Reserve capacity before publishing accepted hashes.

For libcudf: reuse history `filtered_join` objects; rebuilding mutable next is
an explicit measured cost bounded to the owner shard, not the entire layer.
AoS/SoA conversions and internal host synchronization must be counted.

## Dependency policy

CUB/Thrust, CUTLASS and NCCL are existing baseline dependencies. RMM supplies
fixed physical pools. cuCollections and libcudf compete for membership/dedup.
Taskflow is the first scheduling experiment; CUDASTF and cuCascade remain
experimental. Sirius and HeavyDB are executable DB candidates. Kinetica,
SQreamDB, PG-Strom and legacy BlazingSQL are documented alternatives, not
mandatory new dependencies. DuckDB is useful for archive analysis. Explicit
cuGraph/Gunrock/GraphBLAST adjacency is not substituted for implicit generation.

New leaf tests do not certify the production runtime. Compile-only, CPU,
single-T4, two-T4, sanitizer and benchmark evidence are tracked separately.

## Evidence, 2026-09-15 implementation slice

- Branch: `codex/library-first-bfs`; no baseline source was modified.
- Fixed-pool admission: three passing CPU tests, including complete reservation,
  alignment, overflow and the 1 GiB reserve. Successful budget validation is not
  a measurement of CUDA/NCCL/driver residency.
- `OwnerCommitGate`: nine passing CPU tests. Comparison, actual credit grant,
  and successful device completion are distinct; malformed order/count/capacity
  errors poison the gate without publishing pending rows. This gate is not yet
  connected to an executable library owner or the production scheduler.
- Core/runtime/CLI CPU regression passed. Python notebook guard suite: 26 passed.
- Changed Rust files pass rustfmt. Whole-package formatting check still finds
  a pre-existing difference in `crates/mgbfs-runtime/src/route_count.rs`.
- Kaggle `trydotatwo/mgbfs-library-owner-t4` v1 verified two physical T4s but
  failed in ensurepip; v2/v3 installed cuDF/RMM 26.4 successfully but a merged
  stderr diagnostic polluted the path-query output. v4 separates that protocol
  and reaches CMake with CUDA 12.8.93/GCC 11.4, then fails to locate cuDF's CMake
  package in the wheel prefixes. Do not infer absent headers or unsupported
  T4 from that configure failure: package inventory remains to be inspected.
- No new GPU executable has compiled or passed yet. No sanitizer, full BFS,
  memory improvement, speedup, or DB viability result is claimed.

Next: inspect installed wheel file/config inventory (not another blind prefix
change), establish a reproducible C++ SDK environment, pass the characterization
fixture, then implement the first owner adapter against its observed contracts.

### SDK gate follow-up

Kaggle v5 inventory proves the wheels DO contain headers, shared libraries, and
`lib64/cmake/*/*-config.cmake`. v6 discovers those exports after adding their
actual directories to the CMake search; two CPU guard tests cover environment
isolation and discovery of lib64 plus transitive CCCL configs (27 total guards).
v7 with source `61d2c516daca6588161712455f79b38117d25c07` configures successfully
with exact version `26.04.0` and reaches compilation. Compilation fails on
RAPIDS bundled CCCL references to `cudaDevAttrHostNumaMemoryPoolsSupported`
missing from Kaggle CUDA 12.8.93, and on the fixture's missing column-view
null-mask arguments. The latter is corrected in source, not yet GPU-compiled.
Next action: provision an explicitly pinned compatible CUDA SDK and rebuild;
do not mix newer runtime headers with an older toolkit or declare T4 unsupported.
No new performance or GPU-correctness claim follows from successful configure.

### Passing T4 library characterization, Kaggle v9

`trydotatwo/mgbfs-library-owner-t4` v9 completed successfully with C++ source
`3036622b637453bc25d9fe8fd949ee93c31c2aa3`, launcher at `ba0d72c`, and the
checksummed CUDA 12.9 redistributable SDK. Both distinct physical Tesla T4s
passed plain, memcheck, racecheck, initcheck and synccheck: 10 fixture runs,
8 sanitizer runs with zero errors (racecheck also zero warnings).

Each fixture verifies all four hash words, within-batch duplicate elimination,
history-index reuse, accepted-next exclusion and empty inputs. The deliberately
oversized allocation raises the expected RMM OOM; it is not a sanitizer error.
Pool reservation remains 67,108,864 bytes; requested suballocation peak is 2,464
bytes on this tiny fixture. Neither value is whole-device VRAM or BFS capacity.
No full BFS, native owner adapter, overlap or speedup has been demonstrated.

Small raw evidence is under `test_results/library-owner-v9/library-owner/`:
summary, ten individual logs, build/configure, CUDA version, installed-package
list and pip installation report. Exact wheel hashes from that report are in
`experiments/library_owner/requirements-linux-x86_64.lock` for future runs.
The SDK obstacle is resolved; next is implementing the actual owner adapter
with bounded persistent accepted storage, explicit compare/commit and tests.

### Experimental owner implementation and repeated batches

The RED GPU fixture v10 compiled and failed at `OWNER_NOT_IMPLEMENTED`, proving
the new compare/commit test reached its intended missing behavior. Implementation
`ad251654b02a3b0a1e0356b8feeb74814a9c718c` passed the v11 gate on both T4s,
plain plus all four sanitizer tools. `CudfOwner` now owns fixed-capacity accepted
hash columns, borrows immutable history, stages survivors separately, checks
actual grant before commit, and poisons errors instead of permitting retry.
Accepted count is stream-ordered, not proof that GPU publication has completed.

Expanded v12 at `64c3b12adb0d6cfe1820122aeeab766a808fa970` also passed all ten
runs (eight sanitizer runs, zero errors/warnings). It exercises 48 batches of
1,024 rows against an independent host set, including pair duplicates, overlap
with 4,096 history keys and previously accepted keys, and more than 16,000 total
visited keys. Fixed reservation stays 64 MiB; requested suballocation peak on
this fixture is 320,156 bytes. No whole-device or large-BFS inference is valid.

Integration remains incomplete: `distributed_native.rs` uses the AoS
`mgbfs_bounded_owner_compare/commit` descriptor interface with GPU-side reserve,
whereas the experimental cuDF class takes SoA columns and host-visible library
counts. The C ABI, layout conversion and reserve/event bridge must be implemented
and charged to both timing and peak memory, not bypassed in the comparison.

### Shared C ABI gate, Kaggle v14

The v13 RED fixture reached the intended missing implementation and failed at
`OWNER_ABI_CREATE`. The implemented shared library at source commit
`9b34de715799ca996e1b1ba64a95ec7357ed5188` passed v14 on both physical T4s:
ten plain/sanitized fixture runs, including eight sanitizer runs with zero
errors (racecheck also zero warnings). Raw evidence is in
`test_results/library-owner-v14/library-owner/`.

The ABI fixture covers borrowed history, survivor source indices, an empty
second result, stale commit rejection and poisoned-handle rejection. The pool
remains fixed at 67,108,864 bytes; fixture suballocation peak is 320,156 bytes.
The logged oversized-allocation failure is intentional. These measurements
are neither full-device VRAM nor full-BFS performance.

Rust declarations have only type-check evidence, not linking or execution.
Remaining integration includes a Rust-accessible fixed RMM resource context,
AoS-to-SoA conversion, GPU reserve/event bridging and actual runtime dispatch.
This gate does not establish multi-rank/NCCL correctness or a speedup.

The fixed-pool C ABI lifecycle test reached `POOL_ABI_CREATE` in v15 at
`112036087037b74bf8c004eea46fc9b2b1598ea0`: the expected RED result, not a
compiler/dependency failure. Implementation `4759026b6a0e5760fe745aee50dcc9269b92c1c9`
is under GPU validation in v16. It reserves initial=max pool bytes, retains the
previous per-device resource and rejects nested pools or destruction with live
suballocations. The caller must drain streams and destroy owners before pool
release. Rust declarations remain type-check only; runtime integration is pending.

V16 subsequently completed: both distinct T4 devices passed plain and all four
sanitizers (ten runs, eight sanitizer summaries with zero errors and no racecheck
warnings). Evidence: `test_results/library-owner-v16/library-owner/`. The pool
lifecycle fixture proves live-allocation destruction rejection, nested-pool
rejection, fixed-size exhaustion, owner operation under the ABI-installed pool
and restoration of the previous resource. The printed 320,156-byte peak belongs
to the separate characterization pool, not an instrumented full runtime.

### Remaining runtime boundary (source audit)

`distributed_native.rs` currently reserves with owner control stage 1 and
materializes only after stage 2. Its finalization compacts fixed-bucket accepted
AoS hashes, reads the directory/count, then swaps previous/current layers.
The cuDF owner instead holds append-order SoA accepted keys internally.
Therefore integration requires all of the following, not just ABI dispatch:

- Explicit AoS-to-SoA inputs and span-local source ordinals, with bounded scratch.
- Transfer survivor count into stage-1 GPU control, run the existing actual
  StateRing reservation, and read its grant/error before cuDF commit.
- Publish stage 2 only after successful commit, retaining survivor indices until
  materialization has consumed them; failed reservations must not commit keys.
- Expose committed accepted keys for depth finalization, convert/partition them
  into the next layer's directory format, and check its count against reserved
  layer count. Existing fixed-bucket compaction cannot read cuDF private storage.
- Rebuild history from the appropriate depth window only after readers drain;
  account pool plus conversion/finalization scratch in the full VRAM plan.

These synchronization and conversion costs belong in end-to-end timings. No
bridge implementation or measured performance claim follows from this audit.

### Committed export and conversion progress

V17 reached the intended `OWNER_ABI_EXPORT` failure with the export stub. V18
at `94cc3747ec678c7bf3a594b832c77d8f1f269e9f` passed both T4s, plain plus all
four sanitizers (eight zero-error summaries, racecheck zero warnings). Evidence:
`test_results/library-owner-v18/library-owner/`. The ABI now exposes borrowed
append-order committed SoA keys without a host data copy and rejects pending or
poisoned owners. Export does not synchronize; consumers must obey stream/event
ordering. Conversion/partition and actual depth finalization remain unimplemented.

The CPU candidate conversion planner at `00667c8` accounts for five separately
256-byte-aligned u32 planes and rejects capacities outside `1..=INT32_MAX`.
Both new tests failed against the stub before implementation; all five library
memory tests then passed. This planner is not yet charged by the runtime.

V19 pins `08ad47758f69ad7f65904e03d74e19d2dbfee5c8` for the allocation-free
AoS/SoA conversion RED test: 65 rows, explicit plane offsets, source ordinals,
round-trip equality, untouched padding/guard and a short-buffer rejection.
Its conversion functions are still stubs; no conversion GPU pass is claimed.

V19 subsequently reached `LAYOUT_AOS_TO_SOA`, the intended RED failure. The
allocation-free CUDA implementation at `fab8fa3726976b850142aa4eb02751e6304239ee`
passed v20 on both distinct T4s, plain plus all four sanitizer tools with zero
errors (racecheck zero warnings). Evidence is in
`test_results/library-owner-v20/library-owner/`. The fixture verifies all key
words, source ordinals, non-block-multiple rows, plane padding/guard integrity,
round-trip conversion and short-buffer rejection. This is correctness evidence,
not a conversion throughput measurement or full BFS performance result.

V21 pins `3128fa8b3ab487561ce748225432ac9825fc45da` and links the actual
`cuda/state_commit.cu` into the probe: four incoming records include an old key,
a duplicated new key and another new key. The fixture checks two survivor
states paired with two exported hashes after actual native reservation and
materialization, including the ready flag and layer count. Host synchronization
in this fixture is explicit. Its result is pending; this is not yet Rust runtime
dispatch or a multi-rank test.

V21 passed on both T4s with all four sanitizers reporting zero errors and
racecheck zero warnings. Source remains `3128fa8b3ab487561ce748225432ac9825fc45da`;
evidence: `test_results/library-owner-v21/library-owner/`. This establishes the
small C++ integration fixture's state/hash pairing and native publication, not
Rust integration, exhaustive BFS correctness, NCCL execution or performance.
An additional ring-capacity failure fixture is being added before runtime wiring.

V22 at `6f9ad60dc58dfc01c9ae1d4f306ec725a057fee7` passed both T4s and all
four sanitizers (zero errors, racecheck zero warnings). The added fixture forces
two survivors into a one-record StateRing: zero grant, failed commit/export,
unchanged tail/descriptor tail and zero layer count. Evidence:
`test_results/library-owner-v22/library-owner/`.

The Rust `library-owner` feature now exposes `library_native::LibraryShard`,
which wraps the C ABI with `OwnerCommitGate`. New RED/GREEN CPU tests ensure
external FFI failure prevents later publication and comparison cannot replace
an uncompleted borrowed result. All eleven gate tests pass. Feature-enabled
`cargo check` passes, but this is neither linking nor GPU execution evidence.
The first adapter explicitly synchronizes its stream for completion/teardown;
that cost must be measured. It does not own the pool/history/stream and has not
yet been selected by `DistributedNativeBfs` or exercised from Rust on T4.

The library-only Rust feature no longer requires `MGBFS_CUDA_LIB_DIR` or links
the unrelated full native BFS library. Before separation the real build check
failed at that missing variable; afterwards library-only and ordinary native
feature checks both passed. `MGBFS_CUDART_LIB_DIR` selects the pinned CUDA Runtime
directory. This changes build dependencies, not runtime fallback behavior.

V23 pins `fd6a08589379ceae204b43df2e8e6939748fcc17` and adds real Rust linking
and execution alongside the C++ tests. The Rust test uses the actual
`LibraryShard`, removes one duplicate from three keys, verifies all key words
and checks that accepted counts change only after explicit stream completion.
Its private destination is preallocated; native StateRing remains covered by
the separate C++ fixture, not this Rust test. The runner uses the project's
pinned Rust version and locked Cargo dependencies, with plain plus all four
sanitizers per physical GPU. Results are pending.

V23 completed successfully: the actual Rust test executable linked against the
library SDK and passed plain/memcheck/racecheck/initcheck/synccheck on both T4s.
All eight Rust sanitizer runs reported zero errors (racecheck zero warnings).
Source `fd6a08589379ceae204b43df2e8e6939748fcc17`; evidence:
`test_results/library-owner-v23/library-owner/`, including `rust-build.log`,
`rust-version.log` and ten `rust-gpu*` logs. These test durations include test
setup and teardown and must not be reported as BFS timings. The production
distributed scheduler still does not select the library adapter.

V24 at `a526087bb49025d5e82eadb36c6e574c844174dd` validated event-driven Rust
completion on both T4s with all four sanitizers clean. Evidence:
`test_results/library-owner-v24/library-owner/`. Event readiness does not itself
publish host counts: the caller must check native fatal controls before the
separate publication call. The GPU test verifies counts stay unchanged until
that call. This does not establish scheduler overlap or a throughput improvement.

V25 pins `5eb30cc7f2334aa7a50c986121f4b839344736d9` for the separate history
window RED fixture. Previous and current keys differ in the fourth word, and
the candidates include duplicates against each plus a new key. The factory is
still a stub. Once implemented, two borrowed views avoid an explicit combined
history buffer; cuDF index allocation remains part of the fixed pool budget.

V25 reached the intended `OWNER_WINDOW_CREATE` RED failure. V26 at
`a6cf9b8ee63fbe55723cee478a4c261ab5727275` passed C++ and Rust plain plus all
four sanitizer modes on both physical T4s, with zero errors and no racecheck
warnings. Evidence: `test_results/library-owner-v26/library-owner/`. Two
independent immutable views now feed separate cached anti-join indexes; no
explicit combined history array is created. Index and temporary allocations
still belong to the fixed RMM pool and must be charged in full.

V27 pins `217b810d2e0e3d669904881f98ca2cc3935c9846` for a new RED boundary:
seal a completed owner, release history/candidate storage, export the accepted
key unchanged, and reject any subsequent comparison. The seal function is still
a stub. This boundary is needed to overwrite the oldest history buffer during
depth rotation without keeping an extra combined/copy buffer or invalidating
live borrowed history. No full runtime rotation is implemented yet.

V27 reached the intended `OWNER_WINDOW_SEAL` failure. V28 at
`9b47697309f35c3bdb78119bbeb5846d68839a90` passed C++ and Rust plain and all
four sanitizer modes independently on both physical T4s; all sixteen sanitizer
runs reported zero errors, with no racecheck warnings. Evidence:
`test_results/library-owner-v28/library-owner/`. The C++ seal fixture releases
history and candidate tables, then verifies exported accepted keys and rejects
new comparisons. This is a component gate, not NCCL or complete BFS evidence.

The Rust adapter now exposes seal with a host idle guard. A new CPU RED test
failed when that guard allowed pending work; after implementation all fifteen
owner and eight event tests pass. History may be released after successful seal,
but the pool/stream and accepted storage remain live until drained teardown.
The extended Rust GPU fixture checks export and rejected comparison after seal;
its GPU execution is pending. No library backend is selected by the production
distributed scheduler yet.

V29 at `80e762c0504ada1f9e221e886a310a2923675fa8` passed C++ and Rust plain
and all four sanitizers on both physical T4s. The added Rust seal/export test
is now hardware-validated; evidence `test_results/library-owner-v29/library-owner/`.
The complete CPU core/runtime lib/tests command also finished successfully.

V30 at `dfcdf5742c2283b2bc2b6351f46c938499eed7d6` reached the intended Rust
`LIBRARY_FINALIZE_NOT_IMPLEMENTED` failure. Its three-shard fixture has accepted
counts `[2,0,1]` and writes the new SoA layer over the previous history buffer.
It checks full keys, empty-shard offsets and untouched capacity-tail sentinels.
The finalizer now validates idle owners and total capacity, seals ALL history
leases before any write, copies four contiguous planes per nonempty shard,
drains, destroys owners and only then publishes ranges. It does not allocate
GPU storage or concatenate history. GPU validation of this implementation is
pending. Full distributed scheduler selection is still outstanding.

`library_shared_buffers` now accounts four individually aligned planes in each
history bank and one bounded five-plane candidate conversion scratch. It omits
legacy accepted/length/count/selected buffers instead of budgeting both owner
representations. Two new tests failed before implementation and pass now; all
seven distributed-memory tests pass. This is the allocation contract to wire
into the runtime, not an observed VRAM improvement or a full pool budget.

V31 at `5c49e4a44d004067b6cce04012a2ebced4115667` passed C++ and both Rust
component tests, plain and all four sanitizers independently on both T4s.
V32 at `3a05b41663253087302aa27a88d7d47a180b3d29` built and linked the full
native CUDA/CUTLASS/NCCL runtime together with the pinned library SDK. Its new
full-state BFS test reached `LIBRARY_BFS_NOT_IMPLEMENTED` as intended. Evidence
is in `test_results/library-owner-v31/` and `test_results/library-owner-v32/`.

`DistributedNativeBfs::new_library_reference` now explicitly selects the library
owner for DENSE or HASH_FIRST. The original native path stays the default. The
library path skips legacy owner plans/arrays, charges the entire fixed pool and
uses aligned SoA previous/current history. Incoming sorted hashes are processed
in shard-sized ranges; native StateRing grants precede key commit, and native
materialization/request generation consumes library survivor indices. Empty
results complete without passing a null index array to native materializers.
Depth finalization reuses the old history bank; owner handles reopen within the
same fixed pool and reuse preallocated completion events. The initial integration
uses explicit stream/count synchronization, all of which belongs in search time.
This is not an overlap or speedup claim. GPU full-layer validation, archived runs,
actual two-rank exchange and CLI/benchmark selection remain pending.

V33 at `03bba0f453168f1f6222fc5290d1159130bcf7c4` passed the full BFS test
plain and all four sanitizers independently on each physical T4. U4(modulus=2)
full-state layers match the CPU oracle for DENSE/HASH_FIRST and pre-dedup OFF/ON;
the test also checks the default native path after its storage refactor. All
eight full-BFS sanitizer runs reported zero errors, racecheck zero warnings.
Evidence: `test_results/library-owner-v33/library-owner/bfs-gpu*.log`. Durations
include multiple runs, initialization and test assertions; they are NOT BFS A/B
timings. This gate does not validate simultaneous rank exchange or archives.

V34 pins `d37cb51c0d932540274676ef2cd7357aac71f203` and adds a true two-device
NCCL gate using two rank threads in one process, not a torchrun benchmark. It
tests U3(modulus=3) and matrix S4, both profiles, rank maps [0,1]/[1,0], batch=1,
full-state snapshots and checksummed per-rank archives. Archive keys, ownership,
per-depth counts (including empty local layers), and global layer sets are
checked independently. Results are pending. Process-launch, capacity and
performance acceptance remain open even if this gate passes.

Reference-launch selection now has a separate `ReferenceOwner` enum, so a
library request cannot be passed as a native `OwnerBackend`. CUDF_RELATIONAL
requires a compiled library feature and explicit aligned nonzero
`MGBFS_LIBRARY_POOL_BYTES`; native modes reject unused pool settings. The
library benchmark rejects disabled archives. Five CPU selection tests pass,
including RED-before-GREEN coverage of missing dispatch/pool/archive guards.
Production RunConfigV1 and its OwnerBackend enum are unchanged.

The CLI `library-owner` feature enables native CUDA plus the library adapter.
The reference benchmark dispatches explicitly, includes the pool in config
identity and output, and labels library records separately from native records.
HASH_FIRST Tensor generation is now an explicit library constructor option,
with no scalar substitution. The next GPU fixture adds full-layer Tensor checks
and an actual two-process torchrun CLI warmup/measure/archive-verification gate.
Windows type/CPU checks do not compile the Linux-only benchmark module; Linux
build, CLI launch and the new Tensor combination remain pending. Tiny S4 CLI
times will remain correctness-fixture durations, not performance claims.

Reference group parsing now supports `sN` and `uNmM` (for example `u4m12`),
using the existing graph constructors, without changing their generators.
The benchmark stores the canonical group name and rejects permutation codecs
for unitriangular inputs. Two CPU tests cover constructor/order agreement and
invalid inputs; the positive test failed before implementation. The next CLI
gate includes U4(modulus=2) as well as S4. The output explicitly labels its
current cudaMemGetInfo sampling as setup/final only, not a full peak measurement;
external/full-peak profiling is still required for acceptance.

V34 at `d37cb51c0d932540274676ef2cd7357aac71f203` passed the two-device NCCL
and archive fixture plain and all four sanitizers. U3(modulus=3) and matrix S4,
both profiles and both rank maps matched full CPU layer sets; archive ownership,
hashes, counts and commits passed. All sanitizer summaries were clean, including
zero racecheck warnings. Evidence: `test_results/library-owner-v34/library-owner/`.
This is two rank threads on two devices, not yet a process-launch result.
V35 pins `248142359ad42ba64682d88a7de4f17765188fc2` for the added Tensor and
actual torchrun/CLI/archive gates. No speed or peak-memory acceptance is claimed.

The comparison aggregator now retains every durable-time sample and reports its
median and MAD separately from search time. Missing archive timings remain null
for the baseline; mixed missing/invalid samples fail instead of producing a
partial statistic. Per-rank library pool reservations must agree. All 19
distributed-metrics CPU tests pass after the new cases failed against the old
implementation. This changes reporting only; no new GPU measurements are implied.

V35 at `248142359ad42ba64682d88a7de4f17765188fc2` completed with PASS on two
physical Tesla T4s. Downloaded evidence is in
`test_results/library-owner-v35/library-owner/`. The two-device NCCL fixture
passed plain and all four sanitizers (zero errors; racecheck zero warnings).
Linux CLI build and actual two-process torchrun runs completed for S4 DENSE,
S4 HASH_FIRST INT_MMA_SM75, and U4m2 DENSE, with warmup and mandatory archives.
The runner checked global counts 24/24/64 and verified both rank archives for
each scenario. These remain small correctness gates, not a performance panel;
the CLI reports setup/final cudaMemGetInfo only, not full peak VRAM.

V36 launches immutable source `d8e33dc73730d96a13dae3f7f4e98de94bc1b8e6`.
After the correctness gates it runs a single S10 DENSE/cuDF screening sample on
two T4s: batch 32768, per-rank state/hash capacity 3628800, ring 3628800,
fixed RMM pool 1 GiB per rank, pre-dedup ON, matrix state/archive encoding.
Warmup and both rank archives are mandatory. The screening helper retains raw
measurements and failures, verifies archive commits, and labels the external
50 ms device-memory sampler as sampled consumption, not exact peak. Six CPU
screen-contract tests pass. This is not a tuned or repeated baseline comparison;
the actual S10 result is still pending.

V36 completed PASS. The S10 screening produced 3,628,800 states across 46
nonempty depths and verified both rank archives. Search: 7.178855308 seconds;
durable commit: 9.342252056 seconds. External 50 ms nvidia-smi maxima were
1885 MiB on each T4 (3770 MiB sum of per-device maxima, not exact simultaneous
peak). Explicit aligned allocation plan: 1,699,719,936 bytes per rank, including
the fixed 1 GiB library pool; it excludes NCCL/driver and pinned archive memory.
Evidence: `test_results/library-owner-v36/library-owner/screen-s10-dense/`.
This single untuned sample does not establish the <=20% acceptance requirement
or a speed/memory win against the preserved baseline. The full panel is pending.

V40 at `e6a10f8fb56bfac483bf3d79c22b9bd7e5175578` completed the cuco
indexed-Hash128 membership fixture independently on both T4s, plain and all four
sanitizers, with zero errors or racecheck warnings. Equality covered every hash
word and duplicate aliases across all four storage tags. This is not a multi-GPU
owner or BFS result: candidate-slot reuse, stable commit, deterministic provenance
and actual CUCO_INDEXED runtime integration are still outstanding. Full peak VRAM
and the repeated baseline comparison remain required; the goal stays active.

The cuco owner contract passed on both T4s in v42 at
`c6348bba6f7d9f6937c48e06b168f281b7ccf7c7`, including multi-block full-key CPU
oracle checks, candidate-slot reuse, deterministic first-row provenance, and
all four sanitizers with zero errors/warnings. Integration source
`7bd6a6a6f907774ac01b827735770c672c1155a7` adds the common owner ABI and explicit
CUCO_INDEXED reference dispatch; v43 is testing full BFS rather than just the
isolated owner. Its outcome must be checked before any integration claim.

The screening helper now accepts an explicit CUCO_INDEXED or CUDF_RELATIONAL
selection and rejects a returned owner different from the requested one. The
next runner configuration compares one S10 DENSE sample for each with identical
batch, state/ring capacity, pool reservation, pre-dedup and mandatory archives.
This is an untuned paired screen, not the required repeated baseline panel.
All 117 Python tests passed locally. CPU tests for mgbfs-core, mgbfs-runtime,
mgbfs-cli and mgbfs-cuda passed; the workspace-wide command cannot build the
separate root GPU package without MULTIGPUBFS_CUDA_LIB_DIR. Neither CPU result
establishes GPU correctness or performance.

Screen-result validation also checks the requested group, batch, profile,
pre-dedup, world/rank inventory, declared capacities and generation mode against
every rank record. Agreement between ranks alone is insufficient: both ranks
could otherwise have run the same unintended configuration. Ten screen contract
tests pass, including wrong-workload and missing-peer rejection. V43 remains
running; the next screening source is not launched over that live job.

V43 completed PASS at `7bd6a6a6f907774ac01b827735770c672c1155a7` on two
physical T4s. Both owner backends passed full BFS oracle tests on each GPU
and the two-device NCCL fixture, plain and all four sanitizers, with zero
errors and zero racecheck warnings. The cuco common ABI fixture also passed
all tools on both GPUs. Six actual two-process CLI scenarios completed:
S4 DENSE (24 states), S4 HASH_FIRST Tensor (24), U4m2 DENSE (64), for each
owner. All twelve rank archives have VERIFIED checksum/count reports.
Evidence: `test_results/library-owner-v43/library-owner/`.
These are correctness results, not performance acceptance.

The next S10 paired screen retains plain correctness checks but does not
repeat the unchanged sanitizer suite. Its manifest lists executed sanitizer
tools (empty) separately from prior v43 provenance. No native/CUDA/Rust
source changed since that gate. Both owners use identical matrix encoding,
capacities and output contracts; one sample each cannot establish Pareto
acceptance or the <=20% regression condition.

Benchmark timeout cleanup was independently reproduced with real Linux child
processes in the existing `multigpubfs-ref046-green` container (no GPU/network,
read-only source mount). Before the fix, killing the launcher left its child
alive. Launchers now own an isolated process group and cleanup kills that group
on timeout, exceptional exit, or normal completion before another sample starts.
The wait respects the remaining deadline instead of always overshooting by a
20-second polling interval. Two Linux tests pass: timeout and failed launcher
with a surviving child. This is harness correctness evidence, not GPU evidence;
the already-running v44 uses its original immutable harness source.

V44 passed at `89a6873d2ee6afa4bff7a945f02383d6f33b87c5`. S10 DENSE, two
T4s, one matched sample per owner: CUCO_INDEXED search 1.841799048 s, durable
4.126001532 s, sampled device maxima 1815/1815 MiB; CUDF_RELATIONAL search
6.501415079 s, durable 8.445665735 s, sampled maxima 1885/1885 MiB. Both
returned the same 46 layer counts totaling 3,628,800 states, and both rank
archives verified. Fixed pool 1 GiB/rank, state/ring capacity 3,628,800/rank,
batch 32768, matrix_u8, pre-dedup ON. Explicit device allocation for both is
1,699,719,936 bytes/rank; pinned archive reservation is 243,269,632 bytes/rank.
The 50 ms external sampler is not an exact peak measurement. Sanitizers were
not repeated in v44; v43 is the unchanged-native-source sanitizer evidence.
Raw results: `test_results/library-owner-v44/library-owner/`.

Next panel uses five independent fresh-process samples per owner on 1 and 2
T4s, alternating owner order. S10 capacities are explicitly reduced to
1,000,000 state/hash/ring records and a 256 MiB pool per rank, with all other
screen settings retained. This is a fixed-capacity memory experiment, not a
promise to predict an unknown graph peak; overflow remains fatal. Each run
warms up and verifies both/every archive. Native/CayleyPy comparisons and
full-peak acceptance remain outstanding regardless of this panel's outcome.

The following panel adds preserved native commit
`013ed5c979f4225db273e0015fa9ed72fd230c90`, built unmodified with the same pinned
CUDA SDK and CUTLASS as the library reference. The screen helper accepts its
old example-binary argument interface and verifies archives using that commit's
CLI. It requires an explicit native selection, no library pool, and native
backend records; library selection cannot silently invoke the old binary.
Two added CPU tests cover native metadata and launcher/verifier dispatch.
The paired settings remain S10 DENSE, matrix_u8 state/archive, batch 32768,
pre-dedup ON, capacity/ring 1,000,000 per rank, five repeats on each topology.
This is a matched baseline point, not proof against the best tuned native.
V45 is still running the prior two-owner panel; this added native panel has
not yet been submitted or GPU-verified.

V45 failed at immutable source `11230ca009acbd176bda2d5176354349fa7ac446`.
CUDF_RELATIONAL S10 on one T4 completed its first sample and verified the
archive: search 9.672742343 s, durable 11.42103416 s, sampled peak 1097 MiB.
The following CUCO_INDEXED run failed in owner construction: fixed RMM pool
256 MiB exhausted while requesting another 1,573,120 bytes. There are no
completed cuco measurements or five-repeat/topology panel results from v45.
Evidence: `test_results/library-owner-v45/library-owner/`.

Source inspection identifies avoidable replication: each of 64 logical
shards constructs a `CucoOwner` with full `batch * moves` incoming capacity,
allocating its own candidate columns, minima, representatives, flags,
selection, sources, CUB scratch and transient set. The integrated shard loop
is currently serial on one owner stream. At batch 32768 and three moves,
candidate columns alone require 1.5 MiB per shard before the other temporary
buffers. Persistent accepted/hash storage must remain shard-local, but an
explicit shared transient workspace is the next memory optimization candidate.
It must retain candidate data through commit and survivor consumption and
reject overlapping use; tiny single-owner fixtures cannot prove that lifetime.
Do not hide this failed configuration by growing the pool or replacing v45.
The prepared preserved-native comparison is held until that correction is
tested. This does not reject the cuco backend or certify a shared-workspace fix.

Additional local Linux check: all 12 screen contract tests pass in the existing
container. The full Linux Python suite is not green: five modules could not
import missing PyArrow, and the container has no pip. This is a test-environment
limitation, not an archive regression; Windows ran 123 tests with two POSIX
skips, and the two actual Linux process-cleanup tests passed separately.

Shared temporary buffers are now implemented in the isolated C++ owner API:
`CucoWorkspace` owns candidate planes, minima, representatives, flags, selected
rows, source indices, counts and CUB selection scratch. Accepted keys and
persistent/transient hash sets remain per-owner. Optional explicit sharing is
restricted to the same device, stream and incoming capacity; callers must keep
the captured fixed resource alive. The unshared constructor remains compatible.

`CucoWorkspaceLease` keeps the workspace acquired through compare, commit and
downstream result consumption, until an explicit caller-drained complete.
Wrong owner/epoch, overlap, premature completion and reuse after abort are
rejected. Its CPU contract test passes in Linux. The CUDA fixture now covers
two shared owners, independent accepted keys, delayed readers, overlap rejection
and less allocated memory than two private workspaces. Those CUDA checks have
not run yet. The next gate runs the isolated owner on both T4s with all four
sanitizers, not performance or full BFS. C ABI/Rust sharing integration and
the 256 MiB full-BFS rerun remain to be implemented and verified.
