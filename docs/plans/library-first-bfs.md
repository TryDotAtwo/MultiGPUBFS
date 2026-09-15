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
