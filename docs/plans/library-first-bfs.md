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
