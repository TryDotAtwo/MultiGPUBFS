# DB boundary audit (source evidence, not a performance result)

## Sirius

Inspected main `2611fa289d3ce788d9426977629a3699dc472e32` and dev
`1f996e3cd022f7f416bf83fd12c889005e8f730d` separately. They have substantially
different memory layers; conclusions about main must not be applied to dev.

- Main `src/memory/memory_reservation.cpp::request_reservation` enters an
  unbounded condition-variable wait when no memory space can satisfy a request.
  The separate `fail_reservation_limit_policy` is an allocation-over-reservation
  policy, not proof that initial reservation requests fail fast. A main-based
  adapter must not equate these two mechanisms.
- Dev `src/include/memory/sirius_memory_reservation_manager.hpp` derives from
  cuCascade and explicitly handles RMM 26.06 resource ownership changes. The
  already-tested RMM/cuDF 26.04 environment is not a validated Sirius dev SDK.
- Dev `src/include/pin_table.hpp` exposes GPU-owned cuDF table/column structures
  (`materialized_pin`, `device_pin_result`) and ingestion interfaces internally.
  The ordinary `PinTableArgs` entry point is path/table-based (Parquet/DuckDB),
  not an external device-pointer interface. Internal GPU insertion is a candidate
  to investigate; do not claim GPU-native ingestion is impossible just because
  the user-facing pin command takes a path.

Next probe must trace `gpu_ingestible` and scan-manager pinned-entry registration,
then demonstrate one generate/query/consume cycle without D2H intermediate
materialization. It must separately prove fixed physical reservation and fatal
exhaustion rather than wait/spill. No executable Sirius probe has passed yet.

### Dev scan-manager entry point found

At the same dev SHA, `src/include/scan_manager/sirius_scan_manager.hpp` declares
`insert_pinned_entry` taking owned `vector<unique_ptr<cudf::table>>`, explicit
per-chunk cuCascade memory spaces, logical types, statistics and storage metadata.
This is a concrete candidate for GPU-native batch registration, without a
mandatory file round-trip at this call boundary. The complete execution path
and whether registration can bind to an appropriate catalog identity remain
unverified.

Two adapter hazards are explicit in the interface documentation:

- Same-name re-insertion with the same row count may merge additional columns
  while retaining existing columns. It is NOT a guaranteed batch replacement.
  A reused BFS slot must remove the drained old entry or use a fresh identity;
  otherwise equal-sized batches could query stale data.
- `try_match_cached_entry` can fall back to a disk-reading provider on a cache
  miss or malformed entry. Closed-loop tests must assert a cache hit and reject
  fallback; successful SQL alone does not prove GPU-resident BFS execution.

The API also exposes `remove_pinned_entry` and generation-checked late-materialization
handles. Removal/replacement must be serialized after query consumers drain.
These observations identify a plausible adapter, not a benchmark win or a final
DB rejection.

Source: https://github.com/sirius-db/sirius/blob/1f996e3cd022f7f416bf83fd12c889005e8f730d/src/include/scan_manager/sirius_scan_manager.hpp

Implementation follow-up at the same immutable commit:
`src/scan_manager/sirius_scan_manager.cpp:2276-2305` marks a fresh entry GPU-tier,
calls `cudf::table::release()` and moves owned columns into the pinned entry.
There is no host serialization in this registration block. This transfers
ownership of owning cuDF columns; it is not a borrowed raw-pointer API for a
reusable BFS allocation. Allocator compatibility and consumer lifetimes still
need an executed adapter test.

The equal-size merge hazard is confirmed by implementation at 2180-2206:
existing column names cause the new incoming columns to be discarded, retaining
the old data. The return value lists only newly stored columns (2261-2269).
An adapter must check that all submitted columns were stored, not treat a
non-throwing registration as proof of replacement. Explicit removal (2500-2503)
invalidates the late-materialization handle and erases the entry; no GPU stream
drain occurs in that method, so the caller must establish completion first.

Source: https://github.com/sirius-db/sirius/blob/1f996e3cd022f7f416bf83fd12c889005e8f730d/src/scan_manager/sirius_scan_manager.cpp

Source links:
- https://github.com/sirius-db/sirius/blob/2611fa289d3ce788d9426977629a3699dc472e32/src/memory/memory_reservation.cpp
- https://github.com/sirius-db/sirius/blob/1f996e3cd022f7f416bf83fd12c889005e8f730d/src/include/pin_table.hpp
- https://github.com/sirius-db/sirius/blob/1f996e3cd022f7f416bf83fd12c889005e8f730d/src/include/memory/sirius_memory_reservation_manager.hpp

## HeavyDB

Inspected `b348f14049a34b21cdc40f6fe94a80845134d5ad`, `heavy.thrift`.
`sql_execute_df` has explicit device type/id and transport arguments, whereas
`load_table_binary_arrow` accepts a serialized binary Arrow stream, and the
row/column loaders accept Thrift values. This RPC surface is asymmetric: GPU
result export does not itself establish GPU-pointer input support.

Next probe must inspect in-process ingestion/UDTF interfaces before selecting
either a genuinely GPU-resident adapter or a deliberately measured round-trip
baseline. The latter cannot silently count as the required closed-loop native
owner. No HeavyDB execution, rejection, speed, or capacity claim is established.

Source: https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/heavy.thrift

### In-process ingestion and execution follow-up

At the same HeavyDB SHA, a device-to-device ingress primitive does exist:
`AbstractBuffer::write(..., GPU_LEVEL, src_device_id)` dispatches through
`GpuCudaBuffer::writeData` to `CudaMgr::copyDeviceToDevice`. Therefore the RPC
asymmetry is not evidence that all in-process ingestion must traverse the CPU.
The buffer is owned by HeavyDB; this is a copy into its allocation, not adoption
of a caller's recyclable pointer. The primitive alone does not register catalog
columns, fragment metadata, statistics, or establish a query-visible batch.
The next adapter should test those registrations and pin/unpin lifetimes around
an existing GPU chunk, then measure both D2D ingress and output consumption.

`setArrowTable` is not a shortcut to that device-buffer path. Its inspected
`ArrowForeignStorageBase` implementation uses host Arrow allocations, CPU
statistics access and `std::memcpy` in reads. Merely wrapping CUDA addresses
as ordinary Arrow buffers is not a supported device ingestion demonstration.

`TableFunctionManager::allocate_output_buffers` explicitly constructs its
`QueryMemoryInitializer` with `ExecutorDeviceType::CPU`; `makeBuffer` uses
host `checked_malloc`. This particular manager is not proof of a GPU external
pointer table function. It does not rule out separate GPU UDTF execution paths.

Capacity and fallback remain unproved. `GpuCudaBufferMgr::addSlab` calls the
CUDA allocator when adding a slab, so a configured maximum pool is not by
itself a fully preallocated physical reservation. `RelAlgExecutor` guards some
CPU retries with `g_allow_cpu_retry` and query-step retries with
`g_allow_query_step_cpu_retry`. Another inspected `QueryMustRunOnCpu` catch
directly changes `co_copied.device_type` to CPU and retries the subsequence.
A probe must establish which entry point it executes, disable all applicable
fallbacks, and verify device execution; setting only one flag is insufficient
evidence. No executable HeavyDB adapter or benchmark has passed yet.

Sources (all at the immutable SHA above):
- https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/DataMgr/AbstractBuffer.h
- https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.cpp
- https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.cpp
- https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/DataMgr/ForeignStorage/ArrowForeignStorage.cpp
- https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/QueryEngine/TableFunctions/TableFunctionManager.h
- https://github.com/heavyai/heavydb/blob/b348f14049a34b21cdc40f6fe94a80845134d5ad/QueryEngine/RelAlgExecutor.cpp
