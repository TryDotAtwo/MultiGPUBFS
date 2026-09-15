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
