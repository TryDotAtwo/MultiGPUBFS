# CUB route allocation query

`mgbfs_route_query` reports five flat planes plus one shared scratch allocation.
SortPairs and Flagged use the compiled CUB policy's actual size-query APIs on
the current CUDA device. The plan allocates max(sort_scratch, select_scratch),
not their sum, because these operations run serially on its stream. The create
path now consumes this query before its first cudaMalloc. Run kernels and
sorting semantics are unchanged; this is not a throughput optimization claim.

The64-byte C ABI is mirrored by Rust RouteBytes and a named QueryResult adapter.
The adapter checks capacity/plane agreement and shared-scratch accounting.
Output is cleared on query failure. Zero/over-i32 capacities fail before CUB;
CUB errors fail closed. External input/output/count buffers, allocation rounding,
CUDA context and driver overhead are excluded. The caller must select the same
device for query/create/run and bind query provenance to its rank plan. A CUB
query may initialize the CUDA runtime; it is not a no-driver/offline formula.

Validation:

- RED: CPU report test failed before RouteBytes existed; public C++ query test
  failed to link before the query existed.
- GREEN: two Rust report tests; optional-CUDA tests type-check; formatting.
- Public C++ query fixture compiled with CUDA12.5 targeting sm75 and MSVC17.14,
  using `-allow-unsupported-compiler`, then printed ROUTE_QUERY_PASS. Only host
  queries ran on the local RTX3070Laptop context, not CUDA data-plane kernels.
- Added route-query CTest and assertions to the existing GPU route fixture,
  which exercises all128 hash bits, duplicate handling, lengths0/1/31/256/4097.
  These updated GPU fixtures have NOT run on T4 yet.
- Seven Python launcher/evidence tests passed. Future artifacts containing both
  allocation CTests must use verify_primitive_gate.py --require-route-query.

Kaggle v9 is independently running source df42c51 and does NOT include this
change. Its result cannot certify this route create-path change or Rust wrapper.
No production multi-rank runtime or end-to-end A/B result is claimed.

The next prepared launcher pins77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5.
It runs the same80 device/test/tool combinations concurrently across the two
independent physical GPUs. Per-GPU command order and all fixtures remain fixed;
each child gets a disjoint CUDA_VISIBLE_DEVICES. Per-command log paths are
distinct and summary writes are locked. Each mask uses the physical GPU UUID,
not an assumption that CUDA ordinal and nvidia-smi index orders match.
Worker failure prevents a final PASS;
both workers drain before final reporting. This is not multi-rank communication.
RED/GREEN CPU tests verify actual worker overlap, exception propagation and the
complete80-command matrix with device isolation. Ten Python tests pass.
The currently running v9 is unaffected; the next launcher has not run yet.
