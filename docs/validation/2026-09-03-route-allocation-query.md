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
  These updated GPU fixtures subsequently passed on both T4s in version10 below.
- Seven Python launcher/evidence tests passed. Future artifacts containing both
  allocation CTests must use verify_primitive_gate.py --require-route-query.

Kaggle v9 independently passed all80 combinations on source df42c51 and does
NOT include this change. Its result cannot certify this route create-path change
or Rust wrapper.
No production multi-rank runtime or end-to-end A/B result is claimed.

The version10 launcher pins77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5.
It runs the same80 device/test/tool combinations concurrently across the two
independent physical GPUs. Per-GPU command order and all fixtures remain fixed;
each child gets a disjoint CUDA_VISIBLE_DEVICES. Per-command log paths are
distinct and summary writes are locked. Each mask uses the physical GPU UUID,
not an assumption that CUDA ordinal and nvidia-smi index orders match.
Worker failure prevents a final PASS;
both workers drain before final reporting. This is not multi-rank communication.
RED/GREEN CPU tests verify actual worker overlap, exception propagation and the
complete80-command matrix with device isolation. Ten Python tests pass.
Version10 was submitted after all version9 artifacts were downloaded and
verified. Its own complete artifacts subsequently passed the verifier with
--require-route-query and the exact source commit.

## Version10: verified hardware result

Source:77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5. CUDA12.8.93; CUTLASS
ffa119a1255d78998536107466cc7097ecefa393. Private Kaggle kernel:
trydotatwo/mgbfs-native-matrix-primitives-t4, version10.

- GPU0 Tesla T4: GPU-aa6e312e-d526-9c9b-35be-cfe7daf43f2d.
- GPU1 Tesla T4: GPU-8bfa7bd6-de54-2f8c-018b-b0d2392390d3.
- Each GPU15360MiB total,14912MiB free at initial inventory (NOT peak VRAM).
- All80 combinations passed: eight executables, two devices, plain plus four
  sanitizer modes. All64 final sanitizer summaries had zero errors; racecheck
  also had zero warnings/hazards. Exact fixture inventories were checked.
- Both CTests passed: allocation-query and route-query. Generation/hash/route
  Rust wrappers and their ABI queries were exercised by the GPU test binaries.
- Full-state feedback sweeps cover m2..m6 in the regular modes. The documented
  bounded m2..m3 racecheck fixtures remain bounded; no larger racecheck coverage
  or absent production profiles/backends is implied by80/80.
- Actual overlap is recorded, not inferred from config: GPU1 dense/racecheck
  started at971.244s, GPU0 at971.443s, and both emitted concurrent RUNNING
  heartbeats. Their test times were1060.29s and1058.97s respectively.
  The full launcher trace ends at2094.507s including build/setup/logging.
  These are validation/instrumentation timings, NOT search_complete_seconds or
  an end-to-end performance comparison against CayleyPy.
- Fresh local verification also passed:55 Windows CPU contracts, two query
  report tests, ten Python launcher/evidence tests, and Rust formatting.

Artifacts: test_results/native-query-v10/native-primitive-gate/ and its sibling
mgbfs-native-matrix-primitives-t4.log. Verification:

```text
python scripts/verify_primitive_gate.py test_results/native-query-v10/native-primitive-gate --source 77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5 --require-route-query
```

Result: VERIFIED_PRIMITIVE_GATE, combinations80. This closes the hardware
regression for the three allocation-query providers, not the entire architecture.
Bounded sharded owner, the complete rank allocation provider, native transport/
scheduler, production archive integration and full production A/B remain pending.
