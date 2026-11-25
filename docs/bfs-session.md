# MultiGPUBFS session API

The same session accepts a `GraphDefinition`, a schema-2 JSON file or JSON object, and compatible CayleyPy graph definitions and graphs. It accepts a public CayleyPy catalog constructor as `{"catalog": "Family.constructor", "args": [], "kwargs": {}}`. Install `multigpubfs[cayleypy]` for that adapter. Traversals require the matching Linux CUDA native runtime. Path output requires PyArrow. Use `key-info` to see which forward and bidirectional path modes the native executable supports. Path queries that use the Torch fallback require `multigpubfs[search]` with CUDA-enabled PyTorch. The cluster Torch backend requires PyTorch NCCL on every node.

```python
from multigpubfs import open_session

with open_session(gpus=2) as bfs:
    full = bfs.traverse("graph.json", start=[0, 1, 2], preset="balanced")
    nearest = bfs.solve(
        "graph.json", start=[0, 1, 2], targets=[[1, 0, 2], [2, 1, 0]],
        direction="bidirectional", preset="balanced",
    )
    paths = bfs.solve(
        "graph.json", start=[0, 1, 2], targets=[[1, 0, 2], [2, 1, 0]],
        direction="forward", goal_mode="window", extra_steps=2, preset="maxmemory",
        max_paths=100_000, max_output_bytes=1 << 30,
    )

print(full.layer_sizes, full.bfs_completed)
for path in paths.iter_paths():
    print(path["target_index"], path["generator_ids"])
```

Call `paths.verify_paths(cayley_graph)` to replay each saved word with the original CayleyPy graph and compare the endpoint with the reported target. Verification consumes Parquet in batches and returns the number of saved paths checked.

Omit `gpus` to use all visible devices, or pass `devices=[0, 3]` for explicit local CUDA ordinals. Traversals write exact counts for completed distance layers and bounded samples of terminal states. `layers` contains the retained layers with the full expected row count. `layer_samples` also contains partial samples. `solve()` writes `graph.json`, a receipt and, when paths were committed, `paths.parquet`. Output paths are unique unless supplied explicitly. `session.bfs()` is an alias for `session.traverse()`. Defaults go to the user state directory outside the source checkout. Pass `output_root`, set `MGBFS_ARTIFACT_ROOT`, or give a query an explicit `output` path.

`nearest` returns one shortest path to the closest reachable target. `window` emits each distinct sequence of original generator IDs through length `d_min + extra_steps` whose final state is a target. Duplicate generators remain distinct labels. Walks may revisit states and continue through a target. The empty sequence is included when the start is a target. Target indexes refer to the deduplicated target list in the receipt. `INCOMPLETE` means the configured time, output, or memory limit stopped the job. Use `completed_through_length` to identify which lengths are exhaustive. An `UNREACHABLE` result certifies that no target was reached. Bidirectional search needs exact inverse edges. Directed permutation generators provide those edges through inverse permutations. Matrix actions need a supported common domain and invertible generators for the backward search. Forward search also accepts non-invertible actions.

```bash
python -m multigpubfs run graph.json results/traversal --gpus 2 --preset balanced
python -m multigpubfs run graph.json results/paths --gpus 2 \
  --targets '[[1,0,2],[2,1,0]]' --direction bidirectional --goal-mode window --extra-steps 2
```

## Memory, tuning and repeated work

Traversal uses native segmented storage and live GPU admission. `balanced` uses bounded same-graph pilots when a compatible cached profile is unavailable. `maxmemory` considers compact layouts and may select a lossless permutation bitmap when its domain fits the budget. The selector records its same-backend 4× pilot result in `autotune.capacity_slowdown` and exposes it as `same_backend_pilot_slowdown_status`. A measured speed ratio applies to the configurations and graph range in its receipt. The independently selected presets can use different backends, so a same-backend pilot leaves their comparison unverified. `tune="off"` skips performance pilots. `tune="full"` allows the larger existing startup search.

The `tune="auto"` selector shares five seconds across optional catalog lookup, profile admission and up to three pilots. Required native startup and worker teardown have their own deadlines. On a cluster catalog miss, the coordinator can try smaller source batches on the admitted segmented backend and transport. Trials run in separate native processes using each rank's physical free VRAM and cold reserve. The selector compares the same committed BFS prefix and admits any selected batch again for the resident workers. If there are fewer than two comparable prefixes, the launch keeps the admitted default geometry. Every trial process must exit before the next trial or production can start. The trial receipt measures a cold batch comparison for its reported prefix. After a complete run confirms that prefix and the measured batch geometry, the same cluster session can reuse the selection. It runs fresh admission against current memory and reports `REUSED_MEASURED_EXTERNAL_BATCH_PROFILE` when it skips the pilots.

On a single host with two GPUs, large adjacent-transposition permutation workloads using three-bank atomic Pack4 and exact exchange start with eight shards and no more than 98,304 candidate states per producer tile. The runtime still admits the batch and shard geometry against available VRAM. The receipt records this choice under `startup_geometry`. The starting geometry comes from the physical S13 comparison and leaves measured Registry recipes, explicit geometry and full tuning in control.

Native ranks stay open across calls and reuse communicators and compatible allocations. Changing the graph invalidates input-specific data and changing layouts may require new allocations. Switching engines releases idle allocations when necessary. A controlled resource or time stop can preserve the completed prefix and allow a subsequent request. Losing a rank closes the generation.

To reproduce one geometry, pass `native_options` to `traverse()` or `--native-options` with a JSON object to `multigpubfs run`. Explicit options skip the throughput pilots. `capacity` is a hard per-rank bound on the next segmented layer for banked layouts, or on all visited states for a one-bank layout. `segment_rows` bounds the segment extent. `queue_capacity` and `owner_lanes` set requested caps for per-queue rows and active owner streams per rank. Live admission can use a smaller geometry, recorded in the run receipt.

Use `native_options={"transport": "compact", "packet_slots": 3}` to select three transmit and three receive slots per rank. Use `"packet_slots": 1` for one each or `native_options={"transport": "exact"}` for exact exchange. The same JSON works with the CLI's `--native-options` locally and on a cluster.

With three slots, a separate stream packs outgoing data. NCCL waits for the pack event, and local NCCL completion releases the transmit slot. Each receiving shard has two independent input leases into the shared buffers. The unpacking stream can fill one lease while an owner stream processes the other. A receive event releases the NCCL endpoint. The packet and its receive credit stay live until every referenced shard has finished, including spill replay. The packet receipt records allocation, peak occupancy and owner overlap. Direct compact traversal defaults to one transmit and one receive slot. The segmented route launch uses three of each.

For lower-level launches, `MGBFS_SEGMENT_PACKET_RING_SLOTS` and `MGBFS_SEGMENT_PACKET_RX_SLOTS` set the slots separately.

`MGBFS_SEGMENT_PACKET_ROUTE_LOOKAHEAD=1` admits two producer slots. When both pinned metadata buffers are available, the next tile and its readback are queued before the CPU consumes the current tile. The packet receipt reports this path as `route_metadata_lookahead`.

```python
result = bfs.traverse("graph.json", preset="maxmemory", native_options={
    "memory_backend": "permutation_bitmap", "segment_strategy": "atomic",
    "bitmap_pack4": True, "shards": 8, "batch": 8192,
    "queue_capacity": 32768, "owner_lanes": 4,
})
```

For a standalone CUDA build, enable Pack4 with `-DMGBFS_EXPERIMENTAL_PACK4=ON`. The fresh bundled Linux wheel workflow enables it and checks the installed ABI. Pack4 supports atomic permutation bitmap with up to 16 positions at four bits per position. With the adaptive ABI, a smaller alphabet allows up to 64 positions for two symbols, 32 for up to four, or 21 for up to eight. When the binary advertises `graph_path_nearest_resident_v1`, the session keeps a dedicated local worker for forward `nearest` and reuses compatible device allocations. It releases the idle arena when switching to another engine.

The executable advertises separate capabilities for forward and bidirectional `nearest` and `window` through `key-info`. For a supported multi-GPU build, the segmented route engine uses compact packet exchange and exact owner indexes. Explicit `MGBFS_SEGMENT_NEAREST_V1=1` and `MGBFS_SEGMENT_WINDOW_V1=1` require those paths. Set `MGBFS_NATIVE_DISTRIBUTED_NEAREST=0` to use the Torch implementation for distributed nearest, bidirectional queries, and window queries. Both presets use the same exact route engine. The `maxmemory` route preset caps the native batch at 8,192 source rows. The capacity and slowdown comparison for paths is `UNVERIFIED`.

## Profile Registry and Atlas

Pass `profile_registry` and `profile_registry_sha256` to `open_session()`, or use `--profile-registry` and `--profile-registry-sha256` with `multigpubfs run`. A GitHub commit URL or release asset requires a SHA-256 pin. Private GitHub release URLs use the local authenticated `gh` session when anonymous access is denied. The same pin reuses the verified cached bytes in offline mode. Set `MGBFS_PROFILE_REGISTRY_OFFLINE=1` to disable downloads. Explicit native options take precedence.

Catalog recipes match an exact graph, the ordered devices and topology, driver versions, and the native executable and CUDA library hashes. `balanced` tries measured recipes from both presets in speed order. Live admission can reject a saved recipe when the available memory or planned geometry changes. The receipt records the requested preset, the preset used to replay the selected recipe, and whether the runtime used the measured geometry.

Use `multigpubfs profiles export --help` to add an independently verified pair of full runs to a catalog. For runs whose normal state samples truncate a terminal layer, set `MGBFS_PROFILE_TERMINAL_EVIDENCE_DEPTH` to the expected final depth for both arms. A matching native build captures up to 768 KiB of complete terminal evidence, bound to the report by SHA-256. For audited publication, run `multigpubfs profiles publish-github --help`. It checks the evidence and uploads a content-addressed catalog to an existing release, then returns its URL and verified pin.

For a cluster export, pass the coordinator's `launch.json` as `--inventory` and give `--topology` a JSON file with a `rank_metadata` array in global rank order. On each rank, `multigpubfs.profile_selection.cluster_rank_hardware(os.environ, socket.gethostname())` captures the `host`, `topology` from `nvidia-smi topo -m`, `driver` and `network` settings. A `topology_device` field identifies the corresponding `GPU<n>` slot when it differs from the local CUDA ordinal. Ranks sharing a host and local CUDA ordinal require distinct physical slots. Capture both arms with the same rank placement and memory pressure. Export checks the original hostnames and devices against both reports and the ordered hardware against both launch files. The catalog key uses node groups and GPU slots.

From the source checkout, add `--registry-evidence` to the existing preset benchmark to generate a catalog from the same two runs.

```bash
python -m scripts.benchmark_segmented_presets --n 13 --devices 0,1 \
  --registry-evidence --output /path/outside/checkout/s13
```

`pair-1/profile/registry.json` and its `evidence/` directory are written after complete terminal verification, physical VRAM sampling on every rank and a slowdown at or below 4×. The exporter checks that the reduction in sampled peak exceeds the measured baseline drift. Read `pair-1/comparison.json` for the export status and SHA-256 pin. Use `--pressure-mib` for a held allocation. Profiles measured under that pressure are eligible only under the corresponding `MGBFS_PROFILE_PRESSURE_NAMESPACE=cuda_warmed_allocation:<bytes>`.

Install `multigpubfs[atlas]` and call `publish(result, token=token, author="Ivan")` from `multigpubfs`. The returned receipt names the immutable Atlas revision. Set `include_incomplete=True` to publish a partial result. A subsequent upload can advance the proven coverage and preserves the earlier finalized lengths.

## Native request path

1. The session normalizes the graph and query in [api.py](../multigpubfs/api.py). Traversal memory policy is shared in [memory_admission.py](../multigpubfs/memory_admission.py). [launch.py](../multigpubfs/launch.py) drives local allocation and launch. [cluster.py](../multigpubfs/cluster.py) collects external-rank capabilities and memory inventory before distributed admission. Native rank plans bind the graph and effective geometry. The launch checks executable and CUDA library identity across the group.
2. The [segmented runtime](../crates/mgbfs-runtime/src/segmented_distributed_native.rs) partitions the frontier and exact visited state by owner. [Packet exchange](../crates/mgbfs-runtime/src/segmented_distributed_native/packet.rs) carries remote work in bounded slots. [Layer commit](../crates/mgbfs-runtime/src/segmented_distributed_native/layer.rs) publishes completed layers after delivery and owner processing. The [hash index](../crates/mgbfs-runtime/src/segmented_distributed_native/hash_index.rs) bounds the searched series in the sorted segmented hash path and preserves exact state comparison.
3. [Segmented nearest](../crates/mgbfs-runtime/src/segmented_distributed_native/nearest.rs) records one parent for each newly admitted state. Bidirectional search expands complete layers, checks the opposite visited index at the owner, and proves the shortest distance before returning the witness. Bidirectional matrix queries retain goals outside the verified reversible domain in the deduplicated target list. Those goals are unreachable from the validated start.
4. [Segmented Window](../crates/mgbfs-runtime/src/segmented_distributed_native/window.rs) proves the nearest distance, stores layer-scoped states and retains every incoming generator ID in rank-local CSR. [Lineage readback](../crates/mgbfs-runtime/src/segmented_distributed_native/window/lineage.rs) hands completed copies to a bounded [background writer](../crates/mgbfs-runtime/src/path_csr/writer.rs). CUDA can reuse an edge tile after the copy, while the writer retains its host slot through the last CPU read. The layer becomes available after writing and merging finish. The cluster enumerator streams Parquet and fetches the CSR blocks it needs from [owner ranks](../multigpubfs/path_window_files.py). The receipt distinguishes the proven distance, committed lineage depth and last fully written result length.

## Existing GPU allocations

Put the Python code above in `pipeline.py`. Use a shared checkout or the same installation on all ranks. Set `gpus` to the total allocated rank count, or omit it. The coordinator executes the script on rank 0 and dispatches the sequential calls to active workers.

```bash
srun --ntasks="$SLURM_NTASKS" --gpus-per-task=1 python -m multigpubfs cluster pipeline.py
```

With `torchrun`, invoke the following command on each allocated node with the appropriate node ordinal and the same master address. `GPUS_PER_NODE` is the number of local ranks that torchrun starts.

```bash
torchrun --nnodes="$NODES" --node-rank="$NODE_RANK" --nproc-per-node="$GPUS_PER_NODE" \
  --master-addr="$MASTER_ADDR" --master-port=29500 -m multigpubfs cluster pipeline.py
```

The launcher derives four ports from `MASTER_PORT` for the session, native control, path collectives and packet control. Override them with `MGBFS_SESSION_PORT`, `MGBFS_CONTROL_PORT`, `MGBFS_PATH_PORT` and `MGBFS_SEGMENT_PACKET_PORT` when required by the allocation. `MGBFS_CONTROL_TOKEN` can be passed via the scheduler environment. Rank-local directory paths need not match. Only rank 0 publishes output. Native segmented traversal across hosts requires a matching executable advertising `segmented_network_agreement_v1`. Older runtimes can use `balanced` through the external legacy backend. The segmented path weights ownership by admitted per-rank rows after transport setup.

Every external rank of a thin-wheel installation selects the package's pinned universal CUDA library. Each rank checks support for its visible devices. For offline jobs, run `python -m multigpubfs prefetch --sm universal --cache /path/to/cache` in advance and set `MGBFS_NATIVE_CACHE` and `MGBFS_NATIVE_OFFLINE=1` on every rank. Use the same installed package everywhere. An explicit executable or a wheel with a bundled native runtime uses the supplied build, whose native binaries must also match across ranks and support every participating GPU.

With a compatible packet transport ABI on every rank, the cluster can select compact transport for `maxmemory`. Explicit transport options can override the preset. The packet port defaults to `MASTER_PORT + 4`. Use `MGBFS_SEGMENT_PACKET_CONTROL_ADDR` to supply a numeric address reachable from every rank, and `MGBFS_SEGMENT_PACKET_BIND_ADDR` to override the rank 0 listen address. To compare with `exact_exchange`, set `MGBFS_SEGMENT_EXACT_EXCHANGE=1` without an explicit packet address.

Set `lineage_dir` for coordinator scratch storage, `max_lineage_bytes` to cap indexes and temporary files together, and `lineage_cache_bytes` to cap cached file blocks. The receipt records the committed CSR size, peak scratch bytes, peak cached payload and the last fully written length. Native cluster receipts also report remote read requests and payload bytes. The total process RAM includes cached file blocks, search, sorting, and result writing. The Torch fallback builds the disk CSR on rank 0 from bounded metadata chunks.

## Physical acceptance

- **Exact routes on one host, two RTX 3060 GPUs.** Forward and bidirectional nearest and Window passed on the segmented engine. Each Window mode matched all 364 CPU-oracle words with four owner shards. The pressure run committed 3.37 GB of CSR with a 799 MiB conservative rank-0 RSS bound and 3.01 MB of remote block reads. The result kept full coverage through length 25 after stopping partway through length 26. A separate injected loss of rank 1 during Window lineage closed the generation and the next query was rejected. [Nearest](https://github.com/TryDotAtwo/MultiGPUBFS-archive/tree/52048f4/docs/acceptance/2026-11-25-segmented-nearest-public) · [Window and rank loss](https://github.com/TryDotAtwo/MultiGPUBFS-archive/tree/1ed6651/docs/acceptance/2026-11-25-segmented-window)
- **S13 on one host, two RTX 3060 GPUs.** Both Pack4 recipes completed under the same imposed memory pressure. Compared with the corrected explicit `balanced` geometry, the measured `maxmemory` run used 1,458 MiB less peak VRAM per GPU and took 3.20× as long overall and 3.16× on depths 34–44. The original automatic `balanced` run stopped at depth 42. This supports the measured recipes and ranges. Eight-GPU and physical multi-host acceptance of the current session build remain open. [Public preset pair](https://github.com/TryDotAtwo/MultiGPUBFS-archive/tree/c0c3fa9/docs/acceptance/2026-11-25-s13-public-presets) · [Corrected geometry comparison](https://github.com/TryDotAtwo/MultiGPUBFS-archive/tree/f2c3304/docs/acceptance/2026-11-25-s13-balanced-geometry)
- **Automatic balanced S13 on two T4 GPUs.** A cold public request selected eight shards, four owner lanes per rank and Pack4. It completed all 79 exact layers and both full terminal sets in 1,204 s of native BFS. Device-wide VRAM peaked at 7,539 MiB per GPU including a 2 GiB held allocation. [Automatic startup acceptance](https://github.com/TryDotAtwo/MultiGPUBFS-archive/tree/bc0ef76/docs/acceptance/2026-11-25-s13-automatic-kaggle)

## Reproduce the checks

```bash
python -m scripts.benchmark_segmented_presets --n 13 --devices 0,1 \
  --output /workspace/s13-presets
python -m scripts.benchmark_segmented_presets --n 12 --devices 0,1 \
  --packet-control-addr 127.0.0.1:29761 --output /workspace/s12-transport
```

`--pressure-mib` keeps an initialized CUDA allocation on each selected GPU throughout both legs. `--segment-budget-bytes` caps only the segmented storage. Full S13 and depths 34–44 have separate verdicts. Use `--max-depth 44` for the fixed heavy window only. The benchmark writes native rank receipts and checks exact terminal state sets on exhausted runs. The benchmark exercises explicitly selected recipes. Its verdicts do not change public preset selection.

`python -m scripts.benchmark_path_lineage_pressure --help` lists the window pressure test. It uses adjacent swaps and a strictly increasing counter. Its independent oracle checks the CSR byte range and the exact route output. The acceptance requires the committed CSR to be at least twice the declared rank 0 RSS budget and the measured rank 0 peak bound to stay within that budget. Run the script through `multigpubfs cluster` to exercise native reads from owner ranks. Put the lineage directory on the GPU host's scratch disk.
