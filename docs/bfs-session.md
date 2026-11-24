# MultiGPUBFS session API

The same session accepts a `GraphDefinition`, a schema-2 JSON file or JSON object, and compatible CayleyPy graph definitions and graphs. It accepts a public CayleyPy catalog constructor as `{"catalog": "Family.constructor", "args": [], "kwargs": {}}`. Install `multigpubfs[cayleypy]` for that adapter. Traversals require the matching Linux CUDA native runtime. Path output requires PyArrow. Use `key-info` to see which forward and bidirectional path modes the native executable supports. Path queries that use the Torch fallback require `multigpubfs[search]` with CUDA-enabled PyTorch. The cluster Torch backend requires PyTorch NCCL on every node.

```python
from multigpubfs import open_session

with open_session(gpus=8) as bfs:
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

Omit `gpus` to use all visible devices, or pass `devices=[0, 3]` for explicit local CUDA ordinals. Traversals write exact counts for completed distance layers and bounded samples of terminal states. `layers` contains the retained layers with the full expected row count. `layer_samples` also contains partial samples. `solve()` writes `graph.json`, a receipt and, when paths were committed, `paths.parquet`. Output paths are unique unless supplied explicitly. `session.bfs()` is an alias for `session.traverse()`. Defaults go to the user state directory outside the source checkout. Pass `output_root`, set `MGBFS_ARTIFACT_ROOT`, or give a query an explicit `output` path.

`nearest` returns one shortest path to the closest reachable target. `window` emits each distinct sequence of original generator IDs through length `d_min + extra_steps` whose final state is a target. Duplicate generators remain distinct labels. Walks may revisit states and continue through a target. The empty sequence is included when the start is a target. Target indexes refer to the deduplicated target list in the receipt. `INCOMPLETE` means the configured time, output, or memory limit stopped the job. Use `completed_through_length` to identify which lengths are exhaustive. An `UNREACHABLE` result certifies that no target was reached. Bidirectional search needs exact inverse edges. Directed permutation generators provide those edges through inverse permutations. Matrix actions need a supported common domain and invertible generators for the backward search. Forward search also accepts non-invertible actions.

```bash
python -m multigpubfs run graph.json results/traversal --gpus 8 --preset balanced
python -m multigpubfs run graph.json results/paths --gpus 8 \
  --targets '[[1,0,2],[2,1,0]]' --direction bidirectional --goal-mode window --extra-steps 2
```

## Memory, tuning and repeated work

Local traversal uses native segmented storage and live GPU admission. `balanced` uses bounded same-graph pilots when a compatible cached profile is unavailable. `maxmemory` considers compact layouts and may select a lossless permutation bitmap when its domain fits the budget. The selector applies its 4× gate using balanced pilots on the same backend and records the interval in `autotune.capacity_slowdown`. The session exposes this pilot result as `same_backend_pilot_slowdown_status`. The bound between the independently selected session presets remains `UNVERIFIED` pending a matched comparison. `tune="off"` skips performance pilots. `tune="full"` allows the larger existing startup search.

Native traversal ranks stay open across calls and reuse communicators and compatible allocations. Changing the graph invalidates input-specific data and changing layouts may require new allocations. The CUDA path engine reuses PyTorch's allocation cache across jobs. Its `maxmemory` preset lowers candidate workspace pressure. Path enumeration can still be limited by retained lineage or output size, and the 4× traversal policy does not certify path performance. Switching engines releases idle allocations when necessary for the incoming engine.

To reproduce one geometry, pass `native_options` to `traverse()` or `--native-options` with a JSON object to `multigpubfs run`. Explicit options skip the throughput pilots. `capacity` is a hard per-rank bound on the next segmented layer for banked layouts, or on all visited states for a one-bank layout. `segment_rows` bounds the segment extent. `queue_capacity` and `owner_lanes` set requested caps for per-queue rows and active owner streams per rank. Live admission can use a smaller geometry, recorded in the run receipt.

On a source build with compact exchange, `MGBFS_SEGMENT_PACKET_RING_SLOTS=3` enables three shared transmit slots per rank. A separate packing stream prepares each slot. NCCL waits for its pack event, and local NCCL completion releases the slot.

With three transmit slots, `MGBFS_SEGMENT_PACKET_RX_SLOTS=3` also enables three shared receive slots and a separate unpacking stream. Each shard has two independent input leases into the shared buffers. The receive stream can fill one lease while an owner stream processes the other. A receive event releases the NCCL endpoint. The packet and its receive credit stay live until every referenced shard has finished, including spill replay. The packet receipt records allocation, peak occupancy and owner overlap. The default is one transmit and one receive slot.

```python
result = bfs.traverse("graph.json", preset="maxmemory", native_options={
    "memory_backend": "permutation_bitmap", "segment_strategy": "atomic",
    "bitmap_pack4": True, "shards": 8, "batch": 8192,
    "queue_capacity": 32768, "owner_lanes": 4,
})
```

Build the CUDA library with `-DMGBFS_EXPERIMENTAL_PACK4=ON` for Pack4. It supports atomic permutation bitmap with up to 16 positions at four bits per position. With the adaptive ABI, a smaller alphabet allows up to 64 positions for two symbols, 32 for up to four, or 21 for up to eight. When the binary advertises `graph_path_nearest_resident_v1`, the session keeps a dedicated local worker for forward `nearest` and reuses compatible device allocations. It releases the idle arena when switching to another engine.

Sessions select the native rank engine when the executable advertises the requested path mode. Set `MGBFS_NATIVE_DISTRIBUTED_NEAREST=0` to use the Torch implementation for distributed nearest, bidirectional queries, and window queries. Both presets use the same exact route engine. The `maxmemory` route preset caps the native batch at 8,192 source rows and reports `UNVERIFIED` for the capacity and slowdown comparison. The executable advertises its supported modes through `key-info`, with separate capabilities for distributed forward and bidirectional `nearest` and `window`. Bidirectional matrix searches retain goals outside the verified reversible domain in the result with their original indexes in the deduplicated target list. Those goals are unreachable from the validated start. Window queries prove the shortest distance, build rank-local CSR files and stream routes to Parquet. The cluster coordinator requests indexed blocks from the owner ranks as it reads the graph. The native modes have run on two RTX 3060 GPUs on one host. Multi-node acceptance is pending.

The local cache can reuse matching measurements.

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

The launcher derives three ports from `MASTER_PORT` for the session, native control, and path collectives. Override them with `MGBFS_SESSION_PORT`, `MGBFS_CONTROL_PORT`, and `MGBFS_PATH_PORT` when required by the allocation. `MGBFS_CONTROL_TOKEN` can be passed via the scheduler environment. Rank-local directory paths need not match. Only rank 0 publishes output. Native segmented traversal across hosts requires a newly built matching executable advertising `segmented_network_agreement_v1`. Older runtimes can use `balanced` through the existing external legacy backend. The segmented path weights ownership by admitted per-rank rows after transport setup. Cluster `maxmemory` applies compact positions and smaller queues, with slowdown still unverified.

Every external rank of a thin-wheel installation selects the package's pinned universal CUDA library. Each rank checks support for its visible devices. For offline jobs, run `python -m multigpubfs prefetch --sm universal --cache /path/to/cache` in advance and set `MGBFS_NATIVE_CACHE` and `MGBFS_NATIVE_OFFLINE=1` on every rank. Use the same installed package everywhere. An explicit executable or a wheel with a bundled native runtime uses the supplied build, whose native binaries must also match across ranks and support every participating GPU.

With a compatible packet transport ABI on every rank, the cluster can select compact transport for `maxmemory`. Explicit transport options can override the preset. The packet port defaults to `MASTER_PORT + 4` and can be set via `MGBFS_SEGMENT_PACKET_PORT`. Use `MGBFS_SEGMENT_PACKET_CONTROL_ADDR` to supply a numeric address reachable from every rank, and `MGBFS_SEGMENT_PACKET_BIND_ADDR` to override the rank 0 listen address. To compare with `exact_exchange`, set `MGBFS_SEGMENT_EXACT_EXCHANGE=1` without an explicit packet address.

Distributed path queries shard states by exact owners and reuse the NCCL group across consecutive requests. The Torch backend builds the disk CSR on rank 0 from bounded metadata chunks. Native ranks keep their own CSR files available during enumeration. Set `lineage_dir` for coordinator scratch storage, `max_lineage_bytes` to cap indexes and temporary files together, and `lineage_cache_bytes` to cap cached file blocks. The receipt records the committed CSR size, peak scratch bytes, peak cached payload and the last fully written length. Native cluster receipts also report remote read requests and payload bytes. On interruption the receipt tracks the verified shortest-path witness, the committed graph depth, and the last fully written length separately. The cache bound covers file blocks and the total process RAM also includes search, sorting, and result writing. Physical multi-host acceptance and 8-GPU performance of this session API still require CUDA hardware runs.

```bash
python -m scripts.benchmark_segmented_presets --n 13 --devices 0,1,2,3,4,5,6,7 \
  --repeat 2 --output /workspace/s13-presets
python -m scripts.benchmark_segmented_presets --n 13 --devices 0,1,2,3,4,5,6,7 \
  --packet-control-addr 127.0.0.1:29761 --output /workspace/s13-transport
```

`--pressure-mib` keeps an initialized CUDA allocation on each selected GPU throughout both legs. `--segment-budget-bytes` caps only the segmented storage. Full S13 and depths 34–44 have separate verdicts. Use `--max-depth 44` for the fixed heavy window only. The benchmark writes native rank receipts and checks exact terminal state sets on exhausted runs. The benchmark exercises explicitly selected recipes. Its verdicts do not change public preset selection.

`python -m scripts.benchmark_path_lineage_pressure --help` lists the window pressure test. It uses adjacent swaps and a strictly increasing counter. Its independent oracle checks the CSR byte range and the exact route output. The acceptance requires the committed CSR to be at least twice the declared rank 0 RSS budget and the measured rank 0 peak bound to stay within that budget. Run the script through `multigpubfs cluster` to exercise native reads from owner ranks. Put the lineage directory on the GPU host's scratch disk.
