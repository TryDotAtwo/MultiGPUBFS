# Unified graph launch

Current Linux package and immutable installation: [install-native-package.md](install-native-package.md). General permutation/int64-matrix actions use full-state exact equality. Specialized SHARD_AB selection is limited to proved lossless LRX packed-key domains, with a native capability check; other graphs stay on the general exact backend. See [lossless-shard-keys.md](lossless-shard-keys.md).


```python
from multigpubfs import run_graph
report = run_graph(cayleypy_graph, "results")
```

A constructor can also be selected by its exact public catalog name. Install the optional `cayleypy` extra (pinned to the tested CayleyPy0.2.0 API), then use:

```python
from multigpubfs import from_catalog, run_graph
report = run_graph(from_catalog("PermutationGroups.lrx", kwargs={"n": 5}), "results")
```

```bash
multigpubfs PermutationGroups.lrx results --catalog --catalog-kwargs '{"n": 5}'
```

Allowed families are `PermutationGroups`, `MatrixGroups` and `Puzzles`; private or arbitrary dotted names are rejected. Positional arguments use `--catalog-args` as a JSON list. Keyword arguments use a JSON object, so parameters requiring non-JSON objects should be supplied through the Python API or an exact serialized `GraphDefinition`. No user source is evaluated. Constructor selection is startup work and does not enter GPU generation/dedup.

The default uses all CUDA-visible devices on one host. `device=1` selects one
card; `devices=[1,0]` sets explicit placement. `max_seconds` limits work, and
`capacity` overrides rows per card subject to VRAM admission: per-layer rows for proved inverse-closed graphs, cumulative retained rows for directed graphs.
`shards=4` selects independent owner streams. When `shards` is omitted, substantial
workloads measure admitted same-graph GPU profiles:1/4/16 owner shards, quarter-size routing batches, optional parent-origin transport and radix origin ordering. Eligible LRX actions additionally compare SHARD_AB HASH/SORT_MERGE and verified peer transports. Comparable completed depths
and identical layer counts are required; first-collective warmup is excluded.
An alternative must beat the baseline by at least 5% to justify a switch.
`autotune=False` selects the conservative one-shard profile without pilots.
Small/short workloads skip pilots. Profiles are cached by graph, native binary,
GPU identity, driver, topology and relevant communication environment; actual
free-VRAM admission is repeated on every launch. Tuning time is deducted from
the requested work budget. This bounded prefix selection is not a guarantee
of a globally optimal profile across a growing frontier. State generation, routing, equality and dedup run on GPUs.

The general backend retains complete visited history for directed graphs.
Compact output saves at most 1000 states of the last completed layer globally,
and the previous completed layer only when it contains fewer than 1000 states.
The launcher automatically uses three immutable VRAM banks for a proved inverse-closed action: previous, current and future. It retires older keys using a row-to-slot map, preserving probe chains with tombstones; occasional GPU maintenance amortizes retired keys rather than rebuilding history every layer. Generation reads the current bank directly, without a full parent gather. Only the future bank may be partial on a resource stop. The readout staging buffer is capped at 1000 rows. Directed actions and modular matrices whose wrapping arithmetic invalidates an inverse proof retain all visited history. Reports expose `history_layers` (3 or 1) and the actual backend.

Losslessly matched LRX definitions can enter specialized SHARD_AB profile selection through this same API. Other action/root definitions retain the exact general backend.

Memory admission reads actual free VRAM, reserves headroom, and accounts for
the selected history layout, one shared shard-overflow table, row-position maps, routing banks, receive buffers, and control/owner metadata. A finite
mathematical state-space bound avoids VRAM-sized allocations for small graphs. Cyclic permutation actions with inverse/identity/duplicate aliases use exact label periods of the start vector; other permutation actions use the conservative multiset bound.
Automatic multi-device placement uses independent admitted capacities and
weighted hash intervals; explicit capacity overrides retain common capacity.

`COMPLETE` means frontier exhaustion. Deadline, cancellation or resource stop
produce `INCOMPLETE`, never a full enumeration claim. Signals are voted across
ranks at layer boundaries. Errors or peer failure terminate the launch without
restarting BFS. Reports contain graph identity and state checksums.

Verified hardware: one and two RTX 3060 on one host. Logical 8/128-rank memory
geometry tests do not establish hardware acceptance or throughput scaling.
External-rank network launch is connected through WORLD_SIZE/RANK/LOCAL_RANK
and MASTER_ADDR/MASTER_PORT. Its actual TCP/NCCL protocol was checked using two
independent worker directories on one host; physically separate nodes remain
unverified. See external-rank-launch.md.
A Linux x86_64 native wheel was clean-installed and verified outside its source
checkout on one and two RTX 3060, including four-shard placement. Its manifest
verifies executable and CUDA-library SHA256 before launch. The wheel contains
SM86 code and PTX build targets and still requires the compatible external CUDA/NCCL/driver versions stated in its own manifest (published CUDA12 and CUDA13 variants exist).
Bounded GPU profile selection and cache were verified on a complete 40,320-state
graph against a CPU oracle. A complete3.63M-state graph also passed four-profile selection, cache reuse and bound CPU-reference layer/terminal checks. Larger-rank tuning acceptance remains.


Permutation graphs whose start values are all in 0..255 automatically use byte
SoA state storage throughout the retained arena, parent banks and transport
payloads. Permutations preserve this alphabet exactly. The kernels share the
same full equality and value-based hash logic as the int64 path; matrix actions,
negative values and values above 255 keep signed int64 storage. Reports expose
`state_bytes` (1 or 8), and memory admission accounts for that actual width.
This reduces state payload bytes eightfold, not all allocations or total runtime
by eightfold: indices, hash tables and 32-byte route records are unchanged.
The general transport still pads fixed-capacity queues; key-first survivor-only
transport remains a separate performance boundary. Compact history dispatch is automatic for proved inverse-closed actions.

Byte/int64 equivalence was checked on actual one/two RTX 3060, one/four shards,
forced hash collisions, values at 255, repeated elements, and invalid-future
resource stops. Negative/256-valued permutations and modulo-257 matrix actions
were checked through the public launcher and retain the int64 codec.

Three-bank acceptance includes complete CPU-oracle layer/state comparisons for 720/5040-state inverse-closed permutations, a 257-state matrix cycle with capacity 3, wide byte/int64 states, exact resource snapshots, actual one/two-GPU automatic dispatch, cancellation, deadline, and external-rank TCP/NCCL with collective tuning on a complete 40,320-state graph. The larger logical 8/128-rank admission checks are CPU geometry evidence, not physical scaling.


### Wide exact parent-origin transport

`run_graph(..., transport="auto")` remains the default. For sufficiently large wide permutation/int64 workloads, bounded calibration measures full-child transport against parent-origin transport in addition to shard/batch alternatives. Packed<=24byte permutations do not need a duplicate parent-cache alternative. `transport="full"` or `transport="parent"` fixes that choice for reproducibility; shard/batch tuning still applies if enabled. The environment equivalent is `MGBFS_GENERIC_TRANSPORT=full|parent`.

Parent-origin transport all-gathers one immutable SoA parent chunk per source and routes hashes plus exact origin indices. Owners regenerate values for exact equality and write only surviving states directly into retained/rolling VRAM. Parent caches, origin cursors and frontier counts are preallocated and included in admission; no full child plane, CPU child generation, CPU dedup or new host count readback is introduced. All shard owners retire the cache before its next exchange. This trades repeated child traffic for parent replication and destination generation; it may lose at large rank count or expensive actions, so automatic selection uses same-graph measured prefixes rather than assuming a universal win.


### Optional radix ordering and component admission

`candidate_order="auto"` measures unsorted versus GPU radix-ordered candidate indices; `candidate_order="none"|"radix"` fixes ordering. Environment override: `MGBFS_GENERIC_SORT=none|radix`. Exact hashes, states and parent leases remain immutable. Radix sorts coarse hash keys plus uint32 origin indices, then uses the existing exact hash-table dedup. It is not a sorted-history merge implementation. Scratch is preallocated per owner shard and included in memory admission. The optional alternative increases the bounded profile set to five packed or seven wide profiles; only a measured matched-prefix improvement of at least five percent selects a different profile. A losing alternative stays available for reproducibility and different workloads.

Permutation admission counts value arrangements within generator-connected position components. Fixed coordinates contribute a factor of one. This is a startup mathematical upper bound, not CPU enumeration or a claim that all component arrangements are reachable.


### Measured specialized owner selection

`backend="auto"` also measures eligible SHARD_AB HASH and SORT_MERGE owners for losslessly matched LRX permutation definitions on a single host with 1/2/4/8 selected GPUs. Only physical one/two RTX3060 were accepted. Arbitrary int64 labels are restored exactly; other actions and repetition patterns retain the general backend. `backend="generic"` excludes these candidates; `backend="shard_ab_hash"|"shard_ab_sort_merge"` explicitly requests them. Specialized geometry is adaptive and cannot be combined with generic shard/transport/order overrides. External-rank launch retains the general backend. Admission queries actual free VRAM after NCCL warmup. Compact retention streams bounded prefixes through RAM FIFOs, keeps the last two committed layers, and writes only the terminal sample (1000 maximum) plus a previous layer strictly smaller than1000. Resource and deadline stops preserve a validated unprocessed sample. Unknown native errors do not trigger silent fallback or restart.

On the verified two3060 n8r1 acceptance workload, all seven profiles matched through depth28; the generic quarter-batch profile won. Specialized owners are available alternatives, not a claimed universal speedup. The cached second public launch reproduced the complete40320-state graph. LSA capability, multi-host specialized launch and physical4/8/128/Blackwell remain unverified.


The installed package exposes `multigpubfs graph.json new-results --devices 0,1 --seconds 120`, or `python -m multigpubfs` with the same arguments. Input is the validated schema2 definition produced by `GraphDefinition.to_json()`. Omit devices to use all visible GPUs. The result JSON and checksummed compact state receipt are stored in the new output directory; COMPLETE versus INCOMPLETE is retained explicitly. Explicit backend, capacity, transport and candidate ordering are optional. This entrypoint does not enumerate graph states on CPU.


### Peer transport capability and selection

`peer_transport="auto"` (CLI `--peer-transport auto`) observes a pure native LSA build flag. An enabled library must pass a bounded fixed exact24-state GPU graph with known layer and terminal-state receipts before LSA enters performance profiles. AUTO retains HOST as an alternative and scores transports on identical graph prefixes. Compiled-off or the exact NCCL unsupported-device/team error selects HOST; timeouts, unknown native errors and oracle failures are errors rather than silent fallback. `peer_transport="host"` skips the LSA gate. Explicit `peer_transport="lsa"` requires a specialized backend and fails if capability cannot be verified. External ranks retain the general backend; explicit LSA is unsupported there. The cached winning transport is retained and LSA capability is rechecked before use.

Both compiled-off and enabled-but-unsupported fallback paths passed complete exact HASH/SORT_MERGE graphs on twoRTX3060; explicit unsupported LSA rejected. The current hardware cannot validate successful LSA data-plane/throughput. CPU policy tests cover adding and choosing LSA profiles after a verified capability; they are not GPU LSA acceptance. No capability queries or CPU successor enumeration run inside the BFS hot loop.

## External-rank planning checkpoint, 2026-10-10

Collective tuning considers HASH and SORTED_RUNS with1/2/4/8 owner lanes, respects explicitly forced history/order, compares common completed prefixes and passes its selected geometry to production. Sorted allocation queries execute on each rank's own local GPU through graph-local-plan. The shared batch is negotiated before final per-rank allocation; automatic capacities define weighted owner hash cuts. Local device IDs may repeat across hosts/process visibility domains.

Actual tests cover two RTX3060 on one host, TCP/NCCL and independent directories, including one visible GPU per process as local device0. Calculated128-rank contracts do not establish physical128-GPU or multi-host performance. No globally optimal profile or arbitrary-hardware throughput guarantee is claimed.
