# Unified graph launch (current implementation)

```python
from multigpubfs import run_graph
report = run_graph(cayleypy_graph, "results")
```

The default uses all CUDA-visible devices on one host. `device=1` selects one
card; `devices=[1,0]` sets explicit placement. `max_seconds` limits work, and
`capacity` overrides rows per card subject to VRAM admission: per-layer rows for proved inverse-closed graphs, cumulative retained rows for directed graphs.
`shards=4` selects independent owner streams. When `shards` is omitted, substantial
workloads measure four bounded same-graph GPU profiles (1 shard, 4 shards, 16 shards,
and 4 shards with quarter-size routing batches). Comparable completed depths
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

Specialized SHARD_AB remains available through its existing native configuration interface. The general automatic path does not claim to select that specialized kernel for every graph.

Memory admission reads actual free VRAM, reserves headroom, and accounts for
the selected history layout, one shared shard-overflow table, row-position maps, routing banks, receive buffers, and control/owner metadata. A finite
mathematical state-space bound avoids VRAM-sized allocations for small graphs.
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
graph against a CPU oracle. Heavy-graph and larger-rank tuning acceptance remain.


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
