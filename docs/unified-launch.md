# Unified graph launch (current implementation)

```python
from multigpubfs import run_graph
report = run_graph(cayleypy_graph, "results")
```

The default uses all CUDA-visible devices on one host. `device=1` selects one
card; `devices=[1,0]` sets explicit placement. `max_seconds` limits work, and
`capacity` overrides retained rows per card subject to VRAM admission.
`shards=4` selects independent owner streams. When `shards` is omitted, substantial
workloads measure three bounded same-graph GPU profiles (1 shard, 4 shards,
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
The default graph path is general exact retained-history; specialized compact
SHARD_AB remains available through its existing native configuration interface.
Automatic fast-path selection is not yet complete.

Memory admission reads actual free VRAM, reserves headroom, and accounts for
retained history, per-shard tables, routing banks and receive buffers. A finite
mathematical state-space bound avoids VRAM-sized allocations for small graphs.
Current multi-device placement uses a common capacity based on the smallest
available card; heterogeneous weighted capacity tuning remains pending.

`COMPLETE` means frontier exhaustion. Deadline, cancellation or resource stop
produce `INCOMPLETE`, never a full enumeration claim. Signals are voted across
ranks at layer boundaries. Errors or peer failure terminate the launch without
restarting BFS. Reports contain graph identity and state checksums.

Verified hardware: one and two RTX 3060 on one host. Logical 8/128-rank memory
geometry tests do not establish hardware acceptance or throughput scaling.
Multi-host launch and external torchrun dispatch are not connected to this API.
A Linux x86_64 native wheel was clean-installed and verified outside its source
checkout on one and two RTX 3060, including four-shard placement. Its manifest
verifies executable and CUDA-library SHA256 before launch. The wheel contains
SM86 code only and still requires compatible external CUDA12/NCCL2/driver.
Bounded GPU profile selection and cache were verified on a complete 40,320-state
graph against a CPU oracle. Heavy-graph and larger-rank tuning acceptance remain.
