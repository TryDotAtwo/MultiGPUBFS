# Unified graph launch (current implementation)

```python
from multigpubfs import run_graph
report = run_graph(cayleypy_graph, "results")
```

The default uses all CUDA-visible devices on one host. `device=1` selects one
card; `devices=[1,0]` sets explicit placement. `max_seconds` limits work, and
`capacity` overrides retained rows per card subject to VRAM admission.
`shards=4` selects independent owner streams; automatic throughput tuning is
not yet connected. State generation, routing, equality and dedup run on GPUs.

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
Native clean installation and throughput tuning remain acceptance work.
