# BFS scaling regimes: latency, throughput, and capacity

“BFS scales to more GPUs” is not one claim. The same system can reduce the
latency of one fixed traversal poorly, process many independent traversals well,
and solve a larger state space only because aggregate memory increased. These
are different experimental objects.

This note defines the regimes needed for later measurement. It does not select
an implementation or predict an optimal GPU count.

## 1. Fixed-work latency and strong scaling

Choose one immutable workload contract:

```text
same graph/action version,
same source or source set,
same direction and stopping rule,
same output and exactness contract,
same representation semantics.
```

Let `T_p` be end-to-end elapsed time on `p` GPUs. Fixed-work speedup and
parallel efficiency are

```text
S_p = T_1 / T_p,
E_p = S_p / p.
```

This is strong scaling only if the semantic work remains the same. The
per-level frontier and successor counts should agree across `p`; retry,
routing, and synchronization work may change and must be reported separately.

Latency reduction is the direct outcome: how long one declared BFS query takes.
Strong-scaling ratios are one way to describe that reduction relative to a
baseline. If the one-GPU workload does not fit, `T_1` does not exist and a
one-to-`p` speedup must not be invented.

## 2. Capacity scaling

Capacity scaling asks whether more aggregate memory permits a larger exact
visited set, frontier, candidate peak, parent structure, or graph. Success can
mean that a workload previously impossible now completes correctly.

Useful evidence includes:

```text
maximum exact ball/frontier reached,
semantic state count and record widths,
per-GPU resident and scratch peaks,
replication and allocator overhead,
overflow/spill outcome,
validation of the completed result.
```

Aggregate nominal memory is not usable BFS capacity. Replicated tables,
communication buffers, imbalance, temporary sort/compaction space, allocator
headroom, and the maximum-loaded owner reduce it. Capacity scaling can succeed
even when elapsed time grows.

## 3. Throughput scaling over independent queries

Suppose `Q` independent BFS instances are processed in elapsed time `T`:

```text
throughput = Q / T.
```

More GPUs may increase aggregate queries per second by running independent
traversals concurrently even if latency of each query is unchanged. Report
latency distribution as well as throughput: batching can improve utilization
while making an individual query wait longer.

Independent-query batching preserves `Q` separate visited sets and outputs.
It is not multisource BFS. A single BFS seeded with all sources computes

```text
min_i delta(s_i,v),
```

and merges its wavefronts; it does not return `Q` independent distance maps.
Replacing a query batch with multisource BFS changes the problem.

## 4. Weak scaling and why BFS makes it delicate

Weak scaling grows the problem with `p` while attempting to keep a declared
amount of work or data per GPU roughly fixed. For BFS, “problem size per GPU”
must name the quantity:

- total graph vertices or edges;
- reachable ball size;
- frontier states at selected levels;
- logical successor occurrences;
- visited/frontier bytes;
- independent queries.

Holding total vertices per GPU fixed does not hold BFS work per GPU fixed.
Diameter, reachable fraction, frontier profile, duplicate convergence, owner
cuts, and output size can all change with the graph family. A valid weak-scaling
study therefore specifies how instances grow and reports their semantic work
profiles, not only nominal `|V|/p`.

## 5. Capacity scaling is not weak scaling

Both regimes change the workload, but answer different questions:

- weak scaling asks how time or rate changes under a controlled growth rule;
- capacity scaling asks what larger exact workload becomes feasible.

A record such as “four GPUs reached depth 18 while one GPU reached depth 16” is
capacity evidence only after accounting for the different metric balls and
peak intermediates. It is not a speedup measurement, and the two depths need
not represent comparable work.

## 6. Strong scaling is level-profile dependent

For a fixed BFS, additional GPUs act on a changing frontier profile:

- narrow early levels may not fill one GPU;
- wide middle levels may expose abundant parallel work;
- late levels can perform many visited/duplicate checks for little new-state
  progress;
- every strict layer retains a dependency on completion of the preceding one.

Therefore one whole-run efficiency number hides where scaling is gained or
lost. Preserve per-level

```text
semantic work,
max and sum owner work,
local and routed bytes,
stage and critical-path times,
accepted-state progress,
completion/termination cost.
```

The slowest owner and the on-critical-path communication determine a bulk
level, not aggregate throughput alone.

## 7. Superlinear speedup is a diagnosis, not a contradiction

`S_p > p` can occur in a measurement without violating the work/span bounds of
a fixed abstract algorithm, because the physical regime changed. Examples
include:

- the distributed working set fitting into faster memory tiers;
- avoiding spill, paging, or an over-capacity fallback;
- better aggregate cache behavior;
- a representation or batching threshold changing;
- the baseline using a different effective code path.

Such a result should trigger an accounting explanation. It is not evidence that
BFS has negative work, and it is not automatically invalid. The semantic
workload and effective physical paths must first be shown equivalent.

## 8. Scaling efficiency can exceed or fall below capacity efficiency

Define no universal “GPU efficiency” scalar. At least four outcomes coexist:

```text
single-query latency,
fixed-work strong-scaling efficiency,
independent-query throughput,
maximum exact feasible workload.
```

A design may replicate visited information, reducing usable capacity but
improving latency. Another may shard state compactly, increasing capacity while
adding owner routing and latency. Neither outcome dominates without naming the
goal and workload.

## 9. Minimum experiment matrix

A future scaling claim should include separate rows for:

1. same exact workload on every feasible GPU count;
2. increasing workload under a declared weak-scaling rule;
3. maximum feasible workload/cutoff with memory accounting;
4. independent-query batches at controlled arrival and batch sizes;
5. correctness parity and validation scope for every row.

For each row retain topology, device model, clocks/power policy where available,
process placement, peer/transport path, warmup/repetition policy, and end-to-end
timing boundary. Isolated kernel or simulated wire measurements remain lower
rungs of evidence and should not be labeled multi-GPU traversal scaling.

## 10. Rejected implications

- A larger solved ball proves strong scaling.
- Aggregate GPU memory equals exact BFS capacity.
- Higher aggregate states/s means lower latency for one BFS.
- A batch of independent BFS queries is equivalent to multisource BFS.
- Fixed vertices per GPU imply fixed BFS work per GPU.
- Equal depth cutoffs imply comparable workloads across graph families.
- More GPUs imply proportional latency reduction.
- Superlinear speedup alone proves either an error or an algorithmic miracle.

## 11. Current synthesis

Latency scaling asks how fast the same wave finishes. Throughput scaling asks
how many independent waves finish per unit time. Weak scaling changes the graph
under a declared growth rule. Capacity scaling asks how large an exact wave can
be represented at all. Keeping these axes separate prevents memory aggregation,
batch utilization, and genuine single-query acceleration from being reported as
the same achievement.

This note refines notes 07, 13, 29, 47, 51, 54, 103, and 165.

