# REF-006: Pareto study of exact owner partitions

Date: 2026-08-27.

## Question

Is any tested deterministic eight-rank owner function universally best for the
exact `S8` traversal when considering balance, communication, duplicate
convergence, and persistent capacity together?

## Strategies

- `rank_mod`: exact Lehmer rank modulo eight.
- `rank_range`: eight contiguous equal ranges of exact rank.
- `mul_high`: high three bits of rank multiplied by the 64-bit golden-ratio
  constant.
- `mix_00..mix_31`: SplitMix64-style avalanche of exact rank XOR a deterministic
  salted constant, modulo eight.

Every owner function is deterministic and stable for the duration of BFS.

## Metrics

All objectives below are minimized:

- `max_frontier_imbalance`: maximum `max/mean` for levels with at least 128
  frontier states;
- `max_recv_imbalance`: maximum owner receive `max/mean` when at least 128
  source-local unique candidates exist;
- `remote_fraction`: source-local unique candidates whose owner differs from the
  source rank;
- `cross_rank_duplicates`: candidates removed only after source-local sets meet
  at owners;
- `final_visited_imbalance`: final owner visited `max/mean`.

The complete dataset is
[`REF-006-partition-metrics.csv`](REF-006-partition-metrics.csv).

## Named extremes

| strategy | frontier imbalance | receive imbalance | remote fraction | cross-rank duplicates | final visited imbalance |
|---|---:|---:|---:|---:|---:|
| rank modulo | 2.114943 | 2.322581 | 0.635901 | 133,886 | 1.000000 |
| rank ranges | 4.505747 | 4.000000 | 0.333377 | 40,306 | 1.000000 |
| multiplicative high bits | 1.195402 | 1.122807 | 0.911733 | 157,028 | 1.000397 |

Contiguous ranges preserve substantial generator locality but can leave most
work on a few ranks. Multiplicative high bits balance work well but destroy most
locality. Direct modulo lies between those extremes.

## Pareto result

Using all five metrics, 20 of 35 strategies are non-dominated. This large set is
not evidence that the salts are all equally good. It shows that small gains in
one dimension frequently trade against another and that the experiment lacks a
single application-level cost function.

Therefore the hypothesis of a universally best tested owner mapping is rejected.

## Example constrained selection

Suppose an operational requirement is stated explicitly:

```text
max_frontier_imbalance <= 1.30
max_recv_imbalance <= 1.50
```

Three tested strategies satisfy both constraints. Among them, `mix_29` has the
lowest remote fraction:

```text
frontier imbalance       1.282799
receive imbalance        1.389222
remote fraction          0.874467996
cross-rank duplicates    157612
final visited imbalance  1.021627
```

This is a conditional result for this graph and these constraints, not a general
recommendation for salt 29.

## Insights

1. Partition quality is a vector, not a scalar.
2. Range ownership can act like graph partitioning by preserving generator
   locality, but static ranges need not balance individual BFS levels.
3. Strong avalanche mixing makes communication approach random all-to-all.
4. Better balance can increase both remote payload and owner-side duplicate
   convergence because fewer duplicates meet locally.
5. Exact final capacity balance does not imply transient frontier/scratch
   balance; VRAM planning needs peak per-rank level data.
6. Salt sweeping without a declared selection rule risks benchmark overfitting.
   Salts must be validated on other graph families or selected independently of
   the measured workload.

## Limitations

- Counts model ideal exact sets, not GPU execution time.
- The frontier threshold of 128 is an analysis choice.
- Only `S8` adjacent transpositions and eight ranks are used.
- No topology-weighted byte cost, chunking, backpressure, or overlap is modeled.
- A strategy may be non-dominated only because of a tiny numerical difference.

## Next experiment

Repeat a fixed, predeclared subset of owner functions across graph families and
rank counts. Avoid selecting salts on each graph. Add topology-weighted costs
and peak per-rank bytes before translating these counts into a multi-GPU design.
