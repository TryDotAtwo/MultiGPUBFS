# Cayley raw convolution and nonlinear frontier evolution

Normal Cayley ownership makes raw occurrence transport linear at the quotient
level. Exact BFS frontier evolution remains nonlinear because concrete endpoint
sets must be deduplicated and subtracted from visited.

This distinction explains why a quotient routing model can predict traffic
perfectly while still being unable to predict the next frontier.

No implementation or optimizer is introduced.

## 1. Raw candidate histogram is a convolution

Retain the notation of note 170. For quotient block `D`, let

```text
y_(d+1)(D) = number of raw labeled occurrences from F_d ending in D.
```

Summing the coset-to-coset matrix gives

```text
y_(d+1)(D)
 = sum_C M_d(C,D)
 = sum_C f_d(C) mu(C^-1 D).
```

This is right convolution of the frontier block histogram `f_d` with the
generator-image multiplicity `mu`. It is exact before concrete equality and
visited filtering.

Globally,

```text
sum_D y_(d+1)(D) = |S| |F_d|.
```

## 2. Exact BFS applies set semantics after the convolution

At the concrete state level, define

```text
Occ_d = labeled successor occurrences from F_d,
U_d   = distinct endpoint states represented by Occ_d,
F_(d+1) = U_d minus B_d.
```

For each owner block `D`,

```text
f_(d+1)(D) = |(U_d intersect D) minus B_d|.
```

Therefore

```text
0 <= f_(d+1)(D) <= min(y_(d+1)(D), |D minus B_d|).
```

The upper bound can be strict for two independent reasons:

- several occurrences converge to one concrete endpoint in `D`;
- a distinct endpoint in `D` was already visited.

The quotient histogram forgets exactly the within-block identity information
needed to distinguish these causes.

## 3. A same-histogram, different-frontier counterexample

Use directed `G=Z_8`, normal subgroup

```text
H={0,4},
```

and generators

```text
S={+1,+2}.
```

The quotient blocks are `C0=H`, `C1=H+1`, `C2=H+2`, `C3=H+3`. Compare two
exact multisource depth-zero frontiers:

```text
A={0,1},
A'={0,5}.
```

Both have the identical block histogram:

```text
one source in C0,
one source in C1.
```

Consequently the raw convolution is identical:

```text
y_1(C1,C2,C3) = (1,2,1).
```

But concrete expansion differs.

For `A={0,1}`:

```text
0 -> 1,2
1 -> 2,3
```

State `1` is already visited and state `2` occurs twice, so

```text
F_1={2,3}.
```

For `A'={0,5}`:

```text
0 -> 1,2
5 -> 6,7
```

All four endpoints are distinct and unvisited, so

```text
F'_1={1,2,6,7}.
```

The same source block histogram and same raw destination histogram therefore
produce next-frontier sizes two and four. Quotient occurrence counts do not
determine concrete novelty.

## 4. The nonlinear stages

The full level can be viewed as

```text
frontier block counts
  --linear quotient convolution-->
raw destination counts
  --concrete identity quotient-->
unique endpoint states
  --visited set difference-->
accepted next frontier.
```

“Linear” here refers only to count propagation before equality. State
deduplication and set difference are idempotent set operations, not linear
operations on cardinality histograms.

This is the same reason that

```text
|A union B|
```

cannot be recovered from `|A|` and `|B|` without intersection information.
Cross-parent convergence is precisely missing intersection structure.

## 5. When equality with raw counts holds

For block `D`, equality

```text
f_(d+1)(D) = y_(d+1)(D)
```

holds exactly when every occurrence ending in `D` has a distinct concrete
endpoint and none of those endpoints lies in `B_d`.

Across the whole level this is the collision-free, unvisited tree-growth
regime. It can hold in early levels of a high-girth graph, but it is a property
to verify, not a consequence of Cayley regularity or normal quotient structure.

Once relations close words or the wave touches its previous ball, the gap

```text
y_(d+1)(D) - f_(d+1)(D)
```

combines concrete convergence and visited hits. Note 160's waterfall separates
these categories when fuller support-arc information is retained.

## 6. Histograms are sufficient for routing, not identity

The raw histogram is sufficient to reserve or account for logical destination
volume under the quotient contract. It is insufficient to:

- decide which states are new;
- size the final unique frontier exactly;
- recover parent multiplicity;
- infer shortest-path counts;
- prove that a destination bin contains no duplicates;
- replace the authoritative visited set.

Any physical scheme that aggregates records solely by owner must eventually
restore enough concrete identity to perform these duties. Combining transport
containers does not combine semantic states.

## 7. Bounds from block capacity and visited occupancy

If a Cayley block has known size `|H|` and

```text
b_d(D) = |B_d intersect D|,
```

then

```text
f_(d+1)(D) <= |H| - b_d(D).
```

Together with the occurrence bound,

```text
f_(d+1)(D)
 <= min(y_(d+1)(D), |H|-b_d(D)).
```

Neither bound is generally tight. The first ignores whether remaining states
are adjacent to the current frontier; the second ignores endpoint collisions.
They are capacity and occurrence ceilings, not frontier predictions.

For Schreier orbit blocks, replace `|H|` with the actual orbit size from note
167; variable stabilizers make the ceiling block-dependent.

## 8. One- and multi-GPU interpretation

On one GPU, the distinction separates regular expansion volume from exact
identity/visited work. On many GPUs, it separates predictable destination
volume from owner-side convergence and novelty.

Useful per-block telemetry is:

```text
raw occurrences y,
distinct endpoints u,
already-visited distinct endpoints v,
accepted states f_next,
physical records before and after local combination,
owner-received records,
bytes and retries.
```

The exact state-level identity is

```text
raw occurrences
= occurrence aliases + unique endpoints,
unique endpoints
= visited distinct endpoints + accepted states.
```

Its categories must follow the declared label/support contract. A quotient-bin
collision is not automatically an occurrence alias.

## 9. Consequences for modeling

- A quotient convolution can be an exact traffic baseline without being a BFS
  frontier recurrence.
- A learned or analytic model using only block counts cannot be exact for all
  concrete frontiers unless additional within-block intersection state is
  supplied.
- Predicting accepted progress from `y` requires workload-specific collision
  and visited models, which remain hypotheses until validated.
- Equal raw routing across GPU counts does not imply equal owner-side work if
  local precombination changes concrete multiplicities.
- The semantic validation ladder still ends at exact frontier-set equality, not
  histogram equality.

## 10. Rejected implications

- Quotient convolution is the complete BFS recurrence.
- Equal frontier block histograms imply equal next frontiers.
- Equal raw destination histograms imply equal accepted-state counts.
- A destination count of one proves one new state.
- Known block capacity makes the frontier-size upper bound tight.
- Owner aggregation removes the need for concrete identity.
- Cayley regularity prevents cross-parent convergence.
- Histogram parity proves exact frontier-set parity.

## 11. Current synthesis

The Cayley action has a linear count shadow: quotient generator images transport
raw occurrence mass by convolution. BFS itself lives one semantic level deeper:
it takes unions of concrete endpoints and subtracts the accumulated visited
ball. Those idempotent set operations make frontier evolution nonlinear and
information-losing under coarse histograms.

This is a useful division of labor. Quotient algebra predicts where raw work
goes; exact identity determines how much of that work becomes new graph
knowledge.

This note extends notes 04, 25, 51, 54, 157, 160, 163, 165, 167, and 170.

