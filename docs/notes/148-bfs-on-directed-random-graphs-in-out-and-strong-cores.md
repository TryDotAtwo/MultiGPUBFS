# BFS on directed random graphs: IN, OUT, and strong cores

In an undirected graph, one connected-component notion is enough for ordinary
BFS.  In a directed graph, forward reachability, reverse reachability, and
mutual reachability are different objects.  A random directed graph makes this
difference macroscopic.

This note adds no optimized traversal, production SCC implementation,
benchmark, or GPU code.  Its retained Rust program is a semantic probe.

## 1. Model and four root sets

In the directed Erdos-Renyi model `D(n,c/n)`, every ordered pair `u!=v` is an
arc independently with probability `c/n`.  The opposite arc is a separate
random event.  One sampled digraph is frozen before any traversal.

For a root `s`, define

```text
R+(s): vertices reachable from s by following arcs,
R-(s): vertices that can reach s,
SCC(s) = R+(s) intersection R-(s).
```

Forward BFS computes distances inside `R+(s)`.  BFS on the transpose computes
distances inside `R-(s)`.  Their intersection is exactly the root's strongly
connected component, as proved in note 84.  Neither traversal alone proves
mutual reachability.

Weak connectivity, obtained by discarding orientation, is a fifth and different
contract.

## 2. Local forward and reverse branching

Both in-degree and out-degree converge locally to Poisson(`c`).  Consequently,
early forward and reverse explorations are each approximated by a Poisson(`c`)
branching process.  They have the same marginal ensemble law.

That symmetry does not make their realized frontiers equal.  One frozen graph
can give a root many outgoing descendants and no nontrivial predecessors, or
the reverse.  Transposition preserves the distribution, not individual BFS
results.

As before, the branching approximation is local: collisions, shared targets,
and finite-population depletion eventually matter.

## 3. The directed bow tie

Above the directed threshold `c=1`, let `rho` be the positive solution of

```text
rho = 1-exp(-c rho).
```

For the independent-arc model, asymptotically:

```text
GIN:  vertices that can reach the giant SCC       fraction rho,
GOUT: vertices reachable from the giant SCC       fraction rho,
GSCC: giant strongly connected component          fraction rho^2.
```

The names are directional from the strong core: GIN flows into it; GOUT flows
out of it.  Confusing those names reverses the interpretation of forward BFS.

There may also be tendrils and tubes outside the three central sets.  The
simplified fractions above concern the symmetric independent-Poisson model,
not an arbitrary directed network with correlated in/out degrees.

## 4. Root conditioning

A uniformly chosen root has a giant forward reachable set only when it lies in
GIN.  This occurs with probability about `rho`; conditional on that event, its
forward reach occupies about `rho n` vertices.  Hence

```text
E[|R+(s)|/n] approximately rho^2.
```

The analogous statement holds for reverse reach and GOUT.  A root has a giant
SCC only when it lies in GSCC, with probability about `rho^2`; conditional SCC
fraction is also about `rho^2`, giving an unconditioned normalized expectation
near `rho^4`.

These are mixture distributions, not concentrated frontier profiles.  Reporting
only the mean hides whether a traversal was giant or tiny.

## 5. Retained finite observations

`experiments/directed_random_bfs_probe.rs` sampled 20 frozen digraphs per `c`
at `n=2000`.  It ran forward and transpose BFS from root zero, computed all SCCs
for measurement, and ran both traversals from a representative of the largest
SCC.

```text
c    largest SCC   reverse reach of core   forward reach of core
0.8     0.0013             0.0081                  0.0049
1.0     0.0077             0.0645                  0.0407
1.2     0.0892             0.2833                  0.2998
4.0     0.9611             0.9804                  0.9804
```

Only the supercritical rows justify asymptotic GIN/GOUT language.  At and below
the threshold, the probe merely reports reachability to/from the largest finite
SCC; calling those sets giant components would be wrong.

For random root zero:

```text
c    mean forward   mean reverse   mean root SCC   enters core / in core
1.2      0.0692         0.0890          0.0069          5/20 / 2/20
4.0      0.8821         0.9805          0.8645         18/20 / 18/20
```

At `c=4`, all 20 roots were reachable from the core (in GOUT), but only 18
could reach the core (in GIN).  The asymmetry is finite sampling of a symmetric
law, not evidence that forward and reverse models have different parameters.

The representative `c=4` frontier layers were

```text
forward: [1,8,37,152,483,836,377,56,4,1]
reverse: [1,2,8,33,132,407,754,504,108,9].
```

They reach nearly the same total set size through visibly different waves.
Equal final cardinality does not imply equal per-level work or synchronization.

## 6. SCC measurement boundary

The probe uses a transparent two-pass SCC decomposition only to identify the
largest finite SCC and classify roots.  This does not turn ordinary BFS into an
SCC algorithm.  Two BFS traversals from one root recover only `SCC(s)`; a full
partition needs a whole-graph SCC procedure or additional traversals.

Choosing a representative of the largest SCC is an oracle-conditioned
measurement.  An application that does not already know the SCC cannot assume
that root selection for free.

Near `c=1`, the largest finite SCC and its in/out reach have high variance and
nonlinear scaling.  Twenty means are descriptive only.

## 7. Directed multi-GPU interpretation

Direction creates distinct physical and semantic questions:

- forward expansion scans out-degree; transpose expansion scans in-degree;
- storing only outgoing adjacency does not provide reverse BFS for free;
- owner-to-owner routing matrices need not be symmetric;
- a root outside GIN can terminate after little work even when a huge GOUT
  exists elsewhere;
- two traversals with similar total reach may have different frontier peaks and
  synchronization counts;
- intersecting distributed forward/reverse visited sets requires compatible
  stable vertex identity and completed traversals.

Measurements should separate forward and transpose storage, edge scans,
frontier profiles, local/remote candidates, visited outcomes, owner skew,
termination, intersection, and end-to-end time.  This is an observation schema,
not an implementation prescription.

## 8. Docker/Rust probe and retained failure

The probe samples every ordered pair with a deterministic xorshift stream,
builds both adjacency orientations, and validates its claims by complete BFS
and SCC enumeration.  The first Docker gate stopped because `rustfmt --check`
required one expression to wrap.  A later review rejected the first SCC
measurements: its iterative DFS marked sibling vertices too early and therefore
did not guarantee true DFS finish order for Kosaraju.  It was replaced by an
explicit `(vertex,next-edge-index)` stack.  All SCC-derived values were
discarded and recomputed.  Four independent 24-vertex fixtures then exhaustively
checked that two vertices share a reported component exactly when each is
reachable from the other.  The final format, compile, assertions, and execution
gate passed.

The `O(n^2)` sampler is intentionally transparent.  It is not a scalable graph
generator or a traversal benchmark.  The CPU-only container did not request a
GPU and therefore reported no NVIDIA driver.

## Sources

- R. M. Karp,
  [*The transitive closure of a random digraph*](https://doi.org/10.1002/rsa.3240010106),
  *Random Structures & Algorithms* 1(1), 1990, for the directed independent-arc
  model and the unique large SCC of asymptotic fraction `rho^2` above one.
- M. E. J. Newman, S. H. Strogatz, and D. J. Watts,
  [*Random graphs with arbitrary degree distributions and their applications*](https://doi.org/10.1103/PhysRevE.64.026118),
  *Physical Review E* 64, 2001; also
  [arXiv:cond-mat/0007235](https://arxiv.org/abs/cond-mat/0007235), for directed
  in/out generating functions and the bow-tie interpretation.
- Notes 05, 08, 16, 18, 21, 51, 56, 74, 84, 85, 91, 106, 132, 144, and 146
  provide this repository's directed-edge, reverse-search, SCC, owner,
  termination, measurement, support-graph, and random-frontier boundaries.

## Takeaway

Directed BFS answers one-way reachability.  In a supercritical random digraph,
the giant SCC is only the overlap of two larger macroscopic sets: vertices that
can enter it and vertices that can be reached from it.  Root conditioning
decides whether a traversal is giant or tiny, and transposition preserves an
ensemble law without preserving a realized frontier.
