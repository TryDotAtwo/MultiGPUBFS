# BFS versus random-walk hitting and cover time

BFS expands every state at a given minimum distance. A random walk follows one
stochastic trajectory and may revisit the same region many times. Both use the
same transition graph, but depth, hitting time, mixing time, and cover time
measure different phenomena.

This note extends note 33 from walk mass to first-hit and coverage semantics. It
does not implement a sampler or replace BFS with random walks.

## 1. Four different quantities

For a finite connected undirected graph:

- `dist(s,t)` is the length of a shortest `s`-to-`t` path;
- the hitting time `H(s,t)` is the expected number of random-walk steps before
  first visiting `t` from `s`;
- mixing time measures how long the walk distribution takes to approach a
  stationary distribution under a declared norm and tolerance;
- cover time is the expected number of steps needed to visit every vertex,
  commonly maximized over the starting vertex.

BFS distance is deterministic for a fixed graph. The other three depend on a
transition probability rule and are expectations or distributional bounds.

## 2. First hit is not shortest distance

Every random-walk trajectory that first hits `t` after `k` steps supplies an
`s`-to-`t` walk of length `k`, so

```text
dist(s,t) <= k.
```

Deleting cycles from that walk gives a replayable simple path no longer than
`k`. But first hit need not be shortest, and the expectation `H(s,t)` can be
much larger than `dist(s,t)`.

On the path with `n` vertices, started at one endpoint and targeting the other:

```text
dist = n-1,
H = (n-1)^2.
```

BFS advances once through each layer; the walk repeatedly backtracks.

### Scramble length is a witness label, not a BFS distance label

If a length-k move sequence starting at a goal produces x, it certifies
`dist(goal,x)<=k`. In an inverse-closed unit-cost graph, reversing the moves
also gives `dist(x,goal)<=k`. Neither inequality implies equality.
Note 39's C4 example makes this visible without immediate backtracking:
the word `aaaa` has length four but returns to the goal, whose distance is
zero. Forbidding adjacent inverse pairs therefore does not turn scramble
lengths into exact BFS labels.

Direction is essential. In the directed Cayley cycle Z6 with only move +1,
one scramble step sends 0 to 1, but returning from 1 to goal 0 needs five
steps. A forward scramble length is not even an upper bound on the reverse
query unless a valid reverse-path argument is available.

If a dataset uses scramble length as a target, the target describes the
generated trajectory, not automatically the endpoint's shortest distance.
Exact-fit reproduction of such targets would still not certify an admissible
distance heuristic. This is a conditional interpretation rule, not an audit
claim about any current CayleyPy training dataset or model.

#### One retained source example, not checkpoint provenance

Read-only inspection on 2026-08-31 found this concrete training path in
`D:\100XH100\paper\kaggle_review_sweeps\bundle_stage\pilgrim_runtime\pilgrim\trainer.py`:

- lines 97-112 select a random move, excluding the immediately inverse move,
  and apply it by gather;
- lines 114-127 create `depths` from K_min through K_max, start states at V0,
  perform the corresponding number of steps, shuffle, and return states/depths;
- lines 171-180 dispatch the selected sampler to training samples;
- lines 193-197 use Y directly as the loss target;
- lines 224-228 pass generated Y, cast to float, into the training epoch.

This branch supplies trajectory-length targets, not BFS-certified minimum
distances. No BFS correction occurs along the inspected assignment-to-loss
chain. The random-tree branch likewise records its generation depth. These
are statements about this staged source file; no run manifest or checkpoint
link was verified, and they must not be generalized to the production scorer
or every CayleyPy trainer. The earlier requested expert could not read their
checkout and supplied no source-backed provenance; the file evidence here
comes from independent local inspection, not that expert's speculation.

## 3. Coverage is not component exhaustion evidence at a fixed time

On a finite connected graph, a simple random walk visits every vertex
eventually with probability one. This asymptotic probability statement does not
give a deterministic finite stopping time.

After `T` steps, failure to visit `v` means only that this trajectory missed
`v`. It does not prove `v` unreachable. Likewise, observing no new vertex for a
long interval is not an exhaustion certificate unless an independent finite
state count or other complete coverage proof is available.

BFS has a different certificate: when its exact frontier is empty and in-flight
work is globally settled, the reachable closure is complete.

## 4. Complete graph: mixed is not covered

For the complete graph `K_n`, BFS from one root discovers all other vertices in
the first layer. A simple random walk has the uniform stationary distribution,
and after each move its next vertex is nearly a uniform coupon among the other
vertices.

Starting with one visited vertex, the expected cover time is

```text
(n-1) H_(n-1) = Theta(n log n),
```

where `H_k` is the harmonic number. The walk can be perfectly spread in
distribution while still missing some individual coupons. Mixing is about
probability mass, not the event that every state has appeared in one history.

## 5. Worst-case cover time can be cubic

For any connected undirected `n`-vertex graph, classical results bound simple
random-walk cover time by order `n^3`; Feige proved the tight asymptotic upper
constant `4/27` in the worst case. Lollipop-type graphs attain cubic order.

An exact adjacency-list BFS instead processes each reached vertex and edge a
bounded number of times, with ordinary work `O(n+m)`. This is not a claim that
BFS is always cheaper in memory or hardware time. It shows that one-trajectory
random exploration and systematic frontier expansion have fundamentally
different worst-case coverage behavior.

## 6. Stationarity on finite Cayley graphs

For a finite undirected regular Cayley graph, the simple random walk has uniform
stationary distribution. If the graph is bipartite, the non-lazy walk is
periodic; adding a stay-put probability removes that oscillation without
changing reachability.

Uniform stationarity means a stationary-time sample is marginally uniform over
group elements. It does not mean:

- the trajectory has covered the group;
- each BFS sphere was sampled proportionally to its cardinality before mixing;
- the first discovered word for a state is geodesic;
- unvisited states are unreachable.

The state distribution forgets most of the trajectory's duplicate history;
BFS visited state exists precisely to suppress rediscovery.

## 7. Multiple walkers

Launching `p` independent walkers increases sampling throughput but does not
universally divide hitting or cover time by `p`. Walkers can revisit the same
high-probability region, share the same bottleneck, or all miss a rare state.

Their union of visited states becomes an exact coverage certificate only if:

- semantic identity is deduplicated exactly;
- the required total reachable-state count is independently known and matched,
  or another closure proof is supplied;
- every walk uses the declared transition relation;
- probabilistic confidence is not reported as deterministic completeness.

Coordinating a global visited set can reduce redundant walks, but that creates a
different exploration algorithm; it still does not restore BFS distance order
unless minimum-depth scheduling and finalization are enforced.

## 8. Directed and asymmetric Cayley walks

In a directed generator graph, outgoing moves and their probabilities define a
possibly nonreversible Markov chain. A stationary distribution may be nonuniform
or may not be unique if the chain is not irreducible. Hitting some states can
have infinite expectation or probability below one.

A finite strongly connected directed Cayley multigraph with uniform choice
among labeled generators is balanced and has uniform stationary distribution,
but periodicity and cover-time behavior still depend on the alphabet. Parallel
labels affect transition probability even when they share a state endpoint.

Thus state-BFS identity and random-walk transition multiplicity may intentionally
use different graph contracts.

## 9. GPU and multi-GPU boundary

Random walks can be attractive for GPUs because many trajectories have little
frontier synchronization. That operational property does not make them exact
BFS:

- walker steps count generated samples, not unique first discoveries;
- load can be balanced while semantic coverage is highly duplicated;
- independent walkers need no owner routing until global dedup or coverage is
  requested;
- adding global visited communication changes the cost structure;
- a time budget provides probabilistic sampling evidence, not a completed
  radius or unreachable certificate;
- speedup in steps per second says nothing by itself about hitting or cover
  probability for the target states of interest.

Random walks may support sampling, heuristic discovery, or relation probes.
Exact BFS claims still require BFS evidence.

## 10. Evidence checklist

1. Shortest distance, expected hitting time, mixing time, or cover time.
2. Transition probabilities, laziness, labels, and parallel arcs.
3. Expectation, high-probability bound, almost-sure eventual event, or
   deterministic certificate.
4. One target, sampled distribution, or complete component coverage.
5. Known total state count or independent closure evidence.
6. One walker, independent walkers, or coordinated visited exploration.
7. Undirected reversible or directed nonreversible chain.
8. Steps per second versus unique coverage and target-hit probability.

## Sources

- R. Aleliunas, R. M. Karp, R. J. Lipton, L. Lovász, and C. Rackoff,
  [*Random Walks, Universal Traversal Sequences, and the Complexity of Maze
  Problems*](https://doi.org/10.1109/SFCS.1979.34),
  FOCS 1979, 218-223. Classical random-walk coverage bound for undirected
  reachability.
- U. Feige,
  [*A Tight Upper Bound on the Cover Time for Random Walks on
  Graphs*](https://doi.org/10.1002/rsa.3240060106),
  Random Structures & Algorithms 6(1), 1995, 51-54. Tight worst-case cubic
  cover-time upper bound.
- Notes 03, 09, 25, 33, 46, 51, 56, 85, 86, and 94 provide level,
  completeness, fixed-point, walk-mass, spectral, ownership, termination,
  period, exact-length, and boundary context.

## Takeaway

BFS converts edge reachability into deterministic minimum-depth layers and an
empty-frontier closure certificate. A random walk converts edges into a
probability process whose hitting, mixing, and cover times can be far larger
and answer different questions. Parallel walkers improve sample production,
not automatically shortest paths, complete coverage, or unreachable proofs.
