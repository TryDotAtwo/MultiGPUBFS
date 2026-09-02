# BFS versus iterative deepening: storing the frontier or recomputing it

Breadth-first search stores broad shallow progress. Depth-first iterative
deepening (IDDFS) repeatedly recomputes shallow prefixes to obtain a BFS-like
shallowest-solution guarantee with depth-first memory.

They can return the same minimum hop count while traversing very different
objects:

```text
BFS graph search   -> unique state layers with global visited
IDDFS tree search  -> bounded paths/words, often with repeated states
```

This distinction is especially large in Cayley spaces, where many words denote
one group element.

## Iterative-deepening contract

For limits

```text
L = 0, 1, 2, ...,
```

run a complete depth-limited DFS that generates every allowed path of length at
most `L` unless a target is found. Discard/reinitialize the iteration state, then
increase the limit after a cutoff-only result.

The depth-limited search must distinguish:

- **FOUND:** a target was reached;
- **CUTOFF:** some branch stopped only because of the limit;
- **EXHAUSTED:** every branch ended before the limit and no deeper state exists.

Conflating cutoff with exhaustion can stop before a deeper solution.

## Why the first successful limit is shortest

Assume unit edges, finite branching, and complete enumeration at every smaller
limit. If the first success occurs at limit `d`, then:

- the returned path has length at most `d`;
- exhaustive iteration `d-1` proved that no path of length at most `d-1`
  reaches a goal.

Therefore its length is exactly the minimum hop count. This is the same scalar
optimality as BFS target search, obtained by repeated depth bounds rather than
simultaneous storage of `F_0,...,F_d`.

Finite branching matters: an iteration must be able to finish all nodes within
its finite depth bound. Infinite branching at a shallow node can prevent later
siblings from ever being considered.

## Memory and repeated work

On a tree with branching factor `b` and shallowest solution depth `d`:

- BFS stores a frontier that can be on the order of `b^d`;
- depth-limited DFS stores one path plus pending siblings, commonly `O(b*d)`;
- IDDFS regenerates upper levels in each iteration.

For regular exponential trees (`b>1`), the deepest level dominates, so repeated
prefix work is a constant-factor overhead asymptotically. This intuition is not
universal:

- for a chain (`b=1`), repeated work through depth `d` is quadratic rather than
  linear;
- irregular trees can concentrate expensive generation near the root;
- graph transpositions can make the word tree exponentially larger than the
  unique state graph;
- successor computation cost may vary strongly by depth/state.

"IDDFS has BFS time" is an exponential-tree asymptotic statement, not equality
of generated transitions on every graph.

## Tree search versus graph search

Without a global transposition table, IDDFS treats two paths reaching one state
as two tree nodes. Current-path cycle detection can prevent infinite recursion
within one bounded branch, but convergence across different branches is
recomputed.

BFS visited instead quotients the path tree by state identity and stores the
minimum layer at which each state appears. It pays memory to avoid re-expanding
the same state and its suffixes.

Dropping BFS visited is not merely a low-memory representation change. On a
cyclic finite graph, the path tree is infinite even though the state graph is
finite.

## Why boolean visited can break depth-limited DFS

Use limit `3` and edges

```text
s -> a -> c -> x
s -> b -> x -> goal.
```

Suppose DFS explores the `a` branch first. It reaches `x` at depth `3`, marks it
visited, and cannot expand `x -> goal` because that would exceed the limit.
Later it reaches `x` through `s -> b -> x` at depth `2`, but a global boolean
visited suppresses this occurrence. The valid depth-`3` solution

```text
s -> b -> x -> goal
```

is missed.

The first occurrence had zero remaining budget; the second had one. Equal state
does not imply equal search potential under a depth limit.

## Depth-aware transposition dominance

For the same unconstrained state and limit `L`, arrival at smaller depth
dominates arrival at larger depth because it has at least as much remaining
budget:

```text
remaining = L - depth.
```

A bounded-search transposition entry can safely prune an arrival only when an
equal state was already searched with **at least** the same remaining budget
under the same semantic context. Equivalent formulations retain minimum reached
depth or maximum searched remainder.

Important qualifications:

- entries from a smaller previous IDDFS limit do not automatically prove full
  exploration under a larger remaining budget;
- path-dependent constraints require product/history identity from note 20;
- graph-version/generator changes invalidate dominance from note 22;
- hash collision is not state equality;
- evicting an entry usually loses only pruning, while treating an approximate
  hit as exact can lose completeness.

Current-recursion-stack cycle pruning is different from global visited: it only
forbids repeating a state on the same path and does not merge independent
arrivals with different remaining budgets.

## Persistent visited across iterations is dangerous

Iteration `L` proves facts only for its searched remaining-depth budgets. If a
boolean table is retained into iteration `L+1`, states cut off at the old limit
may be incorrectly considered fully explored and their newly available suffixes
skipped.

Safe reuse needs a bound-aware transposition meaning, not "seen sometime in an
earlier iteration." Resetting iteration-local visited is conceptually simpler
but reintroduces repeated graph work.

## IDDFS output is not a complete BFS ball

When IDDFS finds one target at depth `d`, it proves no target existed at smaller
depths. It does not necessarily return:

- every unique state in `B_d`;
- the exact frontier `F_d`;
- all shortest target parents;
- source eccentricity or component exhaustion;
- per-level unique/duplicate statistics comparable to BFS.

Tree-node visits count path prefixes, not unique graph states. Comparing IDDFS
"nodes expanded" directly with BFS accepted states mixes semantic units unless
both are decomposed carefully.

## Ordered IDDFS and shortlex solutions

If generator/edge labels are totally ordered and each depth-limited iteration
enumerates path words in lexicographic order, the first goal at the first
successful depth is the shortlex-least goal word.

Depth-aware transposition pruning can change which word survives even while
preserving minimum distance. A shortlex contract must ensure that pruning keeps
the least representative, not merely any dominating shallow state occurrence.

In a Cayley graph, two equal-depth words representing one element may have
different lex order and different suffix enumeration order. State-distance
dominance alone does not preserve all labeled solutions.

## Cayley word-tree interpretation

For generator alphabet `S`, raw IDDFS explores the rooted word tree `S*` and
evaluates each word into a group element. Relations cause many tree nodes to map
to one Cayley vertex.

Even for a finite group:

- inverse pairs and relations generate infinitely many words;
- each bounded iteration is finite when `S` is finite;
- a finite-depth target is eventually found optimally;
- the search does not "exhaust the group" merely because the word tree keeps
  producing paths;
- proving diameter/component coverage still needs unique state identity/counts
  or independent group knowledge.

Immediate inverse pruning from note 20 removes a class of nongeodesic words and
is safe for unconstrained shortest targets. Stronger algebraic pruning needs a
proof that at least one shortest representative of every relevant element
survives.

## Frontier search is not just BFS without closed

Frontier-search methods reduce memory by retaining Open/frontier information
and using structured duplicate detection and divide-and-conquer reconstruction.
Their correctness comes from those additional mechanisms.

Simply deleting the closed/visited set from graph BFS turns it into path-tree
enumeration with potentially unbounded duplicates. The words "frontier only"
do not supply duplicate or reconstruction semantics.

External delayed duplicate detection from note 15 is another middle ground: it
keeps exact graph identity but moves/batches when duplicates are resolved.

## Bidirectional and iterative-deepening distinctions

Bidirectional BFS reduces effective depth by storing waves from both endpoints.
IDDFS reduces memory by recomputing bounded prefixes. They address different
resources and require different stopping proofs.

A bidirectional depth-bounded scheme must account for how the two path budgets
sum, how meeting states are deduplicated, and whether every smaller total depth
was excluded. Alternating two incomplete DFS trees does not inherit note 08's
complete-ball stopping proof.

## GPU and multi-GPU conceptual implications

IDDFS exposes different parallel work from frontier BFS:

- independent prefix subtrees can be distributed;
- subtree sizes may be highly imbalanced because relations/legality differ;
- cross-prefix transpositions repeat work unless shared exact tables exist;
- a found goal at limit `d` is shortest only after every smaller limit was
  globally completed;
- one path can stop the current limit for a one-solution contract once smaller
  limits are certified, but cancellation/in-flight accounting still matters;
- transposition entries need state plus searched-budget/context semantics;
- tree-prefix traffic and unique-state traffic are different metrics.

This does not select IDDFS or BFS for a GPU. It identifies which work and memory
objects a comparison must measure.

## Counterexamples and rejected shortcuts

- **IDDFS is BFS with a smaller queue.** It recomputes a path tree instead of
  retaining exact unique-state frontiers.
- **Boolean visited is always safe in depth-limited DFS.** A deep first arrival
  can block a shallower occurrence with more remaining budget.
- **Visited can persist unchanged across increasing limits.** Old cutoff does
  not prove exploration under the new budget.
- **Finite graph implies finite tree search.** Cycles create infinitely many
  words/walks.
- **IDDFS success returns the complete ball through that depth.** It proves
  shallowest goal depth, not exhaustive unique-state layers.
- **Constant-factor repeated work is universal.** Chains and costly irregular
  prefixes contradict the exponential-tree intuition.
- **Removing closed yields frontier search.** Exact duplicate and reconstruction
  mechanisms are still required.

## Audit checklist

1. Is the search object a state graph or a path/word tree?
2. Are FOUND, CUTOFF, and EXHAUSTED distinguished?
3. Is branching finite at every bounded depth?
4. What is stored: recursion stack, path-cycle set, or global transpositions?
5. Does a transposition entry record minimum depth/maximum remaining budget?
6. Are entries safely reusable across larger limits and graph versions?
7. Is the output one shortest path or a complete unique-state ball?
8. Are work counts path prefixes, transitions, or unique states?
9. Does pruning preserve any geodesic, the shortlex geodesic, or all geodesics?
10. What global completion certificate covers every smaller distributed limit?

## Sources

- Richard E. Korf, *Depth-First Iterative-Deepening: An Optimal Admissible Tree
  Search*, Artificial Intelligence 27(1), 1985,
  [DOI](https://doi.org/10.1016/0004-3702(85)90084-0), for the classical
  time/space/solution-depth trade-off on exponential tree search.
- David Poole and Alan Mackworth, *Artificial Intelligence: Foundations of
  Computational Agents*,
  [iterative-deepening section](https://www.cs.ubc.ca/~poole/aibook/2e/html2e/ArtInt2e.Ch3.S5.SS3.html),
  for cutoff/exhaustion and finite-graph termination distinctions.
- Richard Korf, Weixiong Zhang, Ignacio Thayer, and Heath Hohwald,
  *Frontier Search*, Journal of the ACM 52(5), 2005,
  [paper](https://ai.dmi.unibas.ch/research/reading_group/korf-etal-jacm2005.pdf),
  for exact memory reduction beyond simply dropping Closed.
- Notes 09, 15, 19, 20, and 22 supply infinite-graph completeness, external
  duplicate detection, shortlex, history state, and version constraints.

## Current synthesis

IDDFS recovers BFS's shallowest-solution guarantee by proving successively larger
depth bounds, not by maintaining BFS frontiers. Its small memory comes from
recomputation of the path tree. On graphs, safe duplicate pruning must compare
remaining search budget and semantic context; a boolean seen bit can destroy
completeness. On Cayley spaces, this is the trade: BFS stores group-element
identity to collapse relations, while raw IDDFS repeatedly walks the much larger
word tree.
