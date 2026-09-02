# Product-state BFS: when the vertex must remember history

Ordinary visited deduplicates by current graph vertex because future outgoing
edges depend only on that vertex. If legal continuation also depends on path
history, the Markov state is larger:

```text
search vertex = (base state, relevant history/memory state).
```

BFS remains ordinary BFS on this product graph. Using only `visited[base]`
merges vertices with different futures and can remove the only valid solution.

## Edge-labeled graph and finite automaton

Let the base graph have labeled edges

```text
v --a--> w,  a in alphabet Sigma,
```

and let a deterministic finite automaton be

```text
A = (Q, q0, transition, Accept).
```

The product graph has vertices `(v,q)` and edges

```text
(v,q) -> (w, transition(q,a))
whenever v --a--> w.
```

Starting at `(source,q0)`, an accepting target is any

```text
(target,q) with q in Accept.
```

Unit-edge BFS in this product finds a minimum-length base walk whose label word
is accepted. The automaton state is not auxiliary metadata: it is part of exact
vertex identity.

For an NFA there may be several successor automaton states, which simply creates
several product edges/states. Epsilon transitions consume no base edge; treating
them as unit graph moves changes the measured length. They require epsilon
closure, elimination, or an explicit zero-cost/0-1 shortest-path model.

## Minimal counterexample to base-only visited

Use labeled edges

```text
s --a--> x
s --b--> x
x --a--> t
```

and accept only the word `ba`. If edge `a` is processed first, base-only visited
marks `x` after word `a` and discards the occurrence reached after word `b`.
It then reports no accepted path.

Product BFS distinguishes

```text
(x, after_a) != (x, after_b)
```

and retains the valid path `s --b--> x --a--> t`. The two records have the same
base state and depth but different future languages.

### Cayley hand trace: equal element, different last-move state

Let `G=Z_2 x Z_2` have commuting involutive generators `a` and `b`, and impose
the semantic rule “the next generator must differ from the previous one.” At
base depth two,

```text
ab = ba
```

as group elements, but the product states are distinct:

```text
(ab,last=b) reached by word ab,
(ab,last=a) reached by word ba.
```

Their legal futures differ:

```text
(ab,last=b) --a--> (b,last=a),
(ab,last=a) --b--> (a,last=b).
```

Base-only visited merges the two arrivals at element `ab` and can erase one of
these residual languages. It also cannot represent a query such as reaching
product target `(a,last=b)`, whose word `bab` is legal, merely by observing
that base element `a` was already reached earlier as `(a,last=a)`.

If “do not repeat” is only pruning for ordinary unconstrained distance, the
generators are involutions and the removed `aa`/`bb` spurs are nongeodesic; the
base element may remain the semantic vertex. If the last label affects legality
or acceptance, the same rule defines a different product graph. Syntax of the
filter does not decide which interpretation is intended.

## Projection changes the meaning of "visited once"

A shortest path in the product never repeats the same product vertex. Its
projection may revisit a base vertex under another memory state:

```text
(v,q1) ... (v,q2), q1 != q2.
```

This is not duplicate work. The visits represent different residual constraints
and may enable different suffixes.

Consequently, product BFS naturally finds a shortest accepted **walk** in the
base graph. If the required base path must be simple (no repeated base vertex),
product-vertex simplicity does not enforce it. Formal-language-constrained
simple-path problems can be much harder than constrained-walk reachability.

## Constraint versus pruning

The same history rule can play two very different roles.

### Semantic constraint

"Solutions must alternate colors" or "move `c` is legal only after `a b`" is
part of the requested path language. Different automaton states at one base
state are genuinely different search vertices.

### Search pruning

"Do not immediately undo the parent move" may be an attempt to remove words
that cannot be shortest in the **unconstrained** original graph. If a proof
shows that at least every geodesic remains representable, the semantic vertex
can stay the base state and the history is only generation context.

Calling both mechanisms "last-move filtering" hides whether the graph was
changed or merely redundant candidates were omitted.

## Why immediate backtrack pruning is safe for unit BFS

In an undirected unit graph, a walk segment

```text
u -> v -> u
```

can be deleted without changing endpoints, shortening the walk by two. No
shortest path contains such an immediate reversal.

For an inverse-closed Cayley alphabet, a geodesic word cannot contain

```text
s s^-1.
```

Therefore skipping the exact inverse of the parent move preserves ordinary
source-to-vertex distances and all geodesic words: it removes only a reducible
two-letter segment. With state-level visited, the skipped destination is also
the already reached parent state at depth one less.

The proof depends on exact inverse semantics, unit/nonnegative cost, and the
unconstrained target metric. It does not justify other history rules.

## Unsafe extrapolations from backtrack pruning

- Forbidding `s s` is safe when `s` is an involution, but not for a general
  generator. In directed `Z` with allowed `+1`, target `2` requires `s s`.
- Forbidding return to any recent state (not only the immediate parent) may
  require remembering an unbounded visited-on-path set if it is a semantic
  simple-path constraint.
- Under regular-language constraints, an accepted shortest walk may deliberately
  backtrack to change automaton state; unconstrained geodesic cancellation no
  longer proves safety.
- In a quotient, the declared inverse must lift to the actual concrete parent;
  equality of canonical keys alone may hide a frame change.
- A move with the same generator ID is not necessarily the inverse if action or
  orientation conventions are inconsistent.

## More general finite-memory rules

Finite automata can encode, for example:

- forbidden adjacent generator pairs;
- bounded cooldown after a move class;
- parity or residue of selected moves;
- required prefix/suffix patterns;
- turn restrictions on a road graph;
- protocol/control-state legality;
- "last generator" or last `k` labels;
- acceptance by a regular expression.

If the memory has `|Q|` states, the explicit worst-case product universe has
`|V|*|Q|` vertices, though an implicit BFS constructs only reachable pairs.
Reachable product geometry can differ radically from multiplying base frontier
sizes by `|Q|` because many combinations are unreachable or merge later.

## Periodic and time-expanded BFS

If edge availability depends on time phase, use states such as

```text
(vertex, time mod p)
```

for a truly period-`p` environment. Travel and optional wait actions advance the
phase. Reaching one base vertex at phase `0` does not dominate reaching it at
phase `1` when different edges will be available next.

Phase-only state is insufficient when costs, deadlines, or availability depend
on absolute time rather than a proved period. A finite horizon may require
`(vertex,time)`; an unbounded nonperiodic schedule may create an infinite state
space. Again, BFS has not changed—the vertex definition has.

## Non-backtracking graph as a product/lift

To treat non-backtracking walks as the semantic object, use directed-edge states

```text
(previous_vertex, current_vertex)
```

and transition to `(current,next)` only when `next != previous`. This is closely
related to the directed line/non-backtracking graph. Two arrivals at the same
current vertex through different previous edges are different states because
they forbid different next reversals.

For ordinary shortest vertex distance, this lifted graph is unnecessary because
a shortest path is already non-backtracking. For counting non-backtracking walks
or applying turn constraints, the lifted state is the correct object.

## Cayley graph times word automaton

For group element `g` and automaton state `q`, a right-action product step is

```text
(g,q) --s--> (g*s, transition(q,s)).
```

The same group element can occur in several product states because different
words representing it leave different accepted suffix languages. Group equality
does not imply product-state equality.

The resulting distance is the length of the shortest generator word that both

1. represents the target group element (or target orbit under the declared
   semantics), and
2. belongs to the accepted language.

It can exceed the ordinary Cayley word metric. It may also require revisiting a
group element under a new automaton state. A group-rank bitmap alone is therefore
lossy; a dense exact product rank could be `group_rank*|Q| + q` when both factors
are finite and densely represented.

Shortlex selection from note 19 applies to accepted product paths when product
frontiers and labels follow the corresponding order. It selects the least
accepted word, not the unrestricted group shortlex normal form.

## Bidirectional product search

Backward search must reverse both factors:

- graph edges use predecessors;
- automaton transitions must represent which earlier states could lead to the
  current state under the reversed label.

A DFA forward transition need not have a unique inverse, so the backward factor
is generally an NFA/subset relation seeded from all accepting target states.
Meeting on base vertex alone is insufficient; forward and backward automaton
states must be compatible with one complete accepted word.

This is the same lesson as symmetry-frame meetings: equal projection does not
prove that hidden semantic state joins.

## GPU and multi-GPU conceptual effects

Product state changes the work identity:

- frontier width counts reachable `(base,q)` pairs;
- duplicate detection must compare both components;
- the base state payload may be shared/recomputed, while `q` remains semantic;
- an owner function based only on base state may colocate phases but must retain
  separate visited entries;
- ownership including `q` may distribute work differently but increase routing
  of repeated base payloads;
- goal detection needs both base target and accepting memory state;
- path records must retain the transition label and automaton predecessor.

Two records with one base key and different `q` are not cross-GPU duplicates.
Pre-communication dedup that drops the memory component destroys correctness.

## Counterexamples and rejected shortcuts

- **The puzzle configuration is always the complete BFS state.** False when
  legal suffixes depend on history, phase, or control state.
- **Reaching a base vertex once dominates every later arrival.** False when the
  residual accepted language differs.
- **Product BFS returns a simple base path.** It returns a simple product path,
  whose base projection may repeat vertices.
- **All last-move pruning is harmless.** Only specifically proved reductions,
  such as exact immediate inverse cancellation for unit geodesics, inherit that
  guarantee.
- **A forward DFA can be run backward deterministically.** Reverse transitions
  may have multiple predecessors.
- **Same base owner/key means duplicate.** Product memory state is part of exact
  identity.

## Audit checklist

1. Does future legality depend only on the current base state?
2. Is history a semantic constraint or a proved geodesic-only pruning rule?
3. What finite memory/automaton state makes the process Markovian?
4. Are epsilon/zero-consumption transitions charged correctly?
5. Is the requested object an accepted walk or a simple base path?
6. Can the same base state with two memory states enable different suffixes?
7. Does immediate-inverse pruning truly return to the concrete parent?
8. What product identity/rank/owner key is exact?
9. For bidirectional search, what forward/backward memory states are compatible?
10. Does target acceptance test both the base state and memory state?

## Sources

- Yanhong A. Liu and Scott D. Stoller, *Solving Regular Path Queries*, MPC
  2002, [author page](https://www3.cs.stonybrook.edu/~liu/papers/RegPathQ-MPC02.html),
  for graph/finite-automaton fixed-point and product-state reasoning.
- Chris Barrett, Riko Jacob, and Madhav Marathe,
  *Formal-Language-Constrained Path Problems*, SIAM Journal on Computing 30(3),
  [DOI](https://doi.org/10.1137/S0097539798337716), for the distinction between
  constrained shortest walks/paths and harder simple-path requirements.
- Omer Angel, Joel Friedman, and Shlomo Hoory,
  *The Non-Backtracking Spectrum of the Universal Cover of a Graph*,
  [arXiv](https://arxiv.org/abs/0712.0192), for directed-edge-state
  non-backtracking walks.
- Notes 06, 12, 16, 17, and 19 supply implicit-state identity, zero-cost
  boundaries, action/inverse, quotient lifting, and shortlex contracts.

## Current synthesis

When future moves depend on history, BFS does not become history-aware by adding
metadata to a base vertex; the mathematical vertex becomes a product state.
Exact visited must distinguish every memory state that changes the future.
Immediate inverse pruning is a special proved shortcut for unconstrained unit
geodesics, not a license for arbitrary last-move filters. On Cayley search, the
product computes shortest accepted words, which can have different geometry,
identity, ownership, and reconstruction requirements from the ordinary group
word metric.
