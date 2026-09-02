# BFS, AND/OR reachability games, attractors, and ranks

Ordinary BFS answers existential reachability: a vertex is useful when at least
one outgoing choice leads toward the target. In a two-player reachability game,
some vertices belong to the reachability player and others to an adversary. At
an adversary vertex, every available response must remain winning.

The resulting layers resemble reverse BFS but use mixed existential and
universal predecessors. They compute an attractor least fixed point and a
worst-case forced-reachability rank, not ordinary shortest-path distance.

This note adds no implementation, optimizer, benchmark, or GPU code.

## 1. Finite reachability-game contract

Let a finite directed arena have vertices partitioned into

```text
V_exists  : reachability player chooses the successor
V_forall  : adversary chooses the successor,
```

with target set `T`. First assume every nonterminal vertex has at least one
successor. The reachability player wins when play visits `T`; an infinite play
that avoids `T` is losing for that player.

This ownership is semantic state. The same underlying directed graph with a
different ownership partition is a different search/game problem.

## 2. Mixed predecessor operator

For `X subset V`, define

```text
Pre_exists(X) = {v in V_exists : exists w, v->w and w in X}
Pre_forall(X) = {v in V_forall : every w with v->w lies in X}.
```

Starting with

```text
A_0 = T,
A_(k+1) = A_k union Pre_exists(A_k) union Pre_forall(A_k),
```

the attractor is

```text
Attr(T) = union_(k>=0) A_k.
```

On a finite graph the increasing sequence stabilizes. This is the least fixed
point of `X -> T union Pre_exists(X) union Pre_forall(X)`.

## 3. Meaning of the layers

Inductively, `A_k` is exactly the set of vertices from which the reachability
player can force a visit to `T` within at most `k` moves. Therefore the rank

```text
rho(v) = min{k : v in A_k}
```

is a worst-case guaranteed arrival bound, not the length of one favorable path.

For finite ranks outside `T`,

```text
rho(v) = 1 + min_(v->w) rho(w)   for v in V_exists,
rho(v) = 1 + max_(v->w) rho(w)   for v in V_forall,
```

where the universal expression is finite only when every successor is winning.

## 4. Ordinary reverse BFS is the all-existential case

If every vertex belongs to `V_exists`, the recurrence reduces to reverse
multi-source BFS from `T`, and `rho(v)=d(v,T)`.

If universal vertices exist, ordinary reverse BFS replaces their `all`
condition by `exists`. It computes vertices having some path to `T`, which is
an over-approximation of the winning attractor:

```text
Attr(T) subset {v : some path v->T exists}.
```

Equality needs an additional theorem or the absence of adversarial choices.

## 5. Minimal counterexample

Let universal vertex `u` have two successors:

```text
u -> t       with t in T
u -> c       with c -> c.
```

Ordinary BFS reports `d(u,T)=1`. But the adversary chooses `c` and stays there
forever, so `u` is outside the attractor.

If ownership of `u` changes to existential, the reachability player chooses
`t` and `rho(u)=1`. The edges and ordinary BFS distance are unchanged; only
the quantifier changed.

## 6. Least fixed point prevents cyclic self-support

Consider universal `u` with successors `t in T` and `u` itself. The equation
"all successors are winning" cannot use `u`'s desired status as its own proof.
Starting from `A_0=T`, `u` never enters because one successor remains outside
every current approximation. Operationally, the adversary can select the
self-loop forever.

The least fixed point rejects such circular justification. Solving only the
set equation with an arbitrary or greatest fixed point would answer a different
property.

## 7. Positional strategies from ranks

At an existential vertex of finite rank, choose a successor of strictly lower
rank. At a universal vertex of rank `k`, every successor has rank at most
`k-1`. Thus every play consistent with the chosen existential strategy reduces
rank on every move and reaches `T` within the starting rank.

No history beyond the current vertex is needed: finite reachability games are
positionally determined. Outside the attractor:

- every existential vertex has no successor inside the attractor;
- every universal vertex has at least one successor outside it.

The adversary can therefore keep play in the complement forever with a
positional choice. The complement is a trap for the reachability player.

## 8. Dead ends require a convention

Universal quantification over an empty successor set is vacuously true, while
existential quantification is false. This matches the common normal-play rule
that the player unable to move loses:

- an existential dead end outside `T` loses for the reachability player;
- a universal dead end outside `T` wins for the reachability player because the
  adversary cannot move.

Other systems may treat deadlock as failure, success, or an explicit terminal
label regardless of owner. Then totalize the arena or modify the target and
predecessor rules explicitly. Silent vacuous truth can otherwise reverse a
result. Under player-to-move-loses semantics, rank means moves to a winning
terminal (target or adversary deadlock); to retain the literal "visit `T`"
interpretation, totalize the deadlock as an explicit winning target state.

## 9. Frontier and finalization semantics

The new attractor frontier is

```text
Delta_k = A_k \ A_(k-1).
```

An existential predecessor becomes winning after one successor is proved
winning. A universal predecessor becomes winning only after all authoritative
successors are proved winning. Consequently, ordinary first-discovery visited
semantics apply to the growing winning set, but the trigger for discovery is
owner-dependent.

For a universal vertex, one observed winning edge is not enough. Its outdegree
and complete successor set are part of the proof certificate.

## 10. Missing and spurious transitions

Successor errors have asymmetric effects:

- missing an existential winning edge can create a false negative;
- adding a spurious existential winning edge can create a false positive;
- missing a universal losing edge can create a false positive;
- adding a spurious universal losing edge can create a false negative.

This sharpens note 55's successor-completeness ladder: universal finalization
requires evidence that no unprocessed losing response exists.

## 11. Strategies and witness objects

An ordinary shortest path is one sequence of vertices. A reachability-game
solution is a strategy plus its guarantee against every adversary response.

For existential vertices, one selected lower-rank edge can witness the strategy.
For universal vertices, the certificate needs all outgoing edges to lead to
lower ranks. A single favorable path through a universal vertex is not a
winning-strategy witness.

Counting paths is also not counting strategies. Different strategy choices can
share paths, and one strategy represents a branching tree of possible plays.

## 12. Relation to alternating automata and AND/OR graphs

An existential automaton/configuration transition behaves like an OR node;
acceptance needs one successful child. A universal transition behaves like an
AND node; all spawned obligations must succeed. Alternating acceptance can
therefore use the same least-fixed-point intuition for finite-word
reachability, subject to its precise run-tree and terminal conventions.

Collapsing an AND node to an ordinary edge set and running BFS changes `all`
into `exists`. Subset construction for an NFA is still existential over runs;
it is not automatically the same as an alternating universal branch.

## 13. Cayley and puzzle-game boundary

Ordinary Cayley BFS assumes the solver chooses each generator, so every state is
existential. Introducing adversarial moves, uncontrollable perturbations, or an
opponent turn creates a game state such as

```text
(group/orbit state, player-to-move, control phase).
```

The result is a forced-reachability rank in that game, not the original Cayley
word metric. Translation symmetry survives only if ownership, allowed moves,
target semantics, and control phase transform compatibly. A fixed target or
asymmetric player rules can destroy the one-root diameter shortcut.

A puzzle statement saying "find a move sequence" is existential. Replacing it
by "succeed for every opponent response" is a materially different problem,
not a more robust implementation of the same BFS.

## 14. GPU and multi-GPU boundary

Any measurement should separate:

- arena construction, ownership, and dead-end convention;
- ordinary reverse-reachable set and winning attractor;
- attractor frontier sizes and rank distribution;
- existential witnesses and universal all-successor certificates;
- complete outdegree/successor validation;
- positional strategy extraction and validation;
- per-device partial evidence and globally finalized vertices;
- communication, reductions, fixed-point rounds, and end-to-end time.

For a universal vertex whose successors are distributed across owners, local
completion is not global completion. Finalization requires a consistent graph
epoch and evidence covering every authoritative successor. For an existential
vertex, duplicate winning witnesses may be reduced without changing boolean
winning status, but witness choice and rank still need a declared policy.

Parallelizing one attractor round, processing independent predecessor updates,
and solving separate games are different workloads.

## Sources

- R. Mazala, *Infinite Games*, in E. Gradel, W. Thomas, and T. Wilke (eds.),
  [*Automata, Logics, and Infinite Games*](https://doi.org/10.1007/3-540-36387-4),
  LNCS 2500, 2002, for reachability games, attractors, traps, and positional
  determinacy.
- N. Fijalkow (ed.),
  [*Games on Graphs*, Chapter 1](https://doi.org/10.1017/9781009500678.002),
  Cambridge University Press, 2026, for the attractor least fixed point and
  rank-based positional strategies.
- Notes 13, 20, 22, 37, 42, 48, 52, 55, 56, 57, 75, 128, and 130 supply this
  repository's multi-source, product, dynamic, contract, bounded, separator,
  visited, successor, distributed-completion, output, direction, bisimulation,
  and subset-state boundaries.

## Takeaway

Ordinary BFS is the all-existential special case of reachability-game
attraction. With adversarial vertices, winning layers use `exists` for the
solver and `all` for the opponent; their rank is a worst-case forced-arrival
bound. The least fixed point prevents cycles from proving themselves winning,
and universal finalization requires complete successor evidence. A favorable
shortest path is not a winning strategy.
