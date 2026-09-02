# Reverse BFS, postdominators, and inevitable targets

Dominance looks forward from one entry. Postdominance looks backward from one
exit: which vertices must be encountered after a given state on every route
that reaches the exit? Reversing the graph makes the algebra familiar, but it
does not make reverse BFS, postdominance, and inevitable termination the same
question.

This note separates those semantics. It adds no postdominator or liveness
implementation.

## 1. Postdominance under a single-exit contract

Let a finite directed graph have a distinguished exit `z`, and initially
restrict attention to vertices from which `z` is reachable. A vertex `p`
**postdominates** `v` when every directed path from `v` to `z` contains `p`.

Under the reflexive convention, `v` postdominates itself and `z` postdominates
every vertex that can reach it. Each non-exit vertex has an immediate
postdominator when the reverse flow-graph assumptions hold, and these edges
form a postdominator tree rooted at `z`.

The restriction to exit-reachable vertices is essential. If `v` has no path to
`z`, the phrase "every `v`-to-`z` path" quantifies over an empty family and can
make naive set-theoretic definitions vacuously true. Compiler treatments avoid
this pathology by imposing or constructing an appropriate exit contract.

## 2. Exact reverse-graph duality

Let `G^R` reverse every arc of `G`. Then

```text
p postdominates v in G
  <=>
p dominates v in G^R rooted at z.
```

Every `v`-to-`z` path in `G` reverses to a `z`-to-`v` path in `G^R`, preserving
the set of vertices on it. Therefore all dominator distinctions from notes 89
and 90 transfer after reversing both graph direction and root/exit roles.

This duality requires the true reverse transition relation. An inverse move
that is algebraically imaginable but absent from the declared directed graph
cannot be inserted silently.

## 3. Reverse BFS is not a postdominator tree

BFS from `z` in `G^R` computes which vertices **can** reach `z` in `G` and their
minimum remaining distance. A reverse BFS parent chooses one shortest suffix
from each vertex to `z`.

It does not identify vertices present on every suffix. In

```text
v -> a -> z
v -> b -> z,
```

reverse BFS may choose `a` on the witness suffix, but `a` does not postdominate
`v`. This is exactly the parent-versus-dominator diamond with time reversed.

Likewise, an immediate postdominator edge need not be an original graph edge or
connect adjacent reverse-BFS layers.

## 4. Every shortest suffix versus every terminating suffix

Use

```text
v -> a -> z
v -> b -> c -> z.
```

Every shortest suffix from `v` to `z` contains `a`, but the longer suffix
through `b,c` avoids it. Thus `a` is a shortest-suffix gateway, not a
postdominator.

Restricting `G^R` to its reverse shortest-path DAG computes mandatory vertices
among shortest completions only. Full postdominance must retain every arc that
can participate in any terminating path, including detours and cycles.

## 5. Postdominance does not imply inevitable termination

Consider

```text
v -> v
v -> z.
```

Every finite path from `v` that reaches `z` contains `z`, so `z`
postdominates `v`. Yet an execution can traverse `v->v` forever and never reach
the exit.

Standard postdominance is termination-insensitive: it constrains paths that do
reach the exit. The liveness statement

```text
every maximal execution from v eventually reaches z
```

also quantifies over infinite paths and dead-ending finite paths. It is
strictly stronger and depends on the execution/fairness model.

For a finite graph with `z` made absorbing and adversarial edge choices, target
inevitability from `v` requires both:

- no reachable directed cycle avoiding `z`;
- no reachable dead end other than `z`.

A reachable avoiding cycle supports an infinite counterexecution. An avoiding
dead end supports a finite maximal counterexecution. If fairness assumptions
exclude some infinite choices, the criterion changes and fairness must be part
of the contract.

## 6. Multiple exits and a virtual exit

For real exits `z_1,...,z_k`, adding a fresh virtual exit `Z` with arcs

```text
z_i -> Z
```

creates a single-exit graph. A real vertex `p` postdominates `v` in this graph
when every terminating path from `v` to any real exit passes through `p`.

This construction changes the question deliberately. Computing separate
postdominator trees for each exit asks which vertices are mandatory conditional
on choosing that exit; the virtual-exit tree asks what is mandatory across all
exit choices. The virtual node itself is bookkeeping and should not be reported
as a semantic state.

Nonterminating branches remain outside the guarantee unless the analysis adds
an explicit nontermination sink or adopts a stronger maximal-path semantics.

## 7. A chosen BFS target is not automatically an exit

Ordinary target-search BFS stops when it has enough evidence about reaching a
goal. The underlying state graph may still contain outgoing transitions from
that goal. Treating the goal as a postdominator exit therefore requires an
explicit transformation, such as making first arrival at the goal absorbing.

Different contracts answer different questions:

- `can reach t`: reverse reachability;
- `minimum steps to t`: reverse BFS distance;
- `p is on every shortest completion`: reverse shortest-DAG gateway;
- `p is on every completion that reaches t`: postdominance after declaring
  `t` as exit;
- `every execution eventually reaches t`: maximal-path liveness.

None can be substituted for another merely because all use backward edges.

## 8. Cayley and product-state boundary

For a finite Cayley graph and selected goal `t`, reverse BFS uses the declared
inverse transition relation to compute minimum completion distance. If the
generator alphabet is asymmetric, the reverse alphabet may differ from the
forward one even though the underlying group has algebraic inverses.

In a one-way directed cycle, after declaring first arrival at `t` as exit, the
states between `v` and `t` in cyclic order postdominate `v`. In a symmetric
cycle, the opposite route removes those nontrivial postdominators. Direction
semantics again outweigh vertex transitivity.

If legality depends on history, postdominance belongs to `(state,history)`.
Projecting to the base Cayley state can merge one completion that must pass
through `p` with another that avoids it.

## 9. Distributed and GPU consequences

- A reverse-reachable bitmap is a may-reach certificate, not must-reach or
  inevitable-termination evidence.
- A reverse BFS parent is one shortest suffix, not a postdominator chain.
- Omitting a longer reverse arc can manufacture a false postdominator while
  preserving the minimum distance.
- A global quiescence detector proves that the search computation has no work;
  it does not prove that all paths in the searched graph terminate.
- Cycles spanning owners are semantic liveness counterexamples even if every
  owner locally has an exit-directed edge.
- Virtual exits and absorbing targets must have globally identical identity and
  transition semantics.

These observations constrain claims; they do not prescribe a kernel or
distributed protocol.

## 10. Evidence checklist

1. Unique real exit, multiple exits, virtual exit, or absorbing search target.
2. Exit-reachable vertex universe and treatment of non-exit-reachable states.
3. True reverse arcs versus algebraic inverses outside the graph contract.
4. One shortest suffix, all shortest suffixes, or all terminating suffixes.
5. Finite terminating paths versus all maximal finite/infinite executions.
6. Fairness assumptions for infinite choices.
7. Base state versus history-expanded product state.
8. Reverse reachability, postdominance, liveness, and search quiescence as
   separate certificates.

## Sources

- J. Ferrante, K. J. Ottenstein, and J. D. Warren,
  [*The Program Dependence Graph and Its Use in Optimization*](https://doi.org/10.1145/24039.24041),
  ACM TOPLAS 9(3), 1987, 319-349. Classical postdominance and control-dependence
  setting.
- T. Lengauer and R. E. Tarjan,
  [*A Fast Algorithm for Finding Dominators in a Flowgraph*](https://doi.org/10.1145/357062.357071),
  ACM TOPLAS 1(1), 1979, 121-141. Dominator-tree theorem applied to the reversed
  single-exit flow graph.
- Notes 09, 20, 28, 41, 56, 84, 89, and 90 provide infinite-graph,
  product-state, shortest-DAG, reverse-graph, termination, SCC, dominance, and
  separator context.

## Takeaway

Reverse BFS proves that an exit is reachable and measures the shortest suffix.
Postdominance proves what every exit-reaching suffix contains. Inevitable
termination additionally rules out infinite and dead-ending counterexecutions.
The three questions share a reversed graph but have different quantifiers and
different failure certificates.
