# Partial-order reduction: reachability, distance, and word loss

## Question

If two actions commute, may BFS keep only one execution order and still claim
the original shortest-path distance?

## Short answer

Commutation alone is not a pruning theorem. It says that two *already valid*
orders can have the same endpoint. To delete outgoing edges safely, a reduction
must also guarantee that every relevant omitted path has a retained
representative.

If that representative is only a permutation of the same unit actions, its
length is unchanged. Therefore an action-preserving reduction preserves the
length of at least one shortest goal path. It need not preserve distances to
every state, every shortest word, the number of shortest paths, or a chosen
canonical word.

## The three different claims

Let `G_r` be the subgraph obtained by expanding only selected actions. Because
`G_r` only deletes edges,

```text
d_G(s,v) <= d_Gr(s,v)
```

whenever `v` remains reachable. Three increasingly strong preservation claims
must not be merged:

1. **Goal reachability:** some goal remains reachable.
2. **Optimal goal cost:** some shortest goal-reaching word has an equally long
   retained representative.
3. **Full BFS metric:** every reachable vertex keeps its original distance.

The first does not imply the second. The second does not imply the third.

## Smallest counterexample: optimal goal distance is not a full BFS metric

Let independent actions `a` and `b` toggle two separate bits from `00`, and let
the goal predicate be “the first bit is one.” Suppose the reduced search keeps
only action `a` at the root:

```text
00 --a--> 10       goal at distance 1
00 --b--> 01       omitted root edge
```

The reduced graph preserves the optimal goal distance exactly. It does not
preserve the original distance to state `01`; that state may be absent or
reachable only by a longer route. Thus “optimality-preserving planning search”
is not automatically “BFS on the original graph with every distance intact.”

## Why a retained trace representative preserves length

For an exact independence relation, swapping adjacent independent actions

```text
u a b v  ->  u b a v
```

changes neither the multiset of action occurrences nor the word length. If a
reduction proves that every solution path `p` has a retained path `p'` that is
a permutation of `p`, then `|p'|=|p|`. Applying this to an original shortest
solution supplies a retained solution of the same length. Since edge deletion
cannot create a shorter solution, the optimum is equal.

This is the core of Xu et al.'s stubborn-set proof: the selected action is
commuted left across omitted actions, and induction retains a permutation of
the original solution sequence. Their optimality corollary is qualified by an
action-set-invariant preference. For unit-cost BFS, length is invariant under
permutation; order-sensitive costs or objectives need an additional argument.

**Source scope check (2026-08-31):** Definition 10 quantifies over solution
sequences from a non-goal state and requires a retained solution using a
permutation of the same actions. Its definition does not separately require
the same final state, only another goal-reaching solution. Therefore the
definition alone establishes neither all-target coverage nor endpoint
identity. The all-finite-path argument below is our direct theorem with an
explicit same-endpoint premise, not a quotation of Definition 10. A permutation
of arbitrary noncommuting actions is not itself an endpoint-equality proof.

## What is lost even in the good case

Keeping one representative per goal-reaching trace can preserve one shortest
goal witness while discarding:

- other shortest interleavings;
- labeled shortest-path counts;
- lexicographically or operationally canonical words;
- intermediate states that occur only in discarded linearizations;
- frontier sizes and duplicate-occurrence geometry of the original graph.

For two commuting moves, `ab` and `ba` reach the same final state at depth two,
but pass through different depth-one states. Retaining only the path `ab` can
retain that endpoint and omit the state reached by `b`. Merely deleting the
second edge of path `ba`, while retaining root edge `b`, does not omit that
state. The omitted path prefix is essential to the counterexample.

### Correction: coverage of all finite traces is stronger

The earlier version incorrectly extended that loss claim to one representative
per *every valid trace*, without restricting the covered path family. If a
retained subgraph contains a valid, equal-length, same-endpoint representative
for every finite path from `s`, then every source distance is preserved.

**Direct proof:** Choose a shortest original path to any reachable `v`. Its
retained representative gives `d_Gr(s,v) <= d_G(s,v)`. Since the retained graph
only deletes edges, the opposite inequality also holds. Thus the distances,
reachable set, and each distinct-state frontier `F_d` are identical. Coverage
of every shortest endpoint path is already sufficient; all finite traces is a
stronger assumption than necessary.

In the two-move example, singleton words `a` and `b` are themselves finite
traces. Covering them forces both depth-one endpoints to remain. One cannot
discard root edge `b` and still claim coverage of every finite trace.

Even this stronger contract does not preserve all shortest words, path counts,
canonical choices, or the number of generated transition occurrences. It does
preserve *distinct-state frontier sizes*. Consequently frontier cardinality and
frontier processing work must not be conflated. This is a mathematical
coverage condition, not proof that an arbitrary history-pruning algorithm with
state-only visited implements it.

## Why pairwise commutation is insufficient

A local fact `ab(s)=ba(s)` does not establish that either order remains enabled
after arbitrary prefixes, nor that every path to the goal crosses a selected
action. Stubborn/ample-set methods add conditions that prevent an omitted
action from being ignored forever and justify moving a selected action across
an omitted prefix. The exact provisos depend on the property being preserved:
reachability, safety, liveness, or optimal planning are not interchangeable.

This also separates two notions already present in the Cayley notes:

- global generator commutation gives a sound length-preserving word swap;
- state-dependent action commutation may hold only at particular states and
  requires enabledness checks along the actual prefix.

In a total Cayley action every generator is enabled, which removes one planning
complication, but does not supply the coverage theorem saying which outgoing
generator occurrences may be omitted while retaining every target of interest.

## GPU and multi-GPU interpretation

POR changes the searched transition graph. It may reduce generated occurrence
records, but dependency/enabledness selection can be irregular and
state-dependent. More importantly, comparing its frontier sizes or throughput
with ordinary BFS is not a same-graph optimization comparison unless the
reported contract is explicitly only the preserved property.

For exact all-state Cayley BFS, ordinary visited dedup already merges equal
endpoints after generating their occurrences. Trace pruning removes paths;
whether it also removes states depends on its coverage contract. An all-finite-
trace same-endpoint guarantee preserves source distances and state frontiers,
whereas a goal-only guarantee need not. Transition occurrence work can differ
in either case. Any performance comparison must identify the preserved output,
not infer a change of distances merely from a change in generated work.

## Sources

- You Xu, Yixin Chen, Qiang Lu, and Ruoyun Huang,
  [*Theory and Algorithms for Partial Order Based Reduction in Planning*](https://arxiv.org/abs/1106.5427),
  2011. Definition 12 makes the reduction a selected-action subgraph; Theorem 1
  proves action preservation by retaining a permutation of each solution
  sequence; Corollary 1 states completeness and qualified optimality
  preservation; Definitions 13--16 distinguish state-dependent and
  state-independent left commutativity.
  Author names were corrected against arXiv metadata on 2026-08-31; the earlier
  attribution to Fern and Yoon was erroneous. Definition 10 was checked in the
  [indexed full-paper copy](https://www.researchgate.net/publication/51913727_Theory_and_Algorithms_for_Partial_Order_Based_Reduction_in_Planning)
  when arXiv's experimental HTML failed to load.
- A. Mazurkiewicz,
  [*Concurrent Program Schemes and their Interpretations*](https://tidsskrift.dk/daimipb/article/view/7691),
  DAIMI PB-78, 1977. This is the trace/partial-order background already scoped
  in note 66.
- Notes 66 and 92 supply the local distinction between trace equality, endpoint
  equality, reachability equivalence, and preservation of the original unit
  metric.

## Takeaway

“Independent actions can be reordered” is an equivalence fact about paths.
“Some orders may be deleted” is a coverage theorem about a property. A
length-preserving representative theorem can preserve the shortest distance to
a goal set; an all-target version preserves the complete source distance map
and distinct-state frontiers. Neither guarantee alone preserves all shortest
words or original transition-occurrence work. The quantifier over covered
paths is the decisive distinction.
