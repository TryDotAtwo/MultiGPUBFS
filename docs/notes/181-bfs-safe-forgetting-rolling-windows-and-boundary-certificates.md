# BFS safe forgetting, rolling windows, and boundary certificates

## 1. Visited is information, not necessarily one permanent table

Permanent `visited=B_d` is the simplest certificate that a candidate has an
already established distance. It is not the only possible certificate. Exact
BFS may forget a closed state when all future computation that could confuse it
with a new state is either impossible or represented by equivalent retained
boundary information.

The safe question is not

```text
has x already been expanded?
```

but

```text
can any future admissible occurrence require distinguishing x from unseen,
and if so, where is that distinction retained?
```

Forgetting is therefore relative to graph orientation, schedule, output,
failure model, and retained certificates.

## 2. Exact rolling window for undirected BFS

In an undirected unit graph, every edge `{u,v}` satisfies

```text
|dist(s,u)-dist(s,v)| <= 1.
```

While expanding the complete frontier `F_d`, every neighbor lies in exactly one
of

```text
F_(d-1), F_d, F_(d+1).
```

Hence scalar distance/reachability BFS does not need layers `F_0,...,F_(d-2)`
for future duplicate rejection. A sufficient rolling state is:

```text
previous = F_(d-1),
current  = F_d,
next     = partially/fully constructed F_(d+1).
```

After `F_(d+1)` is exactly closed and publication obligations retire, rotate
the roles and reclaim `F_(d-1)`.

This theorem permits triangles, self-loops, parallel edges, odd/even cycles,
and arbitrary undirected degree. Bipartiteness merely makes `F_d--F_d` edges
impossible; it is not required for the three-layer result.

### Why all three membership roles can matter

Under the generic contract “enumerate every neighbor, retain only one selected
parent, and construct each unique next state once,” none of the three roles can
be discarded merely because the graph is undirected:

- **previous:** in the diamond `s--a, s--b, a--x, b--x`, let `a` be the selected
  parent of `x`. When `F_2={x}` is expanded, parent exclusion blocks `a` but not
  the other old predecessor `b`. Membership in `F_1` (or equivalent
  information) prevents accepting `b` at depth three;
- **current:** in the triangle `s--a--b--s`, expanding `a` in
  `F_1={a,b}` generates `b`. Current-layer membership prevents accepting that
  same-layer neighbor into `F_2`;
- **building next:** in the diamond, expanding `a` and `b` at depth one produces
  two occurrences of `x`. Next-layer reconciliation makes `x` one state.

These examples prove role-wise information necessity for this generic
interface, not three separately materialized hash tables or a universal memory
lower bound. A bitmap may encode the roles with epochs; a tree certificate can
remove convergence; retaining all shortest predecessors can change which
previous facts are locally available. Any such replacement still owes the
corresponding rejection decision.

### Bipartite specialization: current is work, not a candidate filter

If the reachable undirected component is bipartite, every edge changes BFS
depth parity and therefore joins `F_d` to `F_(d-1)` or `F_(d+1)`. There are no
same-layer edges. Consequently the current frontier must still be retained or
streamed as the work being expanded, but generated endpoints need not be
tested against `F_d` for scalar novelty. Previous-layer rejection and
building-next reconciliation remain separate obligations.

For one connected undirected component the converse also holds: if one exact
rooted BFS has no edge inside any layer, coloring vertices by depth parity is a
valid bipartition. Thus “current membership is never needed as a candidate
filter” is not an accidental queue property; under this interface it exposes
the component's bipartite geometry.

In an inverse-closed Cayley graph, a homomorphism to `Z_2` that maps every
generator to one supplies this parity structurally. Adding a unit-cost
generator of even old word parity can destroy it, introduce same-layer
occurrences, and restore the need for current-layer filtering even though the
state universe is unchanged.

## 3. What the rolling theorem does not preserve

Dropping old layers can preserve future scalar novelty decisions while losing:

- the final reached set as an explicit output;
- random queries `dist(s,x)` for reclaimed `x`;
- parent chains and direct path reconstruction;
- the complete shortest-path DAG and path counts;
- canonical words/parents;
- replay-free checkpoint restart from arbitrary depth;
- later dynamic-graph or new-source updates.

Those outputs need retained metadata, external storage, recomputation, or a new
search. “Exact BFS” must name which of them is requested.

## 4. Directed counterexample: an old state returns

Take

```text
s -> a -> b -> c,
c -> s.
```

At depth three, expanding `c` generates source `s` from depth zero. A fixed
three-layer window that forgot `s` can accept it again at depth four, and the
cycle repeats. Permanent visited or another exact old-membership certificate is
needed.

Cycles are not the only issue. The directed acyclic graph

```text
s -> x,
s -> a_1 -> a_2 -> ... -> a_k,
a_k -> x
```

has `dist(x)=1`, yet expanding depth-`k` state `a_k` regenerates `x`. This is
acyclic when `x` has no path back to the chain, and the backward jump `k-1` is
arbitrarily large. Therefore “the graph is a DAG” does not imply a bounded
backward span under BFS depth. A topological schedule/certificate may help, but
it is a different invariant.

## 5. Backward radial span

For the reachable directed graph and fixed source, define the extended-natural
backward span

```text
beta = sup({0} union
           {dist(s,u)-dist(s,v) : (u,v) is a reachable edge})
     in N union {infinity}.
```

Including zero handles graphs with only forward edges; the supremum handles
unbounded jumps on infinite graphs. Only positive backward jumps matter. If a
proved finite bound `beta <= L` is available,
then while expanding `F_d`, an old destination can lie no earlier than
`F_(max(0,d-L))`. Retaining exact membership for

```text
F_(max(0,d-L)),...,F_d and the building next frontier
```

is sufficient for scalar duplicate rejection under a strict complete-layer
schedule.

This statement is sufficient, not automatically memory-optimal. Computing the
exact `beta` from already known BFS distances can be circular as a prior design
tool; a useful rolling scheme needs a structural finite bound known
independently or validated for the declared graph family.

### Necessity for the pure layer-window interface

Under the narrower interface

```text
enumerate every outgoing edge,
classify an endpoint as old only by membership in the retained BFS layers,
otherwise admit it to the next-layer reconciliation,
```

the same condition is also necessary. If a reachable edge `(u,v)` has

```text
dist(s,u)-dist(s,v) > L,
```

then when `u` is expanded, `v` lies earlier than the oldest retained layer.
The window has no remaining fact that distinguishes this occurrence of `v`
from an unseen state, so it admits `v` again. Therefore a pure exact
`L`-backward-layer filter exists for this fixed rooted graph exactly when
`beta<=L`.

This necessity claim is deliberately interface-relative. It does not rule out
safe forgetting with a different certificate: an operator mask may suppress
the offending edge, a topological certificate may establish that its endpoint
is old, or a retained global summary may answer the same membership question.
Such mechanisms do not refute the missing-layer argument; they preserve the
lost distinction somewhere else.

There is also a quantifier distinction:

- measuring `beta` after a complete BFS characterizes one fixed rooted graph;
- proving a uniform `beta<=L` for a graph family justifies choosing the window
  before traversal;
- observing a small `beta` on sampled instances proves neither the family
  bound nor safety on the next instance.

### Per-layer last-reference depth

A uniform window hides a more exact liveness schedule. For each nonempty BFS
layer `F_j`, define

```text
tau_j = sup({j} union
            {dist(s,u) : (u,v) is a reachable edge and v is in F_j}).
```

The explicit `j` keeps the layer alive through its own expansion even if it has
no incoming edge from an equal or later depth. Under the pure layer-membership
interface, `F_j` may be reclaimed immediately after `F_(tau_j)` has been fully
expanded and all candidates from that expansion have become quiescent.

This is exact for layer-granular reclamation:

- **necessity:** before that point, an unexpanded edge from some layer at most
  `tau_j` can still produce a vertex in `F_j`; without its membership the old
  endpoint can be accepted again;
- **sufficiency:** after that point, no future expansion has an outgoing edge
  into `F_j`, so scalar novelty will never query that layer again.

The backward span is precisely the worst excess lifetime:

```text
beta = sup_j (tau_j-j).
```

Thus a fixed `L`-window replaces the individual facts
`tau_j<=j+L_j` by one uniform envelope `L=sup_j L_j`. In an undirected graph,
`tau_j<=j+1` for every layer. In the directed cycle
`s->a->b->c->s`, the source layer has `tau_0=3`; in the long directed DAG
counterexample, the layer containing `x` has an arbitrarily late `tau_1` even
though no cycle exists.

This formulation also exposes why current-frontier evidence alone is not a
reclamation certificate. Observing that `F_d` does not reference an old layer
says nothing about edges from `F_(d+1),F_(d+2),...`. Online early reclamation
needs a structural upper bound on each future `tau_j`, a permanent substitute
certificate, or predecessor information proving that all possible sources of
the layer have already been exhausted. Computing exact `tau_j` from the fully
labeled graph after BFS is descriptive but circular as a memory-saving policy.

For multi-GPU execution, graph-theoretic last use and physical last use are
separate. Even after depth `tau_j` is logically complete, the layer remains live
until every candidate, retry, remote message, and publication from that depth
is globally retired.

## 6. Cayley inverse-length bound

Consider right multiplication by generators `S` and directed word distance
`d(x)` from identity. For edge

```text
x -> xg,  g in S,
```

suppose `g^(-1)` has an `S`-word of length at most `L_g`. Then

```text
d(x) <= d(xg) + L_g,
```

because a shortest word for `xg` followed by that inverse word reaches `x`.
Therefore

```text
d(x)-d(xg) <= L_g,
beta <= max_g L_g.
```

If `L_g` is chosen as the exact positive-word distance
`d(e,g^(-1))`, this Cayley bound is always attained. Set `x=g^(-1)`.
The generator edge

```text
g^(-1) -> e
```

has radial drop

```text
d(e,g^(-1)) - d(e,e) = L_g.
```

Taking the largest generator-inverse distance gives the exact identity

```text
beta = max_(g in S) d_S(e,g^(-1))
```

for the directed right Cayley graph rooted at the identity. Thus the inverse
length is not merely a convenient upper bound in the full Cayley graph: it is
the minimum exact backward-window length for the pure layer-membership
interface described above.

The equality can contract after quotienting to a Schreier action because
distinct group elements may represent one state. For example, `Z_6` with
positive generator `+1` has Cayley inverse length and backward span five.
Acting on cosets of the subgroup `{0,3}` gives the directed three-cycle, whose
backward span is two. The group-level inverse word `(+1)^5` still reverses every
action edge, but the quotient admits the shorter state-level word `(+1)^2`.
Therefore the group bound remains safe for the action while need not be tight.

Consequences:

- for a symmetric generator set, every inverse is one generator and `beta<=1`,
  reproducing the undirected three-layer window;
- for a nonsymmetric but inverse-generating finite group alphabet, a larger
  finite rolling window may be proved from inverse word lengths;
- if some inverse is not expressible in the directed generator monoid, no such
  finite bound follows from this argument;
- a Schreier action can only reduce concrete endpoint distinctions, but the
  rolling proof must use the actual directed state graph/output identity.

The exact Cayley identity and the safe Schreier upper bound concern old-state
rejection, not whether the resulting window is cheaper than permanent visited.

### Three Schreier quantities that must not be conflated

For a right action and a labeled edge `x -> x s`, define its shortest local
return length

```text
rho(x,s) = dist_S(x s, x).
```

Let

```text
R_action = max_(reachable x, s in S) rho(x,s),
L_group  = max_(s in S) d_S(e,s^(-1)).
```

Then, for the backward radial span from the declared base state,

```text
beta_root <= R_action <= L_group.
```

The first inequality is the directed triangle inequality:

```text
dist(root,x) <= dist(root,xs) + dist(xs,x).
```

The second holds because any positive word representing `s^(-1)` returns
`x s` to `x` at every action state. Thus `R_action` can improve the safe
group-level window without first knowing the root-distance layers, but it is
still only an upper bound on their actual radial drops.

At `x=x_0 a`, with current stabilizer `K=a^(-1) H a`, the local return has the
coset expression

```text
rho(x,s) = min { |w|_S : w in s^(-1) K }.
```

Indeed, `x s w=x` iff `s w` lies in `K`. This makes the action-level
certificate state-dependent through conjugate stabilizers even though every
state applies the same generator collection.

Both inequalities can be strict, as two existing hand-checkable actions show:

- `Z_6`, `S={+1}`, acting on cosets of `{0,3}` has
  `beta_root=R_action=2 < L_group=5`;
- in the three-point `S_3` action of note 158 with
  `S={(12),(13),(123)}` and root `1`, all nonroot states have depth one, hence
  `beta_root=1`. But the edge `2 -> 3` labeled `(123)` needs two positive moves
  to return (`3 -> 1 -> 2`), so `R_action=2`; the group inverse of `(123)` also
  has positive length two, giving `beta_root=1 < R_action=L_group=2`.

Consequently a uniform local-return certificate can be structurally sharper
than the group inverse bound and still wider than the minimum root-specific
rolling window. In the free Cayley action all stabilizers are trivial, so all
three quantities collapse to the exact identity above.

### Vertex transitivity makes the local-return bound exact

Let the directed support graph be strongly connected and vertex-transitive,
and let

```text
R = max_(edge u->v) dist(v,u).
```

For every root, its backward radial span equals `R`. The inequality
`beta_root<=R` is the triangle argument above. For the reverse inequality,
choose an edge `u->v` attaining `R` and a directed-graph automorphism `phi`
with `phi(v)=root`. It maps the edge to

```text
phi(u) -> root
```

and preserves directed distance, so

```text
dist(root,phi(u))
  = dist(phi(v),phi(u))
  = dist(v,u)
  = R.
```

That edge has radial drop exactly `R`. Hence

```text
beta_root = R_action
```

whenever the fixed directed support graph is vertex-transitive. Label-preserving
transitivity is unnecessary for scalar novelty; an automorphism may permute
generator labels as long as it preserves directed support arcs.

Two useful Schreier sufficient conditions are:

- if the base stabilizer `H` is normal, `H\G` is the quotient group and the
  support graph is the Cayley graph of the generator images. Then
  `beta_root=R_action` is exactly the maximum inverse distance in the quotient;
- if the fixed generator collection is invariant under the conjugations used
  by the action, maps `x -> x a` preserve support arcs (while possibly
  relabeling them) and act transitively, again giving `beta_root=R_action`.

Transitivity of the underlying group action alone is insufficient: the fixed
generator set need not make the directed support graph vertex-transitive. The
three-point `S_3` witness above has a transitive action but
`beta_root<R_action`, precisely because its fixed `S` is not
conjugation-invariant and its support profiles differ by state.

## 7. Boundary certificates instead of closed membership

Frontier search deletes Closed nodes but retains, on the active boundary, which
operators/incident transitions have already been used. The “solid boundary”
prevents a future expansion from regenerating forgotten interior states.

This reveals a conservation principle:

```text
forgotten interior membership
is replaced by enough boundary-transition information to block re-entry.
```

The information is moved, not made unnecessary. Applicability depends on being
able to enumerate the relevant neighbors/predecessors and merge used-operator
metadata exactly. Directed graphs can require dummy predecessor information;
large operator alphabets can make boundary metadata expensive.

## 8. Reconstruction is a separate trade

If old parents are deleted, one shortest path may be reconstructed by
divide-and-conquer searches through an intermediate optimal state. This trades
retained memory for repeated search work. It does not recover:

- the original arbitrary parent choices;
- every shortest path;
- the original parallel schedule/order;
- canonical shortlex output without repeating its closure rules.

Thus memory reduction for scalar optimum and preservation of a rich traversal
artifact are different claims.

## 9. Delayed duplicate detection does not imply forgetting

Delayed duplicate detection batches candidates, sorts/partitions them, and
performs exact duplicate/old-state removal before expansion. It changes **when**
membership is consulted. It does not by itself prove that old membership may be
discarded.

External-memory visited files can retain the whole closed set without random
RAM access. Frontier search can discard closed states by a boundary theorem.
These mechanisms solve different memory/I/O problems and may be combined.

## 10. Multi-GPU reclamation boundary

A rolling layer or boundary descriptor cannot be reclaimed merely because one
GPU advanced its local depth. Safe reclamation requires a consistent cut proving:

- every expansion that can reference the retiring layer completed;
- all remote candidates/retries from those expansions were delivered;
- no device buffer, kernel, collective, or spill can still publish such work;
- every owner closed same-layer and next-layer duplicate decisions;
- checkpoints and replicas no longer rely on the retired epoch;
- output metadata that needs the layer was persisted or deliberately waived.

Otherwise a delayed old candidate can arrive after its membership window was
reused and be mistaken for a new state. Layer-bit recycling is therefore an
epoch protocol as well as a graph theorem.

## 11. Capacity and performance interpretation

Rolling windows can change memory from ball volume

```text
sum_(i<=d) |F_i|
```

to a bounded number of recent/boundary layers. But total cost may move into:

- used-operator metadata;
- additional neighbor/predecessor generation;
- sorting/merging;
- reconstruction searches;
- stronger global reclamation barriers;
- checkpoint/replay complexity.

Peak frontier width can still dominate capacity. No universal speed or memory
winner follows from safe forgetfulness alone.

### Three layers need not be a small fraction of the ball

Compare distinct membership states at one declared instant: `F_(d+1)` is fully
constructed, but the old previous layer `F_(d-1)` has not yet been reclaimed.
Ignore storage format, scratch buffers, and duplicated worklist copies. Then

```text
permanent: B_(d+1)
rolling:   F_(d-1) union F_d union F_(d+1).
```

On an undirected path rooted at an endpoint, with sufficient remaining length,
each layer has one state. The ratio is `3/(d+2)` and tends to zero.

On a rooted undirected b-ary tree, before its leaves, `|F_j|=b^j` for `b>1`.
For `d>=1`, summing the geometric series gives the exact ratio

```text
rolling / permanent
  = (b^(d+2) - b^(d-1)) / (b^(d+2) - 1)
  -> 1 - b^(-3).
```

For a binary tree through depth seven, the three retained layers have
`32+64+128=224` states out of `1+2+4+8+16+32+64+128=255`.
As depth grows, this three-role snapshot retains 87.5 percent of the ball, not
a vanishing fraction. The tree is only a hand-checkable growth example; its
separate parent-only certificate can avoid generic visited membership, so this
is not a lower bound on the best tree traversal.

The intuition is that rolling storage saves *old volume*, not a number of
depth indices. When most volume is already near the boundary, forgetting many
old layers saves little of that volume. These are state-count comparisons at
a matching phase, not byte counts, peak-allocation measurements, GPU speedups,
or predictions for puzzle layers whose growth may later saturate or contract.

### Retained Megaminx evidence at the same boundary

The saved 2026-08-28 REF-026/027/028 outputs give layer sizes
`1, 24, 408, 6208, 90144` through depth four for their declared 24-move action.
At the matching instant with `F_4` constructed and `F_2` not yet reclaimed:

```text
permanent B_4:       1 + 24 + 408 + 6208 + 90144 = 96785 states
three layer roles:           408 + 6208 + 90144 = 96760 states
old membership removed:                              25 states
```

This is a hand calculation from retained sphere counts, not a new run or an
observed rolling implementation. It concretely illustrates that shallow, rapid
growth can leave almost nothing to save by discarding old membership alone.
Other phases, deeper layers, metadata, scratch storage, and actual allocations
are not covered by these numbers.

Evidence: `experiments/REF-026-megaminx-depth2-relation-signatures.txt`,
`experiments/REF-027-megaminx-f3-commutation-classes.txt`, and
`experiments/REF-028-megaminx-f4-conjugated-commutators.txt`.

**Cross-artifact compatibility check (2026-08-31):** All three accompanying
reports record input SHA-256
`1780a8368d504fd75f448d25e5bede9adb498b35db6a3251e920bbc8524adfca`.
The inspected probe sources all include `ref025_megaminx_contract_probe.rs`,
take `config.central` and its generator list, and apply permutations by the
same gather `new[j]=state[permutation[j]]`. They retain full vector equality,
not equality of hash fingerprints. REF-026 records 120 distinct position IDs
and 12 face turns with their inverses. These facts support joining the saved
counts for that declared action. They do not freshly authenticate historical
executables or input bytes: no checksum was recomputed and no run was repeated.
The common parser is also shared dependency evidence, not an independent
validation of the physical puzzle's move semantics.

## 12. Rejected implications

- Exact BFS always requires permanent storage of every visited state.
- Undirected cycles invalidate a three-layer rolling BFS.
- Bipartiteness is necessary for bounded visited-layer recycling.
- A directed DAG cannot regenerate an old BFS layer.
- A small graph diameter implies a small rolling-window footprint.
- Delayed duplicate detection proves old visited data can be deleted.
- Deleting Closed preserves parent/path/DAG/count outputs automatically.
- Frontier search stores no information about the forgotten interior.
- One GPU's level completion authorizes global layer-bit reuse.
- A symmetric Cayley generator set needs permanent visited solely because it has
  relations/cycles.

## 13. Evidence boundary and next question

The undirected rolling-window theorem, directed/DAG counterexamples, and Cayley
inverse-length bound are conceptual proofs. The necessity result above is exact
only for the stated pure layer-membership interface. No runtime or memory
measurement is claimed. The inverse-word expression is exact for a full Cayley
graph. For a Schreier action, local return lengths give a possibly sharper
structural certificate, but neither certificate must equal the root-specific
backward span without additional symmetry. Vertex transitivity makes the local
return certificate exact; normal stabilizers and conjugation-invariant move
sets are useful sufficient conditions. A useful next conceptual question is
how structural predecessor information can bound per-layer last-reference
depths without materializing the full reverse graph. Any new executable gate
remains out of scope until the user requests an experiment explicitly.

## Sources

- Richard E. Korf, Weixiong Zhang, Ignacio Thayer, and Heath Hohwald,
  *Frontier Search*, Journal of the ACM 52(5), 2005, 715--748,
  DOI 10.1145/1089023.1089024; accessible copy:
  <https://www.researchgate.net/publication/220430854_Frontier_search>.
- Richard E. Korf, *Delayed Duplicate Detection*, extended abstract:
  <https://citeseerx.ist.psu.edu/document?doi=83da3f4313f067d3c89e28b1191722166f605978&repid=rep1&type=pdf>.
- Rong Zhou and Eric A. Hansen, *Breadth-First Heuristic Search*, Artificial
  Intelligence 170 (2006), 385--408:
  <https://ai.dmi.unibas.ch/research/reading_group/zhou-hansen-aij2006.pdf>.
