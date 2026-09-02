# BFS on Schreier graphs: stabilizer cosets and variable support degree

Note 157 identified same-parent aliases in a nonfree group action. Their exact
shape is controlled by cosets of the current state's stabilizer. This gives a
closed description of loops, parallel labeled successors, and the gap between
constant generator work and variable unique-neighbor work.

No experiment is used. A three-point action supplies a hand-checkable
counterexample.

## 1. Current-state stabilizer

Let a group `G` act transitively on states on the right, and let `S` be the
declared finite generator collection. At state `x`, write

```text
K = Stab(x) = {g in G | x g = x}.
```

For `s,t in S`,

```text
x s = x t
iff x s t^(-1) = x
iff s t^(-1) in K.
```

The last condition is equivalent to equality of right cosets:

```text
K s = K t.
```

Thus the generator labels reaching one endpoint are exactly the elements of `S`
inside one right coset of `K`. Here standard terminology is used: `Ks` is a
right coset, while `sK` is a left coset. The side of the state action and the
side in the coset name must not be inferred from one another.

## 2. Exact successor multiplicities

For every right coset `K t` that intersects `S`, define

```text
mu_x(Kt) = |S intersect Kt|.
```

Then:

- distinct successor states correspond to distinct intersected cosets `K s`;
- the labeled multiplicity of that successor is `mu_x(Ks)`;
- self-loop labels are exactly `S intersect K`;
- total labeled occurrence outdegree is `|S|`;
- total distinct support endpoints, including a possible loop endpoint, are

```text
u(x) = |{K s : s in S}|.
```

The same-parent endpoint excess over all successors is

```text
|S|-u(x) = sum_intersected_cosets (mu_x(C)-1).
```

If simple support degree excludes self-loops, subtract one from `u(x)` whenever
`S intersect K` is nonempty.

## 3. Stabilizers are conjugate, profiles need not be equal

Fix base state `x_0` with stabilizer `H`. At state `x_0 g`,

```text
Stab(x_0 g) = g^(-1) H g.
```

All point stabilizers in a transitive action are therefore conjugate and have
the same group order. But `S` may intersect their cosets differently. Equal
stabilizer size does not imply equal successor multiplicity profile.

This distinction is easy to miss: orbit symmetry belongs to the full group
action, while the fixed move collection `S` may not be preserved by every
conjugation.

## 4. When the profile is invariant

Suppose `S` is invariant under every conjugation relating states in the orbit;
in particular, a union of full conjugacy classes suffices. Conjugation maps

```text
K -> a^(-1) K a,
s -> a^(-1) s a
```

and preserves the equivalence `s t^(-1) in K`. It therefore bijects alias
classes with equal sizes and maps loop labels to loop labels.

Under that condition, every state has the same:

- number of distinct support endpoints;
- loop-label count;
- multiset of same-endpoint generator multiplicities.

Without generator-set conjugation invariance, none of these conclusions follows
from transitivity alone.

## 5. Three-point counterexample

Let `S_3` act on points `{1,2,3}` and use the fixed directed move collection

```text
S = {(12), (13), (123)}.
```

The three labeled successors are:

```text
from 1: 2, 3, 2   -> two distinct endpoints, multiplicities {2,1}
from 2: 1, 2, 3   -> three distinct endpoints, multiplicities {1,1,1}
from 3: 3, 1, 1   -> two distinct endpoints, multiplicities {2,1}
```

Every state performs exactly three generator applications. Yet support endpoint
count is `2,3,2`, and the reasons differ:

- state `1` has one two-label alias class and no loop;
- state `2` has three singleton endpoint classes, one of which is a loop;
- state `3` has both one loop singleton and one two-label alias class.

Thus even equal support endpoint counts at states `1` and `3` do not determine
equal loop versus alias decomposition.

The same fixture shows why total labeled outdegree cannot determine a target
path count. From state `1`, two declared labels reach adjacent state `2`, so
the labeled shortest-path count for `1->2` is two. From state `2`, only one
label reaches adjacent state `1`, so the count for `2->1` is one. Both source
states generate three labeled occurrences and both target distances are one.

In general,

```text
sum_v m(u,v) = |S|
```

is only conservation of total outgoing label mass. A target count depends on
how that mass is distributed among endpoint states and shortest-DAG edges.
Equal `|S|`, equal distance, or even equal support degree does not determine
equal labeled shortest multiplicity.

The action is transitive, and all stabilizers have equal order two. The fixed
`S` is not conjugation-invariant, so the support profiles need not agree.

## 6. Cayley case as trivial stabilizer

For the regular action of `G` on itself,

```text
K = {e}.
```

Every right coset `K s` is the singleton `{s}`. A set of distinct generator
elements therefore has multiplicity one per endpoint, and only an identity
generator creates a loop.

This recovers note 157's cancellation result as the trivial-stabilizer special
case of the coset partition.

## 7. BFS frontier consequences

Constant `|S|` gives constant raw successor occurrences per expanded state. It
does not give constant:

- distinct support neighbors `u(x)`;
- nonloop neighbors;
- outward occurrences into `F_(d+1)`;
- unique next states;
- same-parent label excess;
- cross-parent convergence.

A frontier concentrated in states with large `|S|-u(x)` can spend the same
generator compute as another frontier while producing fewer distinct support
arcs before global visited is even consulted.

Loops belong immediately to the visited-ball occurrence count `Y_d`. Parallel
labels reaching a genuinely new endpoint contribute to `X_d-P_d`. These two
stabilizer effects land in different note-157 counters.

## 8. Quotients can introduce aliases

A full-state Cayley representation may be free. Quotienting states by a symmetry
subgroup changes the vertex set to orbits/cosets and introduces a nontrivial
stabilizer unless the quotient action remains free.

Consequently, reusing the old generator loop over quotient states can create:

- moves that become self-loops after canonicalization;
- distinct moves with the same canonical successor;
- state-dependent useful support degree;
- changed labeled-path identity and replay obligations.

This is not merely an optimization detail. The quotient changes the graph
contract and must carry a distance/path-lifting proof as described in note 17.

## 9. GPU and multi-owner interpretation

Generator-regularity makes raw occurrence work easy to size:

```text
raw occurrences at level d = |S| |F_d|.
```

But support construction and visited traffic depend on the frontier's
stabilizer-coset profiles. Useful telemetry includes histograms of:

- `u(x)` per frontier state;
- loop labels `|S intersect Stab(x)|`;
- alias-class sizes `mu_x`;
- occurrences removed within one parent;
- distinct children after cross-parent combination.

If state layout or owner hashing correlates with stabilizer type, equal frontier
vertex counts can still yield unequal useful output and routing. Conversely,
early same-parent combination can be local and cheap if labels for one parent
are generated together—provided the output does not require them separately.

## 10. What not to infer

- Equal stabilizer order does not imply equal support degree for a fixed `S`.
- Transitive action does not imply the fixed labeled Schreier graph is
  vertex-transitive label-preservingly.
- Constant labeled outdegree does not imply constant simple degree.
- Removing self-loops and aliases for vertex BFS does not preserve every labeled
  path output.
- A quotient with fewer states does not automatically reduce generator work per
  expanded state.

## Sources and internal dependencies

- Note 16 fixes right-action and stabilizer-coset conventions.
- Note 17 supplies quotient and path-lifting boundaries.
- Notes 61-62 separate stabilizers from full-state Cayley identity.
- Notes 64 and 157 separate word, occurrence, support-arc, predecessor, and
  endpoint multiplicities.
- The coset multiplicity formulas follow directly from
  `xs=xt iff st^-1 in Stab(x)`.

## Takeaway

Schreier successor aliases are not mysterious duplicates. They are intersections
of the generator collection with right cosets of the current stabilizer. This
makes raw generator work regular while support degree and useful BFS work may be
state-dependent—and it identifies conjugation invariance as a sufficient
condition that restores a uniform multiplicity profile.
