# Cayley word metrics, generator changes, and BFS growth

Changing a finite generating set changes every local detail seen by BFS, yet
for one finitely generated group the resulting word metrics remain equivalent
at large scale. The useful statement is quantitative: each generator of one
alphabet has a bounded word expansion over the other.

This note develops that bound, its consequences for BFS balls, and its severe
limits for finite puzzle benchmarks and GPU work. It adds no generator-selection
algorithm.

## 1. Two finite symmetric alphabets

Let `S` and `T` be finite symmetric generating sets of the same group `G`. Write
`|g|_S` and `|g|_T` for word length, and define

```text
A = max over s in S of |s|_T,
B = max over t in T of |t|_S.
```

Both constants are finite because both sets generate `G`. Replacing every
letter of an `S`-word by a `T`-word of length at most `A` gives

```text
|g|_T <= A |g|_S.
```

The symmetric argument gives

```text
|g|_S <= B |g|_T.
```

By left invariance, for all `g,h`:

```text
(1/B) d_S(g,h) <= d_T(g,h) <= A d_S(g,h).
```

Thus the identity map between the two Cayley word metrics is bi-Lipschitz and,
in particular, a quasi-isometry. Note 92's nested-alphabet bound is the special
case where one direction has constant one.

## 2. BFS-ball inclusions

Let `Ball_S(r)={g: |g|_S<=r}`. The metric inequalities imply

```text
Ball_S(r) subseteq Ball_T(A r),
Ball_T(r) subseteq Ball_S(B r).
```

For growth functions `beta_S(r)=|Ball_S(r)|`:

```text
beta_S(r) <= beta_T(A r),
beta_T(r) <= beta_S(B r).
```

This is the correct coarse comparison. It rescales radius; it does not pair the
same BFS levels or imply `|Sphere_S(r)|` is close to `|Sphere_T(r)|`.

## 3. What growth information survives

The ball inequalities preserve broad asymptotic growth class under a change of
finite generating set:

- polynomial growth remains polynomial, with the same polynomial degree;
- exponential versus subexponential growth is unchanged;
- intermediate growth remains between polynomial and exponential in the same
  coarse sense.

But exact growth functions and growth series depend on the alphabet. Even the
numerical exponential growth rate per generator step can change because one
step has been rescaled. Generator-independent claims must use the appropriate
coarse equivalence rather than equality of coefficients.

## 4. Exact spheres can change immediately

On `G=Z`, compare

```text
S = {+1,-1}
T = {+1,-1,+2,-2}.
```

Both generate the same group and both have linear ball growth. Yet:

- `Sphere_S(1)` has two elements;
- `Sphere_T(1)` has four elements;
- integer `2` moves from depth two to depth one;
- the `T` Cayley graph contains the triangle `0,1,2,0`, while the `S` graph is
  an infinite line.

So girth, short relations, exact layer counts, parents, duplicate patterns, and
frontier locality are not quasi-isometry invariants.

## 5. Finite groups need a family-level warning

Any two finite metric spaces are quasi-isometric in a coarse existential sense.
That makes the bare phrase "quasi-isometric" nearly vacuous for one finite
puzzle instance: some finite constants always exist.

For useful scaling conclusions across a family `(G_n,S_n,T_n)`, the comparison
constants must be uniform in `n`. They need not be. On a cyclic group, adding
every nonidentity element makes diameter one, while expressing those generators
using `+/-1` requires words as long as roughly half the cycle.

Therefore a bound whose `A_n` or `B_n` grows with puzzle size does not preserve
asymptotic BFS depth, memory peak, or synchronization count across the family.

## 6. Schreier actions inherit the substitution bound

Suppose `G` acts on a state orbit `X`, and both alphabets generate the same
group action. Replacing a generator letter by its word over the other alphabet
produces the same group element and therefore the same action endpoint. Hence
the same constants bound orbit-graph distances:

```text
(1/B) d_S^X(x,y) <= d_T^X(x,y) <= A d_S^X(x,y).
```

Stabilizers may make the actual action distances much smaller than group word
lengths, but they cannot invalidate the substitution upper bound. The graph
must still represent the same action and state identity; quotienting or
history-dependent legality can change the object.

## 7. Directed positive alphabets require mutual simulation

For asymmetric directed alphabets, formal group generation is insufficient.
To obtain a directed distance comparison, every permitted transition of one
alphabet must be expressible as a bounded **positive** word in the other.

On infinite `Z`, alphabet `{+1}` reaches only nonnegative displacement from the
identity, while `{+1,-1}` reaches both directions. They generate the same group
only after formal inverses are admitted, but their directed reachability
relations differ. No two-sided BFS-distance comparison exists on unreachable
pairs.

In a finite group, positive powers eventually express inverses, but the required
bound may be large and family-dependent. Finiteness restores reachability, not
automatically a useful uniform metric constant.

## 8. Geodesics and replay under macro generators

A shortest `T` word can be expanded letter-by-letter into an `S` word whose
length is bounded by `B` times the `T` length. The expanded word is replay-valid
if each substitution is valid, but it need not be `S`-geodesic.

Conversely, storing only the `T` parent label is insufficient for an `S`-move
output contract unless the expansion word, orientation, and state action are
available. Metric comparison proves a length bound; it does not supply a
canonical concrete replay or preserve shortlex order.

## 9. GPU and multi-GPU consequences

Ball inclusions do not predict hardware work at corresponding integer levels.
Changing generators changes at least:

- candidates per parent;
- transformation and ranking cost per candidate;
- relation lengths and duplicate-convergence locality;
- frontier depth, width, and ordering;
- parent-label size and replay expansion;
- number of global level synchronizations;
- partition traffic induced by the new endpoints.

A larger alphabet may reduce semantic depth while increasing edge work per
level. A smaller alphabet may reduce branching while adding synchronization
rounds. The bi-Lipschitz constants constrain state-space geometry, not elapsed
time, memory peak, or communication volume.

Benchmark reports should name the exact alphabet and treat an alphabet change
as a workload change. A uniform family-level metric bound is useful context,
not a performance predictor.

## 10. Evidence checklist

1. Finite symmetric generating sets or directed positive alphabets.
2. Same group, same semigroup reachability, or same Schreier action.
3. Explicit substitution constants in both directions.
4. Fixed finite instance versus uniform constants across a family.
5. Balls/coarse growth versus exact spheres/growth-series coefficients.
6. Group word metric versus orbit-state distance.
7. Macro-label expansion and concrete replay contract.
8. Generator change reported as a graph/workload change.

## Sources

- P. de la Harpe, [*Topics in Geometric Group
  Theory*](https://books.google.com/books/about/Topics_in_Geometric_Group_Theory.html?id=cRT01C5ADroC),
  University of Chicago Press, 2000, especially Chapters IV and VI on word
  metrics, quasi-isometries, and growth.
- C. Loeh, [*Geometric Group Theory: An
  Introduction*](https://loeh.app.uni-regensburg.de/ggt_book/ggt_book_draft.pdf),
  Springer, 2017, Chapters on word metrics, quasi-isometry, and growth. Provides
  a modern explicit treatment of finite-generator word metrics.
- Notes 06, 10, 16, 17, 21, 27, 35, 46, 63, 68, and 92 provide Cayley/Schreier,
  frontier, action, quotient, diameter, girth, growth-series, expansion,
  relation, generator-change, and reachability-equivalence context.

## Takeaway

Finite generating sets of one group define linearly comparable word metrics,
so large-scale growth type survives. Exact BFS layers, girth, relations,
parents, and hardware work do not. On finite puzzle families, only uniform
comparison constants carry scaling meaning; existential quasi-isometry of each
finite instance is not a performance or diameter guarantee.
