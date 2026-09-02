# Cayley generator autocorrelation, common children, and parent coupling

## Question

What algebraic quantity tells us when two Cayley-BFS parents compete for the
same next-layer states, and when that competition can make shortest-parent
choices impossible for any serial first-in BFS?

## 1. Raw common-successor identity

Let `G` be a group, let `S` be a set of distinct generators, and use the right
Cayley digraph

```text
x -> x s,  s in S.
```

The one-step successor sets of `u` and `v` are `uS` and `vS`. Put

```text
h = u^-1 v.
```

Left multiplication by `u^-1` is a bijection, so

```text
|uS intersection vS|
  = |S intersection hS|.
```

Equivalently, a common child has two descriptions

```text
u s = v t
```

exactly when

```text
h = s t^-1.
```

Thus common successors are controlled by the multiplicity with which the
relative displacement `u^-1 v` occurs in the generator difference multiset
`S S^-1`. The function

```text
A_S(h) = |S intersection hS|
```

is a finite-set autocorrelation of the generator indicator. It is translation
invariant: it depends on the relative group element, not the absolute parents.

No inverse-closure assumption is needed. The group inverse `t^-1` exists even
when it is not itself a declared forward generator.

## 2. BFS needs the outward-filtered overlap

Raw common successors need not be new. If `u,v` lie in `F_d`, define

```text
O_d(u,v) = F_(d+1) intersection uS intersection vS.
```

Then `|O_d(u,v)|` counts semantic next-layer states for which these two parents
are both shortest predecessors. The raw autocorrelation gives the ceiling

```text
|O_d(u,v)| <= A_S(u^-1 v),
```

but the BFS layer filter removes inward and same-layer successors. Therefore a
large generator autocorrelation predicts possible convergence; it does not by
itself say where in the radial wave that convergence occurs.

## 3. A sufficient first-in-tree obstruction

Suppose `O_d(u,v)` contains two distinct states `x,y`. Choose shortest-valid
parents

```text
parent(x)=u,
parent(y)=v.
```

Because `v` is also a predecessor of `x`, first-in discovery requires `u` to be
dequeued before `v`. Because `u` is also a predecessor of `y`, it simultaneously
requires `v` before `u`. No serial FIFO order can satisfy both.

Hence

```text
|O_d(u,v)| >= 2
```

is sufficient for the graph to admit a locally geodesic parent assignment that
is not a realizable first-in BFS tree.

This is not a necessary characterization of every obstruction. Larger cycles
can be assembled from several parent pairs even when no pair shares two
outward children. Full tree recognition remains a global ordering problem.

## 4. Recovering the `S_3` example

For all transpositions in `S_3`, choose two distinct transpositions `u,v`.
Their relative displacement `u^-1 v` is a 3-cycle. Exactly three ordered pairs
of transpositions satisfy

```text
u^-1 v = s t^-1.
```

For each of the three choices of `t`, the element `s=(u^-1 v)t` is again a
transposition, giving the three pairs. Equivalently,

```text
uS = vS = {e, (123), (132)},
A_S(u^-1 v) = 3.
```

The raw intersection includes the old identity `e`. Filtering to `F_2` removes
it and leaves the two depth-two 3-cycles, so `|O_1(u,v)|=2`. The abstract
sufficient condition reproduces the `K_(3,2)` layer incidence of note 191.

This identifies the source of the obstruction: not “symmetry” in general, but
high multiplicity of one generator difference at the relevant radial layer.

## 5. Relation and duplicate interpretation

Each equality

```text
u s = v t
```

equates two one-step extensions, or equivalently gives the relation

```text
s t^-1 = u^-1 v.
```

For distinct parents `u,v in F_d` and a common child in `F_(d+1)`, the same
algebraic fact has three interpretations:

1. **BFS work:** two generated occurrences propose one semantic child;
2. **shortest-path output:** the child has at least two shortest predecessor
   occurrences;
3. **first-in realizability:** choosing winners across several shared children
   imposes precedence constraints on their parents.

Without this outward-layer premise, raw convergence still duplicates a
semantic child but need not give either occurrence a shortest-parent role.

Duplicate pressure and tree-order coupling are therefore two projections of
the same incidence structure. Counts alone still lose information: knowing the
total number of duplicate occurrences does not reveal which parents share
which children, and that pattern is what determines precedence cycles.

## 6. Pair counts do not equal duplicate counts

If one child has `k` distinct shortest parents, it contributes:

```text
k generated parent occurrences,
1 accepted semantic state,
k-1 excess occurrences,
C(k,2) parent-pair intersections.
```

Summing `|O_d(u,v)|` over parent pairs therefore counts convergent pairs, not
rejected occurrences. It overweights high-multiplicity children. The exact
next-layer incidence is a hypergraph:

```text
parents ---- child hyperedge of all shortest predecessors.
```

Pairwise autocorrelation is a useful second moment of that hypergraph, not a
complete reconstruction.

## 7. GPU and multi-GPU meaning

`A_S(h)` and its outward-filtered version describe semantic collision geometry
before choosing a kernel:

- high overlap can create same-key atomic contention or sort runs with repeated
  keys;
- parent layout determines whether colliding occurrences are warp-local,
  block-local, device-local, or routed across owners;
- child ownership determines where the authoritative merge occurs;
- state-only union may discard losing occurrences, while DAG/count/canonical
  outputs must reduce them under a richer algebra;
- a pairwise overlap statistic cannot by itself predict traffic bytes or timing
  because it omits record width, ordering, k-way multiplicity, and partition.

For multi-GPU BFS, group translation invariance of raw overlap does not imply
owner locality. A hash or quotient partition can scatter parents and common
children arbitrarily with respect to the Cayley geometry.

## 8. Schreier and representation boundary

For a non-free action, states are not group elements and left cancellation to a
unique `u^-1 v` is unavailable at the state level. Stabilizers can make several
generator elements induce the same state transition. The correct observable is
the action-neighborhood intersection

```text
|N_S(u) intersection N_S(v)|,
```

with occurrence multiplicity retained separately when labels matter.

A group-level `A_S(h)` can bound or explain a lifted computation, but it does
not automatically equal the quotient-state overlap. This is the same
element/state/occurrence separation used throughout CayleyPy analysis.

## 9. Rejected implications

- Large `A_S(h)` means all common successors are in the next BFS layer.
- Pairwise overlap sum equals the number of rejected candidate occurrences.
- Equal duplicate counts imply equal parent-precedence constraints.
- Cayley translation symmetry implies duplicates meet on the same GPU.
- Two common children characterize every non-realizable BFS tree.
- The regular Cayley formula transfers unchanged to Schreier states.

## 10. Evidence boundary

The common-successor identity, outward filter, pair-count conversion, and
two-child obstruction are direct finite set/group proofs. The `S_3` case follows
from the exact cycle-count layers already established in note 138. No code,
enumeration, GPU execution, or performance claim is used.

## Compact conclusion

In a right Cayley graph, generator autocorrelation
`A_S(u^-1 v)=|S intersection u^-1 v S|` counts the raw common successors of two
parents. Filtering to `F_(d+1)` turns it into shortest-parent overlap. Two
shared outward children already permit incompatible first-in parent choices,
while k-way convergence shows why pair overlap, duplicate occurrences, and
GPU contention are related but unequal quantities.
