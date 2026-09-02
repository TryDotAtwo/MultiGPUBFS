# Symmetry quotients: distance to what, and can the path be lifted?

Canonicalization can reduce a BFS state space while preserving an exact search
problem, or it can silently replace the problem by an easier one. The deciding
questions are:

1. do equivalent states have compatible transitions?
2. what target does one quotient vertex represent?
3. can a quotient path be lifted from the requested concrete start?
4. does the lift end at the requested concrete goal?

These are semantic questions, not properties of a hash or canonical encoding.

## Three different maps

Let `pi: V -> Q` map concrete vertices to classes.

### Graph homomorphism

A graph homomorphism maps each original edge to an edge or, in quotient models
that collapse internal edges, possibly to equality of quotient vertices. It
lets an original path project to a quotient walk. Therefore quotient distance
cannot generally exceed original fixed-endpoint distance:

```text
dist_Q(pi(s),pi(t)) <= dist_G(s,t).
```

This gives a lower bound, not a lifting theorem. A quotient edge may have been
witnessed by representatives unrelated to the representative reached by the
previous edge.

### Orbit quotient by graph automorphisms

Let a group `K` act on `G` by direction- and adjacency-preserving
automorphisms. Quotient vertices are orbits `[v]=K*v`, and two distinct orbits
are adjacent when some representatives are adjacent.

Every quotient path can be lifted step by step from any chosen representative
of its first orbit. If `[x] -> [y]` is witnessed by `x0 -> y0` and current
representative is `x=k*x0`, automorphism `k` supplies `x -> k*y0`, with
`k*y0 in [y]`.

The lift need not be unique. Stabilizers and multiple edge orbits may offer
several representatives or labels at each step.

### Graph covering

A covering projection is locally bijective: each incident/outgoing base edge
has exactly one lift at each vertex in the fiber. It therefore provides a
**unique** lifted path after the initial representative is fixed.

Every covering is strong enough for path lifting. A general homomorphism is
not. An automorphism-orbit quotient has existence of lifts for unlabeled paths,
but need not have covering-style uniqueness.

## What orbit-quotient BFS computes

Fix a concrete source `s` and target orbit `[t]`. For an automorphism-orbit
quotient,

```text
dist_Q([s],[t]) = min_(u in [t]) dist_G(s,u).
```

Proof has two directions:

- every concrete `s -> u` path projects, so quotient distance is no larger;
- a shortest quotient path lifts from `s` to some `u in [t]` with the same
  length, so the minimum concrete distance is no larger.

Thus quotient BFS is exact for **distance to an orbit**. It is not automatically
exact for distance to one fixed representative `t`:

```text
dist_Q([s],[t]) <= dist_G(s,t),
```

and the inequality can be strict.

## When fixed-target distance is preserved

Sufficient cases include:

- the target orbit is a singleton;
- every symmetry in `K` fixes the target;
- more generally, every target-orbit representative has the same distance from
  `s` and a shortest quotient lift can be aligned to `t`;
- the intended task explicitly accepts any member of `[t]`.

A particularly clean condition is that `K` fixes the source `s`. Then

```text
dist(s,k*t) = dist(k^-1*s,t) = dist(s,t),
```

so all representatives of `[t]` are equally far from `s`. If `K` does not fix
the concrete source, quotienting both endpoint orbits generally computes the
minimum compatible orbit problem rather than the fixed pair.

These are sufficient semantic arguments, not permission to assume that every
canonicalizer arises from graph automorphisms.

## A strict-distance counterexample

Take the path graph

```text
0 -- 1 -- 2 -- 3
```

and quotient by its reflection `k(i)=3-i`. The orbits are

```text
A={0,3}, B={1,2},
```

so the quotient has one edge `A--B`. With concrete source `s=0` and fixed
target `t=2`,

```text
dist_Q(A,B)=1
dist_G(0,2)=2.
```

The quotient path lifts from `0`, but it ends at `1`, another member of the
target orbit. Nothing is wrong with lifting; the query changed from target `2`
to target set `{1,2}`.

## Transition congruence for arbitrary canonicalization

Suppose `x ~ x'` means `canon(x)=canon(x')`. To define an unlabeled quotient
successor relation independent of representative, require

```text
{[y] | x -> y} = {[y'] | x' -> y'} whenever x ~ x'.
```

This is a transition congruence (or lumpability-like condition). Without it,
expanding only the canonical representative may omit a class reachable from a
different representative.

For labeled moves the requirement is stronger. One may demand the same label
to induce the same class transition, or allow a symmetry-dependent permutation
of move labels and track that permutation as a frame. Preserving unlabeled
distance does not automatically preserve the exact move string.

## Why `canon(move(canon(x), s))` may be insufficient

Let a symmetry `k` map state `x` to canonical representative `k*x`. If `k`
does not commute with move `s`, then

```text
k*(x --s--> x*s)
```

may correspond at `k*x` to a transformed label `k*s*k^-1`, not the original
label `s`. If the generator set is closed under this conjugation, the unlabeled
neighbor orbit can still be correct, but replay must remember how labels were
renamed. If it is not closed, the proposed symmetry is not an automorphism of
the allowed-move graph at all.

Canonical states alone therefore may support distance enumeration while being
insufficient for path reconstruction.

## Lifting a stored parent path

A replay-valid quotient search may need each accepted transition to retain:

- the quotient parent;
- the chosen quotient move/edge orbit;
- the symmetry/frame that maps the concrete lifted parent to the canonical
  representative;
- the resulting frame update;
- enough endpoint information to test the fixed target, if it is not an orbit
  query.

If only canonical parent keys are stored, a sequence of locally valid quotient
edges may fail to compose into the recorded concrete move sequence. A second
lifting pass can work when the quotient map has a proved lifting procedure and
the necessary edge witnesses remain recoverable.

Coverings simplify this because a base edge has a unique lift from the current
concrete vertex. Orbit quotients may require choosing among several lifts and
maintaining the choice consistently.

## Bidirectional search adds a compatibility problem

Forward and backward quotient searches may meet at the same orbit while their
concrete lifts reach different members of that orbit. Joining the two parent
chains then needs a symmetry/frame that aligns the meeting representatives.

For an orbit-target problem, some alignment may be acceptable. For a fixed
start/fixed target path, equality of canonical meeting keys alone does not prove
that the two stored move sequences concatenate. The scalar quotient distance
can be correct for the orbit problem while reconstruction for the concrete
problem fails.

## Multi-source and target-orbit equivalence

Quotienting a target orbit and seeding every target-orbit representative in a
multi-source BFS compute the same scalar object when the quotient is a valid
automorphism-orbit graph:

```text
min_(u in [t]) dist(s,u).
```

They differ physically and in metadata. Multi-source BFS retains concrete
source labels/ties; quotient BFS collapses them unless lifting/frame data are
kept. Neither operation by itself answers distance to one fixed `t` when other
orbit representatives are closer.

## Parity, loops, and collapsed edges

An original edge whose endpoints lie in one orbit becomes a quotient loop or
is collapsed. Consequently:

- quotient path length may ignore concrete motion within a fiber;
- bipartiteness and parity can disappear under a coarse quotient;
- a quotient path of length zero only says source and target are equivalent,
  not that their concrete representatives are identical;
- path reconstruction may need nonzero within-fiber motion even after reaching
  the correct quotient vertex.

This is another reason to state whether the requested output is an equivalence
class or a concrete configuration.

## Counterexamples and rejected shortcuts

### Every equivalence relation defines a safe BFS quotient

False. Equivalent representatives can have different neighbor-class sets. An
expansion from one canonical representative then omits valid transitions.

### A graph homomorphism guarantees path lifting

False. It guarantees projection of original edges, not consistent witnesses for
an arbitrary quotient path.

### An automorphism quotient preserves fixed-target distance

It preserves distance to the target orbit. The reflected four-vertex path gives
strictly smaller distance than the chosen fixed target.

### A quotient parent chain is replay-valid

Not without a lift. Symmetry may rename moves, and adjacent quotient edges may
have been witnessed by incompatible concrete representatives.

### Equal canonical meeting states solve bidirectional reconstruction

They prove an orbit-level meeting. Concrete forward/backward frames may still
need alignment.

## Audit checklist

1. Is equivalence intrinsic state equality, an automorphism orbit, or an
   arbitrary compression?
2. Do equivalent representatives have the same neighbor-class set?
3. Are direction, legality, costs, and move labels preserved or transformed?
4. Does quotient BFS answer a fixed target, target orbit, or minimum over both
   endpoint orbits?
5. Is the map merely a homomorphism, an orbit quotient, or a covering?
6. Does every quotient path lift from the requested concrete start?
7. Is the lift unique, and if not, what choice/frame is stored?
8. Can the lifted path be forced to end at the concrete target?
9. How are quotient parents converted into replay-valid original moves?
10. At a bidirectional meeting, how are the two concrete frames aligned?

## Sources

- Alexander Hulpke, *Computational Group Theory* lecture notes,
  [Chapter VII](https://www.math.colostate.edu/~hulpke/lectures/m501/notes.pdf),
  for group actions, orbits, stabilizers, and Schreier graphs.
- Laurentiu Maxim, *Lecture Notes in Algebraic Topology*,
  [Theorem 4.1.9](https://people.math.wisc.edu/~lmaxim/topbook1.pdf), for the
  path-lifting property of covering projections; the discrete graph-covering
  version is the same local-bijection principle.
- MacArthur et al., *Exploiting symmetry in network analysis*, Communications
  Physics 3 (2020),
  [article](https://www.nature.com/articles/s42005-020-0345-z), for the broader
  warning that shortest-path information is only partially quotient
  recoverable.
- Notes 06, 13, and 16 supply the canonicalization, multi-source orbit, and
  Cayley/Schreier conventions refined here.

## Current synthesis

Quotient BFS is exact only after naming its target semantics. A homomorphism
projects paths and yields a lower bound. An automorphism-orbit quotient lets
paths lift but naturally computes distance to a target orbit. A covering makes
the lift unique from a chosen start. Fixed-target solutions and replay-valid
move strings require additional endpoint and frame information. Canonicalizing
states without these proofs can make BFS perfectly exact on a graph that is not
the puzzle the user asked to solve.
