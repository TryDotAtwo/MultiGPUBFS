# BFS as a least fixed point: balls, deltas, and quiescence

The FIFO queue is a useful execution mechanism, but it is not the mathematical
essence of BFS.  Exact reachability BFS can be viewed as a monotone closure whose
successive differences are the distance frontiers.

Let `G=(V,E)` be a fixed directed graph, let `S subset V` be the source set, and
define the relational image

```text
Post(X) = { v | exists u in X: (u,v) in E }.
```

Put `B_-1 = empty` and

```text
B_0     = S
B_(d+1) = B_d union Post(B_d)
F_d     = B_d \ B_(d-1).
```

The central invariant is

```text
B_d = { v | dist(S,v) <= d }
F_d = { v | dist(S,v)  = d }.
```

Thus `B_d` is a metric ball and `F_d` is its new shell.  `Visited` represents
the accumulated ball; the frontier represents its delta, not a second unrelated
work queue.

## The distance invariant

The base case is immediate: distance zero means membership in `S`.

For the induction step, every member of `B_d` remains in `B_(d+1)`.  Every
successor of a member of `B_d` has a path of length at most `d+1`.  Conversely,
if `v` has a shortest path of length at most `d+1`, then either it was already
in `B_d`, or the predecessor before its final edge has distance at most `d` and
is in `B_d`.  This proves the ball equality; subtracting `B_(d-1)` gives the
frontier equality.

The proof works for directed graphs and multiple sources.  It assumes unit edge
length for the numerical frontier index to equal hop distance.  It does not
assume a particular queue order inside one level.

## Why expanding only the delta is exact

Ordinary frontier BFS uses

```text
F_0     = S
F_(d+1) = Post(F_d) \ B_d
B_(d+1) = B_d union F_(d+1).
```

This is equivalent to repeatedly expanding the whole ball.  Since

```text
B_d = union_(0 <= i <= d) F_i
Post(B_d) = union_(0 <= i <= d) Post(F_i),
```

every successor generated from `F_i` for `i<d` has distance at most `i+1<=d`
and is removed by `\ B_d`.  Only the newest delta can contribute a genuinely
new state.

This is the same logical idea as semi-naive evaluation of a recursive relation:
derive consequences from the newly added facts instead of re-deriving from the
entire relation on every round.  The equivalence is not a generic consequence of
monotonicity alone.  It also uses that relational image distributes over unions.

## Reachability as the least fixed point

Define an operator on the powerset lattice:

```text
T(X) = S union Post(X).
```

Starting from the bottom element gives the Kleene chain

```text
empty <= T(empty) <= T^2(empty) <= ...
```

with `T^(d+1)(empty)=B_d`.  Its union is

```text
R = union_(d < omega) B_d,
```

exactly the states reachable by finite paths.

`R` is a fixed point because relational image preserves arbitrary unions:

```text
Post(union_i X_i) = union_i Post(X_i).
```

It is the least fixed point because any `X` satisfying `T(X)=X` must contain
`S`, and induction then gives `B_d subset X` for every `d`; hence `R subset X`.

The Knaster--Tarski theorem guarantees least and greatest fixed points for a
monotone self-map of a complete lattice.  A separate, stronger property explains
why the countable chain above reaches this particular least fixed point:
`Post` is union-preserving (and therefore omega-continuous).  For an arbitrary
monotone operator, finite/countable iteration from bottom need not suffice;
transfinite stages may be required.  Calling every monotone fixed-point
computation "BFS" would erase this important boundary.

### Explicit monotone operator that needs a stage beyond omega

Let the universe be `N union {infinity}` and define on its powerset

```text
U(X) = {0}
       union { n+1 | n in X intersect N }
       union ({infinity} if N subset X else empty).
```

`U` is monotone.  Starting from empty, every finite iteration contains only a
finite initial segment of the naturals.  Their stage-omega union is all of `N`
but does not yet contain `infinity`.  Applying `U` once more adds `infinity`.
Hence the union of the finite approximants is not a fixed point.

This cannot happen for ordinary graph `Post`: if one vertex has an incoming
edge from some member of a union, that witness already occurs in one member of
the union.  The special "all naturals have arrived" premise in `U` has no finite
witness and is precisely what breaks union preservation.

## Finite and infinite graphs are different operationally

The set-theoretic statements need neither a finite graph nor finite degree:
every finitely reachable vertex appears at its finite path length.

Operational claims need more care:

- On a finite reachable component, the chain stabilizes after finitely many
  nonempty frontiers, no later than the component's maximum finite distance.
- On an infinite but locally finite graph with a finite source set, every finite
  frontier is finite.  Each level can in principle be completely enumerated,
  but the full reachable-set computation may never terminate.
- With infinite branching, even `F_1` can be infinite.  The mathematical shell
  exists, but an ordinary finite-rate machine cannot finish materializing it
  and advance to a completed next level.
- An infinite graph can have finite diameter yet an infinite frontier.  Finite
  stabilization depth is not the same as finite enumerability or finite work.

Local finiteness therefore is not needed for the distance theorem.  It is a
useful sufficient condition for finite per-level work when the source set is
finite.

## Boolean matrix view

Let `A` be the Boolean adjacency matrix and let a Boolean row vector `f_d`
indicate `F_d`.  With logical OR as addition and logical AND as multiplication,

```text
c_(d+1) = f_d A
f_(d+1) = c_(d+1) masked by not b_d.
```

This Boolean semiring product is another notation for `Post(F_d)`.  The
complemented visited mask implements the set difference.  GraphBLAS BFS exposes
exactly this expand-and-mask structure.

Several semantic cautions follow:

- Boolean support says whether at least one predecessor exists; ordinary
  arithmetic multiplication instead counts walks and is not by itself visited
  membership.
- A structural mask and a valued mask can differ when explicitly stored false
  entries exist; the declared mask semantics matter.
- Matrix orientation determines whether row-vector multiplication follows
  outgoing or incoming edges.
- Replacing the Boolean semiring can change the computed object: path counts,
  minimum costs, and provenance require their own algebra and termination
  arguments.

### Positive count support projects exactly to Boolean existence

Define `phi:N->B` by `phi(0)=false` and `phi(n)=true` for `n>0`. For
nonnegative exact path counts,

```text
phi(x+y) = phi(x) OR phi(y),
phi(x*y) = phi(x) AND phi(y).
```

The first identity says a combined family is nonempty iff at least one family
is nonempty. The second uses the absence of zero divisors in `N`: a Cartesian
combination of prefix and suffix families is nonempty iff both are nonempty.
Thus support is a semiring homomorphism from exact counting to Boolean
existence. Projecting `sigma>0` preserves whether a shortest-prefix family
exists, while irreversibly discarding its multiplicity.

This does not make an arbitrary numeric representation a reachability oracle.
Modulo `M`, exactly `M` paths have residue zero although their support is
nonempty. For example, the two paths in a diamond give `sigma(t)=2` but
`sigma(t) mod 2=0`. A modular residue remains exact for a declared modular
count output; testing that residue for nonzero is not exact reachability.

For BFS layers, the Boolean projection still needs the distance/visited mask
that selects first arrivals. Otherwise arithmetic powers include later walks
whose positive support does not mean a new frontier state.

### Finite distance also projects exactly to reachability

On the nonnegative min-plus domain `R_(>=0) union {infinity}`, define
`psi(x)=false` for `x=infinity` and `psi(x)=true` for finite `x`. Then

```text
psi(min(x,y)) = psi(x) OR psi(y),
psi(x+y)      = psi(x) AND psi(y).
```

A minimum is finite iff at least one alternative is finite; a concatenated
route has finite length iff both pieces do. Thus finite-support projection is a
semiring homomorphism from min-plus shortest-distance algebra to Boolean
reachability.

The projection forgets all arrival times. The path `s--a--b` and a graph with
the direct edge `s--b` both mark `b` reachable, while their distances are two
and one. Therefore a final reachable set cannot reconstruct BFS frontiers,
eccentricity, or shortestness. Boolean reachability is an exact quotient of
distance semantics, not an interchangeable representation of the full metric.

### Structural versus valued complemented masks

A sparse Boolean vector may contain an explicitly stored `false`. A valued mask
interprets that position as false; a structural mask interprets the position as
true because a tuple is present. Complementing the two masks reverses different
predicates:

```text
stored false at v
complemented valued mask:     v is allowed
complemented structural mask: v is rejected.
```

Therefore a structural visited mask is exact only when its stored pattern is
itself the visited set. “Boolean type” alone does not establish that invariant;
the representation must exclude stored false entries or use valued semantics.

### Replace versus merge when reusing a frontier vector

GraphBLAS masked assignment can either replace the output outside the mask or
retain old output entries there. This matters when the same vector object is
reused across BFS levels.

On the directed edge `0->1`, let current frontier `q={0}` and visited `{0}`.
The candidate product is `{1}` and the complemented visited mask disallows
position zero. With replace semantics, the new `q` is exactly `{1}`. Without
replace, the old output entry at masked-out position zero can remain, producing
stored support `{0,1}` rather than the next frontier.

Using a freshly empty output object avoids that particular stale-output case,
but it is a different allocation/publication contract. An accumulator can also
combine previous output with new candidates and thereby compute a union or
other annotated object instead of the BFS delta. The accumulator, replace flag,
and output initialization must be declared together.

When input and output alias, the library operation must still behave as if its
declared inputs were read from the operation snapshot. A hand-written in-place
loop that immediately consumes newly written entries can cascade through
several hops in one call and is no longer one BFS expansion.

### Stored entries versus Boolean support at termination

GraphBLAS `nvals` reports the number of stored tuples, not the number of values
that evaluate to true. A vector containing only explicitly stored false values
has positive `nvals` but empty Boolean support. Consequently

```text
nvals(frontier)==0
```

is an exact emptiness test only under a no-explicit-false representation
invariant. Otherwise zeros must be removed or valued truth must be checked.

These examples show that semiring choice alone does not identify the computed
frontier. Orientation, mask kind/complement, replace, accumulator, aliasing,
stored-zero policy, and termination test jointly form the operation contract.

Linear algebra is therefore a representation of the same relational recurrence,
not an automatic correctness proof for every choice of operators and masks.

## Datalog view

Reachability from sources can be written schematically as

```text
Reach(x) :- Source(x).
Reach(y) :- Reach(x), Edge(x,y).
```

Bottom-up evaluation repeatedly applies the immediate-consequence operator
until no new facts appear.  Naive evaluation joins against all accumulated
`Reach`; semi-naive evaluation propagates only the newly derived `DeltaReach`.
For this linear reachability rule, BFS frontiers are those deltas grouped by
derivation depth.

The analogy has limits.  General recursive programs can have several mutually
recursive predicates, nonlinear rules with multiple recursive inputs, negation,
aggregates, or weighted annotations.  Their deltas and semantics need not be
ordinary BFS layers.

## Synchronous emptiness versus a fixed point

In a level-synchronous single-process BFS, after all of `F_d` has been expanded,
deduplicated, and filtered against complete `B_d`, the condition
`F_(d+1)=empty` proves

```text
Post(B_d) subset B_d,
```

so `B_d` is closed under successors and is the reachable-set fixed point.

The timing of the observation is part of the proof.  An empty output buffer
before every producer has finished does not prove an empty mathematical delta.
Nor does a capacity-truncated empty buffer.

## Asynchronous and distributed deltas

An asynchronous execution can converge to the same reachable set if accepted
updates are monotone and idempotent and every causal work item is eventually
delivered and processed.  But its transient work sets need not equal synchronous
distance frontiers.  If exact distances are also required, distance labels need
relaxation/finalization reasoning beyond set reachability.

Common invalid inferences are:

- one worker's empty queue means the global delta is empty;
- no message is currently visible, therefore no message is in flight;
- every owner has observed local closure, therefore all observations belong to
  one consistent global cut;
- duplicate delivery is harmless without making insertion idempotent;
- fairness detects termination;
- a lost delta can be reconstructed merely because the update rule is monotone.

Fairness and reliable eventual processing are liveness assumptions: they ensure
that enabled consequences are not postponed forever.  They do not constitute a
termination detector.  Global quiescence additionally needs evidence that every
participant is passive and no work/message is in flight, usually relative to a
consistent protocol state.

On a finite graph, only finitely many first insertions exist, but a faulty
protocol can still duplicate messages forever or declare completion too early.
On an infinite reachable graph, correct fair execution may make progress
forever and never reach global quiescence.

## A useful proof vocabulary

These terms separate claims that are often compressed into "BFS finished":

- **inflationary:** accumulated reached state never disappears;
- **monotone:** more known input cannot derive fewer consequences;
- **sound:** every inserted state has a valid source path;
- **complete through depth d:** every state of distance at most `d` is present;
- **closed:** `Post(B) subset B`;
- **fixed point:** `T(B)=B`;
- **least fixed point:** no smaller set contains the sources and is closed;
- **quiescent execution:** no local or in-flight work remains;
- **terminated protocol:** every participant has safely learned the required
  global termination fact.

Closure proves a fixed point only together with source containment.  Soundness
plus closure then identifies the least reachable fixed point: soundness gives
`B subset R`, while source containment and closure give `R subset B`.

## Sources and independent check

- Alfred Tarski,
  [A Lattice-Theoretical Fixpoint Theorem](https://people.csail.mit.edu/carroll/probSem/Documents/Tarski.pdf),
  gives the complete-lattice fixed-point theorem for increasing maps.
- Francois Bancilhon and Raghu Ramakrishnan,
  [An Amateur's Introduction to Recursive Query Processing Strategies](https://ftp.cs.wisc.edu/pub/techreports/1988/TR772.pdf),
  surveys naive and semi-naive evaluation of recursive relations.
- Aydin Buluc,
  [Breadth-First Search in GraphBLAS](https://people.eecs.berkeley.edu/~aydin/Buluc_GRA22_Keynote.pdf),
  shows Boolean-semiring frontier expansion with a complemented visited mask.
- The
  [GraphBLAS C API Specification v2.1](https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf)
  defines valued/structural and complemented masks, replace descriptors,
  accumulators, and stored-tuple counts used in the cautions above.
- The `autolean` expert independently checked the ball/frontier induction,
  least-fixed-point argument, finite/infinite assumptions, and the distinction
  between fairness and quiescence.  The argument above retains the important
  qualification that monotonicity alone does not imply the delta recurrence or
  convergence at stage omega.

## Current conclusion

BFS is a distance-stratified computation of the least successor-closed set
containing the sources.  `Visited` is its accumulated approximation; `frontier`
is its exact new delta.  Queue, Boolean matrix, recursive query, and distributed
message schedules are different realizations of that object, and each must
separately prove that no delta was lost and that an observed empty delta is
globally real.
