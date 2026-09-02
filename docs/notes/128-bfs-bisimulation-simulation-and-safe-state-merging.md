# BFS, bisimulation, simulation, and safe state merging

An implicit BFS often wants to merge states that appear behaviorally
equivalent. The relevant question is not whether their encodings look alike,
but whether every transition from either representative can be matched from
the other and whether the goal predicate is constant on the merged class.

Strong bisimulation supplies that two-sided transition condition. It preserves
exact shortest reachability to class-saturated goals, but not identity,
fixed-representative distance, path multiplicity, or arbitrary metadata.

This note deepens note 17's transition-congruence boundary and adds no
implementation, optimizer, benchmark, or GPU code.

## 1. Labeled transition-system contract

Let states `X` have observations `obs(x)` and labeled transitions

```text
x --a--> y.
```

For ordinary unlabeled BFS, use one common label. For puzzle moves, labels are
generator names. A relation `R` is a strong bisimulation when `x R x'` implies:

1. `obs(x)=obs(x')`;
2. for every `x --a--> y`, some `x' --a--> y'` has `y R y'`;
3. symmetrically, every `x' --a--> y'` is matched by an `x --a--> y` into the
   same related class.

When `R` is an equivalence relation, its quotient has one state per class and
a labeled class transition whenever a representative has such a transition.
The standard transition-system quotient retains class self-loops; suppressing
them changes trace length even when minimum distance to another class remains
unchanged. The two-sided condition makes class behavior independent of which
representative is currently held.

## 2. Projection and lifting

With quotient self-loops retained, every concrete path projects to a quotient
path of the same number of labeled steps. Conversely, a quotient path can be
lifted step by step from any chosen representative of its initial class:
bisimulation matches the next quotient transition from the representative
reached so far.

Therefore the quotient neither loses nor invents finite labeled traces at the
class level. The concrete endpoint of a lift is only guaranteed to lie in the
final class, and different matching choices can give different representatives.

This is existence of a lift, not covering-style uniqueness and not preservation
of the number of paths.

## 3. Exact BFS distance to a saturated goal

Let goal set `T` be saturated by the equivalence: if `t in T` and `t R x`, then
`x in T`. Equivalently, goal membership is part of `obs` and each quotient
class is entirely goal or entirely non-goal.

For concrete start `s`, projection and lifting give

```text
d_X(s,T) = d_(X/R)([s], T/R).
```

Projection proves the quotient distance is no larger. Lifting a shortest
quotient goal path reaches some concrete member of a goal class; saturation
makes that endpoint a genuine goal, proving the reverse inequality.

Hence quotient BFS is exact for minimum step count to the saturated property.

## 4. A fixed representative is not saturated

If the requested target is one concrete state `t` but its class contains
another state `t'`, quotient arrival at `[t]` can lift to `t'`. Then

```text
d_(X/R)([s],[t]) = min_(u in [t]) d_X(s,u)
```

under the path-lifting conditions, not necessarily `d_X(s,t)`. Note 17's
reflected four-vertex path is the concrete strict counterexample.

Marking only `t` as a goal would split its class during observation-respecting
bisimulation refinement. Keeping the class merged while testing singleton
membership violates the premise of the distance theorem.

## 5. Simulation is only one direction

A simulation requires each transition of one system to be matched by the
other, but not conversely. It can justify projection of concrete behavior into
an over-approximating abstract system. Abstract paths may then have no concrete
lift, producing spurious reachability or a distance smaller than any concrete
goal path.

Thus simulation can support a lower-bound or may-reach abstraction under an
explicit mapping and target contract. It does not by itself license exact BFS
on merged states. Calling two systems "simulation equivalent" also needs care:
two simulations need not provide one common stepwise relation with the same
lifting strength as bisimulation.

## 6. Bisimulation ignores multiplicity

The matching clauses require existence, not a bijection of outgoing edges.
Several transitions may all be matched by one transition into a related class.
Consequently bisimulation preserves whether a labeled class transition exists,
but not automatically:

- degree;
- number of parallel transitions;
- number of shortest paths;
- predecessor-DAG multiplicity;
- probability mass or transition rate.

Probabilistic lumpability and weighted/rate bisimulations add quantitative
conditions. The word "lumpable" must not be used as if all these contracts were
the same.

### Diamond witness: exact distance, changed path count

Take the unlabeled graph

```text
s -> p -> t
s -> q -> t,
```

with `t` the only goal. States `p` and `q` are strongly bisimilar: they have the
same non-goal observation and each has a transition to the same goal state.
Merging them into class `C={p,q}` gives

```text
s -> C -> t.
```

Both graphs have goal distance two. The concrete graph has two distinct
shortest vertex paths, while the ordinary support quotient has one class path.
Retaining two parallel `s -> C` occurrence identities could preserve this
particular multiplicity, but that is extra quantitative structure not supplied
by the existence-based bisimulation relation.

This does not contradict deterministic DFA minimization preserving accepted
words. In a DFA, each input word determines one run, so language preservation
preserves which distinct words are counted. A nondeterministic graph can have
several concrete state paths for the same unlabeled or labeled trace; plain
bisimulation may merge those paths.

## 7. Equitable partitions are stronger in a simple graph

For a finite simple unlabeled graph, a partition is equitable when any two
vertices in one cell have the same number of neighbors in every cell. This
implies ordinary graph bisimulation: equal positive counts provide a matching
neighbor class.

The converse fails because bisimulation ignores counts. On the path `P_3` with
no observations, the universal one-class relation is a bisimulation: every
vertex has at least one successor and every successor remains in the one
class. Yet endpoints have degree one and the middle has degree two, so the
partition is not equitable.

Stable color refinement computes an equitable refinement of its initial
coloring. It can therefore distinguish count differences that plain
bisimulation deliberately forgets. Neither partition is an exact state key.

## 8. Observations prevent meaningless collapse

Without observations, every state of an **unlabeled** transition system with at
least one outgoing transition per state can belong to the universal
bisimulation: any step is matched by some step whose target remains universally
related. In a labeled system this requires, at minimum, the same set of enabled
labels at every related state.

This is not a paradox. Bisimulation preserves the observations and modal
behavior that were declared; if nothing distinguishes states and only
existence of a next step is observed, very little remains observable.

For BFS reachability, at minimum include goal/non-goal status. Depending on the
output, observations may also need error states, accepted labels, depth cost,
resource phase, parity, ownership-independent state type, or other properties
that must survive merging.

## 9. Costs and zero-length abstraction

Strong step bisimulation preserves unit transition count because every matched
move consumes one step. If transitions carry weights, matching only labels or
destinations does not preserve shortest cost. A cost-preserving relation must
match the relevant weight, or the quotient computes a different metric.

Stutter bisimulation can collapse internal steps while preserving selected
temporal properties, but then quotient hop count is not original BFS distance.
It belongs with contraction-like lower-resolution semantics from note 124, not
with exact unit-step distance preservation.

## 10. Direction and bidirectional BFS

Strong bisimulation matches outgoing transitions. That supports forward BFS.
Backward BFS runs on reversed transitions and requires the analogous incoming
class condition. In an undirected graph the two coincide. In a directed graph,
an outgoing bisimulation need not remain a bisimulation after reversal.

A bidirectional quotient search therefore needs both directions proved safe,
plus concrete frame/representative alignment at the meeting class as described
in note 17.

## 11. Automorphism orbits and coverings

Orbits of label-preserving graph automorphisms form bisimulation classes; the
automorphism gives a bijection of neighbors and therefore even equal per-orbit
counts. Such an orbit partition is equitable in the finite simple setting.

An arbitrary bisimulation class need not arise from any automorphism. Its
states can be behaviorally indistinguishable under the declared observations
without being globally symmetric.

A graph covering is stronger locally: from a chosen concrete representative,
each base edge has a unique lift. Bisimulation promises at least one matching
lift, while an automorphism orbit supplies structured but not necessarily
unique choices.

## 12. Cayley and Schreier transition congruence

For a deterministic labeled Cayley transition `g --a--> g*a`, an equivalence
supports a label-preserving quotient exactly when

```text
g ~ h  implies  g*a ~ h*a  for every allowed generator a.
```

Then each label induces a well-defined class transition. Quotients by a normal
subgroup give the familiar quotient group; suitable coset relations can give
Schreier actions even without a quotient group, subject to the chosen action
side.

If a symmetry permutes generator labels by conjugation, it is an unlabeled or
frame-dependent quotient rather than strong same-label bisimulation. Replay
must track that permutation. A canonical hash collision supplies none of these
facts.

## 13. Frontier and visited semantics

In a valid quotient BFS, authoritative visited is keyed by equivalence class.
First class discovery gives minimum quotient depth and, for a saturated goal,
minimum concrete goal depth. It does not enumerate every concrete state in the
class or give the distance to each representative.

Candidates from different concrete states may merge into one class. Counting
them as duplicate states is valid only for the quotient output contract.
Reporting the quotient visited count as the number of original reachable states
is generally false unless class sizes are separately known and fully covered.

## 14. GPU and multi-GPU boundary

State merging can reduce frontier and visited cardinality only after the
equivalence has been proved and its class key computed consistently. Report
separately:

- relation/partition construction and validation;
- observation and target-saturation checks;
- concrete candidates and quotient-class candidates;
- class-level duplicates and concrete multiplicities;
- quotient traversal and concrete path lifting;
- representative/frame metadata;
- partition ownership and cross-owner class reconciliation;
- original and quotient memory, work, communication, and depth.

A per-device local refinement is not a global bisimulation partition if a
remote transition can split a class. Class finalization needs a globally
consistent transition epoch. This is a semantic synchronization condition,
not a recommendation for one implementation.

## Sources

- C. Baier and J.-P. Katoen,
  [*Principles of Model Checking*, Chapter 7](https://www.labri.fr/perso/anca/Verif/Baier-Katoen.pdf),
  MIT Press, 2008, for bisimulation, simulation, quotient transition systems,
  and partition refinement.
- R. Paige and R. E. Tarjan,
  [*Three Partition Refinement Algorithms*](https://doi.org/10.1137/0216062),
  SIAM Journal on Computing 16(6), 1987, for relational coarsest partitions.
- Notes 06, 16, 17, 20, 28, 32, 37, 52, 57, 64, 101, 123, and 124 supply this
  repository's canonicalization, action, quotient, product-state, fingerprint,
  intersection-profile, contract, visited, output, multiplicity, refinement,
  covering, and contraction boundaries.

## Takeaway

Bisimulation makes quotient transitions liftable in both directions and
therefore preserves exact unit BFS distance to goals that are unions of whole
classes. It does not preserve a fixed representative, transition counts,
shortest-path multiplicity, weights, or undeclared observations. Simulation is
one-sided and may admit spurious abstract paths; equitable partitions and color
refinement preserve counts that plain bisimulation ignores. Safe state merging
starts by declaring which behavior and target property must remain observable.
