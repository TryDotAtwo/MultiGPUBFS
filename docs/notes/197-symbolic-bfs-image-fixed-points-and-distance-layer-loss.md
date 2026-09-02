# Symbolic BFS: image fixed points and distance-layer loss

## Question

When does symbolic reachability compute actual BFS layers and shortest
distances, rather than only the final reachable set?

The representation may be symbolic while the BFS invariant remains exactly
set-theoretic. Distance is preserved only if iteration boundaries or equivalent
certificates are retained.

## 1. Symbolic state and transition predicates

For a finite Boolean-encoded state space, let:

- `x` be current-state variables;
- `x'` be next-state variables;
- `T(x,x')` be the transition relation;
- `F_d(x)` describe the current frontier;
- `R_d(x)` describe the reached ball.

The symbolic forward image is

```text
Image_T(F_d)(x') = exists x . F_d(x) and T(x,x').
```

After renaming `x'` to the current-state variables, exact BFS is:

```text
F_(d+1) = Image_T(F_d) and not R_d,
R_(d+1) = R_d or F_(d+1).
```

Boolean existential quantification performs union over all predecessor states.
Logical `or` deduplicates endpoints idempotently. These are symbolic forms of
`unique(expand(frontier)) minus visited`.

## 2. Accumulated-image iteration reaches the same balls

One may instead iterate only the accumulated predicate:

```text
R_0     = S,
R_(d+1) = R_d or Image_T(R_d).
```

For unit edges, induction gives exactly the radius-`d+1` ball. Earlier reached
states are re-imaged logically, but idempotent union prevents them from becoming
new. The layer can be recovered during the iteration as

```text
F_(d+1) = R_(d+1) and not R_d.
```

Thus frontier-only and accumulated-ball symbolic iterations can have the same
metric semantics while very different intermediate formula work.

## 3. The final fixed point loses distance

At convergence, one retained predicate

```text
R = least X . S or Image_T(X)
```

answers reachability membership. It does not say at which iteration a state
first entered.

Two graphs can have the same reachable set and different source distances. If
only the final characteristic function is retained, no operation on that
function alone reconstructs the missing edge relation and layer chronology.

Therefore:

```text
exact symbolic reachability != stored symbolic BFS distance map.
```

Distance output needs layer predicates, a symbolic integer-valued distance
relation, predecessor certificates, or recomputation against the transition
relation.

## 4. Synchronous rounds versus saturation schedules

A synchronous image iteration associates one logical transition hop with one
outer round. First entry at round `d` then certifies distance `d`.

Symbolic engines may use partitioned transition relations, variable clusters,
or saturation/chaotic fixed-point schedules that apply several relation pieces
until local stability before another outer boundary. Such schedules can compute
the same least reachable set while allowing information to traverse several
graph edges during one implementation “round.”

The round counter then measures schedule work, not graph distance. To recover
shortest layers, the engine needs an explicit hop-indexed recurrence or a proof
that its scheduling unit advances at most one semantic edge.

This is the symbolic analogue of k-hop GPU batching: fixed-point equality can
survive while naive iteration-depth labels do not.

## 5. Idempotent set union discards path multiplicity

BDD/formula disjunction represents membership support. If ten shortest parents
lead to one child, the result predicate still contains one satisfying child
assignment.

Ordinary symbolic reachability therefore preserves:

- reached-state membership;
- layer membership when differences are retained;
- scalar shortest distance derived from first layer membership.

It does not automatically preserve:

- every shortest predecessor;
- parallel transition labels;
- shortest-path counts;
- one canonical word;
- uniform path-sampling weights.

Those require relations or values over richer algebras than Boolean support.

## 6. Witness extraction is a separate phase

If a target first appears in `F_D`, one shortest witness can be reconstructed by
repeatedly finding a predecessor in

```text
F_(d-1)(x) and T(x,current).
```

This needs retained layer predicates or recomputation of them. A final reachable
predicate alone can supply candidate predecessors but does not force depth to
decrease and can lead around cycles.

As in Lee's wave/search versus trace distinction:

```text
symbolic fixed point/layers -> prove membership and distance,
symbolic predecessor choices -> extract one witness.
```

## 7. BDD compactness is structural, not cardinal

A BDD can represent a set containing exponentially many Boolean states with far
fewer nodes when the characteristic function has exploitable regularity. It can
also become exponentially large under an unfavorable function or variable
order.

Therefore none of these implications is valid:

```text
many represented states -> large BDD,
small frontier cardinality -> small BDD,
small BDD at depth d -> small image at depth d+1.
```

The symbolic work coordinates are node counts, apply/quantification work,
intermediate diagram peaks, cache behavior, transition partitioning, and
variable order—not explicit edge occurrence count alone.

## 8. Classic BDDs do not solve the uncountable case automatically

With `n` Boolean state variables, a BDD describes a subset of a finite universe
of at most `2^n` valuations. The represented set may be astronomically large but
is still finite.

Note 196's uncountable-frontier boundary requires a representation language for
an uncountable domain, such as exact arithmetic constraints or another
appropriate symbolic theory. “Symbolic” names a representation strategy, not a
guarantee that every cardinality or transition theory is decidable.

## 9. Exactness obligations of a symbolic representation

A symbolic BFS claim needs all of:

1. an exact encoding of semantic states;
2. an exact transition relation for the declared graph;
3. exact existential image computation;
4. exact Boolean union, intersection, complement/difference, and emptiness;
5. a declared synchronous layer or alternative distance certificate;
6. target predicates matching the semantic goal;
7. witness lifting/replay when concrete paths are returned.

An abstraction or quotient may intentionally merge concrete states. It then
needs the same distance/goal congruence and lifting proofs as any other BFS
abstraction; BDD storage does not provide them automatically.

## 10. GPU and multi-GPU interpretation

Symbolic BFS shifts work from one-record-per-state traversal to operations on
shared representation nodes. GPU relevance then depends on the symbolic
backend:

- irregular BDD pointer chasing and unique-table operations;
- bulk bitset or truth-table operations;
- SAT/QBF/theory solving;
- relational-product kernels;
- partitioned relation scheduling;
- descriptor communication and canonicalization across owners.

The number of semantic states represented by one node is not a throughput
count. Multi-GPU exactness needs authoritative canonical symbolic nodes or an
exact reconciliation protocol, plus a globally consistent fixed-point/layer
boundary. Replicating a compact predicate can be cheaper than sharding it, or
far more expensive if intermediate images explode; neither follows from state
cardinality alone.

## 11. Rejected implications

- Every symbolic reachability fixed point stores BFS distances.
- One symbolic iteration always equals one graph edge.
- Boolean disjunction preserves shortest-path counts.
- A compact final reachable BDD implies compact intermediate frontiers.
- BDD size is proportional to represented-state count.
- Symbolic storage automatically supports uncountable domains.
- Exact symbolic encoding makes an abstraction distance-preserving.
- GPU state throughput predicts symbolic relational-product throughput.

## 12. Evidence boundary

The recurrences and distance-loss argument are direct set proofs. The retained
primary source establishes symbolic state/relation representation with BDDs and
fixed-point model checking on very large finite systems; this note does not
attribute a specific shortest-layer algorithm or GPU claim to that paper. No
symbolic package was run and no representation-size performance claim is made.

## Sources

- Jerry R. Burch, Edmund M. Clarke, Kenneth L. McMillan, David L. Dill, and
  L. J. Hwang, [*Symbolic Model Checking: 10^20 States and
  Beyond*](https://www.cs.cmu.edu/~emc/papers/Conference%20Papers/symbolic%20model%20checking%2010%20%20states%20and%20beyond.pdf),
  *Information and Computation* 98(2), 142--170, 1992,
  DOI 10.1016/0890-5401(92)90017-A. Primary source for symbolic state/relation
  representation, BDDs, and fixed-point computations.
- Note 25 supplies the union-preserving graph fixed-point proof; notes 57 and
  172 supply output and merge-algebra distinctions; note 196 supplies the
  explicit-enumeration/cardinality boundary.

## Compact conclusion

Symbolic BFS is ordinary BFS set algebra executed on predicates rather than
state lists. Exact image and reached-set difference preserve layers, but one
final reachable fixed point forgets first-entry depth, and more aggressive
fixed-point schedules need separate hop certificates. Symbolic compression
changes representation work; it does not relax identity, transition, output, or
distance-finality obligations.
