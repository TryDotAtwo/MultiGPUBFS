# Short relations and BFS duplicate signatures

Short group relations do not create one generic kind of "duplicate." They
leave distinguishable signatures depending on how words or an extra edge meet
the current metric ball. REF-022 makes this concrete on `C_31`, `Z_8 x Z_8`,
and the adjacent-generator Cayley graph of `S_3`.

## Words, states, and expansion occurrences

For a symmetric generator set of degree `q`, the non-backtracking word tree has

```text
T_0 = 1
T_d = q(q-1)^(d-1),  d>=1.
```

`T_d` counts reduced words; BFS frontier size `|F_d|` counts distinct group
elements at word distance `d`. Relations make the word-to-element evaluation
noninjective.

Even without a nontrivial relation, every positive-depth vertex in an inverse-
closed tree has an edge back toward its parent. Such visited occurrences are
inverse backtracking, not collisions between two new reduced words.

## A counter taxonomy

Partition completed-frontier transition occurrences into:

```text
parent return
visited non-parent hit
new candidate occurrence.
```

Partition new candidates again into first occurrences and additional
occurrences converging on the same new state.

- **parent return:** immediate inverse cancellation under the selected tree;
- **alternate predecessor:** another edge from `F_(d-1)` to a state in `F_d`;
- **same-level edge:** an edge internal to `F_d`, often an odd-cycle boundary;
- **older-ball edge:** a non-parent edge from `F_d` deeper into `B_(d-1)`;
- **candidate convergence:** several paths from `F_d` reach one new state;
- **finite closure:** transitions continue but no new state remains.

One duplicate ratio merges all six and loses the relation geometry.

## Equal-length words and even relators

If distinct reduced words `u` and `v`, both length `r`, represent one state,
then `u v^-1` is an identity word of length at most `2r`. Their BFS paths can
first appear as convergence while producing `F_r` from `F_(r-1)`.

An even cycle of length `2r` through the root likewise has an opposite vertex
reached by two length-`r` arcs. Examples:

- `ab=ba` gives a length-four commutator and convergence into `F_2`;
- `s_0s_1s_0=s_1s_0s_1` gives a length-six braid relation and convergence into
  `F_3`.

An even shortest relation therefore often appears at half its length. "Often"
is essential: a written relator may freely reduce, have a shorter consequence,
revisit vertices, or fail to be shortest for the actual generator collection.

## Odd cycles have a different signature

For an odd cycle of length `2r+1`, the two vertices at distance `r` from the
root are distinct, have unique shortest root paths, and are connected by the
final cycle edge.

At `girth=2r+1`, depth-`r` paths need not converge. Expanding `F_r` instead sees
a same-level visited edge. `C_31` at `r=15` records:

```text
convergence_duplicates = 0
visited_nonparent = 2 directed occurrences.
```

A detector watching only collisions among new candidates can miss the first
odd relation boundary entirely.

## Reading REF-022

### `C_31`

The frontier remains two states through depth fifteen. No new candidates
converge. The length-31 closure appears as an internal edge of `F_15`.

### `Z_8 x Z_8`

Expanding four depth-one states produces 12 candidate occurrences but eight
unique depth-two states. Four occurrences are commuting-square convergence.

By depth four, non-backtracking word count is 108 while the frontier has 14
states. At depth eight, they are 8,748 and one. Relation overlap and finite
wraparound replace tree growth with torus spheres.

### `S_3`

The two length-three braid words converge on the unique opposite element while
producing `F_3`. Expanding it later exposes the second geodesic as a non-parent
visited edge back to `F_2`.

## Girth constrains onset, not the whole profile

Girth can delimit an initial tree-like radius, but it does not determine:

- how many shortest relations exist;
- how their translations overlap;
- later sphere sizes, diameter, or saturation;
- alternate-predecessor versus same-level-edge counts;
- candidate locality under a physical frontier order;
- owner distribution or GPU contention.

Two Cayley graphs with the same degree and girth can have different growth
series and duplicate signatures. A presentation supplies hypotheses; exact
frontier profiles measure their combined geometric consequences.

## Relations as falsifiable predictions

1. Fix the actual generator alphabet, inverse convention, labels, and action.
2. Freely reduce proposed relators.
3. Identify equal-length word equalities and odd closed words.
4. Predict the earliest affected frontier or internal-ball edge.
5. Measure parent returns, visited-nonparent hits, convergence, and accepted
   states separately.
6. Replay word equalities on an independent state action.
7. Leave shorter consequences open unless girth is independently proved.

## GPU and multi-GPU boundary

Algebra predicts equal states, not their hardware co-location. Equal candidates
may appear within a warp, far apart in a frontier, or on different GPUs and
meet only at an authoritative owner. Layout and ownership determine where the
semantic equality becomes removable.

REF-016/017 show locality dependence; REF-010 shows convergence migrating from
source-local to owner-side as rank count changes. No optimization follows from
knowing a relation alone.

## Current conclusions

1. Inverse returns, candidate convergence, alternate predecessors, same-level
   edges, and saturation are different BFS events.
2. Equal length-`r` words naturally converge while producing `F_r`.
3. An odd cycle `2r+1` can preserve unique radius-`r` geodesics while adding a
   same-level edge inside the ball.
4. Candidate-only duplicate counters miss odd boundaries.
5. Girth bounds initial tree-likeness but not later growth.
6. Relations predict semantic coincidences; layout determines exploitable
   locality.
7. REF-022 validates three finite fixtures, not Cube, Megaminx, or arbitrary
   presentations.
