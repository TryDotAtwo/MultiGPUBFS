# BFS on Johnson graphs: fixed-weight exchange frontiers

The Johnson graph `J(n,k)` models exact BFS over all `k`-subsets of an
`n`-element universe. Each move preserves cardinality by removing one selected
element and adding one unselected element. It is a clean model for implicit
state spaces with a conservation invariant.

No optimizer, production implementation, benchmark, or GPU code is added.

## 1. State and adjacency

Vertices are the `k`-subsets of `[n]`. Two states `A,B` are adjacent when

```text
|A intersection B| = k-1,
```

equivalently when one selected and one unselected element are exchanged. Thus

```text
|V| = C(n,k),
degree = k(n-k).
```

Complementing every subset gives `J(n,k) ~= J(n,n-k)`, so calculations may
normalize to `k<=n/2` without changing the abstract graph.

## 2. Distance is exchange deficit

Fix root `R`. For another `k`-set `A`, define

```text
i = |R minus A| = |A minus R| = k-|R intersection A|.
```

One exchange can repair at most one missing root element, giving distance at
least `i`. Pair each unwanted element of `A` with a missing element of `R` and
exchange them to attain `i`. Therefore

```text
d(R,A)=k-|R intersection A|=|R symmetric_difference A|/2.
```

The diameter is `min(k,n-k)`. For normalized `k<=n/2`, it is `k`.

## 3. Exact frontier sizes

A state at depth `i` omits `i` elements of `R` and contains `i` elements from
outside `R`. Hence

```text
|F_i| = C(k,i) C(n-k,i),
0<=i<=min(k,n-k).
```

Vandermonde's identity gives

```text
sum_i C(k,i)C(n-k,i)=C(n,k),
```

so the layers exhaust the whole state universe.

The frontier ratio is

```text
|F_(i+1)|/|F_i| = ((k-i)(n-k-i))/(i+1)^2.
```

This locates the peak without traversing the graph, but it predicts neither
queue order nor hardware locality.

## 4. Intersection numbers

At depth `i`, classify one exchange by whether its removed and added elements
belong to the root:

- remove an outside element and restore a missing root element:
  `c_i=i^2` inward neighbors;
- exchange within the two mismatch categories without changing their count:
  `a_i=i(n-2i)` same-layer neighbors;
- remove a still-shared root element and add a new outside element:
  `b_i=(k-i)(n-k-i)` outward neighbors.

They sum to `k(n-k)` and depend only on depth. Thus Johnson graphs are
distance-regular. Edge balance

```text
|F_i| b_i = |F_(i+1)| c_(i+1)
```

is exactly the frontier-ratio formula.

## 5. Same-layer edges and triangles

Except for the single edge `J(2,1)`, nontrivial connected Johnson graphs are not
bipartite. Already at depth one,

```text
a_1=n-2,
```

which is positive for `n>=3`. Three `k`-sets sharing `k-1` elements form a
triangle whenever three choices remain for the final element.

Therefore a current-frontier omission from visited is exposed immediately.
The fixed-cardinality invariant does not imply parity alternation.

## 6. Exact candidate multiplicity

Each next-layer state at depth `i+1` has

```text
c_(i+1)=(i+1)^2
```

shortest predecessors. Expanding all of `F_i` therefore produces exactly
`(i+1)^2` outward occurrences per new semantic state.

An expanded state at depth `i` also emits `i^2` previous-layer occurrences and
`i(n-2i)` same-layer occurrences. These should be reported separately from
next-layer convergence: all are visited rejections, but they arise from
different geometry.

## 7. Shortest paths

For a target at distance `i`, a shortest path must remove each of the `i`
unwanted root elements once and insert each of the `i` desired outside elements
once. Choose an order of removals and independently an order of insertions;
pairing their positions specifies the exchanges. Thus

```text
sigma(R,A)=(i!)^2.
```

This agrees with the recurrence from `c_i=i^2`. One-parent BFS keeps one path,
while the shortest-path DAG retains all `i^2` predecessors at depth `i`.

## 8. Relation to the binary hypercube

Represent a `k`-subset by its binary indicator word. The states are one
fixed-weight layer of `H(n,2)`, but the Johnson graph is not the ordinary
induced subgraph of the hypercube: flipping one bit changes the weight, so that
induced subgraph has no edges.

A Johnson edge is a two-bit `10 <-> 01` exchange and has hypercube distance two.
For fixed-weight states,

```text
d_Hamming(A,B)=2 d_Johnson(A,B).
```

Replacing each exchange by two bit flips introduces intermediate states outside
the invariant set and changes both frontier semantics and visited capacity.

## 9. Schreier, not generally Cayley

The symmetric group `S_n` acts transitively on `k`-subsets. With all
transpositions as labeled moves, this is a Schreier action with a large
stabilizer, not generally a regular Cayley action.

For a given subset:

- `k(n-k)` transpositions cross membership and produce distinct Johnson
  neighbors;
- `C(k,2)+C(n-k,2)` transpositions swap two selected or two unselected labels
  and fix the subset.

If an implicit generator loop emits all `C(n,2)` transpositions, those
stabilizer occurrences are self-loops. Silently dropping them changes measured
candidate work but not the simple Johnson adjacency graph. The oracle contract
must state which object is measured.

## 10. Dense identity and symmetry

Combinatorial-number-system ranking can bijectively map `k`-subsets to
`[0,C(n,k))`. A raw `n`-bit mask is simpler for small `n`; a combinadic rank is
denser but costs arithmetic. Neither changes the graph distance.

Vertex transitivity makes the frontier profile root-independent. It does not
make subset ranks uniformly local under an arbitrary owner partition, and it
does not identify two different subsets with the same intersection size.

Intersection size is sufficient for scalar root distance but not for visited
identity.

## 11. Spectrum and association scheme

The adjacency eigenvalues are

```text
theta_j=(k-j)(n-k-j)-j
       =k(n-k)-j(n+1-j),
```

with multiplicity

```text
C(n,j)-C(n,j-1),
```

for `0<=j<=min(k,n-k)` and `C(n,-1)=0`. These are Johnson-scheme quantities.
They constrain global linear propagation but do not replace exact frontier
identity or visited membership.

## 12. GPU and multi-GPU interpretation

Johnson graphs are useful calibration workloads because they combine:

- fixed degree `k(n-k)`;
- exact depth-conditioned candidate classes;
- combinatorial rather than full-cube state count;
- dense bit-mask or combinadic identity;
- abundant same-layer and convergent candidates;
- a transitive but non-free action.

They can expose the difference between generator occurrences and distinct
neighbors, especially when stabilizer moves are present. They remain unusually
regular and do not model arbitrary legality tests, transformation costs, or
puzzle relations.

For sharded traversal, report state ownership, cross-owner exchanges,
stabilizer/self-loop occurrences, previous/same/next-layer candidates, accepted
states, bytes, synchronization, and end-to-end time separately.

This is a conceptual workload decomposition, not an optimized pipeline.

## 13. Docker/Rust probe

`experiments/johnson_graph_bfs_probe.rs` exhaustively traverses every normalized
`J(n,k)` for `2<=n<=12` and `1<=k<=floor(n/2)`: 36 complete fixtures.

It confirmed with zero mismatches:

```text
state count C(n,k);
distance k-|R intersection A|;
frontiers C(k,i)C(n-k,i);
intersection counts i^2, i(n-2i), (k-i)(n-k-i);
shortest-path count (i!)^2.
```

The largest fixture was

```text
J(12,6): states=924, degree=36, diameter=6,
layers=[1,36,225,400,225,36,1].
```

The probe is an exact small-state oracle, not a benchmark.

## Sources

- A. E. Brouwer, A. M. Cohen, and A. Neumaier,
  [*Distance-Regular Graphs*](https://doi.org/10.1007/978-3-642-74341-2),
  Springer, 1989, for Johnson graphs, intersection arrays, and association
  schemes.
- A. Moon, [*The Graphs G(n,k) of the Johnson Schemes Are Unique for n at
  Least 20*](https://doi.org/10.1016/0095-8956(84)90070-4), *Journal of
  Combinatorial Theory, Series B* 37(2), 1984, for the Johnson-scheme graph
  definition and distance-regular context.
- Notes 10, 11, 16, 17, 20, 32, 46, 61, 64, 69, 111, 123, 141 provide this
  repository's frontier, path, Schreier, quotient, product-state,
  distance-regular, capacity, equality, history, Cartesian-product, Hamming,
  and covering boundaries.

## Takeaway

In `J(n,k)`, BFS measures half the symmetric-difference size. Layer `i` has
`C(k,i)C(n-k,i)` states, each with `i^2`, `i(n-2i)`, and
`(k-i)(n-k-i)` inward, same-layer, and outward neighbors, and each target has
`(i!)^2` shortest paths. The family shows how a conservation invariant changes
the state universe and turns an apparent Cayley generator set into a Schreier
action with stabilizer work.

