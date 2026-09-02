# BFS orderings: Cuthill--McKee, bandwidth, and profile

BFS defines distance layers but leaves the order inside each layer open.
Cuthill--McKee uses that freedom to build a linear vertex ordering intended to
place adjacent vertices near one another in a symmetric sparse matrix. This is
a graph-layout heuristic built on BFS, not a different shortest-distance
algorithm. This note adds no implementation, optimizer, benchmark, or GPU code.

## 1. From a graph to a symmetric sparsity pattern

For a symmetric matrix, associate one vertex with each row/column and an
undirected edge `{u,v}` with each off-diagonal structural nonzero. A simultaneous
row/column permutation is a bijection

```text
pi: V -> {1,...,n}.
```

It changes storage position, not graph adjacency. The bandwidth of an ordering
is

```text
bw(pi) = max_{{u,v} in E} |pi(u)-pi(v)|.
```

The graph bandwidth is the minimum of this value over all orderings. Finding
that optimum is NP-complete in general, so a fast ordering heuristic must not
be described as an exact bandwidth minimizer.

## 2. Matrix profile is a different objective

For a symmetric pattern with diagonal entries included, one common profile
convention is

```text
profile(pi) = sum_i (i - first(i)),
```

where `first(i)` is the smallest numbered column containing a structural
nonzero in row `i` at or before the diagonal. Bandwidth measures the single
largest edge span; profile sums row-wise left-envelope lengths.

Two orderings can have the same bandwidth and different profiles. Fill-in,
wavefront, cache locality, and factorization time are further objectives and do
not follow from bandwidth or profile alone.

## 3. Cuthill--McKee is degree-ordered BFS

From a chosen root, Cuthill--McKee performs a BFS-like traversal. When expanding
a vertex, its undiscovered neighbors are placed into the queue in nondecreasing
degree order, with a declared tie-break. The resulting dequeue order is the CM
numbering.

The distance invariant remains ordinary BFS:

- vertices appear in nondecreasing root distance;
- every layer is contiguous in the ordering;
- an undirected edge joins only the same or adjacent layers;
- degree sorting changes only within-layer discovery order, not distances.

For disconnected patterns, each component needs its own start and traversal;
the order in which components are concatenated is another policy choice.

## 4. A layer-width upper bound on the produced bandwidth

Let `w_i=|F_i|` for the CM root. Because adjacent vertices lie in `F_i` and
`F_i` or in `F_i` and `F_(i+1)`, any ordering that keeps layers contiguous obeys

```text
bw(pi) <= max_i (w_i + w_(i+1) - 1),
```

with absent boundary layers treated as zero. Same-layer edges alone span at
most `w_i-1`; adjacent-layer edges fit inside the union of two consecutive
blocks.

This is an upper bound for that layer ordering, not an equality and not a bound
on optimal graph bandwidth. Within-layer tie-breaking determines which actual
edges approach the block extremes.

## 5. Why low-degree-first is heuristic

Low-degree-first ordering tries to keep the next frontier narrow and avoid
numbering a highly branching vertex too early. It does not optimize the global
maximum edge span by proof. Future connections among already queued vertices
and later layers are only partially visible to the local rule.

A star is a calibration. Rooting at a leaf yields layers of sizes `1,1,n-2`.
A layer-contiguous order puts the center next to the root but far from some
other leaves, producing bandwidth `n-2`. Yet placing roughly half the leaves on
each side of the center gives optimal bandwidth about half as large. Exact BFS
layers can therefore constrain an ordering away from the global bandwidth
optimum.

By contrast, an endpoint-rooted path produces its natural order and bandwidth
one. A complete graph has bandwidth `n-1` under every ordering, so no heuristic
can improve it.

## 6. Reverse Cuthill--McKee preserves bandwidth exactly

Reverse Cuthill--McKee assigns the CM list in reverse. If

```text
pi_R(v) = n+1-pi(v),
```

then for every edge

```text
|pi_R(u)-pi_R(v)| = |pi(u)-pi(v)|.
```

Therefore CM and its exact reversal have identical bandwidth. Reversal is used
because profile and envelope are directional and can change, often favorably.
Any claim that reversal itself reduced bandwidth must involve a different root,
tie-break, preprocessing step, or measurement error.

The comparative literature studies conditions under which reverse ordering has
a smaller profile. It is not a universal theorem that every RCM profile is
strictly smaller.

## 7. Root choice and pseudo-peripheral vertices

A peripheral vertex has eccentricity equal to graph diameter. Starting near a
periphery tends to produce a long, narrow level structure, which is often useful
for CM-style orderings.

Pseudo-peripheral procedures repeatedly:

1. run BFS from a candidate;
2. choose a low-degree vertex from the last layer;
3. run BFS again;
4. continue while the number of levels increases.

Termination proves that this local sweep rule found no level-count improvement
under its candidate policy. It does not generally prove that the returned
vertex is truly peripheral or that the observed eccentricity is the diameter.
This is the same certificate boundary exposed by note 21's double-sweep
counterexample.

Root choice, neighbor degree order, and tie-breaking are separate inputs to the
final numbering. Reproducibility requires all three.

## 8. BFS correctness versus ordering quality

CM can be perfectly correct as BFS while producing a poor bandwidth/profile
ordering. Conversely, a useful ordering does not strengthen the traversal's
distance certificate. The two validation layers are:

### Traversal validation

- every component vertex appears exactly once;
- recorded depth is exact;
- queue order is nondecreasing in depth;
- every edge spans depth difference at most one.

### Ordering validation

- `pi` is a bijection;
- bandwidth and profile are recomputed from the original pattern;
- the exact root, degree convention, and tie-break are recorded;
- comparison includes the original, CM, and RCM orders under the same metric;
- no heuristic output is labeled globally optimal without an independent bound.

## 9. Cayley and Schreier interpretation

A finite simple Cayley graph is regular, so the CM degree sort supplies no
preference. Root translation also preserves layer sizes. The ordering is then
driven mainly by generator enumeration and tie-breaking, not by the celebrated
low-degree rule.

Every Cayley vertex is peripheral because every vertex has eccentricity equal
to the diameter. Pseudo-peripheral root search is therefore semantically
unnecessary once genuine Cayley transitivity and exact diameter scope are
established. It still cannot solve the within-layer ordering problem.

Schreier graphs may be irregular and need not inherit the same graph
automorphisms, so degree order and root choice can matter. The action orbit alone
does not justify Cayley conclusions.

For an implicit puzzle graph, a CM permutation of the entire state space first
requires a complete enumeration or another exact ranking of all relevant
states. This preprocessing object is distinct from the compact state rank used
by ordinary implicit BFS.

## 10. What reordering preserves

A vertex permutation preserves:

- adjacency up to relabeling;
- all distances and shortest-path multiplicities;
- component structure, diameter, girth, and spectrum;
- BFS layer cardinalities from corresponding roots.

It can change:

- adjacency-array locality and edge span in storage;
- owner assignment if partitioning hashes the numeric ID;
- communication pattern under range partitioning;
- deterministic parent and frontier order;
- compression behavior of sorted IDs.

Thus a reorder can alter hardware work without altering the mathematical BFS
problem. If owner mapping depends on IDs, comparing before and after reordering
changes both layout and partition policy unless ownership is held invariant.

## 11. GPU and multi-GPU boundary

CM/RCM preprocessing has its own repeated frontier, degree-ordering, and global
numbering costs. Applying the resulting permutation may require rewriting
adjacency and remapping every state-indexed artifact.

For explicit graphs, a narrower storage band may improve locality for some
kernels, but bandwidth alone does not predict coalescing, cache reuse, warp
divergence, or total traversal time. For implicit Cayley graphs, computing and
storing a global CM order may cost more than the traversal it was intended to
support.

In multi-GPU execution, a global BFS ordering requires consistent layer and
tie-break decisions across owners. Local CM orders concatenated independently
are not the same as one global CM order. Report separately:

- ordering construction and permutation cost;
- bandwidth, profile, wavefront, and fill metrics;
- graph storage and owner mapping before and after;
- BFS frontier/visited correctness;
- measured traversal and communication effects.

No matrix-layout heuristic should be reported as a BFS throughput improvement
without an end-to-end measurement on the permuted representation.

## Sources

- E. Cuthill and J. McKee,
  [*Reducing the Bandwidth of Sparse Symmetric Matrices*](https://doi.org/10.1145/800195.805928),
  Proceedings of the 24th ACM National Conference, 1969. Introduces the
  degree-ordered level heuristic.
- E. Cuthill, D. J. Rose, and R. A. Willoughby,
  [*Comparative Analysis of the Cuthill--McKee and the Reverse Cuthill--McKee Ordering Algorithms*](https://doi.org/10.1137/0713020),
  SIAM Journal on Numerical Analysis 13(2), 1976. Separates direct and reverse
  ordering behavior for sparse symmetric systems.
- A. George and J. W. H. Liu,
  [*An Implementation of a Pseudoperipheral Node Finder*](https://cs.uwaterloo.ca/research/tr/1976/CS-76-44.pdf),
  ACM Transactions on Mathematical Software 5(3), 1979. Develops practical
  pseudo-peripheral root selection.
- C. H. Papadimitriou,
  [*The NP-Completeness of the Bandwidth Minimization Problem*](https://doi.org/10.1007/BF02280884),
  Computing 16(3), 1976. Establishes the exact optimization boundary.
- Notes 4, 10, 21, 46, 51, 71, 72, and 120 supply this repository's queue,
  frontier, eccentricity, width, ownership, arbitrary-profile, peripheral, and
  distance-histogram distinctions.

## Takeaway

Cuthill--McKee is ordinary BFS plus a low-degree tie policy that turns layers
into a linear numbering. BFS proves the levels; it does not prove that the
numbering minimizes bandwidth or profile. Exact reversal preserves bandwidth
and changes only directional objectives such as profile. In regular Cayley
graphs degree and root heuristics largely disappear, leaving generator order,
tie-breaking, and the cost of materializing a global permutation as the real
questions.
