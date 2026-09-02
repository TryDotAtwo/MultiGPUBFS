# Cayley versus Schreier BFS: what is the vertex?

The phrase "Cayley search" can hide three different vertex models:

1. a group element `g in G`;
2. a concrete puzzle configuration `x` in an orbit of a group action;
3. a coset representing all group elements that produce the same configuration.

BFS is exact only relative to the chosen vertex identity. Confusing these
models changes the graph before any frontier algorithm begins.

## Right Cayley convention

Fix a generator collection `S` and directed labeled edges

```text
g --s--> g*s.
```

A path with labels `s1, s2, ..., sk` ends at

```text
g*s1*s2*...*sk.
```

Starting from the identity, BFS depth is the minimum positive-word length over
the allowed generator alphabet. If `S=S^-1`, this is the usual symmetric word
metric. If inverse generators are absent, it is a directed distance even though
every group element is algebraically invertible.

For right edges, left multiplication is a graph automorphism:

```text
a*g --s--> a*g*s.
```

Consequently

```text
dist_S(x,y) = length_S(x^-1*y)
```

for directed or undirected distance with the same allowed alphabet. This is the
precise reason that a Cayley search from arbitrary `x` to `y` can normalize to
identity-to-`x^-1*y`.

The same translation preserves more than the minimum length. A labeled path

```text
x --s1,...,sk--> y
```

exists exactly when `s1*...*sk=x^-1*y`. Left multiplication by `x^-1` keeps
every label and gives a bijection with identity-rooted paths to `x^-1*y`.
Therefore, under the same vertex/labeled-path convention,

```text
sigma_x(y) = sigma_e(x^-1*y),
```

and the complete shortest-path structures translate as well. Inverse closure
is unnecessary; the statement holds for a directed positive alphabet on its
reachable pairs.

The order is convention-sensitive. For left Cayley edges `g->s*g`, right
multiplication by `x^-1` is the label-preserving translation and the normalized
target is `y*x^-1`. In a noncommutative group, substituting `y*x^-1` for
`x^-1*y` under the right-edge convention can query a different element and a
different path count.

Right multiplication by a common `a` is generally **not** an automorphism:

```text
(g*s)*a = g*(s*a)
(g*a)*s = g*(a*s).
```

They agree only under extra commutation/conjugation conditions. Cayley metrics
are left invariant under this convention, not automatically bi-invariant.

## Left Cayley convention

If edges are instead

```text
g --s--> s*g,
```

then labels `s1,...,sk` produce

```text
sk*...*s2*s1*g.
```

The visible label order and the multiplication order are reversed. Right
multiplication is now the obvious graph automorphism, and

```text
dist_S(x,y) = length_S(y*x^-1).
```

Neither convention is more correct. A system becomes incorrect when its move
application, stored label order, parent reconstruction, and replay use
different conventions. One-move and commuting-move tests cannot reveal this;
a noncommuting two- or three-move word can.

## Directed generators: an inverse operation need not be a forward-allowed move

Suppose `s` is a permutation, so `s^-1` exists algebraically, but only `s` is in
the allowed move collection. The edge `g -> g*s` does not imply the edge
`g*s -> g` in the original forward graph. The reversed graph always contains
the reverse of every original edge: there, applying `s^-1` traverses from
`g*s` to its predecessor `g`, even when `s^-1` is not forward-allowed. The
stored witness from that predecessor uses the original forward label `s`.

Consequences:

- forward BFS still gives exact shortest directed move count;
- reverse BFS must use predecessor generators `S^-1`;
- `dist(x,y)` need not equal `dist(y,x)`;
- algebraic word simplification must not introduce unavailable inverse moves;
- an undirected treatment silently solves a different move problem.

There is a finite/infinite distinction. In a finite group, if `S` generates
`G` as a group, positive powers can express each generator inverse, so the
directed Cayley graph is strongly connected, though distances need not be
symmetric. In the infinite group `Z` with `S={+1}`, `S` group-generates `Z`
when inverses are allowed algebraically, but directed search from `0` reaches
only nonnegative integers. "Generates the group" must therefore say whether it
means group or positive-monoid generation.

## From a group action to a Schreier graph

Let `G` act on configurations on the right:

```text
(x*g)*h = x*(g*h).
```

Fix a base configuration `x0` and its stabilizer

```text
H = {h in G | x0*h = x0}.
```

The reached puzzle vertices are orbit states `x0*g`, with moves

```text
x0*g --s--> x0*g*s.
```

Two group elements represent the same state exactly when

```text
x0*g1 = x0*g2
iff g1*g2^-1 in H
iff H*g1 = H*g2.
```

Thus a right action corresponds to **right cosets** `H*g` in `H\G`, and the
state graph has edges `H*g -> H*g*s`. For a left action, the analogous model
uses **left cosets** `g*H` in `G/H`. The position of the translating element
`g`, not the position of `H` in the quotient notation, determines the name.

If `H` is trivial, the action on the orbit is free and the orbit graph can be
identified with the Cayley graph. If `H` is nontrivial, several group elements
and words are one puzzle vertex. Group-level visited and state-level visited
then answer different questions.

## Why a Schreier graph is not merely a smaller Cayley graph

At each state there are still `|S|` labeled move occurrences, but after state
identity is applied:

- a generator may create a loop because it lies in the current state's
  conjugate stabilizer;
- distinct generators may reach the same neighboring state;
- unique simple-graph degree can vary even though labeled outdegree is fixed;
- relations modulo the stabilizer create collisions not present as equality of
  group elements.

Most importantly, the left-translation symmetry used by a right Cayley graph
does not automatically descend to arbitrary right cosets. Mapping
`H*g -> H*a*g` is not even well-defined for arbitrary `a`; mapping by right
`a` generally conjugates the generator labels rather than preserving fixed
`S`. A connected Schreier graph need not have the full vertex-transitivity or
distance normalization of its Cayley cover.

## Arbitrary start and goal in an orbit

Let

```text
start = x0*a
goal  = x0*b.
```

A move word `w` solves the puzzle exactly when

```text
x0*a*w = x0*b
iff a*w*b^-1 in H
iff w in a^-1*H*b.
```

So the normalized group-level target is generally a **set/coset-like subset**,
not the single element `a^-1*b`. The single target is valid only when the action
is free (`H={e}`) or when an additional choice/proof reduces the query without
changing its distance.

This also shows why choosing arbitrary representatives for configurations can
change a naive `a^-1*b` result: replacing `a` or `b` by another representative
changes the element but not the orbit state. The correct target condition must
be representative-independent.

### Shortest labeled-path counts target the whole coset-like set

Let `D` be the minimum word length among words in `a^-1*H*b`. Under labeled
path semantics,

```text
sigma(x0*a, x0*b)
  = |{label words w of length D : w in a^-1*H*b}|.
```

Choosing one representative can preserve neither distance nor multiplicity.
Even if it happens to choose a nearest representative, other representatives
of the same state may have equally short label words and contribute additional
paths.

For a minimal example, take additive `Z_4`, stabilizer `H={0,2}`, and generators
`{+1,-1}`. From state `H` to state `H+1={1,3}`, both one-letter words solve the
same state query: `+1` reaches representative `1` and `-1` reaches `3`. The
labeled shortest-path count is two, while searching only for representative
`1` reports one. If parallel labels are collapsed into one simple-support edge,
the vertex-path count may instead be one; that is a different graph/output
contract.

The same statement factors through identity-rooted Cayley counts. Let
`ell(g)` and `sigma_C(e,g)` be the shortest length and labeled shortest-word
count in the Cayley cover. For a finite target fiber,

```text
D = min_(h in H) ell(a^-1*h*b),

sigma_Sch(H*a,H*b)
  = sum_(h in H : ell(a^-1*h*b)=D)
      sigma_C(e,a^-1*h*b).
```

Each label word evaluates to one group element, so the nearest representatives
partition the shortest solution words into disjoint endpoint classes. In the
`Z_4` example, representatives `1` and `3` each contribute one Cayley word at
depth one, giving Schreier labeled count two. This sum does not compute a
simple-support vertex-path count after label aliases have been collapsed.

The two count conventions are related by local edge multiplicities. In the
directed state support, define

```text
m(u,v) = number of declared generator-label occurrences s with u*s=v.
```

A fixed support vertex path `P=(v_0,...,v_k)` has exactly

```text
product_(i=0)^(k-1) m(v_i,v_(i+1))
```

labeled lifts: at each step one independently chooses a label realizing that
support edge. Therefore

```text
labeled shortest count
  = sum over shortest support paths P
      product over edges of P m(u,v).
```

For the `Z_4/H` witness, the sole one-edge support path has multiplicity two,
recovering the two labeled words. The formula counts declared label
occurrences; duplicate physical delivery is not another label. Unit-cost loops
do not occur on a positive-length shortest path between distinct states, though
they remain part of labeled transition work and of other output conventions.

Equivalently, labeled shortest counts obey a weighted recurrence on the support
shortest-path DAG:

```text
sigma(v)
  = sum_(u : d(u)+1=d(v)) sigma(u) * m(u,v).
```

Every shortest prefix ending at `u` can be extended by each declared label that
realizes support edge `u->v`. Thus integer edge multiplicity is sufficient to
continue a count-only computation without treating parallel labels as separate
support neighbors in the mathematical recurrence.

The scalar `m(u,v)` is not sufficient for label-sensitive outputs. It cannot
reconstruct which moves realize the edge, choose a canonical label, enumerate
words, or retain per-label metadata. Multiplicity compression preserves the
sum, not the identities of its summands.

Nor do separate histograms of prefix counts and edge multiplicities determine
the next count. Use a support diamond with

```text
sigma(a)=2, sigma(b)=1
```

and the same next-boundary multiplicity multiset `{2,1}` in two labeled graph
instances. If the larger edge multiplicity is attached to `a`, then

```text
sigma(t)=2*2 + 1*1 = 5.
```

If the multiplicities are crossed between the two parents, then

```text
sigma(t)=2*1 + 1*2 = 4.
```

Both instances can keep the same support diamond, per-layer frontier sizes,
total outward multiplicity, and multiplicity histograms. The missing datum is
the pairing between each parent's accumulated prefix mass and its endpoint
multiplicity. Marginals do not determine their weighted dot product.

The marginals still give sharp pairing-only bounds. Sort nonnegative prefix
masses and endpoint multiplicities as

```text
x_1 <= ... <= x_n,
y_1 <= ... <= y_n.
```

For any unknown pairing `pi`, the rearrangement inequality gives

```text
sum_i x_i*y_(n+1-i)
  <= sum_i x_i*y_(pi(i))
  <= sum_i x_i*y_i.
```

Opposite sorting minimizes the dot product and equal sorting maximizes it. A
two-pair swap proves the direction: aligning `x_i<=x_j` with `y_p<=y_q`
instead of crossing them changes the sum by
`(x_j-x_i)*(y_q-y_p)>=0`. For multisets `{1,2}` and `{1,2}`, the interval is
exactly `[4,5]`, matching the two counterexample instances.

These are sharp over all abstract pairings consistent with the marginals. A
particular Schreier action, support topology, or generator set may forbid some
pairings and narrow the attainable range. Histograms therefore provide an
envelope, not the exact graph-specific count.

### A three-state counterexample

Write `G=Z_6` additively, allow only the directed generator `S={+1}`, and take
the subgroup/stabilizer `H={0,3}`. The Schreier states are the three cosets

```text
H={0,3}, H+1={1,4}, H+2={2,5},
```

forming a directed 3-cycle under `+1`. From `H`, the state `H+1` has distance
one. But the same target state may be represented by `b=1` or by `b=4`:

```text
directed Cayley distance 0 -> 1 = 1
directed Cayley distance 0 -> 4 = 4
Schreier distance H -> H+1 = 1.
```

The exact normalized target set is `H+1={1,4}`, whose nearest member has length
one. Selecting the arbitrary representative `4` and searching only for it
returns the wrong state distance even though every group operation is valid.

## Stabilizer is not an optional symmetry quotient

The stabilizer expresses transformations that already leave the same concrete
base state unchanged under the declared action. Identifying `g` and `h*g` for
`h in H` is therefore state equality.

By contrast, deciding that rotated or reflected configurations should count as
equivalent introduces a new equivalence relation and usually a new quotient
problem. It may change a fixed-orientation goal into a goal orbit. These two
operations must not be conflated merely because both reduce a state count.

## Parity and bipartiteness

For an undirected, loopless Cayley graph with inverse-closed `S`, bipartiteness
means word-length parity is well-defined: every word representing the identity
has even length. Equivalently, there is a homomorphism

```text
chi: G -> Z_2
```

that maps every generator to `1`. Its two fibers are the bipartition.

This explains the adjacent-transposition `S_n` example: permutation sign maps
every adjacent swap to odd parity. Adding an even 3-cycle as a one-edge
generator destroys that particular parity rule and creates odd cycles relative
to the new alphabet. Adding the identity generator creates loops and therefore
also destroys graph bipartiteness, even though it does not change distances.

### Why checking defining relators is enough

Suppose the group has a complete presentation `G=<S | R>` and every symbol in
the Cayley generator alphabet is intended to change color. On the free group
over `S`, assigning every generator to `1 in Z_2` defines a homomorphism;
inverses also map to `1` because `-1=1` in `Z_2`. A relator maps to zero exactly
when its written length is even.

If every defining relator in `R` has even length, their conjugates, inverses,
and products also map to zero. Thus the whole normal closure of `R` lies in the
parity kernel, and the map descends to `G`; no odd derived relation can appear.
Conversely, one odd defining relator prevents the map from descending.

```text
Z_3 = <a | a^3=e>   fails: the defining relation has odd length;
Z_4 = <a | a^4=e>   passes: parity is a well-defined homomorphism.
```

This is only as sound as the presentation and alphabet declaration. An
incomplete relation list is not a proof. Adding a redundant Cayley generator
also adds a symbol required to map to `1`: declaring `b=a^2` introduces the
odd-length relator `b a^-2`, so the enlarged alphabet loses word parity even
when the original `a`-only Cayley graph had it.

For Schreier graphs, group parity alone may not descend to states: two elements
in one coset must have the same parity. A sufficient and necessary compatibility
condition for this parity map is `H subset ker(chi)`. If the stabilizer contains
an odd element, one state has representatives of both parities and cannot carry
a well-defined Cayley word parity.

## Small semantic tests

1. Declare right/left action, permutation-array meaning, and composition order.
2. Replay a noncommuting word of length at least two in both the oracle and an
   independent composition model.
3. Check that depth one equals the unique legal one-move state set, while also
   recording labeled multiplicities and loops.
4. Verify reverse-search predecessors by replaying the original forward label.
   Use a directed three-cycle as a missing-inverse fixture: the original graph
   has asymmetric distances, while predecessor traversal still works.
5. Search for a nonidentity stabilizer element and compare group-element count
   with unique orbit-state count.
6. Verify that two representatives of the same state give the same normalized
   target **set**, not necessarily the same target element.
7. Compare direct start-to-goal BFS with identity normalization only on an
   instance where free-action/Cayley assumptions are independently known.
8. Replay every returned parent path on concrete configurations, not only on
   ranks or group words.
9. Check whether a claimed parity invariant is constant across every coset.
10. State explicitly whether duplicate generator labels and shortest-word
    multiplicities are part of the output contract.

## Counterexamples

### Testing only one move

Left/right composition bugs pass every one-move round trip. Noncommuting `a,b`
with `a*b != b*a` expose the reversed replay order.

### Treating every configuration as a unique group element

If `h != e` fixes `x0`, then `e` and `h` are distinct Cayley vertices but the
same configuration. A group-element visited table explores a covering graph
with redundant semantic states.

### Normalizing a Schreier query to `a^-1*b`

When `H` is nontrivial, valid solution words form `a^-1*H*b`. One representative
may not be the shortest member and may even make results depend on arbitrary
encoding choices.

### Adding inverse edges because moves are permutations

This converts a directed allowed-move graph into its symmetric closure and can
shorten or create paths that the puzzle rules forbid.

### Assuming regular unique degree

A Schreier graph has one labeled outgoing occurrence per generator, but loops
and coincident destinations can make its simple-state neighbor count smaller
and state-dependent.

## Sources

- Alexander Hulpke, *Computational Group Theory* lecture notes,
  [Chapter VII](https://www.math.colostate.edu/~hulpke/lectures/m501/notes.pdf),
  for orbit, stabilizer, Schreier graph, and Schreier tree relationships.
- Henry Wilton, *Topics in Geometric Group Theory*,
  [Cayley graphs and word metric](https://dec41.user.srcf.net/h/IV_M/topics_in_geometric_group_theory/1_1),
  for right-edge Cayley conventions, left invariance, and the word metric.
- Yaroslav Vorobets, *Notes on Schreier graphs*,
  [PDF](https://people.tamu.edu/~yvorobets/Research/Schreier.pdf), for actions,
  stabilizers, coset graphs, and their relation to Cayley graphs.
- The `multigpu_beam` expert highlighted replay convention, nontrivial
  stabilizers, directed inverse handling, and small cross-start tests. The
  single-element normalization suggestion was refined here to the full target
  condition `w in a^-1*H*b` for a non-free right action.

## Current synthesis

A Cayley BFS searches words as group elements; a Schreier BFS searches the
effect of words on configurations. They coincide only for a free action on the
chosen orbit. Right/left convention determines replay order, invariant side,
and normalization formula. Before choosing ranks, hashes, frontiers, or GPU
ownership, the search must state whether a vertex is a group element, a coset,
or a concrete configuration—otherwise a fast exact BFS can be exact for the
wrong graph.
