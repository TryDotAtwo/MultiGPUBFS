# BFS on de Bruijn and Kautz overlap digraphs

De Bruijn and Kautz digraphs turn a word into its shifted suffix plus one new
symbol. They have constant degree and logarithmic diameter in their number of
vertices, yet their BFS frontiers are not tree levels: distinct appended-word
histories quickly collide on the same suffix state.

These graphs make the BFS distinction between path histories and state identity
especially concrete. This note adds no optimizer, production implementation,
benchmark, or GPU code. A tiny exhaustive Rust probe checks finite examples.

## 1. Directed de Bruijn contract

For alphabet `A` of size `d`, the directed de Bruijn graph `B(d,n)` has all
length-`n` words as vertices. For every `a in A`,

```text
x_1 x_2 ... x_n  ->  x_2 ... x_n a.
```

Thus

```text
|V| = d^n,      indegree = outdegree = d.
```

Loops are possible, for example `000 -> 000`. This note uses directed distance
and keeps those loops in the transition relation. Removing orientation or
coalescing arcs defines a different graph.

## 2. State identity forgets history

After `k` moves, the state contains the last `n-k` symbols of the source
(when `k<n`) followed by the `k` appended symbols. The symbols shifted off the
left are no longer part of state identity.

There are `d^k` append histories of length `k`, but they need not yield `d^k`
new states:

- a history can return to an already visited word;
- histories from different parents can end in the same suffix word;
- some targets may already have been reached at a smaller depth through a
  source/target overlap.

Candidate count, unique next states, and new BFS frontier size are therefore
three different quantities even though every vertex has fixed outdegree.

## 3. Exact overlap-distance formula

For words `x,y` of length `n`, let

```text
ov(x,y) = max{k in [0,n] : suffix_k(x) = prefix_k(y)}.
```

Then directed de Bruijn distance is

```text
d_B(x,y) = n - ov(x,y).
```

Upper bound: retain the longest matching suffix of `x` and append the remaining
symbols of `y`. Lower bound: after fewer than `n-ov(x,y)` shifts, a longer
suffix of `x` would remain as a prefix of `y`, contradicting maximality.

This formula explains BFS without simulating a queue, but it does not remove
the semantics: it is a closed form for this particular implicit graph.

## 4. Diameter and the Moore-bound viewpoint

Every target can be reached by appending its `n` symbols, so diameter is at most
`n`. For `d>=2`, the pair `a^n,b^n` with `a!=b` has zero overlap and distance
`n`; hence the directed diameter is exactly `n`.

Since `|V|=d^n`, this is logarithmic in the state count. The graph is a standard
large directed degree/diameter construction, but note 119's distinction still
applies: small diameter constrains the number of sequential BFS levels, not the
number of vertices or edges processed.

## 5. Root-dependent frontier profiles

Diameter, regular degree, and total state count do not determine a root's layer
sizes. The layer

```text
F_k(x) = {y : ov(x,y)=n-k}
```

must exclude words with any longer overlap. Those exclusions depend on the
border and periodicity structure of `x`.

For `B(2,3)`, exhaustive BFS gives

```text
root 000 : [1, 1, 2, 4]
root 010 : [1, 2, 3, 2].
```

Both traversals cover the same eight vertices and have eccentricity three, but
their intermediate frontier geometries differ.

## 6. Kautz removes equal adjacent symbols

The directed Kautz graph `K(d,n)` uses an alphabet of size `d+1` and length-`n`
words with unequal consecutive symbols. It has the same shift-and-append edge,
but the appended symbol must differ from the current last symbol. Therefore

```text
|V| = (d+1)d^(n-1),      indegree = outdegree = d.
```

The overlap formula remains valid. If a positive overlap is retained, the next
target symbol differs from its predecessor because the target is a Kautz word.
If overlap is zero, the source's last symbol differs from the target's first;
otherwise overlap one would exist. Thus appending the unmatched target suffix
is always legal, and the same lower-bound argument applies.

For the tested `K(2,3)`:

```text
root 010 : [1, 2, 3, 6]
root 012 : [1, 2, 4, 5].
```

Again, regularity and diameter do not imply identical BFS layers at every root.

## 7. Why Kautz has more states at the same degree and diameter

At outdegree `d` and word length/diameter `n`, de Bruijn has `d^n` vertices,
while Kautz has

```text
(d+1)d^(n-1) = (1 + 1/d)d^n.
```

The adjacent-symbol restriction removes loops and changes allowable words, yet
the larger alphabet supplies more vertices. This is a degree/diameter property,
not a statement that every Kautz BFS frontier is larger than the corresponding
de Bruijn frontier.

## 8. Iterated line-digraph interpretation

`B(d,n)` is the `(n-1)`-fold line digraph of the complete directed graph on `d`
symbols with one loop per symbol. `K(d,n)` is the corresponding iterated line
digraph of the complete symmetric loopless digraph on `d+1` symbols.

A length-`n` word records a directed walk of `n-1` base arcs; shifting moves the
window forward by one arc. This connects the construction to note 88, but the
state here is a fixed-length window, not an unbounded trail-history state.

## 9. De Bruijn sequences are not BFS orders

A cyclic de Bruijn sequence contains every length-`n` word exactly once as a
cyclic substring. It corresponds to a Hamiltonian cycle in `B(d,n)` (or an
Eulerian-cycle construction one dimension lower).

BFS instead groups vertices by minimum directed distance from a root. A
Hamiltonian enumeration and a BFS enumeration can both visit every vertex, but
their ordering invariant and certificate are different. Short diameter does not
turn a Hamiltonian cycle into a shortest-path tree.

## 10. Directed and undirected conventions diverge

The shift operation is directional. Passing to the underlying undirected graph
adds reverse traversal of every shift arc and can reduce distances, change
diameter, merge predecessor/successor roles, and alter frontiers.

A result for the directed de Bruijn or Kautz digraph must not be reported as a
result for an undirected graph with the same labels unless the conversion is
explicitly part of the contract.

## 11. These labels are not canonical Cayley generators

For a fixed appended symbol `a`, the map

```text
x_1...x_n -> x_2...x_n a
```

is not a permutation: different first symbols collapse to the same successor.
Therefore these labeled moves are not group actions and do not directly define
a Cayley graph. The natural state transition forgets information, whereas a
Cayley generator acts bijectively.

Special graph isomorphisms or alternative group presentations require separate
proof. Constant indegree/outdegree, compact word encoding, or superficial
symmetry is not enough to transfer Cayley translation arguments such as one-root
frontier invariance.

## 12. Relation to implicit BFS

The successor oracle is compact:

```text
shift left; append each legal symbol.
```

But compact generation does not make the reachable state set a tree. Exact BFS
still distinguishes:

- word state identity;
- appended-symbol edge labels;
- duplicate candidates;
- first-discovery distance;
- parent/path multiplicity if requested.

The closed-form overlap distance is available only because this oracle has very
special structure. Treating an arbitrary implicit puzzle transition as a shift
register would be an unjustified model change.

## 13. Bounded Rust probe

`experiments/debruijn_kautz_bfs_probe.rs` exhaustively enumerates all states and
all ordered source-target pairs for `B(2,3)` and `K(2,3)`. It compares ordinary
queue BFS against the overlap formula and reports selected frontier profiles.

Observed in Docker with Rust 1.85.1:

```text
B(2,3) states=8 diameter=3 overlap_mismatches=0
B(2,3) root=[0, 0, 0] layers=[1, 1, 2, 4]
B(2,3) root=[0, 1, 0] layers=[1, 2, 3, 2]
K(2,3) states=12 diameter=3 overlap_mismatches=0
K(2,3) root=[0, 1, 0] layers=[1, 2, 3, 6]
K(2,3) root=[0, 1, 2] layers=[1, 2, 4, 5]
```

This is exhaustive evidence only for those two finite instances. The general
formula and diameter statements rely on the proofs above, not extrapolation
from the probe.

## 14. Failed-run observation

The first two Docker invocations used `bash -lc`; this local image's login shell
reset `PATH`, hiding the present `/usr/local/cargo/bin/rustc` rustup link. Both
failed before compilation with `rustc: command not found`. Calling the confirmed
toolchain by absolute path under `bash -c` succeeded. No algorithmic result was
inferred from the failed runs.

## 15. GPU and multi-GPU boundary

These graph families suggest, but do not prove, favorable execution properties:

- fixed outdegree bounds generated candidates by `d|F_k|`;
- fixed-length words admit compact encodings;
- successor generation is regular;
- small diameter bounds synchronization rounds for a full level-synchronous
  traversal.

They do **not** imply tree-like unique output, uniform frontier size, balanced
owner partitions, low communication, or high end-to-end throughput. Duplicate
rate and root-dependent layer profiles remain relevant.

A de Bruijn/Kautz graph used as a logical processor network is also distinct
from the BFS state graph being computed. Mapping owners onto such a topology
adds routing, congestion, link, and failure questions; it does not change the
logical BFS metric unless communication availability is made part of the state
transition model.

Any measurement should separate generated histories, unique candidates, new
states, layer profile, owner routing, physical hops, synchronization, and total
time. This is a conceptual measurement contract, not an optimization proposal.

## Sources

- N. G. de Bruijn,
  [*A Combinatorial Problem*](https://pure.tue.nl/ws/portalfiles/portal/4442708/597473.pdf),
  Proceedings of the Section of Sciences of the Koninklijke Nederlandse
  Akademie van Wetenschappen 49, 1946, for de Bruijn cycles/sequences.
- J. Bang-Jensen and G. Gutin,
  [*Digraphs: Theory, Algorithms and Applications*](https://www.math.ucdavis.edu/~saito/data/digraphs/bang-jensen-gutin_digraph-book.pdf),
  for the word-overlap definition and iterated-line-digraph view of de Bruijn
  and Kautz digraphs.
- G. J. M. Smit, P. J. M. Havinga, and P. G. Jansen,
  [*On the Design of Kautz Networks*](https://ris.utwente.nl/ws/portalfiles/portal/142753599/Smit91on.pdf),
  1991, for Kautz degree/diameter structure and interconnection-network context.
- Notes 27, 29, 35, 39, 44, 46, 64, 85, 88, 93, and 119 supply this
  repository's girth, complexity, growth, word/state, GPU-transfer, memory,
  multiplicity, directed-period, line-digraph, generator, and Moore-bound
  boundaries.

## Takeaway

De Bruijn and Kautz BFS is controlled by suffix-prefix overlap. Constant degree
and diameter `n` coexist with early duplicate collisions and root-dependent
frontier profiles. Their shift labels forget information and are not ordinary
Cayley generators. Compact implicit successors and attractive logical-network
geometry therefore do not by themselves prove uniform or efficient GPU BFS.
