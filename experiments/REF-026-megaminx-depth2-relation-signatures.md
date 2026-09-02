# REF-026: current Megaminx depth-two relation signatures

Date: 2026-08-28

Status: pass after two intentional RED cycles and one formatting-only failed gate

## Question

Which concrete short relations explain the first divergence between reduced
move words and exact state frontiers in the current 24-move Megaminx Cayley
graph?

## Fixed input and equality

REF-026 uses the same read-only current configuration as REF-025:

```text
D:\100XH100\data\puzzle_info.json
sha256: 1780a8368d504fd75f448d25e5bede9adb498b35db6a3251e920bbc8524adfca
logical positions: 120 unique values
moves: 12 face turns plus their 12 inverses
```

All state grouping uses full `Vec<u8>` equality. Hashes select Rust `HashMap`
or `HashSet` buckets but are not accepted as state equality.

## Test-first Docker evidence

The first RED compile failed on the absent depth-two word and transition-profile
functions. A second RED cycle added a stricter face-pair contract and failed on
the intentionally absent fields. After minimal implementation, five tests
passed, including both inherited REF-025 tests. The first full gate requested
only rustfmt line wrapping; the final gate passed.

```text
image: multigpubfs-rust-toolchain:dev
image id: sha256:764a443c2ddc39b28b8fbb0b1495656984ea5ee8dd82f7f435f2069a6574ce69
workspace mounts: read-only
tests: 5 passed, 0 failed
GPU requested: no
```

## Forming `F2`: every convergence is commutation

Expanding the 24 depth-one states gives `24*24=576` labeled occurrences.
Exactly 24 are immediate inverse returns, leaving 552 non-backtracking
length-two words.

Those 552 words reach 408 states:

```text
552 word occurrences
-408 unique endpoints
=144 convergence extras.
```

The exact endpoint multiplicity classification is unusually clean:

- 264 states have one reduced length-two word;
- 144 states have exactly two;
- no state has multiplicity above two;
- every two-word group is exactly `(a,b)` versus `(b,a)`;
- no other length-two collision occurs.

After stripping move direction, the 144 groups are 36 unordered face pairs,
and every pair supplies all four direction combinations:

```text
ab       = ba
a^-1 b   = b a^-1
a b^-1   = b^-1 a
a^-1b^-1 = b^-1a^-1.
```

Thus all first candidate convergence is explained by commuting face turns.
It is not merely compatible with commutation; the full word-pair audit leaves
no unclassified depth-two collision.

## Expanding `F2`: the order-five boundary

The 408 states produce 9,792 labeled transition occurrences:

```text
backward to F1       552
same-level in F2      24
older than F1          0
forward candidates  9216
```

Every one of the 24 directed same-level occurrences was checked against one
move orientation `g` and has the form

```text
g^2 --g--> g^3 = g^-2.
```

The reverse direction is supplied by `g^-1`. Hence these are the 24 directed
views of 12 undirected boundary edges closing the face-turn 5-cycles `g^5=e`.
No other same-level edge occurs in `F2`.

This realizes both note 60 signatures in one real generator set:

- length-four commutators appear as candidate convergence while forming `F2`;
- length-five power relations appear as same-level edges while expanding `F2`.

The graph therefore has girth four, even though its individual generators have
order five.

## What the 3,008 next-layer convergences do not yet mean

The 9,216 forward occurrences from `F2` form 6,208 unique `F3` states, leaving
3,008 convergence extras. This count is exact, but its presentation-level
interpretation is not yet classified.

It must not be read as “3,008 new primitive relations.” In a Cayley graph,
translations and overlaps of the 36 already known commutation squares create
later collisions. Several word pairs can also be consequences of a smaller
set of relators. REF-026 records the count and leaves witness classification
open.

## BFS lessons

1. Reduced word count, labeled occurrence count, and unique frontier size are
   three different objects even at depth two.
2. Candidate convergence and same-layer edges can expose different relations
   during the same frontier expansion.
3. The sum of backward occurrences from `F2` is 552, exactly the number of
   accepted reduced-word occurrences that formed `F2`; multiple shortest
   parents account for `552-408=144` alternate predecessors.
4. In an undirected graph an edge from `F_d` cannot reach earlier than
   `F_(d-1)`; the observed `older_ball=0` is required by the BFS distance
   inequality, not a special Megaminx property.
5. Algebraic equality does not state whether equal candidates meet in one warp,
   one batch, one owner, or different GPUs.

## Artifacts

- `experiments/ref026_megaminx_relation_probe.rs`
- `experiments/REF-026-megaminx-depth2-relation-signatures.txt`

The retained text output has SHA-256
`34264f9f6eb5e542f509a1f98e61596187b364aa1616edd7f8cff1df192d796e`
and matched a fresh Docker run after newline normalization.

## Scope

- current config checksum above;
- full-state exact analysis through expansion of `F2`;
- no classification yet of the 3,008 `F3` convergence extras;
- no timing, GPU, multi-GPU, or optimization conclusion;
- no change to CayleyPy or production search code.
