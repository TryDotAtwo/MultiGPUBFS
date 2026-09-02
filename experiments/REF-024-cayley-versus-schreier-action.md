# REF-024: the same generators under Cayley and Schreier BFS

Date: 2026-08-28

Status: pass after one formatting-only failed gate

## Question

How does the BFS signature change when the adjacent transpositions
`s0=(01), s1=(12)` act on group elements of `S3` versus on one marked point?

## Declared graphs

- **Cayley:** vertices are all six permutations; an edge applies `s0` or `s1`
  to the current permutation.
- **Schreier:** vertices are the three possible positions of point `0`; the
  same labeled moves act on that point.
- Both are directed labeled occurrence graphs with involutive moves. Loops are
  retained as labeled occurrences and counted separately from simple neighbors.

The Rust file is a standalone semantic probe, not a reusable implementation,
benchmark, or optimization.

## Test-first Docker evidence

The RED compile failed on the intentionally absent action, distance, and row
functions. After minimal implementation, all three semantic tests passed. The
first full GREEN gate then stopped only because `rustfmt --check` requested
three line wraps. After that correction, tests, formatting, compilation, and
execution all passed in the read-only workspace mount of:

```text
image: multigpubfs-rust-toolchain:dev
image id: sha256:764a443c2ddc39b28b8fbb0b1495656984ea5ee8dd82f7f435f2069a6574ce69
rustc: 1.75.0
tests: 3 passed, 0 failed
```

All compilation and calculation occurred in Docker. No GPU was requested.

## Exact observations

### Regular Cayley action

The frontiers are `1,2,2,1`. There are no loops. While forming depth three,
two candidates converge on the longest permutation, witnessing

```text
s0 s1 s0 = s1 s0 s1.
```

This is equality of two group elements represented by different words.

### Point Schreier action

The frontiers are `1,1,1`. At the root, `s1` is a loop:

```text
s1 != identity in S3
0 * s1 = 0.
```

Thus the first closed state word has length one, long before the length-six
Cayley relation underlying the hexagon. It witnesses membership in the base
state stabilizer, not group identity. At point `2`, `s0` similarly becomes a
loop because stabilizers are conjugated across the orbit.

### Representative distance

The state distance from point `0` to point `2` is two, via `s0 s1`. The chosen
group representative `s0 s1 s0` maps point `0` to the same target but lies at
Cayley distance three. Exact state distance therefore minimizes word length
over every representative of the target coset; an arbitrary representative
can overstate it.

## Counter interpretation

| Signal | Cayley S3 | Schreier point action |
|---|---:|---:|
| root loop | 0 | 1 |
| first candidate convergence | forming depth 3 | none |
| total unique states | 6 | 3 |
| frontier sequence | 1,2,2,1 | 1,1,1 |

Keeping labeled occurrences matters: deleting loops yields the simple path on
three point states and erases the shortest stabilizer witness, although
distances between distinct point states remain unchanged.

## Artifacts

- `experiments/ref024_cayley_schreier_probe.rs`
- `experiments/REF-024-cayley-versus-schreier-action.txt`

## Scope

- exact only for these two finite actions;
- no Cube or Megaminx inference;
- no claim that loops should be retained by every output contract;
- no timing, GPU, multi-GPU, or performance claim.
