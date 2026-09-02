# REF-039: BFS frontiers in random regular samples

## Question

How long do random 3- and 4-regular BFS waves follow their regular-tree bounds,
and does equal degree imply equal radial work inside a layer?

## Method

- `n=2000`, degrees three and four, 20 deterministic seeds each.
- Pair shuffled stubs and reject loops/parallel edges.
- Verify exact final degree for every vertex.
- Run exact BFS from vertex zero.
- Record generation attempts, connectivity, frontier profile, root
  eccentricity, outward multiplicity, and per-layer radial-count ranges.
- Docker-only Rust execution.

## Retained failures

1. Rust 1.85 rejected unstable `std::iter::repeat_n`; replaced with stable
   `repeat().take()`.
2. The added nested-tuple instrumentation failed `rustfmt --check`; the exact
   formatter spacing was applied.

Both command chains stopped before the affected computation. The final gate
passed.

## Result

```text
r=3: connected 20/20, attempts mean 6.45 [1,37],
     root eccentricity mean 12.90 [12,13]
r=4: connected 20/20, attempts mean 46.80 [2,134],
     root eccentricity mean 9.00 [9,9]
```

Representative profiles were

```text
r=3: [1,3,6,12,24,46,90,171,288,436,497,328,92,6]
r=4: [1,4,12,36,106,282,582,716,258,3]
```

The representative 4-regular depth-four layer already had inward range `[1,2]`,
same-layer range `[0,1]`, and outward range `[2,3]`, rejecting
degree-regular-to-distance-regular transfer.

## Status

Pass after one toolchain-compatibility failure and one formatting-only failure.
This is finite pseudorandom evidence, not a uniformity or performance test.

