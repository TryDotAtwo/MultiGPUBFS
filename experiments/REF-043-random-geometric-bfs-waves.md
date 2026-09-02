# REF-043: spatial BFS waves in square and torus random geometric graphs

## Question

How do radius, geometric boundary, root location, and owner partition change
components, hop-distance stretch, frontier waves, and cross-owner edges?

## Method

- `n=2000`, 20 deterministic point sets.
- Unit square and flat torus graphs built on identical points.
- Radius multipliers `0.8,1.0,1.3,2.0` of
  `sqrt(log(n)/(pi n))`.
- Exact all-pairs radius decisions followed by exact BFS.
- Roots nearest the geometric center and corner.
- Assert `d_G>=ceil(d_E/r)` for every reached center-root pair.
- Compare vertical spatial ownership with ID-parity ownership.
- Rust build and execution only in Docker.

The all-pairs sampler is transparent but deliberately nonscalable.  This is not
a neighbor-index or BFS performance benchmark.

## Retained failures

1. The first Docker gate stopped because `rustfmt --check` requested three
   mechanical changes.
2. The first completed run produced `NaN` stretch for sparse samples with no
   eligible reached pair beyond one radius.  Those aggregates were rejected.
   The corrected probe accumulates ratio sums plus explicit pair counts and
   recomputes every row.

The final format/compile/assert/run gate passed.

## Result

```text
mult  topology  connected  largest  degree  stretch  spatial remote
0.8   square       0/20     0.6184    4.74    2.452       0.0120
0.8   torus        0/20     0.8979    4.86    2.901       0.0239
1.0   square       0/20     0.9930    7.36    1.563       0.0146
1.0   torus        6/20     0.9986    7.58    1.557       0.0286
1.3   square      16/20     0.9999   12.34    1.344       0.0191
1.3   torus       19/20     1.0000   12.83    1.342       0.0371
2.0   square      20/20     1.0000   28.63    1.240       0.0295
2.0   torus       20/20     1.0000   30.39    1.239       0.0565
```

ID-parity ownership remained near `0.50` remote in every row.  At multiplier
`1.3`, square center/corner mean finite eccentricities were `20.55/38.55`, while
the torus values were both `19.95`.  The representative square corner wave
remained wholly on one spatial owner through depth 14.

## Interpretation boundary

Stretch is pair-weighted over reached center-root pairs with `d_E>=r`; sparse
rows do not compare identical pair populations.  Finite eccentricity in a
disconnected graph is component eccentricity, not diameter.  The results do not
estimate a sharp threshold, prove an optimal partition, or measure GPU speed.

## Status

Pass after formatting-only failure and rejected zero-denominator aggregation.
