# REF-035: exact BFS on Hamming graphs

## Question

Do complete small Hamming graphs realize the closed distance, frontier,
intersection, and shortest-path formulas used as a BFS calibration model?

## Method

- Rust with dense base-`q` integer states.
- Generate every one-coordinate symbol replacement.
- Complete BFS from the all-zero word.
- Audit every state against Hamming weight, expected intersection counts, and
  factorial shortest-path multiplicity.
- Run `d=1..5`, `q=2..4` only in Docker.

## Result

All 15 fixtures passed. The largest was `H(5,4)` with 1,024 states, degree 15,
diameter 5, and layers `[1,15,90,270,405,243]`. Every fixture had zero distance,
intersection, and shortest-path-count mismatches.

The complete console output is reproducible from
`experiments/hamming_graph_bfs_probe.rs`.

## Status

Pass. This is exhaustive only for the declared small fixtures and is not
performance evidence.

