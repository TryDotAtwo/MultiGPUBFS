# REF-044: BFS wave branching under additive small-world shortcuts

## Question

How can a small number of long-range unit edges change BFS depth, frontier
width, affected destinations, and owner participation while barely changing
mean degree?

## Method

- `n=4096`, root `n/4`, base graph `C_n^2` with offsets `+-1,+-2`.
- Add exactly `0,4,16,64,256,1024` uniformly proposed nonloop, nonduplicate
  shortcut edges.
- Twenty deterministic samples per shortcut count.
- Assert every new distance is no larger than exact baseline
  `ceil(cyclic_distance/2)`.
- Record eccentricity, mean distance, frontier peak, benefited vertices,
  contiguous/striped owner routing, and first mixed-owner depth.
- Rust build and execution only in Docker.

This fixed-count additive model is explicitly related to, but not identical
with, canonical Watts-Strogatz rewiring or random-count Newman-Watts ensembles.

## Retained failure

The first Docker gate stopped before compilation on three `rustfmt --check`
changes.  The mechanical correction was applied; the complete
format/compile/assert/run gate passed.

## Result

```text
shortcuts degree eccentricity mean distance peak  benefited first mixed remote
0         4.000    1024.00       512.25      4.00   0.0000     512.00    0.0007
4         4.002     552.90       279.24     16.25   0.6755     136.90    0.0010
16        4.008     255.50       138.63     42.30   0.8582      58.30    0.0017
64        4.031      85.75        46.47    130.40   0.9538      12.25    0.0045
256       4.125      33.70        19.77    414.05   0.9801       5.50    0.0159
1024      4.500      14.65         9.30    999.15   0.9908       1.95    0.0562
```

Striped ownership stayed near one-half remote.  About half of shortcut edges
crossed the contiguous cut, but they caused work to reach the second owner far
earlier than the local ring wave.

## Interpretation boundary

These finite samples illustrate a crossover from local waves to shortcut-seeded
wave branching.  They do not estimate a universal scaling function, measure
clustering, choose a partition, or benchmark BFS/GPU throughput.  Added unit
edges change the graph metric and are not an implementation-only optimization.

## Status

Pass after one formatting-only failure.
