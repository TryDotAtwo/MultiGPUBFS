# REF-041: size-biased BFS frontiers in configuration-model multigraphs

## Question

How do exact degree sequences with the same mean degree change the degrees seen
by BFS, local branching, component coverage, and per-layer occurrence work?

## Method

- `n=2000`, 20 deterministic pairings per distribution.
- Degree multisets: all four; half two/half six; half one/half seven.
- Shuffle degree labels, construct a uniform stub pairing, and retain loops and
  parallel-edge occurrences.
- Assert exact adjacency-list degrees.
- Run exact BFS from vertex zero and exact largest-component enumeration.
- Record theoretical excess mean, root conditioning, and representative
  layer size/degree/radial occurrence profiles.
- Rust build and execution only in Docker.

The direct pairing model is a multigraph.  This is intentional and is not a
claim about a simple-graph-conditioned sample.

## Retained failure

The first Docker gate stopped before compilation because `rustfmt --check`
required one function signature to wrap.  The formatter-only correction was
applied and the full gate passed.

## Result

```text
distribution       E[D]   nu    E[D*]  largest  root giant  root fraction
all degree 4        4.00  3.00    4.00   1.0000     20/20       1.0000
half 2, half 6      4.00  4.00    5.00   0.9999     20/20       0.9999
half 1, half 7      4.00  5.25    6.25   0.9380     18/20       0.8435
```

The representative half-1/half-7 frontier mean degrees evolved as

```text
1.00, 7.00, 7.00, 6.50, 6.08, 5.70, 3.80, 1.10, 1.00.
```

It began from a leaf, immediately entered hubs through the size-biased edge
law, and ended in a leaf-heavy boundary after hubs were depleted.

## Interpretation boundary

The samples illustrate root-versus-edge degree laws and finite depletion.  They
do not validate the asymptotic branching approximation, measure GPU performance,
establish a power-law result, or select a load-balancing policy.  The 18/20 root
giant count is retained rather than converted into a precise probability claim.

## Status

Pass after one formatting-only failure.
