# REF-040: BFS frontiers in two-block stochastic graphs

## Question

Can graphs with the same expected degree and total branching eigenvalue have
different type-resolved BFS waves and different owner-routing behavior?

## Method

- Two equal blocks, `n=2000`, 20 deterministic seeds per case.
- Independent frozen undirected edges with probability `c_ab/n`.
- Four matrices with `within+across=8`: segregated, assortative, neutral, and
  disassortative.
- Exact BFS from vertex zero plus exact largest-component enumeration.
- Typed frontier layers and remote-edge fractions for block and striped owners.
- Rust build and run only in Docker.

This `O(n^2)` sampler is transparent and intentionally small.  It is neither a
scalable SBM generator nor a BFS performance benchmark.

## Retained failure

The first Docker gate stopped before compilation because `rustfmt --check`
required one long expression to wrap.  The formatter-only change was applied;
the corrected format/compile/run chain passed.  A preceding sandboxed launch
also failed to access the Docker named pipe and was rerun with approved Docker
access; that was infrastructure, not an algorithm result.

## Result

```text
segregated:     eigs=(4.0, 4.0), largest=0.4912, root0=0.4902,
                block_remote=0.0000, striped_remote=0.5012
assortative:    eigs=(4.0, 3.5), largest=0.9796, root0=0.9796,
                block_remote=0.0626, striped_remote=0.4996
neutral:        eigs=(4.0, 0.0), largest=0.9792, root0=0.9792,
                block_remote=0.4987, striped_remote=0.4982
disassortative: eigs=(4.0,-3.5), largest=0.9798, root0=0.9798,
                block_remote=0.9376, striped_remote=0.5000
```

The representative disassortative layers alternated type dominance:

```text
(1,0),(2,6),(23,4),(12,72),(266,67),(189,547),(435,168),...
```

The segregated root never reached type one.  Assortative layers mixed slowly;
neutral layers were near balanced after the first steps.

## Interpretation boundary

The observations illustrate the two-type branching matrix and owner-cut
trade-off.  They do not establish asymptotic convergence, community-detection
quality, GPU throughput, or an optimal partition.  In particular, zero remote
edges in the segregated/block-owned case coincided with only one owner holding
the reachable component of a single-source traversal.

## Status

Pass after one formatting-only failure and one sandbox infrastructure failure.
