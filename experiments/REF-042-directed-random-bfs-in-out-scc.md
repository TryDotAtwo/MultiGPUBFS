# REF-042: forward, reverse, and strong reach in directed random graphs

## Question

How do forward BFS, transpose BFS, and root SCC size differ across the directed
Erdos-Renyi threshold, and how does root conditioning expose the IN/OUT bow tie?

## Method

- `D(n,c/n)` with independent ordered arcs, `n=2000`.
- `c` in `0.8, 1.0, 1.2, 4.0`, with 20 deterministic samples each.
- Freeze adjacency and transpose adjacency before traversal.
- Run forward and reverse BFS from root zero.
- Use a transparent two-pass SCC decomposition to identify the largest SCC.
- Run forward/reverse BFS from one largest-SCC representative to measure its
  finite outgoing/incoming reach.
- Build and run Rust only in Docker.

## Retained failure

The first Docker gate stopped before compilation because `rustfmt --check`
required one iterator expression to wrap.  Subsequent review found that the
initial iterative DFS marked sibling vertices on push and did not guarantee a
valid Kosaraju finish order.  Those SCC results were rejected.  The DFS was
replaced by an explicit `(vertex,next-edge-index)` stack and every sample was
recomputed.  Four 24-vertex fixtures exhaustively compared every reported SCC
pair against mutual forward/reverse reachability; the final
format/compile/assert/run gate passed.

## Result

```text
c    largest SCC   core reverse   core forward   root forward   root reverse
0.8     0.0013        0.0081         0.0049          0.0018         0.0014
1.0     0.0077        0.0645         0.0407          0.0079         0.0111
1.2     0.0892        0.2833         0.2998          0.0692         0.0890
4.0     0.9611        0.9804         0.9804          0.8821         0.9805
```

At `c=4`, root zero reached the largest SCC in 18/20 samples, was reachable
from it in 20/20, and belonged to it in 18/20.  Representative forward and
reverse layers were respectively

```text
[1,8,37,152,483,836,377,56,4,1]
[1,2,8,33,132,407,754,504,108,9].
```

## Interpretation boundary

For `c>1`, core reverse/forward reach approximate GIN/GOUT.  For `c<=1`, they
are only sets leading to or from the largest finite SCC and must not be called
giant components.  The retained finite means do not validate asymptotic laws,
measure SCC performance, or imply a GPU implementation strategy.

## Status

Pass after one formatting-only failure and one rejected SCC-order instrument.
