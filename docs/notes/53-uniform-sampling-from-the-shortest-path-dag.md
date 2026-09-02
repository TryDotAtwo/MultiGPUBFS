# Uniform sampling from the shortest-path DAG

BFS can compactly represent exponentially many shortest solutions through
distance labels, predecessor edges, and path counts. Those counts also support
uniform sampling of one shortest path without enumerating them all.

Uniformity is relative to a declared path identity: vertex sequences,
edge-labeled paths, Cayley generator words, and source-labeled paths are
different sample spaces.

## The shortest-path DAG and prefix counts

For one source `s`, exact BFS distances define predecessor edges

```text
P(v) = {e=(u -> v) | d(u)+1=d(v)}.
```

Let `sigma(v)` count shortest paths from `s` to `v` under the chosen edge/path
identity:

```text
sigma(s)=1
sigma(v)=sum_(e=(u->v) in P(v)) sigma(u).
```

The sum is over predecessor **edges**. In a simple graph there is at most one
edge per ordered endpoint pair. In a labeled multigraph, two labels connecting
the same `u` and `v` contribute separately if they define distinct paths.

The recurrence is a dynamic program because predecessor depth strictly
decreases. It can be evaluated level by level after or during a complete BFS
layer construction.

## Backward uniform sampling

To sample a shortest path ending at target `t`, start at `v=t`. While `v!=s`,
choose one predecessor edge

```text
e=(u->v)
```

with probability

```text
Pr[e | v] = sigma(u)/sigma(v),
```

then set `v=u`. The recurrence guarantees that the probabilities of all
incoming predecessor edges sum to one.

Consider one particular shortest path

```text
s=v_0 -> v_1 -> ... -> v_D=t.
```

The probability of selecting all its edges backward is

```text
product_(i=1..D) sigma(v_(i-1))/sigma(v_i)
= sigma(s)/sigma(t)
= 1/sigma(t).
```

The intermediate factors telescope. Every shortest path therefore receives
the same probability, so the sample is exactly uniform.

This proof also shows why choosing predecessors uniformly by **vertex** is
generally wrong: predecessor subtrees can contain very different numbers of
complete prefixes.

## Minimal biased-choice example

Let target `t` have two predecessor vertices `a` and `b`, with

```text
sigma(a)=1
sigma(b)=9.
```

There are ten shortest paths to `t`. Choosing `a` or `b` with probability
`1/2` gives the unique path through `a` probability `1/2`, while each path
through `b` receives only `1/18` after uniform recursion there.

Correct predecessor probabilities are

```text
Pr[a->t]=1/10
Pr[b->t]=9/10.
```

Uniform local choice is not uniform global path sampling.

## Parallel edges and generator labels

Suppose two distinct move labels `m_1,m_2` both map `u` to `v`. If paths are
edge-labeled, they are two predecessor edges and each carries weight

```text
sigma(u).
```

Their combined probability mass is `2*sigma(u)/sigma(v)`. If paths are only
vertex sequences, the two edges collapse to one transition and should not
double the count.

For a Cayley graph:

- vertices are group elements or declared puzzle states;
- edge-labeled shortest paths are geodesic generator words;
- duplicate generator labels can multiply word paths without changing states;
- relations can give many geodesic words for one element;
- quotienting configurations can merge state paths while leaving several lifts.

Therefore the phrase "uniform shortest Cayley solution" must specify whether
uniformity is over generator words, state sequences, symmetry classes, or
concrete lifted replays.

## Multiple sources

For source set `S`, initializing

```text
sigma(s)=1 for each distinct s in S
```

makes `sigma(v)` count all nearest-source shortest paths, treating the source
identity as part of the path. Backward sampling terminates at a source with
probability proportional to the number of shortest paths contributed by that
source.

This is not uniform over nearest sources. A source with more shortest paths gets
more mass. To sample sources uniformly first and then paths conditionally, one
needs per-source counts or a different two-stage distribution.

Declared source multiplicities/weights similarly change the sample space.

## Forward suffix counts

For a fixed target `t`, one may instead define

```text
tau(t)=1
tau(v)=sum_(e=(v->w), d(w)=d(v)+1, w lies on a shortest route to t) tau(w).
```

Choosing successor edge `v->w` with probability `tau(w)/tau(v)` samples
uniformly forward. This requires target-specific suffix counts and the sub-DAG
satisfying

```text
d_s(v)+dist(v,t)=dist(s,t).
```

Prefix counts `sigma` are convenient for backward sampling from any reached
target; suffix counts are convenient when paths must be emitted forward without
first constructing a reversed edge list. Either representation needs complete
counts in the chosen shortest-path DAG.

## Exact integer sampling avoids floating bias

At vertex `v`, predecessor edge weights are positive integers summing to
`sigma(v)`. An exact conceptual sampler can:

1. draw an unbiased integer `R` uniformly from
   `0..sigma(v)-1`;
2. scan or index cumulative predecessor weights;
3. choose the unique interval containing `R`.

This avoids rounding probabilities such as `sigma(u)/sigma(v)` to floating
point. It still requires an unbiased random-integer procedure for arbitrarily
large bounds and a declared random seed/source for reproducibility.

Using `random_word mod sigma(v)` is biased unless the random word range is an
exact multiple of `sigma(v)`; rejection sampling or another proved mapping is
needed for exact uniformity.

Floating sampling may be an acceptable approximate output, but it should not be
labeled exact uniform merely because counts were exact before conversion.

## Overflow and compressed counts

Shortest-path counts can be exponential in a linear-size DAG. Consequently:

- fixed-width wraparound corrupts probabilities unpredictably;
- saturation makes high-count branches appear artificially equal;
- modular counts preserve a residue, not relative path mass;
- logarithmic counts can estimate entropy but do not provide exact weights;
- approximate sketches can support approximate sampling only with an error
  model.

Exact uniform sampling needs exact integer counts or another exact combinatorial
sampler. Detecting overflow and returning `COUNT_OVERFLOW/UNKNOWN` is safer than
silently sampling from distorted weights.

The information content of one uniform path index is

```text
log2(sigma(t)) bits
```

up to integer rounding. A single parent pointer deliberately discards that
multiplicity and cannot reconstruct a uniform sample over the lost alternatives.

## Complete predecessor coverage is required

Correct distances do not imply correct sampling. If a parallel visited winner
retains only one parent of `v`, the resulting tree still gives a valid shortest
path but changes

```text
sigma(v)
```

and every downstream sampling probability.

The builder must distinguish:

- first discovery of a child at depth `d+1`;
- another predecessor edge from depth `d` into that same child;
- an edge from an older/nonpredecessor layer;
- replay/retry duplicates of the same predecessor contribution.

Every semantic predecessor edge contributes once. Note 30's fault semantics are
important: addition is not idempotent, so retrying one contribution twice
overcounts paths unless it carries a deduplicated contribution identity.

## Sampling from a quotient is not sampling concrete paths

An abstract or symmetry-quotient path can have zero, one, or many concrete
lifts, and different abstract paths can have different lift multiplicities.
Uniformly choosing an abstract shortest path therefore need not produce a
uniform concrete shortest path after lifting.

For uniform concrete sampling through a quotient, weights must count the number
of valid concrete completions represented by every abstract choice, including
orientation/frame constraints. A quotient distance proof alone supplies no such
multiplicity measure.

This parallels the PDB distinction in note 49: projection is sufficient for a
lower bound, not for concrete path-generation probabilities.

## Bidirectional counting needs a unique cut

Suppose forward and reverse searches have exact prefix/suffix counts. A meeting
vertex `x` satisfying

```text
d_f(x)+d_b(x)=D
```

represents

```text
sigma_f(x) * sigma_b(x)
```

shortest path combinations, assuming the forward and reverse path identities
and edge labels are compatible.

A crossing edge `u->v` with

```text
d_f(u)+1+d_b(v)=D
```

has weight

```text
sigma_f(u) * multiplicity(u->v) * sigma_b(v).
```

However, summing over **every** equality vertex can count one complete path many
times—once at every vertex on that path. To partition paths exactly, choose one
fixed distance cut, such as all crossing edges from forward depth `k` to
`k+1`, so every shortest path crosses the cut exactly once.

Then:

1. sample a connector with probability proportional to its path-combination
   weight;
2. sample the forward prefix and reverse suffix conditionally by their counts;
3. replay the joined labels under the correct orientation.

Uniformly choosing a meeting connector is biased when connector weights differ.

## Multi-GPU count reduction

A child owner can collect predecessor contributions from many source ranks.
Exact global counting requires:

```text
one contribution per semantic predecessor edge
exact addition without overflow
completed depth-d contribution set before sigma at depth d+1 is final
retry deduplication
matching edge-label/path identity across ranks.
```

Integer addition is associative and commutative, which permits reordering, but
it is not idempotent. Duplicate messages must not be mistaken for harmless BFS
candidate duplicates.

Sampling can occur after counts are finalized and available along the selected
path. If counts are sharded, each backward choice may require owner lookup or a
materialized path-specific cache. That is a latency/placement question, not a
change to the probability proof.

## GPU interpretation without an implementation

Distance-only GPU BFS can discard losing duplicate candidates after one visited
winner. Uniform shortest-path sampling needs richer evidence:

```text
predecessor-edge occurrences retained/deduplicated
count additions and maximum bit width
overflow/approximation status
cross-rank contribution records
count-finalization boundary
random seed and exact/approximate sampling mode
sample replay and empirical distribution checks on small exact cases.
```

Big-integer arithmetic and variable predecessor lists may fit GPUs poorly, but
that is a representation measurement question. It cannot justify silently
switching the requested distribution.

## Counterexamples and rejected shortcuts

### Choose a predecessor uniformly

This is biased whenever predecessor sub-DAGs contain different numbers of
shortest prefixes.

### One BFS parent is enough to sample all shortest paths

The parent tree has already discarded every alternative predecessor.

### Correct distances imply correct path probabilities

Missing or duplicated predecessor contributions leave distances unchanged and
corrupt `sigma`.

### Modular or saturated counts are harmless

They do not preserve relative integer path mass and generally bias sampling.

### Uniform abstract path plus arbitrary lift is uniform concrete sampling

Abstract paths and choices can have unequal numbers of concrete lifts.

### Uniform bidirectional meeting gives a uniform path

Meeting connectors carry different products of prefix and suffix counts, and an
unfixed meeting set can count each path multiple times.

## Sources and evidence

- Ulrik Brandes,
  [A Faster Algorithm for Betweenness Centrality](https://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf),
  uses BFS predecessor sets and shortest-path counts as a compact dynamic
  program over the shortest-path DAG.
- Donald Knuth,
  [The Art of Computer Programming, Volume 2](https://www-cs-faculty.stanford.edu/~knuth/taocp.html),
  provides exact random-integer and rejection-sampling background.
- Notes 11, 13, 17, 20, 30, 36, 49, and 52 provide the predecessor/count,
  multi-source, quotient, path-identity, retry, representation, abstraction,
  and distributed-filter boundaries used here.

## Current conclusions

1. Exact prefix counts on the complete shortest-path DAG permit uniform backward
   sampling by weighting each predecessor edge with its prefix count.
2. The probability of every full shortest path telescopes to `1/sigma(t)`.
3. Uniformity must name vertex paths, labeled edges/generator words, sources,
   and any quotient/lift semantics.
4. Exact uniform sampling requires exact predecessor coverage and count
   arithmetic; distance correctness alone is insufficient.
5. Multiple sources, quotients, and bidirectional connectors require weights
   proportional to the number of concrete complete paths they represent.
6. Distributed addition can reorder contributions but must deduplicate retries
   and finalize a whole predecessor layer before sampling.
