# BFS output contracts and finalization boundaries

"Exact BFS" is incomplete as a specification. The same distance layers can
support several outputs, and a run that is finished for one output may still be
missing mandatory evidence for another.

This note organizes those outputs as information contracts. It asks what must
be mathematically finalized, not how to build an optimal CPU, GPU, or
multi-GPU representation.

## First declare graph and path identity

Before choosing output metadata, define what constitutes a vertex, edge, and
path.

Possible path identities include:

```text
vertex sequence
edge-occurrence sequence
visible edge-label sequence
Cayley generator word
source-labeled path
concrete lift of an abstract/quotient path.
```

These are not interchangeable. If two labeled generators both map `u` to `v`,
there is one endpoint transition in a simple-graph view, two edge occurrences
in a multigraph view, and possibly one or two visible words depending on label
equality. Distance can remain unchanged while counts and enumeration change.

The output contract must therefore inherit:

- graph/move and identity versions;
- directed orientation and edge cost convention;
- simple, labeled, or multigraph multiplicity;
- source and target semantics;
- quotient/lifting semantics, if any;
- total orders used for canonical choices.

## The output lattice is only partially ordered

Some outputs contain enough information to derive others, but there is no one
linear ladder in every representation.

```text
reachable set / distance labels
          |
          +--> one arbitrary shortest witness
          |
          +--> one canonical shortest witness
          |
          +--> complete shortest-predecessor DAG --> path counts
                                                    |
                                                    +--> uniform sampling
                                                    +--> explicit all paths

multi-source distance --> arbitrary owner | canonical owner | all tied owners
```

A complete predecessor DAG plus source initialization can derive exact counts,
but counts alone cannot reconstruct the predecessor edges. One canonical path
does not contain all arbitrary paths. A set of nearest sources does not select
a coherent parent forest without additional choices.

Thus richer and more expensive are not always synonyms for set inclusion: each
output is a named mathematical object.

## Contract 1: reachability or scalar distance

For source set `S`, the canonical scalar is

```text
d_S(v) = min_(s in S) dist(s,v).
```

Target-only distance requires a matching lower-bound stopping certificate;
full distance output requires exhaustion or completion of a declared radius.
Within-layer order, first claimant, parent, and nearest-source winner are
irrelevant to this scalar.

Sufficient semantic evidence includes:

- every required successor occurrence was considered without loss;
- exact state identity and visited membership;
- complete closure through the claimed distance/radius;
- a valid stopping lower bound for early target termination;
- explicit `UNREACHABLE`, `OUTSIDE_RADIUS`, and `UNKNOWN/FAILED` distinctions.

A visited bitmap may represent reachability, but the representation alone does
not prove expansion completeness or closure.

## Contract 2: one arbitrary replayable shortest path

For every selected non-source vertex `v`, retain one predecessor edge

```text
e=(u -> v),  d(u)+1=d(v).
```

In an implicit/labeled graph this generally includes the move label and enough
orientation/frame information to replay `u -> v`. Following parents strictly
decreases depth, so the witness cannot cycle and reaches a source.

One first-arriving same-depth claimant may be sufficient. Other shortest
parents may be discarded after their contribution to frontier membership has
been handled. The selected path can vary with adjacency order, thread timing,
rank count, or owner routing and still satisfy this contract.

A replayable chain is an upper-bound witness. Shortestness additionally comes
from the BFS distance/closure proof; replay alone cannot exclude a missed
shorter route.

## Contract 3: one deterministic or canonical shortest path

"Deterministic" must name the total order. Examples are:

```text
minimum (source_id, parent_state_id, move_id)
shortlex-least generator word
minimum edge-occurrence sequence
stable but implementation-specific discovery order.
```

These select different paths. Minimum parent state ID is not generally the
shortlex-least word because full parent-word order need not agree with state-ID
order.

A canonical winner is final only after every equal-distance proposal capable
of beating it under the declared order has been accounted for. Atomic
first-winner is therefore not a canonical reduction. A stable result on one
launch geometry is not automatically stable across GPU counts or ownership
partitions.

Canonical source, canonical path word, and canonical parent can also be
separate keys. Their lexicographic priority must be explicit.

## Contract 4: complete shortest-predecessor DAG

With exact distances, retain every semantic edge

```text
E_sp = {(u,v) in E | d(u)<infinity and d(u)+1=d(v)}.
```

Finite depth increases on every such unit-cost edge, so `E_sp` is acyclic.
Edges between unreachable vertices are excluded even if sentinel arithmetic
would make `infinity+1=infinity`. A Boolean
visited claim is insufficient to construct it: a later same-depth proposal to
an already discovered child may be another required predecessor rather than a
discardable duplicate.

The diamond

```text
    a
  /   \
s     t
  \   /
    b
```

has two shortest predecessors of `t`. Either one gives a valid tree; retaining
only one corrupts the complete-DAG contract while leaving every distance
correct.

For a labeled multigraph, predecessor **edge occurrences** rather than only
parent vertices may be required. Retry duplicates of the same occurrence must
not be mistaken for distinct semantic edges.

## Contract 5: exact shortest-path counts

Let `P(v)` be the complete semantic predecessor-edge collection. Then

```text
sigma(s)=1
sigma(v)=sum_(e=(u->v) in P(v)) sigma(u).
```

Counts require every contribution exactly once under the chosen path identity,
but an implementation need not retain the whole DAG after a contribution has
been safely aggregated. Conversely, a DAG without initialized and evaluated
count recurrence is not itself a count result.

Arithmetic is part of the contract:

- exact unbounded integer;
- checked fixed-width result or explicit `OVERFLOW`;
- saturated, modular, logarithmic, or approximate value with a different
  declared meaning.

Addition is not idempotent. Replaying one distributed predecessor contribution
twice overcounts even though replaying an ordinary Boolean visited claim may be
harmless.

## Contract 6: explicit enumeration of all shortest paths

A linear-size shortest-path DAG can encode exponentially many paths. In a
layered two-choice graph, `O(k)` structure can represent `2^k` source-target
paths. Therefore explicit enumeration has an unavoidable output-proportional
time and byte cost.

"All paths" must state whether output is:

- streamed or retained in memory;
- ordered canonically or arbitrarily;
- deduplicated by vertices, labels, or edge occurrences;
- complete after interruption/restart;
- bounded, paginated, or deliberately truncated.

A cap without a three-valued completion result changes `ALL_PATHS` into
`SOME_PATHS` while possibly leaving the scalar distance exact.

## Contract 7: uniform shortest-path sampling

Uniform sampling is neither one arbitrary path nor all paths. It needs complete
predecessor alternatives and exact prefix/suffix multiplicities for the chosen
sample space. Choosing predecessor vertices uniformly is biased when their
sub-DAG path counts differ.

Counts may permit sampling without materializing every path, but missing one
predecessor or overflowing one count changes the distribution. A quotient path
also needs concrete lift multiplicities before it can support a claim of
uniform concrete sampling.

Note 53 gives the telescoping proof and exact integer-selection conditions.

## Multi-source output contracts

Multi-source BFS always has a canonical scalar distance to the set, but source
ownership is additional output. Define

```text
A(v) = argmin_(s in S) dist(s,v).
```

Legitimate contracts include:

1. no source label—only `d_S(v)`;
2. one arbitrary member of `A(v)`;
3. one canonical member under a declared source order;
4. the entire set `A(v)`;
5. path counts grouped by tied source;
6. a parent forest coherent with the selected source labels.

At a tie vertex, all six can share the same distance and disagree in metadata.
Physical rank/GPU ownership is not nearest-source ownership.

### Pointwise labels versus a coherent forest

Choosing a valid nearest source independently for every vertex can produce a
label whose usable shortest predecessors all carry another label. The labels
remain pointwise correct but do not define a same-label path back to the source.

A coherent forest additionally requires

```text
label(parent(v)) = label(v).
```

Canonical equal-distance source improvements must be resolved before expansion
or propagated to descendants. Distance finality alone does not finalize the
Voronoi label.

## Finalization boundaries

Suppose a target has optimal distance `D`. Completion differs by output:

| Requested result | Work that must be finalized |
|---|---|
| distance only | enough lower-bound/closure evidence to exclude distance `<D` |
| one arbitrary path | distance proof plus one replay-valid length-`D` witness |
| canonical path | every equal-distance competitor that can beat the witness under the total order |
| predecessor DAG to target | every semantic predecessor edge in the target-relevant shortest sub-DAG |
| exact count | every required predecessor contribution, once, in depth order |
| all paths | complete DAG/count basis plus output-complete enumeration |
| uniform sample | complete exact weights plus unbiased selection for the declared sample space |
| all nearest sources | every equal-distance source claim, not merely the first |

For full-graph outputs, replace target-relevant completion with the requested
reachable component or radius.

### Which layer must finish in ordinary FIFO BFS?

For unit edges with mark-on-enqueue, let the only edges be
`s->a`, `s->b`, `a->t`, `b->t`, `t->z`, and process a before b.
After expanding s the queue is [a,b]. Expanding a first discovers t at depth
two, gives it one shortest-path contribution, and leaves queue [b,t].
The distance two and one witness are already final. Expanding b adds the
second contribution and leaves [t]. At this point all depth-one predecessors
have been processed: t's shortest-parent set and count two are final before
t itself is expanded. Its outgoing edge to z need not be inspected to answer
these target-only questions.

Thus for the standard forward counting recurrence, completing expansion of
`F_(D-1)` suffices for the target at distance D; expanding all of `F_D` is
unnecessary. A sequential FIFO implementation that checks the target when it
is first dequeued has already completed those predecessor expansions. Checking
at first discovery has not. These statements assume equal-depth contributions
are accumulated rather than dropped by visited, and do not transfer to an
asynchronous queue with pending predecessor messages. For D=0, the source
initialization supplies the distance/count base case without an earlier layer.

In bidirectional search, `a+b>=mu` can finalize distance and one witness. Work
at total length `mu` may still contain alternative connectors, canonical
winners, predecessor edges, or path-count mass. Strict inequality alone is not
a universal all-output rule: completion must cover the actual equality-boundary
objects from which the requested output is derived.

## Duplicate handling changes with the output

Candidate records with the same child can have several meanings:

| Candidate relationship | Distance/frontier | One arbitrary path | Canonical path | DAG/counts |
|---|---|---|---|---|
| exact retry of same occurrence | deduplicate | deduplicate | deduplicate | deduplicate contribution ID |
| different older-depth route | discard | discard | discard | discard |
| another same-depth parent | collapse child | one winner | compare total key | retain semantic predecessor |
| parallel label from same parent | collapse child | one winner | compare label/occurrence | retain if path identity distinguishes it |

Therefore "duplicate" is not a property of the child key alone. It is a
relation between records under the declared output equivalence.

## Distributed and GPU consequences

This is an ownership/accounting boundary, not a backend prescription.

- The authoritative child owner may decide novelty, but source ranks may hold
  required losing parent/label proposals.
- Distance-only communication can omit metadata that canonical/DAG/count
  outputs require.
- Parent finalization needs evidence that every relevant producer and in-flight
  message for that depth has retired.
- Fixed-capacity metadata buffers need explicit overflow states independent of
  frontier capacity.
- Retry semantics must distinguish idempotent visited claims from
  non-idempotent counts and output records.
- Repartition/restart must preserve graph, identity, path, source, and tie-order
  epochs, not just state IDs.
- Determinism across device counts requires a semantic total reduction, not
  reproducible timing on one topology.

The cost of richer output should be reported separately from traversal work:
metadata bytes, competing proposals, reductions, predecessor records, integer
width, reconstruction traffic, and output bytes are distinct quantities.

## Failure-state matrix

One run can have different validity by output column:

```text
distance:              VALID
one arbitrary path:    VALID
canonical path:        UNKNOWN (tie proposals incomplete)
predecessor DAG:       INVALID (buffer overflow dropped edges)
path count:            INVALID (depends on incomplete DAG)
all-path enumeration:  PARTIAL
```

Collapsing this to one `success=true` flag hides which theorem survived. Each
artifact should name the strongest finalized contract and explicit weaker or
failed columns.

## Minimal artifact fields

```text
graph/move/identity/path-equivalence version:
source/target/radius and directed orientation:
requested output contract:
distance/closure or stopping certificate:
parent and replay-label convention:
canonical total order and finalized tie boundary:
predecessor occurrence identity and completeness:
count arithmetic/overflow semantics:
nearest-source tie and coherent-forest semantics:
retry/dedup contribution identity:
frontier and metadata capacity outcomes:
enumeration/sample completeness and ordering:
validation status per output column:
```

## Counterexamples to common substitutions

### Correct distances imply correct parents

Distances permit several shortest parents; a stored parent can also carry the
wrong replay label even when its endpoints are adjacent.

### One valid path implies canonical path

A valid first winner may lose under the declared parent, source, or word order.

### One parent per vertex represents all shortest paths

The diamond loses one shortest route immediately; layered diamonds lose
exponentially many.

### All predecessor vertices represent all labeled paths

Parallel labels between the same endpoints collapse at the vertex level.

### Exact counts imply the predecessor DAG is available

Counts can be aggregated and predecessor identities discarded.

### All nearest-source labels are incidental

Facility ownership, Voronoi boundaries, source-conditioned counts, or coherent
forests may consume those labels as semantic output.

### A global visited set makes every retry idempotent

Visited membership is Boolean-idempotent; count additions and emitted parent
records are not unless contribution identities are deduplicated.

## Sources and relation to existing notes

- Notes 05, 08, 11, 13, 19, 30, 53, and 56 establish the variant, stopping,
  DAG/count, multi-source, canonical-word, replay, sampling, and partial-layer
  facts synthesized here.
- Thomas H. Cormen et al., *Introduction to Algorithms*, BFS chapter, gives the
  standard distance and predecessor-tree contract.
- Ulrik Brandes, "A Faster Algorithm for Betweenness Centrality,"
  <https://doi.org/10.1080/0022250X.2001.9990249>, uses complete predecessor
  sets and shortest-path counts as distinct BFS metadata.
- The Graph500 BFS specification separates the traversed graph and root from a
  predecessor-array validation contract:
  <https://graph500.org/?page_id=12>.

## Current conclusions

1. Exact BFS is a family of output contracts, not one Boolean property.
2. Scalar distance is insensitive to many ties that are semantic for paths,
   labels, source ownership, counts, and sampling.
3. One arbitrary witness, one canonical witness, a predecessor DAG, exact
   counts, and explicit all-path enumeration are different objects.
4. Path identity determines whether parallel labels and generator occurrences
   are duplicates or separate solutions.
5. A stopping certificate finalizes only the output named by its proof;
   equality-boundary work may remain for richer outputs.
6. Multi-source distance does not uniquely determine a source label, and
   pointwise labels do not automatically define a coherent forest.
7. Retry, overflow, and capacity validity must be assessed separately for each
   output column.
8. Richer output creates metadata and synchronization obligations but does not
   prescribe an optimal implementation.
