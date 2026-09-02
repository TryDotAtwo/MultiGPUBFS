# BFS on preferential-attachment graphs: age, hubs, and core entry

Preferential attachment adds a dependency that a degree histogram alone cannot
show: the graph remembers its construction order.  Older vertices have had more
opportunities to receive edges, so a BFS wave moves not only through degree
classes but also through an age-correlated geometry.

This is a source-backed conceptual note.  No experiment was run: Docker was
unavailable at the single permitted readiness check, and infrastructure repair
was deliberately kept outside the BFS investigation.

## 1. Model before slogan

"Scale free" is not a graph generator.  A reproducible preferential-attachment
claim must specify at least:

- the seed graph;
- how many edges a new vertex contributes;
- whether endpoints are sampled sequentially or simultaneously;
- whether loops and parallel edges are allowed;
- the attachment weight, for example `degree`, `degree + delta`, or a nonlinear
  function;
- the root rule and whether vertex birth times are exposed to the traversal.

The classical Barabasi-Albert mechanism grows the graph and chooses existing
endpoints proportionally to their degree.  Rigorous variants differ in small
details that can change exact finite distributions.  Consequently, a theorem
for one formal process must not be attached to every graph called BA.

## 2. What BFS samples

For any fixed undirected graph, traversing an edge exposes an endpoint through
the size-biased degree law discussed in note 147.  Preferential attachment adds
correlations:

```text
birth time -> opportunity to accumulate degree -> position in the hub core.
```

The arrows describe statistical dependence, not a deterministic ranking.
Young vertices can have atypical degree and an old vertex need not be a global
hub.

A useful qualitative decomposition of a root BFS is:

1. escape from the root's local young neighborhood;
2. entry into progressively older and typically higher-degree vertices;
3. rapid spreading through a small, densely connected-by-paths core;
4. outward filling of younger peripheral branches.

This is not a new BFS algorithm.  It is a possible shape of ordinary exact BFS
layers under a growth-correlated graph law.

## 3. Same degree tail, different radial process

A configuration model can be constructed with a degree multiset resembling a
preferential-attachment sample.  It then randomizes pairings conditional on
those degrees.  Preferential attachment does not: endpoint choices and birth
times leave degree-degree and age-degree correlations.

Therefore these data do not determine one another:

```text
degree histogram
root frontier profile
distance distribution
time to first major hub
owner-routing matrix by level.
```

The configuration model is a useful null model for asking what the degrees
alone explain.  A difference between the two ensembles is evidence about
correlations only after size, degree sequence, root conditioning, and graph
semantics are controlled.

## 4. Tree intuition and its failure

Before collisions, a root in a locally sparse region can look tree-like. Once
the wave enters a hub-rich cyclic core, candidate multiplicity can grow much
faster than the number of new vertices:

```text
scanned edge occurrences >> unique next frontier.
```

This gives a characteristic trap for intuition.  A very high-degree frontier
can simultaneously mean:

- abundant parallel work;
- a large candidate buffer;
- heavy convergence onto already visited core vertices;
- only modest growth in unique states.

This collision mechanism does not apply to the simple `m=1` tree case: every
nonparent edge there reaches a distinct child. Note 152 derives the exact
frontier recurrence and isolates this correction.

The relevant branching quantity is neither the root's degree nor the global
mean degree.  It evolves with the degree and age composition of the current
frontier and with depletion of the finite graph.

## 5. Distances need model qualifications

Rigorous preferential-attachment models can have much shorter typical
distances than bounded-degree graphs.  For standard linear models with at
least two outgoing edges per arriving vertex, results of order
`log(n)/log(log(n))` are model-specific asymptotic statements, not a universal
law of every power-law network.  The tree case, different attachment offsets,
different exponents, and diameter versus typical distance require separate
statements.

For BFS this distinction is operational:

- typical distance describes a random vertex pair;
- root eccentricity describes the last vertex reached from one root;
- diameter is a maximum over all pairs;
- number of levels does not determine edge-scan work or peak frontier memory.

Calling all four quantities "small world" loses the actual BFS contract.

## 6. Ownership consequences

If vertex IDs are birth order, contiguous ID ownership is also an age
partition.  Old hubs may then concentrate authoritative visited records,
incident-edge scans, and incoming routed candidates on the first owner.  A
vertex-balanced partition can be extremely work-imbalanced.

Hash ownership disperses ages but turns many hub edges into remote traffic.
Degree-balanced partitioning can still fail per level because BFS reaches
degree classes nonuniformly.  Thus total edge balance, total vertex balance,
per-level work balance, and communication volume are four different claims.

An apparent multi-GPU benefit from early core entry may be additional frontier
parallelism, while an apparent loss may be owner hotspotting or duplicate
routing.  Neither follows from the degree exponent alone.

## 7. A bounded future probe

When Docker is available, a small Rust probe can compare two frozen graphs:

1. a precisely declared linear preferential-attachment multigraph;
2. a degree-preserving randomized pairing or switching null model.

The purpose would be explanatory, not performance optimization.  Retain per
depth:

- frontier vertices and scanned edge occurrences;
- unique next states and collision classes;
- frontier birth-time and degree quantiles;
- first depth hitting declared old/core sets;
- owner work and routing for birth-contiguous versus hashed ownership.

The comparison must either preserve the realized degree sequence or state
exactly which degree statistic differs.  Otherwise it cannot isolate the role
of growth correlations.

## 8. Retained non-run

The Docker readiness command

```text
docker info --format '{{.ServerVersion}}'
```

failed with permission denied on `dockerDesktopLinuxEngine`.  No restart,
configuration edit, or Docker repair was attempted.  REF-045 records the probe
as not run rather than converting an infrastructure failure into data.

## Sources

- A.-L. Barabasi and R. Albert,
  [*Emergence of Scaling in Random Networks*](https://doi.org/10.1126/science.286.5439.509),
  *Science* 286 (1999), for the growth-plus-preferential-attachment mechanism.
- B. Bollobas, O. Riordan, J. Spencer, and G. Tusnady,
  [*The degree sequence of a scale-free random graph process*](https://doi.org/10.1002/rsa.1009),
  *Random Structures & Algorithms* 18 (2001), for a precise process and
  rigorous degree-sequence analysis.
- B. Bollobas and O. Riordan,
  [*The diameter of a scale-free random graph*](https://doi.org/10.1007/s00493-004-0002-2),
  *Combinatorica* 24 (2004), for model-qualified distance and diameter results.
- Note 147 supplies the size-biased endpoint baseline; notes 46, 47, 51, 73,
  119, and 144-150 supply capacity, work/span, ownership, queue/frontier, and
  random-wave boundaries.

## Takeaway

Preferential attachment makes BFS history-sensitive at the ensemble level:
birth order shapes where hubs tend to live, and entering that old core can turn
a narrow peripheral wave into a wide, collision-heavy wave.  A power-law
degree histogram alone cannot predict that transition, its depth, or its
multi-owner cost.
