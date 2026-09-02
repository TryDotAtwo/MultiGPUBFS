# BFS on percolated Cayley graphs and random frontiers

Percolation first samples a random subgraph and then asks connectivity questions
inside that realization. Conditional on the sampled open graph, BFS remains an
ordinary exact deterministic traversal. Across samples, its frontiers, cluster
size, depth, and exhaustion event become random variables.

This note separates those two levels of reasoning and calibrates them on a
regular tree. It adds no percolation simulator or optimized traversal.

## 1. Bond and site percolation

In independent bond percolation with parameter `p`, every semantic edge is
retained (open) independently with probability `p` and deleted (closed) with
probability `1-p`. Site percolation instead retains vertices, with an explicit
condition needed for whether the BFS source itself is open.

One realization `omega` defines one fixed graph `G_omega`. Exact BFS from `s`
then computes:

```text
F_r(omega) = {v : d_(G_omega)(s,v)=r},
C_s(omega) = union_r F_r(omega).
```

The BFS invariant is unchanged after conditioning on `omega`; the input graph
is what became random.

## 2. Quenched and averaged statements

A **quenched** statement concerns one fixed realization: its exact reached set,
frontier, distance, or empty-frontier certificate. An **annealed** statement
averages over realizations, such as

```text
E_p[|F_r|]
```

or the probability that `s` belongs to an infinite open cluster.

These cannot be interchanged:

```text
E[|F_r|] is not the frontier of an "expected graph",
E[|C_s|] is not one realized cluster size,
positive expectation is not survival of a particular sample.
```

An empty frontier exactly exhausts the source cluster in that realization. It
does not prove the underlying unpercolated graph disconnected or the parameter
subcritical.

## 3. Exact calibration on a regular tree

Take the infinite `q`-regular tree, `q>=3`, with bond probability `p`. The root
has `Binomial(q,p)` open children. Every later discovered vertex has
`Binomial(q-1,p)` forward children, independently. The open root cluster is a
Galton-Watson branching process after its first generation.

For `r>=1`:

```text
E[|F_r|] = q p ((q-1)p)^(r-1).
```

The critical probability is

```text
p_c = 1/(q-1).
```

- below `p_c`, extinction occurs almost surely and expected frontiers decay;
- at `p_c`, expected frontier size stays constant after the first factor, yet
  extinction still occurs almost surely;
- above `p_c`, expected frontiers grow exponentially, but the root cluster still
  has a positive probability of early extinction.

This is a precise counterexample to reading one expectation as a per-run
completeness or infinite-survival certificate.

## 4. General graphs are not independent branching trees

On a graph with cycles, different generated branches can reach the same state.
They also expose shared edge variables and become dependent after conditioning
on the explored history. A naive offspring estimate such as

```text
p * (degree-1)
```

counts potential forward edge occurrences, not unique next-frontier states.

Tree branching can provide a comparison or local approximation under stated
conditions, but collisions, short relations, bottlenecks, and finite saturation
can invalidate equality. An observed or expected branching factor above one is
not a universal proof of a giant or infinite component.

## 5. Infinite cluster, giant component, and long finite survival

These are different claims:

- an **infinite cluster** exists only in an infinite graph model;
- a **giant component** means a component occupying a positive fraction in a
  declared finite-graph sequence;
- reaching radius `R` means only that this realization contains an open path of
  length at least `R` from the source.

A nonempty frontier at every tested radius does not prove an infinite cluster.
A finite puzzle instance cannot contain one at all. Conversely, a trial whose
root cluster dies immediately can occur even at a supercritical parameter.

Critical probability, finite-size scaling, and per-run cluster exhaustion must
therefore be reported separately.

## 6. Cayley symmetry before and after sampling

The unpercolated Cayley graph is vertex-transitive, so under invariant
independent percolation every root has the same cluster-size and survival
distribution. A particular sampled open cluster is generally not
vertex-transitive or regular.

Changing the generator set changes the edge graph and hence can change:

- the critical probability;
- short-cycle collision structure;
- open-cluster distances and frontier law;
- existence or uniqueness regimes of infinite clusters.

"Same generated group" does not define the same percolation model, just as it
does not define the same BFS metric in notes 92-93.

For a Schreier graph, the action orbit and stabilizer again define a different
percolation substrate from the Cayley graph of group elements.

## 7. Static percolation is not transient message loss

In static bond percolation, an edge's open/closed status is sampled once for the
entire traversal. Retrying a closed edge does not make it open. In an independent
message-loss model, the same semantic edge may succeed on a later attempt.

The two models have different path events and stopping evidence:

- static percolation explores reachability in one quenched subgraph;
- transient loss explores a communication process over time;
- resampling an edge on every BFS expansion creates an annealed temporal graph,
  not ordinary bond percolation.

Reproducibility therefore requires stating whether randomness is keyed to a
semantic edge, a directed occurrence, a trial, or each transmission attempt.

## 8. Finite experiments and evidence

Repeated trials can estimate distributions of cluster size, maximum depth, peak
frontier, and exhaustion. They do not by themselves prove a critical threshold
or asymptotic giant-component law.

Useful retained evidence includes:

- graph/generator identity and percolation type;
- `p`, seed, trial count, and confidence procedure;
- paired undirected-edge versus directed-arc random identity;
- complete per-trial layer histogram and explicit capacity failures;
- separate sample mean, quantiles, tail events, and exact theoretical values
  where known.

The mean peak frontier alone can hide rare memory-heavy realizations. Capacity
planning needs tail or worst-declared-quantile evidence.

## 9. GPU and multi-GPU interpretation

For an implicit percolated Cayley graph, every owner must agree on the same
open/closed result for the same semantic edge. In an undirected model, opposite
orientations must share one random edge identity; in a directed model they may
be independent by contract.

Distributed concerns remain separate:

- random pruning changes semantic candidate work before owner routing;
- owner balance can vary strongly by realization;
- identical `p` does not imply identical work across seeds;
- one dropped candidate from overflow is not a sampled closed edge;
- capacity or communication failure must not be folded into the percolation
  probability;
- performance comparisons need either paired realizations or an explicit
  statistical design.

No choice of random-edge representation or GPU kernel is prescribed here.

## 10. Evidence checklist

1. Bond, site, directed, or temporal resampling model.
2. One fixed realization versus expectation/probability over trials.
3. Semantic edge identity, inverse orientation, labels, and parallels.
4. Finite cluster, giant component in a family, or infinite cluster.
5. Exact frontier versus generated open-edge occurrences.
6. Tree comparison assumptions and cycle/duplicate corrections.
7. Seeded reproducibility, trial count, confidence, and tail statistics.
8. Sampled closure versus overflow, loss, or execution failure.

## Sources

- G. Grimmett, [*Percolation*, 2nd
  edition](https://doi.org/10.1007/978-3-662-03981-6),
  Springer, 1999. Bond/site models, critical probability, tree calibration, and
  subcritical/supercritical distinctions.
- I. Benjamini and O. Schramm,
  [*Percolation Beyond `Z^d`, Many Questions and a Few
  Answers*](https://doi.org/10.1214/ECP.v1-978),
  Electronic Communications in Probability 1 (1996), 71-82. Percolation on
  Cayley, transitive, and nonamenable graphs.
- Notes 10, 22, 27, 35, 46, 55, 63, 71, 92, 93, 94, 96, and 97 provide
  frontier, temporal, girth, growth, expansion, successor-validation, relation,
  arbitrary-profile, generator, amenability, message-loss, and end context.

## Takeaway

Percolation randomizes the graph; BFS remains exact after the realization is
fixed. Expected frontier growth, survival probability, one trial's exhaustion,
finite giant components, and infinite clusters are different statements. Tree
branching gives an exact calibration, while Cayley relations, finite saturation,
and distributed failures must not be disguised as percolation randomness.
