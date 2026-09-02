# BFS trails, edge history, and line digraphs

A trail forbids repeated edges but permits repeated vertices. It lies strictly
between an unrestricted walk and a vertex-simple path. As with simple paths,
future legality is history dependent; the history now records used semantic
edges rather than used vertices.

This note clarifies the state contract and the limited role of line-graph
transformations. It does not propose a trail-search implementation.

## 1. Walk, trail, and simple path

Under one fixed edge-identity convention:

- a **walk** may repeat vertices and edges;
- a **trail** may repeat vertices but not edges;
- a **simple path** may not repeat vertices and therefore cannot repeat an edge.

Thus

```text
simple path => trail => walk,
```

and neither converse holds in general.

Every shortest unit-edge walk is a simple path by note 87, hence also a trail.
Ordinary BFS already finds a shortest trail whenever it finds a shortest path.
The distinction matters for exact-length, long, enumerated, or edge-covering
queries, not for ordinary shortest reachability.

## 2. Exact trail state

Let `F` be the set of edge identities already traversed. A direct Markov state
for a trail is

```text
(v,F).
```

An outgoing semantic edge `e=(v,x)` is legal only when `e notin F`, producing
`(x,F union {e})`.

For `m` semantic edges, the raw endpoint/history universe can contain up to

```text
n 2^m
```

states before reachability and length restrictions. A global edge-used bit does
not work across alternative histories: one history's use of `e` must not ban
`e` in every other history.

## 3. Equal endpoint, different legal edge continuation

Use directed vertices `s,a,b,v,x,t` and arcs

```text
s->a, a->b, b->v,
s->v, v->x, x->v,
x->t.
```

Two length-three trails end at `v`:

```text
H_1: s->a->b->v
H_2: s->v->x->v.
```

`H_2` repeats vertex `v`, which is legal for a trail, but it has already used
arc `v->x`. The continuation

```text
v->x->t
```

is legal after `H_1` and illegal after `H_2`. Merging by endpoint `v`, or
marking `v->x` globally after one history uses it, can discard the only
length-five trail through that continuation.

This is the edge-history analogue of note 87's used-vertex witness.

## 4. Directed line graph equivalence

For a directed graph `D`, its line digraph `L(D)` has:

- one vertex for every directed arc of `D`;
- an arc `e->f` when the head of `e` equals the tail of `f`.

A nonempty directed trail

```text
e_1,e_2,...,e_k
```

in `D` uses distinct arcs and consecutive incidence, so it is exactly a simple
directed path through vertices `e_1,...,e_k` of `L(D)`. Conversely, every
simple directed path in `L(D)` lifts to a directed trail in `D`.

The transformation exchanges edge history for vertex history; it does not make
long simple-path search into ordinary BFS. Source/target vertex constraints
also need explicit entry/exit handling because line vertices represent arcs,
not original vertices.

## 5. Ordinary undirected line graphs lose orientation compatibility

The ordinary line graph of an undirected graph makes two edge-vertices adjacent
when the original edges share any endpoint. A simple line-graph path need not
admit one continuous trail orientation in the original graph.

Take the three-edge star with center `c` and leaves `a,b,d`. Its line graph is a
triangle, so all three edge-vertices can be ordered as a simple path. But no
trail can traverse all three star edges: after moving leaf-to-center-to-leaf
through two edges, the walk is stranded away from the third edge.

One can add oriented-incidence state, but for an undirected trail the two
opposite orientations of one edge must still share one used-edge identity.
Therefore the directed-line equivalence cannot be copied to an ordinary
undirected line graph without extra semantics.

## 6. Eulerian trails are a structured special case

An Eulerian trail uses every semantic edge exactly once. Despite carrying the
largest possible used-edge set, its existence has special polynomial structural
criteria and constructive algorithms:

- in an undirected connected non-isolated support, an Euler circuit requires
  all degrees even; an open Euler trail permits exactly two odd-degree vertices;
- in a directed graph, balanced in/out degrees together with the appropriate
  connectivity condition characterize an Euler circuit, with the standard
  endpoint imbalances for an open trail.

Hierholzer's construction exploits this global degree structure. It does not
show that arbitrary prescribed-length or constrained trail queries reduce to
BFS, nor that generic `(v,F)` histories can be merged.

## 7. Cayley edge identity

In a labeled directed Cayley graph, a natural arc occurrence is

```text
(g,s): g -> g s.
```

Reusing generator label `s` at another source is not reusing the same arc. If
two labels produce the same endpoint, a labeled multigraph contract may still
treat them as distinct arc occurrences.

For an undirected inverse-paired model, `(g,s)` and `(gs,s^-1)` may denote two
orientations of one semantic edge. Traversing one direction then forbids the
reverse direction in an edge-simple trail. A directed-arc model instead treats
them as distinct unless the contract explicitly pairs them.

Thus trail counts and legality depend on simple/labeled/multigraph and
directed/undirected identity choices already emphasized in notes 06 and 37.

## 8. Finite directed Cayley graphs and Euler circuits

Counting labeled generator arcs, every vertex of a finite directed Cayley graph
has one outgoing and one incoming occurrence for each generator. Therefore
indegree equals outdegree. If the positive-alphabet digraph is strongly
connected, it has an Euler circuit through every labeled arc occurrence.

This circuit can revisit each group element many times. It is neither a simple
state path nor a shortest-path object, and its existence says nothing about BFS
frontier depth. It is an edge-coverage statement enabled by regular balance.

For a Schreier action or a projection that merges labeled parallels, degrees
and edge identities must be recomputed in the declared graph rather than
borrowed from the Cayley multigraph.

## 9. Distributed and GPU boundary

For generic exact trails:

- `(v,F)` histories with the same endpoint are distinct work records;
- routing by endpoint does not authorize deduplication;
- a device-global edge bitmap incorrectly shares one history's exclusions with
  all others;
- an undirected paired-edge ID must be consistent across owners;
- labeled parallel arcs need stable occurrence identities if they count
  separately.

Eulerian traversal uses different structural state and coordination. Neither
case is implemented or optimized here.

## 10. Evidence checklist

1. Walk, directed-arc trail, undirected-edge trail, or simple path.
2. Semantic edge identity, labels, parallels, loops, and inverse pairing.
3. Exact length, shortest, longest, enumeration, or all-edge coverage.
4. Endpoint-only versus `(endpoint,used_edges)` state.
5. Directed line digraph versus ordinary undirected line graph.
6. Entry/exit mapping between original vertices and line vertices.
7. Cayley versus Schreier degree and edge-occurrence semantics.
8. Per-history exclusion versus global traversal bookkeeping.

## Sources

- C. Hierholzer,
  [*Ueber die Moeglichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren*](https://doi.org/10.1007/BF01442866),
  Mathematische Annalen 6 (1873), 30-32. Classical Euler-trail construction.
- F. Harary and R. Z. Norman,
  [*Some Properties of Line Digraphs*](https://doi.org/10.1007/BF02854581),
  Rendiconti del Circolo Matematico di Palermo 9 (1960), 161-168. Foundational
  line-digraph treatment.
- Notes 06, 20, 34, 37, 39, 51, 64, and 87 supply graph identity,
  history-product, incidence, contract, nonbacktracking, ownership,
  record-multiplicity, and simple-path distinctions used here.

## Takeaway

A trail's future depends on which semantic edges its own history used. The
direct state is `(v,F)`, so neither vertex visited nor one global edge bitmap is
generically exact. Directed trails correspond to simple paths in a directed
line graph, but this only moves the history constraint; ordinary undirected
line graphs can even lose traversal orientation compatibility. Eulerian trails
are a polynomially structured all-edge special case, not ordinary BFS.
