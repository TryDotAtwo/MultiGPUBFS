# Local certificates for BFS distance labels

A completed BFS run can be checked without reproducing its queue schedule. The
key is to validate the mathematical output: upper-bound path witnesses plus a
lower-bound inequality on every graph edge.

This gives a compact way to understand what large CPU, GPU, and distributed BFS
validators are actually proving.

## Directed edge geometry

Let `delta(v)` be shortest directed distance from source `s`. For every directed
edge `u -> v` with finite `delta(u)`, appending that edge to a shortest path to
`u` gives

```text
delta(v) <= delta(u) + 1.
```

This is one-sided. A directed edge may point from a deep layer to a much earlier
layer. For example,

```text
s -> a -> b -> c
^              |
|______________|
```

has distances `0,1,2,3`, while edge `c -> s` spans three levels backward. Thus
the frequently used test

```text
abs(delta(u) - delta(v)) <= 1
```

is false for general directed graphs.

For an undirected graph, every edge can be traversed both ways. Applying the
one-sided inequality in both orientations gives

```text
abs(delta(u) - delta(v)) <= 1.
```

This is why Graph500's undirected BFS validator can scan every input edge and
require its endpoints to differ by at most one level.

## A complete local certificate

Suppose a result supplies labels

```text
L: V -> nonnegative integers union {infinity}.
```

For a complete single-source directed BFS result, check:

1. **Root:** `L(s)=0`, and no other vertex has label zero.
2. **Predecessor witness:** every finite-labeled `v != s` has an incoming edge
   `u -> v` with `L(u)=L(v)-1`.
3. **Edge feasibility:** every edge `u -> v` with finite `L(u)` satisfies
   `L(v) <= L(u)+1`; in particular, `v` cannot be labeled infinity.

These three local conditions prove that every label is the true shortest
distance and that every reachable vertex is included.

### Upper-bound half

Repeatedly following the predecessor witness decreases the nonnegative label by
one, so it reaches `s` after exactly `L(v)` steps. Every link is a real directed
edge. Hence there is an actual path of length `L(v)`:

```text
delta(v) <= L(v).
```

### Lower-bound half

Take any directed path

```text
s = x_0 -> x_1 -> ... -> x_k = v.
```

Starting from `L(s)=0`, edge feasibility inductively gives `L(x_i)<=i`, so

```text
L(v) <= k.
```

This holds for every path to `v`, therefore `L(v)<=delta(v)`. Combining both
halves yields `L(v)=delta(v)`.

Edge feasibility also proves reachability closure: a finite-labeled vertex
cannot have an outgoing edge to an infinity-labeled vertex. Starting at `s`, no
reachable vertex can escape the labeled set.

## Why either half alone is insufficient

A parent chain alone proves only an upper bound. On

```text
s -> v
s -> a -> v
```

the real path `s->a->v` can be stored with label two even though the true
distance is one. Every parent edge is valid, but edge `s->v` violates
`L(v)<=L(s)+1`.

Edge inequalities alone prove only that labels are not too large relative to
paths. Assigning label zero to every vertex satisfies all unit-edge
inequalities, but clearly does not describe source distances. Root uniqueness
and decreasing predecessor witnesses exclude such underestimates.

A matching histogram of level counts supplies neither half: missing and
spurious vertex identities can cancel in every count.

## Parent array versus existential witness

If the output includes `parent[v]`, condition 2 can be checked by requiring

```text
parent edge exists: parent[v] -> v
L(parent[v]) = L(v)-1.
```

An arbitrary-parent BFS needs only one such witness. A canonical parent,
shortlex path, all-predecessor DAG, or shortest-path count needs additional
checks over every eligible equal-depth predecessor. The distance certificate
does not silently validate richer metadata.

Parent pointers also must be well-founded. Strict label decrease already
excludes a parent cycle among finite nonnegative labels, so a separate cycle
test becomes redundant if labels and every parent decrement are trusted. A
validator deriving levels only from parents, as Graph500 permits, must instead
detect cycles explicitly.

## Local recurrence certificate for shortest-path counts

### Boundary: substituting weights requires a rooted witness

The unit certificate above is not generalized merely by replacing `+1` with
`+weight`. With zero-cost edges, vertices other than `s` may correctly have
label zero, so root uniqueness must not be imposed on weighted distances.
Also, parent equality need not strictly decrease the label.

For a finite directed graph with nonnegative finite weights, an exact
single-source distance certificate can instead require:

1. `L(s)=0`, and all labels are nonnegative finite values or infinity.
2. Every finite-labeled non-source vertex has a real tight parent edge, and
   its selected parent chain terminates at `s` in finitely many steps.
3. Every edge from finite-labeled `u` has a finite-labeled endpoint `v` with
   `L(v)<=L(u)+weight(u,v)`.

Summing tight parent equalities along the rooted chain gives a real path of
cost `L(v)`, proving `dist(s,v)<=L(v)`. Summing edge inequalities along any
source path gives the reverse bound. Finite-successor closure excludes missed
reachable vertices. The two proof halves survive; strict unit-depth decrease
is replaced by the explicit root-termination certificate.

Why the replacement is necessary: take an isolated source and an unreachable
zero-cost two-cycle `p->q->p`. False labels `L(p)=L(q)=1`, with parents pointing
to each other, pass root uniqueness, real-edge checks, tightness, and every
weighted edge inequality. They fail only the rooted-witness requirement.
Conversely, a genuine zero edge `s->a` makes `L(a)=0` correct, contradicting
the unit-only unique-zero requirement. Neither example concerns a runtime
implementation here; they test the logical certificate.

This certifies weighted distances, not path-count recurrences across zero-cost
cycles. Numerical tolerances or floating-point approximate equality would
require their own error contract rather than the exact equalities above.

### Unit-depth count recurrence

Assume the distance labels `L` have already passed the complete certificate and
the validator can enumerate every graph edge under the declared path identity
convention. Check

```text
sigma(s)=1,
sigma(v)=0 when L(v)=infinity,
sigma(v)=sum sigma(u)
         over every edge u->v with finite L(u) and L(u)+1=L(v),
         for finite-labeled v!=s.
```

Parallel labeled edges contribute separately exactly when they define distinct
requested paths. Because every eligible predecessor has depth one smaller,
induction over `L=0,1,2,...` uniquely fixes every `sigma(v)`. There is no cycle
in which incorrect positive counts can support one another: the recurrence
always bottoms out at the source layer.

The finite-label restriction is essential, not just an implementation guard.
Take an isolated source `s` and an unreachable directed cycle `p->q->p`.
Correct distances are `L(s)=0`, `L(p)=L(q)=infinity`. If an unrestricted
recurrence treats `infinity+1=infinity`, it mistakenly admits both cycle edges
as shortest predecessors. Then `sigma(p)=sigma(q)=7` satisfies those two
circular equations even though neither vertex has a path from `s`. Setting
unreachable counts to zero and restricting predecessor eligibility to finite
depth restores the well-founded induction. A numeric sentinel must implement
this semantic guard rather than relying on arithmetic overflow behavior.

Completeness of the predecessor scan is essential. In the diamond

```text
s->a->t
s->b->t,
```

a claimed predecessor list containing only `a->t` is internally consistent
with `sigma(t)=1`; the full graph recurrence gives two. Checking the recurrence
only over reported parents validates the report against itself, not against the
graph. Exact arithmetic also matters: a modular or saturated recurrence
certifies only that declared arithmetic output, not the exact integer count.

## Complete traversal versus a bounded ball

For a claimed exact ball through radius `R`, edges leaving the boundary are
allowed. Replace full closure by:

```text
for every edge u -> v with L(u) < R:
    v is labeled and L(v) <= L(u)+1.
```

Vertices at label `R` need valid predecessor chains, but their outgoing
neighbors may be outside the materialized ball. This certifies every distance
through `R`; it does not claim component exhaustion.

### Absent from a bounded table is not an infinity certificate

An exact ball through radius `R` certifies for an absent target only
`dist(s,t)>R`, with infinity still one possible value. It does not select
infinity over a larger finite distance. For radius one, compare the two graphs

```text
G1: s -> a -> t
G2: s -> a       t isolated
```

Both have the same exact retained ball `{s,a}` and labels `0,1`. The bounded
certificate need not scan edges leaving boundary vertex `a`, so it accepts
that same ball in both graphs. Yet `dist_G1(s,t)=2` and `dist_G2(s,t)=infinity`;
their shortest-path counts to `t` are respectively one and zero.

Therefore three statuses must stay distinct:

| Evidence | Distance claim for t | Full shortest-path count claim |
|---|---|---|
| t retained with a valid finite-distance certificate | Exact finite distance | Exact only after the count recurrence is also certified |
| t absent from exact B_R | Greater than R, possibly infinity | Unknown in general; no path within radius R |
| t absent from a certified successor-closed reached set containing s | Infinity | Zero |

The zero-for-infinity count rule earlier in this note applies to the last
status, not every missing table entry. A timeout/OOM/cancelled traversal may
certify even less than `dist>R` unless a completed radius is separately known.
One numeric sentinel can encode these cases in storage only if the result's
scope and completion metadata keep their meanings distinct. This is a
semantic observation, not an API or implementation change.

For a target-only result at distance `D`, one valid parent chain proves an
upper bound `D`. Proving minimality still requires a lower-bound certificate
covering all paths that might reach the target in fewer than `D` steps. A scan
only along the returned path cannot provide that.

## Reachability labels and infinity

The infinity convention needs care in directed graphs:

- an edge from finite `u` to infinity-labeled `v` contradicts completeness;
- an edge from infinity-labeled `u` to finite `v` is harmless because `u` need
  not be reachable from `s`;
- an edge between two infinity-labeled vertices says nothing;
- in an undirected graph, one finite and one infinite endpoint always
  contradicts component closure.

This is another reason not to apply an undirected absolute-difference validator
to directed output.

## Explicit graph validation

For an explicit edge list or CSR graph, the certificate can be checked by:

```text
one scan of vertices for roots and parent witnesses
one scan of edges for feasibility and reachability closure.
```

The work is `O(V+E)` and can be much cheaper operationally than recreating the
solver's frontier data structures. It is still linear in the claimed graph;
there is no general sublinear exact validator that can ignore an adversarial
uninspected edge capable of creating a shorter path.

The [Graph500 benchmark specification](https://graph500.org/?page_id=12) uses
this style for its undirected output: it validates the root/tree, parent edges,
one-level edge differences, and spanning of the source component. Its contract
is important context—the symmetric endpoint inequality relies on Graph500's
undirected input.

## Implicit graph validation

For an implicit graph, “scan every edge” means regenerating every successor of
every relevant state. This may cost roughly the same transition work as the
original traversal. More importantly, using the exact same successor code in
solver and validator can preserve a shared omission.

Confidence is stronger when validation has an independent basis, such as:

- a separately specified move/action interpreter;
- replay against source puzzle rules;
- exhaustive small-domain enumeration with an independent rank;
- generator permutation and inverse-composition checks;
- metamorphic relations known from the group presentation;
- cross-representation parity, for example explicit oracle versus implicit
  generation on a tractable subgraph.

Path replay checks witnesses but not successor completeness. Independent edge
generation is what makes the lower-bound half credible.

## Hash-indexed output

The certificate quantifies over semantic vertices. If labels, parents, or the
edge scan address vertices only by a colliding hash, all checks may agree on the
same unintended quotient graph. Exact validation therefore needs an injective
rank, full-state collision resolution, or another proof that the tested key is
exact for the claimed domain.

Aggregate XORs or fingerprints can detect many accidental corruptions but do
not replace quantified membership and edge checks.

## Parallel and multi-GPU validation

The conditions decompose naturally across owners:

- each owner validates its local vertices and outgoing edges;
- remote endpoint labels must be fetched or communicated exactly;
- cross-rank parent edges require source label and adjacency confirmation;
- every rank reports violation counts and first witnesses;
- a global reduction proves zero violations only after all partitions and
  in-flight validation messages are complete.

This is embarrassingly parallel in arithmetic but not communication-free.
Owner imbalance follows the distribution of vertices and checked edges, while
remote-label traffic follows the edge cut. A fast all-reduce of one final flag
does not account for the cost or completeness of obtaining remote labels.

A robust validator should retain concrete witnesses, not only a Boolean:

```text
violating edge (u,v)
L(u), L(v)
owner/version of both labels
missing or invalid parent
graph/generator version
```

That turns a failed correctness gate into useful diagnostic evidence.

## What this certificate does not prove

Even a perfect pass does not by itself prove:

- a particular BFS schedule was used;
- deterministic processing or parent order;
- all shortest parents or correct path counts;
- claimed work, timing, memory, or scaling measurements;
- correctness under a different graph version or generator convention;
- completeness beyond a declared bounded radius.

It proves the distance/reachability object identified by its graph and scope.
That is already a strong and implementation-independent result.

## Compact validator contract

```text
graph direction and version:
source and infinity convention:
claimed scope: target / radius / component:
semantic vertex identity:
root uniqueness:
parent-edge and one-level decrement checks:
directed edge feasibility or undirected absolute difference:
reachability closure / permitted radius boundary:
independent successor evidence:
distributed completion reduction:
retained failure witnesses:
richer metadata checks, if requested:
```

## Current conclusions

1. BFS distances admit a local certificate consisting of real decreasing
   predecessor witnesses and one feasibility inequality per edge.
2. The absolute one-level edge rule is an undirected specialization, not a
   universal BFS invariant.
3. Parent replay proves upper bounds; complete edge feasibility supplies the
   missing lower bound.
4. Bounded balls require closure only below the final radius, not across its
   outer boundary.
5. For implicit and hash-indexed graphs, successor completeness and exact
   semantic identity remain the hard parts of validation.
6. Distributed validation can partition the scans, but global zero violations
   is meaningful only after every owner and remote-label obligation completes.
