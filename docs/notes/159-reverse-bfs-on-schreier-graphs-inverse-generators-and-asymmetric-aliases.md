# Reverse BFS on Schreier graphs: inverse generators and asymmetric aliases

Backward search in a directed implicit graph must enumerate predecessors. For a
right group action, every generator action is a bijection, so predecessors are
generated exactly by inverse moves. In a nonfree Schreier action, however, the
alias profile of `S^-1` at a state need not match the forward profile of `S`.

This matters to bidirectional BFS: the two sides can have equal raw generator
count but different loops, distinct support degree, convergence, and routing.

No experiment is used. The three-point action from note 158 gives a complete
hand-checkable witness.

## 1. Reverse successor oracle

The forward labeled occurrence is

```text
x --s--> x s,  s in S.
```

If `y s=x`, then applying the inverse gives `y=x s^-1`. Therefore the reverse
graph occurrence oracle is

```text
x --s^-1--> x s^-1,  s in S,
```

or equivalently the generator collection

```text
S_rev = S^-1 = {s^-1 : s in S}.
```

Reusing `S` for backward search computes forward distance from the target, not
distance to the target, unless the support relation happens to be symmetric.

## 2. Forward aliases use right cosets

At state `x` with stabilizer `K`, note 158 gives

```text
x s = x t  iff  K s = K t.
```

Forward alias classes are intersections of `S` with right cosets `K s`, with
multiplicity

```text
mu_f(Ks) = |S intersect Ks|.
```

## 3. Reverse aliases expose left-coset intersections

Reverse labels are `s^-1`. Their right-coset class at `x` is `K s^-1`, with

```text
mu_b(Ks^-1) = |S^-1 intersect K s^-1|.
```

Inverting every element maps the set `K s^-1` to the left coset `s K`, so

```text
|S^-1 intersect K s^-1| = |S intersect s K|.
```

Thus:

```text
forward profile samples S in right cosets K\G,
reverse profile samples S in left cosets  G/K.
```

When `K` is nonnormal, those partitions differ. Equal stabilizer size and equal
raw occurrence count `|S|` do not force equal support profiles.

## 4. Loop counts nevertheless agree

Forward loops use `S intersect K`. Reverse loops use `S^-1 intersect K`.
Because `K` is closed under inversion,

```text
|S^-1 intersect K| = |S intersect K|.
```

So the number of loop labels at one state is direction-invariant even when the
nonloop alias classes differ. Equal loop count must not be promoted to equal
support outdegree or equal useful frontier work.

## 5. Conditions restoring profile symmetry

Two useful sufficient conditions are:

### Inverse-closed generator collection

If `S=S^-1` as a labeled collection up to the declared inverse pairing, forward
and reverse occurrence sets coincide. The support graph is symmetric and all
per-state alias profiles match.

### Normal stabilizer

If `K` is normal, left and right cosets coincide and inversion permutes the
quotient cosets. Forward and reverse alias-class-size multisets therefore agree,
although labels and endpoint correspondence still need explicit mapping.

Neither condition is necessary in every finite instance: accidental equality
of the realized support profiles can occur. They are structural sufficient
conditions, not an iff theorem for arbitrary `S`.

## 6. Three-point directional counterexample

Use the transitive point action of `S_3` and

```text
S      = {(12),(13),(123)},
S^-1   = {(12),(13),(132)}.
```

Forward successor endpoints from note 158 are

```text
state 1: 2,3,2  -> 2 distinct
state 2: 1,2,3  -> 3 distinct
state 3: 3,1,1  -> 2 distinct
```

Reverse successor endpoints are

```text
state 1: 2,3,3  -> 2 distinct
state 2: 1,2,1  -> 2 distinct
state 3: 3,1,2  -> 3 distinct
```

Therefore support endpoint profiles are

```text
forward: 2,3,2
reverse: 2,2,3.
```

Each state still applies three labels in either direction, and each direction
has the same loop count at that state. The alias mass moves between states and
changes which frontier composition is expensive.

## 7. Bidirectional correctness and work are separate

For source `s_0` and target `t`, forward BFS over `S` computes
`dist(s_0,v)`. Backward BFS over `S^-1` computes `dist(v,t)`. Their meeting
certificates and stopping bounds require those exact meanings.

Choosing which side to expand is a work policy. Comparing only
`|F_f|` and `|F_b|` ignores:

- raw occurrence work `|S||F|`;
- state-dependent support degree;
- same-parent alias excess;
- cross-parent convergence;
- owner routing and metadata requirements.

The smaller state frontier can produce more distinct support arcs or more
remote traffic than the larger frontier. This does not invalidate the
smaller-frontier heuristic; it defines why it is not universally work-optimal.

## 8. Path reconstruction

A backward tree edge labeled `s^-1` represents the original forward move `s`
in the opposite traversal direction. Stitching one path must convert labels and
orientation consistently.

If several reverse labels reach one predecessor state under a stabilizer alias,
one label suffices only for one replayable path. A labeled shortest-path DAG or
path count may require every distinct semantic inverse occurrence. Canonical
state equality alone cannot reconstruct the lost move label later.

## 9. Multi-owner consequences

The same vertex owner function may see different traffic matrices from the two
sides because forward and reverse support endpoints differ. Useful per-side,
per-level telemetry includes:

- occurrence frontier size `|S||F|`;
- distinct support arcs and endpoints;
- loop and alias histograms;
- owner-to-owner routed occurrences;
- distinct parent/label contributions retained at meetings;
- label inversion used for replay.

Balancing forward ownership does not prove balanced reverse ownership. The
authoritative state identity may be shared, while direction-specific occurrence
streams differ.

## 10. Cayley and quotient boundary

In a free Cayley action with distinct generators, stabilizers are trivial, so
neither side has same-parent aliases. Directed asymmetry can still remain when
`S` is not inverse-closed: forward and reverse reachability profiles use
different generator collections.

After a symmetry quotient, nontrivial stabilizers can add the left-versus-right
coset asymmetry described above. A bidirectional implementation copied from the
covering Cayley graph must therefore revalidate reverse generation, labels,
support degree, and path lifting on the quotient.

## Sources and internal dependencies

- Notes 05, 08, 40, and 56 require reverse arcs and sound bidirectional stopping.
- Note 16 fixes right-action, inverse-generator, and coset conventions.
- Notes 57 and 64 define replay and labeled-output multiplicity.
- Notes 157-158 give occurrence and stabilizer-coset accounting.
- The forward/reverse profile formulas follow by inversion of right cosets.

## Takeaway

Reverse Schreier BFS is generated by `S^-1`, not by wishful reuse of `S`.
Forward aliases inspect right cosets of the stabilizer; reverse aliases correspond
to left-coset intersections after inversion. The two search sides can therefore
be semantically exact yet physically asymmetric even with identical raw
generator counts.
