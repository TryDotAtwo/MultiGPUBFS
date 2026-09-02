# Cayley and Schreier ownership: cosets, orbits, and routing

An implicit Cayley graph has no stored adjacency array to partition. Ownership
is therefore a function of semantic state identity. Two broad choices expose a
useful tension:

- hash ownership aims at statistical balance but usually destroys generator
  locality;
- algebraic ownership can make a declared generator subgroup local but does
  not guarantee balanced BFS frontiers.

This note derives the exact locality statements. It does not recommend a
partition or implement one.

## 1. Right Cayley action and the matching cosets

Let vertices be elements of a finite group `G`, with directed labeled edges

```text
g --s--> g s,    s in S.
```

Choose a subgroup `H <= G` and assign one algebraic owner block to each left
coset

```text
gH.
```

Then an occurrence labeled `s` stays in the same block exactly when

```text
g s H = g H
<=> s H = H
<=> s in H.
```

Therefore every label in `S intersect H` is local at every vertex, and every
label in `S minus H` crosses to another left coset at every vertex. At the raw
labeled-occurrence level,

```text
local fraction  = |S intersect H| / |S|,
remote fraction = |S minus H| / |S|.
```

This is an exact algebraic count, not a timing prediction. It assumes one block
per left coset and counts generator occurrences, including any labels repeated
in the generator manifest.

For a left action `g -> s g`, the matching construction uses right cosets `Hg`.
These are the standard names: `gH` is a left coset and `Hg` is a right coset.
Mixing the formulas for the action and coset sides destroys the simple locality
proof; the action side does not determine the coset's name.

## 2. Equal global blocks do not imply balanced waves

Every left coset has `|H|` elements, so the full Cayley vertex set is divided
into `[G:H]` equally sized blocks. This proves exact total-state balance when
there is one owner per coset.

It does not prove per-level balance. Define

```text
f_(d,C) = |F_d intersect C|
```

for coset block `C`. The values can be highly unequal even though every block
has the same final capacity. A BFS rooted in `H` begins entirely on that one
owner. Generators inside `H` expand only within it; other blocks are reached
through quotient-crossing labels later.

Thus algebraic locality can create long underutilized phases. Total vertex
balance, current-frontier balance, occurrence balance, byte balance, and
critical-time balance remain separate claims.

## 3. Normality changes quotient predictability, not basic locality

If `H` is normal, left and right cosets coincide and form the quotient group
`G/H`, and each label induces a well-defined quotient transition

```text
gH -> (gH)(sH) = gsH.
```

Owner-level motion can then be studied as a Cayley walk on `G/H` with generator
images `sH`; labels in `H` become quotient loops.

If `H` is not normal, the partition and the exact local-label theorem still
hold. What is lost is a representative-independent right group action of
`G/H`: the destination block of an outside label can depend on the particular
state inside the source coset. Normality is therefore a quotient-structure
boundary, not a prerequisite for coset ownership.

## 4. Mapping many cosets to fewer GPUs

Usually `[G:H]` is not the device count. A second map

```text
left coset -> GPU
```

coalesces algebraic blocks. All `H`-label edges remain local. Some outside-label
edges may also become device-local because their two cosets were assigned to
the same GPU.

Now exact local/remote traffic depends on both the quotient/coset transition
profile of the current frontier and the block-to-GPU map. The lower algebraic
guarantee is retained, but the fraction `|S intersect H|/|S|` is no longer the
complete device-local fraction.

Capacity also depends on the sum of assigned coset sizes plus frontier and
scratch peaks. Equal numbers of cosets per GPU give equal total state capacity
for Cayley vertices, but still not equal frontier work.

## 5. Schreier actions replace cosets by subgroup orbits

Now let states be right cosets `Kg` in `K\G`, with the right action

```text
Kg --s--> Kgs.
```

To make every generator in `H` local, group states by their right `H`-orbits:

```text
(Kg)H = {Kgh : h in H}.
```

These owner blocks correspond to double cosets

```text
K\G/H.
```

Every `h in H` stays inside the orbit by construction. Unlike Cayley left
cosets, orbit sizes need not be equal. The orbit-stabilizer formula gives

```text
|(Kg)H| = [H : H intersect g^-1 K g].
```

State stabilizers therefore control both same-parent generator aliases and the
capacity of algebraic owner blocks. This is a substantive Cayley/Schreier
boundary: a subgroup-local partition that is perfectly equal on `G` can become
intrinsically imbalanced on `K\G`.

An outside generator `s notin H` can sometimes remain in the same `H`-orbit;
that occurs when `Kgs=Kgh` for some `h in H`. Therefore the Cayley equivalence
“local iff `s in H`” does not transfer unchanged to a nonfree Schreier action.

## 6. Hash ownership

Let an exact canonical identity determine

```text
owner(x) = hash(x) mod P.
```

Under an ideal independent uniform model, distinct endpoints are balanced in
expectation, and a distinct edge is remote with probability `1-1/P`. This is a
model statement, not a deterministic guarantee for a structured frontier or a
particular hash.

Hash ownership can spread early and late frontiers more evenly than an
algebraic block partition, while routing most support arcs. It can also place
many convergent candidates at their common authoritative owner, which is
necessary for exact global novelty but may concentrate contention.

The owner hash must be based on exact semantic identity or followed by full
collision resolution. Statistical balance cannot compensate for aliased
states.

## 7. Routing matrices expose what scalar cut fractions hide

For each completed level record a source-owner to destination-owner matrix for:

```text
raw generator occurrences,
distinct support arcs,
unvisited support arcs,
accepted unique states,
payload and protocol bytes.
```

The matrices distinguish several situations with the same scalar remote
fraction:

- traffic evenly spread across peers;
- one hot destination owner;
- many occurrences converging to few states;
- few wide records carrying most bytes;
- balanced totals but a long critical tail.

For algebraic ownership, also retain per-frontier coset/orbit occupancy. For
Schreier actions, record stabilizer and orbit-size distributions.

## 8. Authority and replication remain separate

Partition choice answers where identity is authoritative. Replicas or advisory
filters can reject some old states before routing, but stale negatives cannot
claim global novelty. A locally generated state that belongs to another owner
must still reach an authoritative decision or an equivalent exact protocol.

Algebraic knowledge that a label is local removes routing for that occurrence;
it does not remove visited lookup, duplicate convergence, frontier publication,
or global layer completion.

## 9. Rejected implications

- Equal Cayley coset sizes imply balanced BFS levels.
- A low generator cut implies high multi-GPU utilization.
- Normality is required for subgroup-local left-coset ownership under right
  multiplication.
- Cayley coset balance transfers directly to Schreier graphs.
- An outside generator always crosses a Schreier `H`-orbit.
- Hash ownership deterministically balances every frontier.
- Fewer remote occurrences proves fewer bytes or lower level time.
- A locality-preserving owner function makes distributed visited unnecessary.

## 10. Current synthesis

For right Cayley multiplication, left cosets give an exact and unusually clean
locality law: membership of the generator in `H` decides locality everywhere.
Normality upgrades the block graph to a quotient group, but does not create
frontier balance. For Schreier actions, the correct blocks are `H`-orbits or
double cosets, whose stabilizers create variable capacity and alias behavior.

Hash and algebraic ownership therefore exchange different kinds of structure;
their value can only be assessed with per-level semantic and routing matrices,
not total state balance or one edge-cut scalar.

This note extends notes 07, 16, 51, 52, 158, 159, 160, and 165.
