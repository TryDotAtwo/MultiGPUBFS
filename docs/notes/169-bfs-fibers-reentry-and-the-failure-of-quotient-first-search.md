# BFS fibers, re-entry, and the failure of quotient-first search

A quotient can identify the first depth at which a fiber is reachable. It does
not turn the original BFS into a two-stage algorithm that first chooses a
quotient block and then searches only inside that block.

Original shortest paths may enter the same fiber through several boundary
states, leave it, and later re-enter it. Owner blocks are partitions of state
authority, not automatically independent subproblems.

No hierarchical implementation is proposed here.

## 1. First fiber arrival is only one scalar

For a normal Cayley quotient and coset `C=gH`, note 168 proved

```text
alpha(C) = min_(x in C) d_G(e,x) = d_(G/H)(H,C).
```

This identifies the nearest state in `C`. The fiber actually carries the
distance multiset

```text
Delta(C) = { d_G(e,x) : x in C }.
```

`alpha(C)` is only its minimum. Quotient BFS does not recover the other values,
their multiplicities, their parent fibers, or the levels at which the owner
will receive work.

## 2. A fiber can disappear and reappear

Take the undirected cycle `Z_6` with generators `{+1,-1}` and normal subgroup

```text
H={0,3}.
```

The left-coset owner blocks are

```text
C0={0,3}, C1={1,4}, C2={2,5}.
```

BFS from `0` has

```text
F0={0}       in C0,
F1={1,5}     in C1,C2,
F2={2,4}     in C2,C1,
F3={3}       in C0.
```

Owner block `C0` is active at depths zero and three with no states at depths one
or two. Its quotient distance is zero, but that does not mean all its concrete
states are settled at depth zero or that the owner can be retired after its
first activation.

The quotient frontier describes first block arrivals. Original frontiers can
continue to occupy any previously activated block.

## 3. Local subgroup generators may not generate the fiber

Under left-coset ownership `gH`, labels in `S intersect H` are local. But the set

```text
T = S intersect H
```

need not generate `H`. In the `Z_6` fixture above, `T` is empty even though the
fiber `H` contains two states. State `3` is reached only by leaving the block
and re-entering it.

Therefore “all subgroup labels are local” does not imply “the owner can reach
all states of its fiber using only local labels.” The former is an edge-locality
fact; the latter is a generation claim requiring `<T>=H`.

## 4. Even a generated fiber can have external shortcuts

Let `G=Z_20`, let `H` be the even residues, and use directed generators

```text
S={+2,+1,+5}.
```

Here `T=S intersect H={+2}` generates `H`. Inside the local `+2` subgraph,
reaching state `6` from `0` takes three steps:

```text
0 -> 2 -> 4 -> 6.
```

But the original Cayley graph has the two-step path

```text
0 --+1--> 1 --+5--> 6,
```

which leaves `H` and re-enters it. A local-only BFS inside the target fiber
returns distance three although the exact global distance is two.

Thus `<S intersect H>=H` is still insufficient for local fiber distances to
equal global distances. One would need an additional geodesic-convexity or
isometric-subgraph property for the fiber under the full generator metric.

## 5. Quotient-first plus local correction is not generally additive

A tempting formula is

```text
d_G(e,g)
?= d_(G/H)(H,gH) + local_distance_within_gH(entry,g).
```

It is not generally well-defined because a quotient path can lift to different
entry representatives. It is not generally exact because:

- the shortest quotient path need not choose the best entry for the fixed
  target;
- several equal quotient paths can enter through different states;
- the best concrete path may revisit quotient blocks;
- moving kernel/subgroup portions of a noncommutative word to the end can alter
  the concrete word and its generator length;
- local fiber distance can exceed a path that leaves and re-enters.

The exact target distance is a minimum over compatible lifted words, not a sum
of two independently minimized scalars.

## 6. Owner blocks are authoritative shards, not search phases

In distributed BFS, an owner may receive states for many original depths and
from many source owners. Its durable responsibilities include:

- exact identity and visited authority for every state assigned to it;
- convergence of all same-depth proposals required by the output contract;
- publication of accepted states into the correct global depth;
- retention or recovery of state needed by later re-entry;
- participation in global completed-layer or quiescence evidence.

“This owner was already visited” is not meaningful. Visited applies to semantic
states, not to the whole ownership block. Block-level first contact cannot
replace state-level novelty.

## 7. Multiple entries and shortest-path metadata

Suppose a fiber first appears at depth `a`. Later states in the same fiber can
have shortest parents:

- inside the fiber;
- in a different fiber;
- in several fibers simultaneously;
- through labels that are distinct but alias at the Schreier state level.

A complete shortest-path DAG therefore cannot be reconstructed from one
quotient parent per block. Scalar fiber distance retains only the first entry
depth and loses concrete predecessor multiplicity and compatible representatives.

For bidirectional BFS, meeting in the same owner block is likewise not a state
intersection. The forward and reverse records must identify a common semantic
state or supply a proven, replayable connector inside the fiber.

## 8. When hierarchical reasoning can be exact

A quotient/fiber decomposition needs explicit extra structure, for example:

- path lifting with retained representatives;
- an isometric or geodesically convex fiber for the relevant targets;
- a product or semidirect-product metric theorem with compatible generators;
- dynamic programming over boundary states rather than one scalar per block;
- a proven covering with the required lift semantics.

These conditions are problem-specific. Normality alone gives a quotient group
and exact distance to the fiber, not an additive distance to each member.

## 9. Multi-GPU measurements implied by re-entry

For each block/owner record:

```text
first active depth,
all active depths,
frontier states per depth,
entries from each source block,
local versus leaving/re-entering shortest-parent arcs,
accepted states and duplicate convergence,
resident state retained between active intervals,
bytes and critical time per active interval.
```

This distinguishes a block touched once in one broad wave from a block with
sparse repeated activation. Equal total owned states can hide very different
temporal load profiles.

## 10. Rejected implications

- First quotient arrival settles every state in the fiber.
- A previously active owner block can be retired permanently.
- `S intersect H` necessarily generates `H`.
- If `S intersect H` generates `H`, local distances are globally shortest.
- Normality makes quotient and fiber distances additive.
- One quotient parent reconstructs the concrete shortest-path DAG.
- Forward and reverse searches meeting in one owner block have met in a state.
- Owner-level visited can replace state-level visited.

## 11. Current synthesis

Quotient BFS compresses where the wave first touches a fiber. It does not
compress the entire time evolution inside that fiber. Original BFS layers cut
across owner blocks, and shortest paths can cross the same block boundary more
than once.

The safe mental model is therefore “owners shard state authority while one
global metric wave evolves,” not “the quotient chooses a block and the block is
then solved locally.”

This note extends notes 08, 11, 16, 17, 51, 57, 93, 167, and 168.
