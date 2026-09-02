# CayleyPy and DeepCubeA Cube actions: a conjugacy audit

## Question

What must be checked before a Cube BFS result obtained with CayleyPy's move
tables can be transferred to DeepCubeA's environment?

This is a source and semantics audit, not a performance study. No runtime was
executed because the authoritative Docker server was unavailable, and the
study rule forbids replacing a Docker calculation with a host calculation.

## Pinned sources already used by REF-029

REF-029 pinned two independently written actions:

- CayleyPy commit `fbdde24b891d956b9ec905b939f848ecab711978`, whose
  `cayleypy/puzzles/cube.py` explicitly lists six 54-position face
  permutations;
- DeepCubeA commit `919489f14ecbbc80dc1bf1539ac0a462ffaca7c5`, whose
  `environments/cube3.py` constructs the affected source and destination
  indices geometrically.

The current workspace contains the CayleyPy source snapshot but not a
DeepCubeA checkout. The latter was therefore re-read from the official
repository at the already pinned commit. This is enough to refine the semantic
question, but not to claim executable equivalence.

## Correction: both runtime states are 54 sticker IDs

The earlier wording “coordinate conjugacy” could suggest a sticker-to-cubie
conversion. That is not the representation used by the pinned DeepCubeA
environment:

- `goal_colors = arange(54)` gives every sticker position a distinct byte;
- `Cube3State.colors` stores that length-54 array;
- equality is exact array equality;
- `_move_np` copies selected old positions into selected new positions;
- only `state_to_nnet_input` applies integer division by nine, turning the
  unique IDs into six face classes for the neural network.

Thus search identity in DeepCubeA is the unique-sticker state. Its neural
input is a many-to-one observation of that state and must not be substituted
for exact BFS visited identity.

CayleyPy's REF-029 fixture likewise used 54 unique sticker labels for the free
permutation action and separately studied the repeated-color quotient. Those
two CayleyPy models must not be conflated with DeepCubeA's exact runtime state
and its derived neural observation.

## What differs syntactically

The two sources do not share position numbering or necessarily the same sign
name:

- CayleyPy orders its six faces according to its own flat sticker layout and
  states a full permutation by cycles;
- DeepCubeA flattens `(face,row,column)` with face order
  `U,D,L,R,B,F`, generates moves named `U-1,U1,...,F-1,F1`, and updates
  `new[new_idx] = old[old_idx]`;
- REF-029's Rust fixture uses `new[i] = old[p[i]]` after expanding CayleyPy's
  cycles and creates inverse labels explicitly.

Matching the same face letter is therefore only a hypothesis. A reversed sign
or a different viewpoint convention can preserve order four and early sphere
counts while still making word labels incompatible.

## Exact transfer criterion

Let `P_C(g)` be CayleyPy's position action and `P_D(h)` DeepCubeA's. Runtime
equivalence requires one bijection `q` of all 54 positions and one bijection
`lambda` of the 12 signed move labels. For unique-sticker state `x`, the same
`q` must rename both array positions and sticker IDs:

```text
T_q(x)(q(i)) = q(x(i)).
```

Then the required equation is

```text
T_q(P_C(g)(x)) = P_D(lambda(g))(T_q(x))
```

for every unique-sticker state `x`. Renaming positions alone would generally
fail to map solved state to solved state. Because these are permutation
actions, it is sufficient to check the corresponding 54-entry permutations
for all 12 generators after normalizing action convention. The same single
`q` must work simultaneously; choosing a separate renaming per face proves
nothing about words using several faces. Note 185 develops this criterion.

The comparison should report `lambda` explicitly. Without it, equality of
unlabeled graphs transfers distances and sphere sizes at most, not move words,
canonical parents, or replay artifacts.

## Why the existing REF-029 evidence is insufficient

REF-029 established for the CayleyPy action:

- exact QTM sphere sizes through radius four;
- the first order-four relation signature;
- bounded agreement between unique-sticker and repeated-color balls.

DeepCubeA source inspection independently shows the same broad Cube/QTM
contract. These facts are necessary sanity checks, but they do not determine
the simultaneous conjugacy. Many differently labeled permutation actions have
six order-four face generators and the same shallow unlabeled sphere counts.

## Future bounded Docker gate

When Docker becomes available naturally, a small Rust probe should consume
two immutable 12-by-54 permutation fixtures and:

1. validate that each table is a permutation and every declared inverse pair
   composes to identity;
2. solve for or ingest a declared position bijection `q` and signed-label map
   `lambda`;
3. check all 12 generator conjugacy equations entry by entry;
4. replay a fixed mixed-face word in both actions through the mapping;
5. emit the complete maps, source commits, fixture hashes, and first mismatch.

This is an oracle-validation fixture, not an optimized solver. It needs no GPU
and must not trigger Docker repair. A failed comparison should first test sign,
action-side, and flattening conventions before being interpreted as a
different puzzle graph.

## Consequences for BFS study

1. Exact visited identity and neural observation are separate contracts even
   inside one application.
2. Equal shallow BFS counts are a useful fingerprint but not an action proof.
3. An unlabeled graph isomorphism transfers distances; a labeled simultaneous
   conjugacy is needed to transfer paths as move words.
4. Application-scale performance is not meaningful evidence until successor,
   equality, goal, and label semantics pass this boundary.
5. This audit narrows the next experiment; it does not promote REF-029's
   DeepCubeA status from unverified to validated.

## Sources

- CayleyPy, `cayleypy/puzzles/cube.py`, pinned by REF-029 at commit
  `fbdde24b891d956b9ec905b939f848ecab711978`.
- DeepCubeA, `environments/cube3.py`, official repository, pinned by REF-029 at
  commit `919489f14ecbbc80dc1bf1539ac0a462ffaca7c5`.
- REF-029 report and Rust fixture in `experiments/`.
