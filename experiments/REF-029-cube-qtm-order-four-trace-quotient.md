# REF-029: Cube QTM order-four relation propagation

Date: 2026-08-28

Status: pass after infrastructure, intentional RED, and formatting failures

## Question

Does the Megaminx progression from static commutation to later geodesic
relations transfer unchanged to an independently specified 3x3x3 Cube action?

## External action and metric

The fixture is transcribed from the explicit six 54-sticker face-cycle
permutations in the clean upstream CayleyPy checkout:

```text
repository: https://github.com/cayleypy/cayleypy.git
commit: fbdde24b891d956b9ec905b939f848ecab711978
cube.py SHA-256: 2f4bb00375fe38090ed8e1fcc3ed4b0a0f8a787beee2da873181f928baa5b630
metric: QTM
generators: U,U',D,D',L,L',R,R',B,B',F,F'
```

The Rust probe reproduces CayleyPy's `new[i]=old[p[i]]` convention, constructs
each inverse, and checks that all 12 signed moves have order four and pair with
their declared inverse.

A second clean source, DeepCubeA commit
`919489f14ecbbc80dc1bf1539ac0a462ffaca7c5`, independently computes 12 Cube3
face moves from geometric indexing. Its source SHA-256 is
`79fcea960ca5f6d81ce7ff3d12238d214f19af3a5819a0ca5221f206b6f038a9`.
This pass audited that contract but did not execute a coordinate conjugacy
comparison: the existing Rust Docker image lacks NumPy, and importing the
external Python oracle failed explicitly for that reason.

## Two state models

The same moves were explored from:

1. 54 unique sticker labels, which makes the permutation action free and gives
   a genuine Cayley graph of the generated subgroup;
2. CayleyPy's six repeated face colors, which is formally an orbit/Schreier
   representation until trivial stabilizer is established.

Both models have exact sphere sizes

```text
1, 12, 114, 1068, 10011
```

through `F4`, agreeing with the published QTM prefix. Equality of every ball
cardinality through radius four makes the orbit map injective on that ball.
Consequently this color representation has no nonidentity stabilizer word of
length at most eight: such a word could be split into two words of length at
most four whose unique-sticker elements would collide under the color map.

This is a bounded result, not a global proof that the color action is free.

## Static trace quotient

The first layers for the unique-sticker model are:

| Depth | Candidate records | States | Geodesic words | Static trace classes | Extra trace classes |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 12 | 12 | 12 | 0 |
| 2 | 132 | 114 | 132 | 120 | 6 |
| 3 | 1,236 | 1,068 | 1,416 | 1,176 | 108 |
| 4 | 11,580 | 10,011 | 15,144 | 11,532 | 1,521 |

Unlike current Megaminx, Cube QTM leaves the static trace quotient already at
depth two. The six witnesses are exactly

```text
g g = g' g'
```

for the six faces. Since `g` has order four, both sides are the same half turn:
`g^2 = g^-2`.

## Quotient by commutation plus the order-four rewrite

The probe next closes each fixed-length word under only:

- adjacent swaps of commuting face moves;
- `g g <-> g' g'` for each face.

The result is exact through F4:

| Depth | Classes after both rewrites | States | Remainder |
|---:|---:|---:|---:|
| 1 | 12 | 12 | 0 |
| 2 | 114 | 114 | 0 |
| 3 | 1,068 | 1,068 | 0 |
| 4 | 10,011 | 10,011 | 0 |

Thus all 108 F3 and 1,521 F4 extra static trace classes are translations,
insertions, commuting reorderings, or combinations of the six order-four
half-turn equalities. They are not evidence for 1,629 new primitive relation
families.

The F4 distribution still becomes richer: 1,467 states have more than one
static trace class and one state has up to four. The known-relation quotient
collapses all of them to one class per endpoint.

## Contrast with Megaminx

Generator order changes the earliest BFS signature:

- order four: `g^2=g^-2` gives two geodesic length-two words for one state;
- order five: `g^2` and `g^-2=g^3` have different lengths, so the power cycle
  first appears as a same-level edge between the two F2 vertices rather than an
  F2 convergence.

This is a parity/metric effect, not a claim about cubes versus dodecahedra in
general. Changing the move metric by making a half turn one generator would
change the graph and the signature again.

## Evidence record

The intentional RED compile failed on absent audit functions. The first full
GREEN gate passed three tests and stopped only on formatting. A second
test-first cycle introduced the order-four quotient; the final Docker gate
passed four tests, formatting, optimized compile, and execution.

Two infrastructure failures were retained:

- one Docker orchestration cell lost the output of a completed command;
- the external Python smoke failed with `ModuleNotFoundError: numpy`.

No package was installed and no external checkout was modified.

```text
image: multigpubfs-rust-toolchain:dev
image id: sha256:764a443c2ddc39b28b8fbb0b1495656984ea5ee8dd82f7f435f2069a6574ce69
workspace mount: read-only
GPU requested: no
```

## Artifacts and limits

- `experiments/ref029_cube_qtm_relation_onset.rs`
- `experiments/REF-029-cube-qtm-order-four-trace-quotient.txt`
- probe source SHA-256:
  `ccb3a4a0847c1dc5241d5ced4a12bae191c0a836ac95ed71c40a15a929640032`;
- raw output SHA-256:
  `d3006bcacad338cc7d243805838f04566992cf121bdb8494b91253a0f8f151c5`;
- exhaustive only through depth four;
- exact full-state equality, no hashing;
- DeepCubeA runtime equivalence remains unverified;
- no production implementation, timing, GPU work, or optimization.
