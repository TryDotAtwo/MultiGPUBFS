# Generator order parity and early BFS signatures

A generator's order does not by itself determine a Cayley graph, but it does
predict one small and useful family of local signatures.

Let a symmetric unit-cost generator alphabet contain `g` and `g^-1`, and let
`g` have finite order `m`.

## Even order

If `m=2k`, then

```text
g^k = g^-k.
```

Both words have length `k`. If no shorter relation reaches that element, they
are two geodesics converging at F_k. For Cube QTM, `m=4` and `k=2`, producing
the six F2 equalities

```text
g g = g^-1 g^-1.
```

This is convergence: two same-depth words have the same endpoint.

## Odd order

If `m=2k+1`, then

```text
g^(k+1) = g^-k.
```

The two natural directions around the power cycle have lengths `k+1` and `k`.
The Cayley edge always exists:

```text
g^k --g--> g^(k+1)=g^-k.
```

It joins two vertices in F_k only if the length-`k` power words are geodesic
in the full declared generator alphabet. Symmetry gives the two endpoints
equal distance from the identity, but their common distance can be less than
`k` because of other generators.

For current Megaminx, `m=5` and `k=2`; REF-026 observed exactly these F2
same-level edges. The signature is not a convergence of two length-two words.

## Why metric matters

The algebraic power identities do not depend on the metric. Their proposed
BFS layers require the geodesicity condition above under the full alphabet.
If `g^2` is added as a unit generator, the graph changes:

- distances change;
- the layer at which the power relation appears changes;
- candidate, frontier, and duplicate counts change.

Shortcuts can also use generators outside the cyclic subgroup. For example,
in additive `Z_7 x Z_2`, take `g=(1,0)` and the symmetric unit alphabet
`{(1,0),(-1,0),(0,1),(3,1),(-3,1)}`. Here `ord(g)=7`, but
`g^3=(3,0)=(0,1)+(3,1)` and `g^-3=(-3,0)=(0,1)+(-3,1)` both have distance
exactly two: neither is the identity or a generator. Thus their power-cycle
edge is in F2, not F3, although the alphabet's only moves in `<g>` are `g`
and `g^-1`.

Thus QTM and HTM are not performance modes of one BFS graph. They define
different generating sets and therefore different word metrics.

## Propagation is not novelty

REF-029 found 6 extra static trace classes at Cube F2, 108 at F3, and 1,521 at
F4. Yet closing words under static commutation plus the same six F2 half-turn
relations leaves zero remainder through F4.

The lesson is general: a growing duplicate or relation-witness count does not
count new primitive relations. One short relation can be translated, embedded
in longer words, reordered through commuting moves, and combined with itself.
BFS counters observe occurrences in the explored ball, not a minimal group
presentation.

## State representation boundary

REF-029 also compared unique sticker labels with six repeated face colors.
Their exact balls agree through radius four, proving injectivity of the color
orbit map on that ball and excluding a nontrivial stabilizer word of length at
most eight. This is strong bounded evidence, but not a global free-action
proof.

For an arbitrary puzzle encoding, repeated symbols should therefore trigger a
Schreier/stabilizer audit rather than an automatic claim that the explored
graph is a Cayley graph.
