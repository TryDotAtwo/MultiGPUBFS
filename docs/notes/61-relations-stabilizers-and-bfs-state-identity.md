# Relations, stabilizers, and BFS state identity

A move word can collide in BFS for two fundamentally different algebraic
reasons: it can represent the same group element, or distinct group elements
can act identically on the represented state. The second phenomenon is not an
implementation duplicate. It follows from the chosen vertex identity.

## The orbit map

Let a group `G` act on the right of a base state `x0`, and let

```text
H = Stab(x0) = {h in G : x0*h = x0}.
```

The map

```text
pi(g) = x0*g
```

projects the Cayley graph onto the Schreier state graph. It preserves every
labeled transition:

```text
pi(g*s) = pi(g)*s.
```

But it is injective exactly when the action is free on the orbit. In general,

```text
pi(g1) = pi(g2) iff H*g1 = H*g2.
```

The sets `H*g` are **right cosets**, and their space is denoted `H\G`.
They carry the right action `H*g -> H*g*s`; left cosets instead have the form
`g*H` and form `G/H`.

Consequently every Cayley walk projects to a valid state walk, while several
Cayley vertices and paths may become one state vertex or path endpoint.

## What distance is minimized

For a target state `x0*g`, state BFS computes

```text
dist_state(x0, x0*g) = min { |w|_S : w in H*g }.
```

It does not generally compute `|g|_S` for an arbitrary representative `g`.
For start `x0*a` and target `x0*b`, the valid solution words are

```text
w in a^-1 * H * b.
```

Thus a single-element Cayley target is a sound normalization only for a free
action or after a separate representative-independence proof.

Projection immediately gives the one-way metric inequality

```text
dist_state(pi(a), pi(b)) <= dist_Cayley(a,b),
```

for the selected representatives. Equality is extra structure, not a default.

## Identity relations versus stabilizer words

Starting at the Cayley identity, a word closes exactly when it evaluates to the
group identity. Starting at `x0` in the action graph, it closes when it evaluates
to any element of `H`.

At another state `x0*g`, the closing words belong to the conjugate stabilizer
appropriate to the fixed right-action convention. Therefore a generator can be
a loop at one state and a genuine edge at another even though every state has
one labeled occurrence per generator.

This changes the interpretation of the signatures from note 60:

- candidate convergence can witness equality only modulo a stabilizer;
- a loop can be the shortest nonidentity stabilizer word;
- a same-level edge can close a state cycle without yielding the corresponding
  identity relation in `G`;
- frontier shrinkage can come from action non-freeness before short group
  relations become visible.

## Labeled graph versus simple graph

The Schreier action supplies one outgoing occurrence per generator. Applying
state equality may produce loops or several labels with one endpoint. A simple
graph view may remove loops and merge parallel endpoints.

That simplification can preserve distances between distinct vertices while
changing all of the following:

- labeled word counts;
- shortest closed-word witnesses;
- generator-specific predecessor multiplicity;
- occurrence-based work estimates;
- bipartiteness if loops were part of the declared graph.

The graph contract must therefore distinguish move occurrences from unique
simple neighbors.

## REF-024 as the smallest concrete picture

With `G=S3` and adjacent swaps `s0=(01), s1=(12)`:

- the regular Cayley action is free, has six vertices, frontiers `1,2,2,1`,
  and first candidate convergence at the length-three braid endpoint;
- the action on point `0` has stabilizer `{identity,s1}`, three vertices,
  frontiers `1,1,1`, and a root loop labeled `s1`;
- point `2` is two state moves away, but one group representative mapping `0`
  to `2` is three Cayley moves away.

The same generators therefore do not determine the BFS geometry without the
vertex/action declaration.

## Consequences for puzzle research

Before transferring a relation signature to Cube or Megaminx, record:

1. whether a vertex is a full move-group element, a concrete configuration, a
   partially observed configuration, or an optional symmetry class;
2. the precise right/left action and move composition order;
3. the stabilizer of the chosen base state or evidence that it is trivial;
4. whether loops and duplicate labeled destinations are retained;
5. whether the target is one concrete state, an intrinsic coset, or an optional
   symmetry orbit;
6. whether returned words are replayed on the concrete state action.

Only then can an observed collision be called an identity relation, a
stabilizer relation, or a quotient artifact.

## Sources and evidence boundary

- Alexander Hulpke, *Computational Group Theory*,
  [Chapter VII](https://www.math.colostate.edu/~hulpke/lectures/m501/notes.pdf),
  for group actions, orbits, stabilizers, and Schreier graphs.
- Yaroslav Vorobets, *Notes on Schreier graphs*,
  [PDF](https://people.tamu.edu/~yvorobets/Research/Schreier.pdf), for
  coset/action graph conventions.
- Notes 16, 17, 27, and 60 establish the existing action, quotient, girth, and
  duplicate-signature contracts. REF-024 validates the finite S3 example.

The algebraic statements above are proved from the orbit map. The measured
counts cover only REF-024's six-element group and three-state action; they do
not quantitatively characterize any production puzzle.

## Current synthesis

BFS always quotients the word tree by its declared state identity. In a Cayley
graph that identity is group-element equality. In a Schreier graph it is
equality of action states, equivalently coset equality. Relations explain the
first; stabilizers add further collisions in the second. Optimizing visited or
duplicate handling before fixing that identity risks making the wrong graph
fast.
