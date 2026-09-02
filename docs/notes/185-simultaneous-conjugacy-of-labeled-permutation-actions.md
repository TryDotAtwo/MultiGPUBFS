# Simultaneous conjugacy of labeled permutation actions

## Purpose

Note 184 reduced CayleyPy/DeepCubeA transfer to a simultaneous conjugacy
question. This note develops that criterion far enough to prevent a future
oracle check from proving the wrong equivalence.

The subject is exact action semantics, not an optimized isomorphism algorithm.
No calculation or runtime experiment is reported.

## 1. States carry two copies of the position convention

Let `Omega_C` and `Omega_D` be two 54-position coordinate systems. A
unique-sticker state in convention `C` is a bijection

```text
x : Omega_C -> Omega_C,
```

where `x(i)` is the identity of the sticker currently occupying position `i`.
The solved state is the identity map.

A coordinate correspondence `q : Omega_C -> Omega_D` must rename both roles:

```text
T_q(x)(q(i)) = q(x(i)).
```

Equivalently, `T_q(x)=q x q^-1`. Renaming only array positions gives
`y(q(i))=x(i)` and generally maps the solved state to `q^-1`, not to the target
solved state. That one-sided reorder is therefore the wrong cross-runtime state
oracle for unique sticker IDs.

For repeated colors, sticker values are not another copy of the position set.
Then the value transformation is instead a declared color map. The unique-ID
and color-quotient oracles are different contracts.

## 2. Pull-action convention

Suppose a move table `p` uses

```text
A_p(x)(i) = x(p(i)),
```

the `new[i]=old[p[i]]` convention used by REF-029. If target move table `r`
uses the same convention, then

```text
T_q(A_p(x)) = A_r(T_q(x)) for every x
```

holds exactly when

```text
r = q p q^-1.
```

Thus checking the 54-entry move tables is sufficient for every reachable
state and every word, provided action side and table meaning were normalized
first. Comparing rendered turns by eye is not that normalization.

## 3. A labeled action is a tuple, not six independent moves

Write the source tuple as `(p_g)_(g in Sigma_C)`. With a signed-label bijection
`lambda : Sigma_C -> Sigma_D`, exact labeled equivalence means

```text
p_(lambda(g))^D = q p_g^C q^-1     for every g.
```

The same `q` must satisfy the whole tuple. Individual permutation conjugacy is
only a necessary test: it says corresponding generators have the same cycle
type.

A small counterexample uses three positions:

```text
source tuple: ((12), (13), (23))
target tuple: ((12), (13), (13)).
```

Every paired component is a transposition and hence individually conjugate.
No single `q` conjugates the tuples, because conjugation is injective and
cannot map the two distinct source permutations `(13)` and `(23)` to the same
target permutation.

Both tuples even generate the same abstract group `S_3`. Abstract generated
group isomorphism therefore does not by itself identify a labeled action.

## 4. Labeled graph equivalence and weaker projections

The tuple criterion is exactly an isomorphism of the directed labeled
transition structures after applying `lambda` to labels. It preserves:

- exact reachability and distance;
- every word endpoint after label translation;
- inverse pairs and relations;
- labeled shortest paths, canonical word order after translating its alphabet,
  and replay artifacts;
- loops, parallel labeled transitions, and stabilizers.

Several weaker objects forget information:

- the simple support graph forgets labels, multiplicity, and loops;
- sphere counts forget adjacency inside and between layers;
- generator orders forget mixed-generator relations;
- a list of selected relation words is only a fingerprint;
- the abstract generated group forgets which permutation action is used.

These weaker comparisons may reject equivalence, but passing them does not
construct the required `q`.

## 5. Orbit propagation

Assume `lambda` is fixed. Choose a source anchor `i` and a proposed target
anchor `j=q(i)`. The equations force

```text
q(p_w^C(i)) = p_(lambda(w))^D(j)
```

for every generator word `w`.

Consequences:

1. On a transitive action, one anchor image determines every value of `q` if
   the result is well-defined.
2. Well-definedness fails if two source words reach the same position but
   their translated target words reach different positions.
3. Injectivity fails if distinct source positions are forced onto one target
   position.
4. For several orbits, at least one compatible anchor choice is needed per
   orbit, and orbit sizes and labeled structure must match.

This propagation is a semantic description of the proof obligation, not a
performance prescription. A future probe may use another exact method.

## 6. Stabilizer formulation

For a transitive action and fixed anchors `i,j`, propagation is well-defined
precisely when every word fixing `i` in the source also fixes `j` after label
translation. For finite orbits of equal size and an invertible label/group
translation, this becomes equality of the corresponding point stabilizers.

This explains why relation checks at the group identity are not enough. The
action also depends on point stabilizers. Cayley regular actions have trivial
stabilizer; Schreier and repeated-color actions generally do not.

## 7. Existence is not uniqueness

If `q_1` and `q_2` both conjugate the same labeled source tuple to the same
target tuple, then

```text
c = q_2 q_1^-1
```

commutes with every target generator. Conversely, composing one solution with
such a commuting permutation gives another solution. Hence all solutions form
a coset of the centralizer of the target action.

For a transitive action, fixing one anchor removes any commuting automorphism
that moves that anchor: a commuting permutation fixing the anchor must fix its
entire generated orbit. In a regular Cayley action there can be many unanchored
coordinate conjugacies, while the image of the identity selects one.

Therefore a comparison report should not silently treat the first discovered
`q` as canonical. It should state the anchor convention or report the remaining
action automorphisms relevant to output reproducibility.

## 8. Generator-label ambiguity

Before solving for `q`, `lambda` itself may be unknown. Face names narrow the
possibilities but do not prove signs or action side. Necessary filters include:

- inverse pairing;
- cycle type of each signed generator;
- commutation/noncommutation matrix;
- cycle type or equality of short mixed words;
- declared face/opposite-face structure.

These filters can reduce candidates but remain fingerprints. The final gate is
still the simultaneous 54-entry equation for every signed generator.

Canonical words add another condition: if source and target alphabets have
different lexical orders, translating a source-shortlex word need not yield
the target-shortlex word even when it remains a shortest valid word.

## 9. Exact future evidence contract

A bounded CayleyPy/DeepCubeA comparison should emit:

1. normalized pull/push convention for both tables;
2. the complete signed-label map `lambda`;
3. the complete position map `q`;
4. confirmation that `q` renames both positions and unique sticker IDs;
5. all 12 simultaneous generator equations;
6. solved-state and mixed-word replay checks;
7. anchor/canonicalization policy and any remaining ambiguity;
8. immutable source commits and fixture hashes;
9. the first exact mismatch on failure.

Passing only sphere prefixes, move orders, one-face traces, or independent
per-generator conjugacy must be reported as partial evidence.

## 10. GPU and multi-GPU boundary

Action conjugacy is an oracle-correctness gate that should precede performance
comparison. It does not require a GPU. Once proved, it allows corresponding
frontiers and visited sets to be compared after `T_q`; it does not imply equal
layout locality, hash distribution, owner routing, memory traffic, or elapsed
time. Those physical properties can change under a semantically exact
renumbering.

## Sources and dependencies

- Notes 16 and 123 define Cayley/Schreier action and covering distinctions.
- Notes 128 and 168 distinguish safe quotient behavior from concrete labeled
  action identity.
- Note 184 and REF-029 provide the pinned Cube action question.
- The conjugacy, orbit, stabilizer, and centralizer statements above are direct
  finite permutation-action derivations under the declared conventions.

## Compact conclusion

Cross-runtime Cube equivalence is equality of a whole labeled permutation
action up to one coordinate bijection, not six unrelated visual move checks.
For unique sticker states the bijection acts on both array positions and
sticker IDs. Orbit propagation exposes existence, point stabilizers expose
well-definedness, and the action centralizer exposes nonuniqueness. None of
these semantic facts predicts GPU performance.
