# BFS, geodesic languages, and automatic Cayley graphs

A finite automaton can recognize a language of generator words. That is not yet
an exact state BFS. The missing question is what the accepted words mean after
evaluation in the group or puzzle action.

This note isolates the exact conditions under which automaton enumeration
matches BFS spheres. It proposes no implementation or optimizer.

## 1. Four properties that must not be conflated

Let `S` be a finite generator alphabet and

```text
pi : S* -> G
```

evaluate a word as a group element. For a language `L subset S*`, ask
separately:

1. **coverage:** `pi(L)=G`;
2. **uniqueness:** `pi` restricted to `L` is injective;
3. **geodesicity:** every `w in L` has `|w|=|pi(w)|_S`;
4. **prefix closure:** every prefix of a word in `L` also belongs to `L`.

Regularity means only that a finite automaton recognizes `L`. It implies none
of the four properties above.

## 2. When accepted length really is a BFS layer

If `L` has coverage, uniqueness, and geodesicity, then evaluation gives a
bijection

```text
{w in L : |w|=r}  <->  {g in G : |g|_S=r}.
```

The right side is exactly the radius-`r` BFS sphere around the identity in the
Cayley graph. In this special semantic sense, enumerating accepted words does
not need a global `visited` set to remove two accepted representatives of the
same element: uniqueness has already proved that collision impossible.

Prefix closure is a further property. With it, deleting the last letter of a
nonempty accepted word stays inside `L`; the accepted prefixes form a canonical
rooted geodesic tree. Without it, the length-`r` accepted set can still match
the sphere, but a generator that emits only accepted complete words is not
automatically an online frontier traversal through accepted prefixes.

## 3. DFA counts and rational growth

For a finite automaton, let `M` count labeled transitions between states, `u`
select the start state, and `v` select accepting states. The number of accepted
words of length `r` is

```text
a_r = u^T M^r v.
```

Therefore

```text
sum_(r>=0) a_r z^r = u^T (I-zM)^(-1) v,
```

a rational formal power series. This becomes the spherical growth series of
the Cayley graph only when coverage, uniqueness, and geodesicity have all been
proved. Otherwise it counts syntax -- accepted words -- rather than distinct
states at shortest distance.

## 4. Three small counterexamples

### Regular and unique does not imply geodesic

Take `Z=<a>` with `A=a^-1` and

```text
L = { a^(n+2) A A : n>=0 } union { A^k : k>=1 }.
```

This is regular and gives exactly one representative of every integer. The
first branch represents nonnegative `n`, while the second represents negative
integers. But `aaAA` represents the identity with length four, and every word
on the first branch contains avoidable cancellation. Unique normal forms need
not preserve BFS depth.

### Regular and geodesic does not imply unique

Let two distinct labels `a` and `b` both act as `+1` on the integer state line,
with corresponding inverse labels. Both `a^r` and `b^r` are geodesic words for
the same state `r`. A regular language containing both overcounts every
positive sphere even though each word is shortest.

This is also a warning that labeled transitions and unique neighbor states are
different graph contracts.

### Freely reduced does not imply geodesic in a quotient

In `C_3=<a | a^3=e>`, write `A=a^-1`. The word `aa` has no adjacent inverse
cancellation, yet it represents `A`, whose word length is one. Free reduction
removes only the universal inverse relations, not the defining relations of
the quotient group.

## 5. What automaticity does and does not say

An automatic structure uses a regular representative language together with a
uniform fellow-traveler or multiplier-automaton condition for nearby group
elements. This is much stronger than merely having some regular language that
maps onto the group, but the chosen representatives are not geodesic or unique
unless the particular structure says so.

For word-hyperbolic groups, geodesic languages over finite symmetric generating
sets are regular and yield automatic structures. This is a special theorem,
not a property of arbitrary finitely generated groups or arbitrary generator
presentations. Even for a finite group, choosing one geodesic word per element
produces a finite and hence regular language; that per-instance observation is
vacuous for scalability unless automaton size and construction are controlled
across the family.

## 6. This is not product-state BFS

Note 20 uses an automaton to constrain admissible paths. Its semantic vertex is
`(graph_state, automaton_state)`, because the automaton phase can change which
continuations are legal.

Here the automaton instead attempts to enumerate canonical representatives of
ordinary Cayley vertices. It can replace the duplicate-filled word tree only
after coverage, uniqueness, and geodesicity are established for the evaluated
states. If any condition is unknown, exact equality and `visited` remain the
safe state-BFS mechanism.

## 7. Cayley normal forms can collide in a Schreier graph

A puzzle state often belongs to an orbit `G/H`, not to `G` itself. Distinct
group elements `g` and `gh`, with `h in H`, act to the same orbit state. Thus a
language that is unique over group elements can still emit many words for one
puzzle state.

Replacing state-level `visited` would require a proved unique geodesic
transversal for the relevant cosets under the actual action convention, not
merely a group normal form. This is the same stabilizer distinction exposed in
notes 16, 61, and 62.

## 8. Reverse and bidirectional qualifications

Regular languages are closed under formal reversal, but that fact alone does
not prove that reversed accepted words are canonical geodesic representatives
under the reverse action. One must also handle inverse letters, multiplication
orientation, coverage, and uniqueness.

In bidirectional search, equality of group or orbit states remains the meeting
predicate. Equality of automaton control states is not equality of endpoints.

## 9. GPU and multi-GPU interpretation

A small DFA may reject many syntactically redundant word extensions before
state materialization. That does not by itself prove less total state work or
higher throughput:

- automaton states add control context and may change divergence or batching;
- automaton size can grow across a puzzle family;
- canonical word ownership and canonical state ownership are different;
- a Schreier stabilizer can recreate duplicates after group evaluation;
- a faster word generator is not an exact BFS unless the semantic bijection is
  proved.

These are conceptual boundaries, not a proposed GPU design. Any performance
claim would need bounded measurements under an explicit language, state
identity, action, and hardware contract.

## 10. Evidence checklist

1. Alphabet, inverse convention, and evaluation orientation.
2. Coverage of the actual group or orbit state space.
3. State-level uniqueness, not merely unique strings or group elements.
4. Geodesicity in the declared generator metric.
5. Prefix closure if online frontier generation is claimed.
6. Automaton size and construction cost across the intended family.
7. Exact meeting/equality semantics for reverse or bidirectional use.
8. Separate accepted-word counts, evaluated states, and unique BFS states.

## Sources

- D. B. A. Epstein, J. W. Cannon, D. F. Holt, S. V. F. Levy, M. S. Paterson,
  and W. P. Thurston, [*Word Processing in
  Groups*](https://books.google.com/books/about/Word_Processing_in_Groups.html?id=lH2NEQAAQBAJ),
  Jones and Bartlett, 1992. Automatic structures, regular normal forms,
  multiplier automata, and shortlex structures.
- J. W. Cannon, [*The combinatorial structure of cocompact discrete
  hyperbolic groups*](https://doi.org/10.1007/BF00146825), Geometriae Dedicata
  16 (1984), 123-148. Finite cone types and regular geodesic structure in the
  hyperbolic setting.
- Notes 16, 19, 20, 23, 35, 39, 61, 62, 64, 81, 92, and 93 provide the action,
  shortlex, product-state, word-tree, growth, nonbacktracking, stabilizer,
  record-multiplicity, BFS-tree, generator, and word-metric context.

## Takeaway

A finite automaton recognizes words, while BFS discovers semantic states at
minimum distance. The two coincide only after coverage, uniqueness, and
geodesicity are proved; prefix closure is additionally needed for the most
direct canonical frontier tree. In CayleyPy-style orbit actions, a group normal
form is not enough because stabilizers can collapse distinct group elements to
one puzzle state.
