# BFS, Myhill--Nerode equivalence, DFA minimization, and residual languages

In a deterministic finite automaton, the exact future information carried by a
state is its residual language: the set of suffixes that lead from that state
to acceptance. Two memory states can be merged for constrained BFS precisely
when every future word is accepted or rejected identically.

This is stronger than equal current depth, equal accepting status, or equal
distance to an accepting state. It is also narrower than concrete graph-state
identity: minimizing the constraint automaton does not merge distinct base
vertices.

This note adds no implementation, optimizer, benchmark, or GPU code.

## 1. Residual language of a DFA state

Let a complete DFA be

```text
A=(Q, Sigma, delta, q0, F).
```

The right or residual language of state `q` is

```text
L_q = {z in Sigma* : delta*(q,z) in F}.
```

Define

```text
q equivalent r  iff  L_q=L_r.
```

Equivalent states agree not only on whether the empty suffix is accepted, but
on every possible labeled continuation. In particular, equivalence respects
acceptance and is stable under each symbol:

```text
q equivalent r  implies  delta(q,a) equivalent delta(r,a).
```

The latter follows because `L_(delta(q,a))` is the left derivative
`a^(-1)L_q`.

## 2. Myhill--Nerode view from prefixes

For a language `L`, prefixes `u,v` are Myhill--Nerode equivalent when

```text
for every suffix z:  uz in L iff vz in L.
```

These classes are exactly distinct residual languages after prefixes. The
Myhill--Nerode theorem states that `L` is regular exactly when it has finitely
many such classes. For a regular language, those reachable classes form the
unique minimal DFA up to state renaming.

Thus a minimal DFA state is not a remembered prefix. It is the entire future
acceptance behavior shared by possibly many prefixes.

## 3. Why ordinary visited is safe on a DFA

BFS from `q0` over symbol transitions reaches automaton states in minimum word
length. If two words reach the same DFA state, every future suffix has the same
acceptance result from either prefix. Keeping only the first minimum-depth
arrival is therefore sufficient to find the minimum length of any accepted
word.

Acceptance must be checked at depth zero because the empty word may be in the
language. With a fixed total alphabet order and ordered FIFO expansion, the
first accepted word is the shortlex-least accepted word under the conditions of
note 19.

This visited rule answers existence or one shortest witness. It does not
enumerate all accepted words: different prefixes reaching one state are
distinct words and cycles may generate infinitely many of them.

## 4. Minimal DFA preserves shortest accepted length

Quotienting a DFA by residual-language equivalence preserves the language
exactly. Therefore it preserves:

- whether any accepted word exists;
- the minimum accepted word length;
- the set and count of accepted words at every length;
- the shortlex-least accepted word for a fixed alphabet order.

It does not preserve the original state names, prefix provenance, or internal
run-state sequence. A witness word can be replayed through the original DFA if
that metadata is needed.

The minimum DFA is minimal for recognizing this language, not necessarily for
some richer output such as transition counts, diagnostics, tagged histories,
or probabilities.

## 5. Equal distance to acceptance is too weak

Suppose nonaccepting state `p` accepts suffix `a` and nonaccepting state `q`
accepts suffix `b`, with neither accepting the other's one-letter suffix. Both
have minimum distance one to acceptance, but

```text
L_p != L_q.
```

Merging them preserves one scalar distance while changing which labeled words
are legal. Equal BFS layer, equal eccentricity, equal acceptance bit, and equal
nearest-goal distance are all weaker than residual-language equality.

This is the automata analogue of note 128's warning that undeclared
observations permit an overly coarse behavioral quotient.

### One-step similarity is still too weak

Over alphabet `{a,b}`, let nonaccepting states `p` and `q` both have minimum
distance two to acceptance. Their `b` transitions go to the same rejecting sink
`D`; their `a` transitions go to nonaccepting states `r` and `u`:

```text
p --a--> r,   q --a--> u,
p --b--> D,   q --b--> D.
```

From `r`, symbol `a` accepts and `b` rejects. From `u`, symbol `b` accepts and
`a` rejects. Thus every immediate successor of `p` and `q` is nonaccepting,
both have the same labeled one-step acceptance profile and the same nearest
accepting distance, yet

```text
aa in L_p and aa notin L_q,
ab notin L_p and ab in L_q.
```

Comparing only current observations or one expansion layer cannot prove safe
history-state merging. The condition is recursive: corresponding successors
must themselves remain equivalent for every future suffix. Residual-language
equality packages that unbounded continuation requirement into the state
equivalence.

## 6. Relation to deterministic bisimulation

View a complete DFA as a deterministic labeled transition system whose only
state observation is accepting versus rejecting. Residual-language equivalence
is then the coarsest strong same-label bisimulation respecting that observation:

- equal residuals imply equal observations and equivalent `a`-successors;
- bisimilar states accept exactly the same finite suffixes by induction.

This coincidence relies on deterministic total symbol transitions and the
acceptance observation. General nondeterministic transition systems,
probabilities, weights, or additional observations require their own
equivalence.

## 7. Partial DFAs and dead sinks

A partial DFA treats a missing symbol transition as rejection of that
continuation. Standard minimization can first complete it with one rejecting
dead sink that loops on every symbol. Omitting the sink physically is safe only
if the same missing-transition semantics remains explicit.

Two partial states that have different missing labels can still be equivalent
only if those differences lead to the same residual behavior. Comparing only
stored outgoing edges without the implicit rejecting sink can misclassify them.

## 8. NFA and epsilon boundaries

An NFA configuration is generally a set of active NFA states, not one state.
Subset construction makes those reachable sets deterministic; residual
equivalence can then minimize the resulting DFA. Merging individual NFA states
by a naive DFA rule is not the standard minimal-DFA theorem, and minimum NFA
size is a different problem.

Epsilon transitions consume no input symbol. Counting them as unit BFS edges
changes accepted word length. Epsilon closure, elimination, or an explicit
zero-cost model must precede any claim about shortest accepted words.

## 9. Minimizing the automaton factor of product BFS

In note 20's labeled graph product, states are `(v,q)` and a base edge labeled
`a` moves to `(v',delta(q,a))`. Replacing `A` by an equivalent minimal DFA
preserves which base-walk label words are accepted. Therefore it preserves the
minimum accepted walk length to the declared base target.

The exact product key becomes

```text
(base vertex, residual-language class).
```

Distinct base vertices remain distinct even when their automaton classes are
equal. Conversely, the same base vertex paired with two inequivalent residual
classes must remain two product states.

Automaton minimization is a semantic reduction of memory state, not permission
to use base-only visited.

## 10. Product reachability can hide unreachable DFA states

The globally minimal DFA may contain states unreachable from `q0` only if the
input automaton was not first trimmed; a true minimal accessible DFA does not.
Even its reachable states need not all appear in one particular graph product,
because the base graph may not realize every alphabet word.

One can reason about a product-specific reachable quotient, but its correctness
must be relative to the actual base labeled transition system. The standalone
minimal DFA is canonical for the language and independent of one base graph;
a product-specific reduction may not transfer to another graph epoch.

## 11. Cayley constraints and normal-form languages

For a Cayley product `(g,q)`, DFA minimization preserves the regular constraint
language exactly and therefore preserves the shortest accepted generator word
that evaluates to the target group element or orbit state.

It does not prove that the accepted language is a geodesic unique normal form.
Coverage, uniqueness under group or Schreier evaluation, geodesicity, and
prefix closure remain the separate conditions of note 99. A smaller automaton
recognizing the same language cannot repair a semantically unsuitable language.

Nor does residual equivalence identify group elements: automaton states encode
future word acceptance, while Cayley vertices encode evaluated group state.

## 12. Reverse and bidirectional search

Reversing a DFA language generally produces an NFA because one forward state
may have several predecessors under a symbol. Minimizing the forward DFA does
not make its transitions invertible.

Bidirectional constrained search must still construct compatible reverse
memory semantics and test whether forward and backward residual conditions can
form one accepted word. Equality of minimized forward-state IDs at a base
meeting is not a general compatibility theorem.

## 13. Frontier and multiplicity interpretation

DFA minimization can reduce the number of product frontier records by merging
memory states with identical futures. It preserves accepted label words, but
the physical quotient frontier no longer records how many original automaton
states or prefixes mapped into each residual class.

Separate:

- distinct DFA states;
- residual classes;
- distinct prefix words;
- base walks carrying those words;
- unique product states;
- unique base states.

These counts answer different questions. A state-level BFS queue is not a word
enumerator or path-counting dynamic program.

## 14. GPU and multi-GPU boundary

Any later measurement should report separately:

- original DFA size, reachable trim, and minimized size;
- alphabet and total/partial transition convention;
- minimization construction and validation;
- base and automaton product cardinalities;
- original and minimized product frontiers;
- accepted words, concrete paths, and quotient records;
- owner key for `(base,residual class)`;
- cross-owner reconciliation and witness replay;
- preprocessing, traversal, and end-to-end time.

Different devices must use one identical class numbering or a stable canonical
class identity. Locally minimized fragments are not globally equivalent if a
remote suffix distinguishes their states. This is the automata form of note
128's global partition-epoch requirement.

## Sources

- J. Hopcroft,
  [*An n log n Algorithm for Minimizing States in a Finite Automaton*](https://doi.org/10.1016/B978-0-12-417750-5.50022-1),
  Theory of Machines and Computations, 1971, for partition-refinement DFA
  minimization.
- R. R. Williams,
  [*The Myhill--Nerode Theorem*](https://people.csail.mit.edu/rrw/6.045-2020/lec6-before-class.pdf),
  MIT 6.045 lecture notes, for residual equivalence and uniqueness of the
  minimal DFA.
- J. A. Brzozowski,
  [*Derivatives of Regular Expressions*](https://doi.org/10.1145/321239.321249),
  Journal of the ACM 11(4), 1964, for derivatives as residual-language states.
- Notes 12, 19, 20, 37, 53, 57, 64, 99, 101, and 128 supply this repository's
  zero-cost, shortlex, product, contract, sampling, output, multiplicity,
  geodesic-language, refinement, and bisimulation boundaries.

## Takeaway

Myhill--Nerode equivalence merges DFA states exactly when every possible suffix
has the same acceptance result. Minimization therefore preserves the language
and shortest accepted-word BFS, and safely reduces the memory factor of a
constrained product search. It does not merge base vertices, enumerate all
prefixes, make an NFA minimal, invert transitions, or prove that a Cayley
language is geodesic and unique.
