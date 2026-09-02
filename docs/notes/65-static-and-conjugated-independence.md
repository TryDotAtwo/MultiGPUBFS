# Static and conjugated independence in Cayley BFS

REF-027 and REF-028 expose two different meanings of "independent moves."

## Static independence

For generators `a` and `b`, static independence is

```text
ab = ba.
```

It can be stored as a fixed matrix over the generator alphabet. Adjacent swaps
using that matrix generate a trace monoid equivalence. Through current
Megaminx depth three, this equivalence explains every equality among shortest
words.

## A commuting composite after conjugation

At depth four, exact equalities first appear in the form

```text
g (h k h^-1) = (h k h^-1) g.
```

The composite group element `h k h^-1` commutes with `g`, even though the
spelling does not expose an adjacent statically commuting pair that connects
the two words. In a permutation interpretation, conjugation transports the
support of `k` by `h`.

This must not be described as a generator whose legality or meaning changes
with the current BFS state. Under the right Cayley convention, an edge from
`x` labeled `k` is always `x -> xk`; the generator is global and fixed. What
changed is the vocabulary: `h k h^-1` is a composite group element, not one
letter in the original trace alphabet. A dependence relation fixed solely on
single move names can therefore be sound but incomplete for equality of whole
group words.

## What BFS keeps and discards

State-only BFS still keeps exactly what the distance problem requires:

- first discovery fixes the shortest distance in an unweighted graph;
- visited dedup safely merges all later occurrences of the same state;
- the next frontier needs the state, not a presentation of every word.

What it discards is equally concrete:

- how many shortest histories reached the state;
- whether those histories are connected by static commuting swaps;
- which conjugated or longer relations caused additional merging.

Hence "duplicate" is not one mathematical phenomenon. Candidate duplication,
shortest-word multiplicity, static trace equivalence, and equality in the
represented group are successive quotients. Their counts need not agree.

## Consequence for future probes

A compact next experiment should not merely propagate one static trace normal
form per state. REF-028 supplies a counterexample to that abstraction. A more
honest probe might track a bounded relation signature or selected conjugated
elements, while retaining state equality as the source of truth. This is an
open research direction, not yet an implementation plan.
