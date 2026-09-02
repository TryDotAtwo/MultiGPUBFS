# BFS, NFA subset states, antichains, and dominance

An NFA can occupy several states after reading one word. The deterministic
search vertex is therefore the entire set of possible current NFA states, not
one chosen run state and not the union of configurations reached by different
words. BFS over these reachable subsets finds shortest accepted words exactly.

Subset inclusion creates useful monotonicity, but it is not an unconditional
visited rule. Pruning by inclusion depends on existential versus universal
acceptance, search depth, base-graph position, and the requested witness or
enumeration contract.

This note adds no implementation, optimizer, benchmark, or GPU code.

## 1. NFA and subset transition

Let an epsilon-free NFA be

```text
A=(Q, Sigma, Delta, I, F),
```

where `I subset Q` is the initial set and `Delta(q,a)` is a set of successors.
For a subset `P subset Q`, define

```text
post_a(P) = union_(q in P) Delta(q,a).
```

The deterministic subset automaton starts at `I`, follows `post_a`, and accepts
subset `P` exactly when

```text
P intersect F is nonempty.
```

After reading word `w`, its subset is exactly the NFA states reachable by some
run labeled `w`. This is one deterministic semantic configuration for that
word.

## 2. BFS on reachable subsets

Treat every reachable subset as one graph vertex and every alphabet symbol as
a unit edge. Ordinary BFS from `I` then gives the minimum word length reaching
an accepting subset. With fixed alphabet order and the ordered-frontier
conditions of note 19, it can select the shortlex-least accepted word.

Authoritative visited must compare full subsets. Reaching NFA state `q` as one
member of two different subsets does not make those subset configurations
equal, because their other active states can enable different suffixes.

The empty subset is the rejecting dead state of a complete determinization. It
may be stored implicitly, but its transition and rejection semantics remain.

## 3. A frontier is a set of subsets

At word depth `d`, subset BFS can contain many distinct configurations:

```text
F_d subset 2^Q.
```

The union

```text
union_(P in F_d) P
```

is generally not a substitute for `F_d`. It combines NFA possibilities that
belong to different input prefixes into one fictional configuration.

For example, suppose prefix `a` reaches `{p}`, prefix `b` reaches `{q}`, state
`p` can accept suffix `x`, and state `q` can accept suffix `y`. Replacing the
two configurations by `{p,q}` and attaching it to prefix `a` falsely makes
`ay` appear accepted; attaching it to `b` falsely makes `bx` appear accepted.

Frontier union preserves neither word provenance nor deterministic
configuration identity.

## 4. Exponential state space is a semantic possibility

An `n`-state NFA has at most `2^n` subset states, and classical families reach
exponentially many of them under determinization. The reachable subset graph
may be much smaller for one automaton, but `n` NFA states alone do not imply an
`O(n)` deterministic visited universe.

Subset frontier width, total reachable subsets, active-state population inside
each subset, accepted words, and accepting runs are separate quantities.

## 5. Inclusion monotonicity

Subset transition is monotone:

```text
P subset R  implies  post_a(P) subset post_a(R).
```

By induction this holds for every suffix word `z`. Under existential
acceptance, the suffix language from a subset is the union of suffix languages
from its members, so

```text
P subset R  implies  L(P) subset L(R).
```

A superset has at least the accepting possibilities of a subset. This is the
order-theoretic fact behind some antichain methods. Which side of the order to
retain depends on the decision problem and fixed-point direction.

## 6. Qualified dominance for one shortest accepted path

Consider constrained product records

```text
(v,P) reached at depth d_P
(v,R) reached at depth d_R.
```

For existential acceptance, if

```text
P subset R and d_R <= d_P,
```

then `(v,R)` dominates `(v,P)` for the output "existence and length of one
shortest accepted continuation to the declared base target." Any base suffix
walk from the same `v` whose labels are accepted from `P` is also accepted from
`R`, and the replacement prefix is no longer.

Every premise matters:

- different base vertices can have different available suffix walks;
- a superset reached later can lose optimal total length;
- universal acceptance reverses or changes order reasoning;
- shortlex comparison also needs the prefix-word order;
- enumerating all accepted words or paths must retain distinct prefixes;
- preserving a particular NFA run needs more than existential acceptance.

This is a semantic dominance theorem, not a proposed pruning implementation.

## 7. Equal size and overlap prove nothing

Two subsets of equal cardinality may have completely different residual
languages. Large intersection does not imply equivalence or dominance. A
compact hash, popcount, Bloom signature, or sorted-state prefix can be an
advisory filter, but exact visited needs exact subset equality unless a stronger
proved quotient or dominance contract applies.

Likewise, one subset accepting now and another rejecting now says only whether
the empty suffix is accepted. Their nonempty suffix languages can still relate
in either direction.

## 8. Residual equivalence after determinization

Different reachable subsets can recognize the same residual language. After
subset construction, note 129's DFA minimization can merge exactly those
language-equivalent subsets. Inclusion is only an order; equality of residual
languages is the equivalence needed for language-preserving minimization.

Therefore there are three distinct reductions:

1. exact visited merges identical subsets;
2. qualified dominance may discard an ordered subset for one query;
3. DFA minimization merges different subsets with identical residual language.

They have different evidence and output contracts.

## 9. Epsilon closure and measured depth

For an epsilon-NFA, the initial deterministic state is `eclose(I)`, and after
symbol `a` the next state is

```text
eclose(post_a(P)).
```

Epsilon reachability consumes zero input symbols. Expanding epsilon arcs as
ordinary unit BFS edges measures NFA transition count rather than word length.
Epsilon cycles also require closure/fixed-point handling so that zero-cost
reachability terminates before the next symbol layer is finalized.

## 10. Accepted words versus accepting runs

Subset construction preserves the accepted language: a word is accepted iff
at least one NFA run accepts it. It does not preserve the number of accepting
runs. Multiple runs ending in one NFA state collapse under set semantics, and
multiple members can accept the same word.

Counting accepted words, counting accepting runs, sampling runs, and finding
one shortest accepted word are different outputs. Set membership is idempotent;
run multiplicity is not.

## 11. Product-state BFS

For a labeled base graph, the exact constrained vertex is `(v,P)`. A base edge

```text
v --a--> v'
```

induces

```text
(v,P) -> (v',post_a(P)).
```

Two records at one base vertex with different subsets can enable different
accepted suffixes. Two identical subsets at different base vertices remain
different because base successors and target status differ.

Subset inclusion dominance is valid only when all remaining semantic fields
match, including base vertex, goal convention, resource phase, and graph epoch.

## 12. Cayley and Schreier constraints

For Cayley search, exact product state is `(g,P)` under the declared generator
action. Determinization preserves the NFA's regular constraint language, so
subset BFS finds the shortest generator word that both reaches the target and
has an accepting NFA run.

It does not remove group-word collisions or Schreier stabilizer collisions.
Different accepted words can evaluate to one state, and the same group state
with different active subsets remains distinct when future acceptance differs.

Inclusion dominance requires the same concrete group/orbit state. Applying it
across merely canonical-looking states reintroduces note 17's quotient and
fixed-target problems.

## 13. Reverse and bidirectional subsets

Reversing an NFA swaps initial/final roles and reverses transitions. The reverse
configuration is again generally a subset, seeded from the relevant accepting
states. Forward subset equality is not automatically the meeting predicate.

A bidirectional meeting must prove that one forward prefix subset and one
reverse suffix subset admit a compatible middle automaton state and one common
accepted word. Base-state equality alone is insufficient.

## 14. GPU and multi-GPU boundary

Subset configurations may be represented compactly, but representation does
not change their semantic identity. Report separately:

- NFA states, transitions, epsilon closure, and alphabet;
- reachable subset count and subset cardinality distribution;
- subset frontier width and exact-equality duplicates;
- inclusion comparisons and every applied dominance premise;
- accepted words, NFA runs, product states, and base states;
- subset ownership and stable cross-device keys;
- remote transition data used to form complete `post_a(P)`;
- determinization/minimization preprocessing, traversal, and replay time.

If NFA transitions are sharded, a device's partial union is not the complete
next subset. Finalizing visited or dominance before all authoritative member
successors are included can silently drop active states and accepted suffixes.

## Sources

- M. O. Rabin and D. Scott,
  [*Finite Automata and Their Decision Problems*](https://doi.org/10.1147/rd.32.0114),
  IBM Journal of Research and Development 3(2), 1959, for nondeterministic
  automata and powerset determinization.
- M. De Wulf, L. Doyen, T. A. Henzinger, and J.-F. Raskin,
  [*Antichains: A New Algorithm for Checking Universality of Finite Automata*](https://doi.org/10.1007/11817963_5),
  CAV 2006, for implicit determinization and antichain order reasoning, with a
  different universality contract from shortest existential BFS.
- Notes 12, 19, 20, 28, 37, 42, 52, 57, 64, 99, 128, and 129 supply this
  repository's zero-cost, shortlex, product, fingerprint, contract, bounded,
  visited, output, multiplicity, Cayley-language, bisimulation, and residual
  equivalence boundaries.

## Takeaway

The deterministic NFA search state after one word is the full set of possible
run states. BFS frontier is a set of such subsets; unioning configurations from
different prefixes invents behavior. Inclusion gives monotone suffix-language
containment, but pruning needs the same base state, a no-later dominating
depth, existential acceptance, and a compatible output contract. Exact subset
identity, inclusion dominance, and residual-language minimization are three
different operations.
