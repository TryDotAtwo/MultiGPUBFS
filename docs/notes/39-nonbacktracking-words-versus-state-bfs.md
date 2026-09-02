# Non-backtracking words versus state BFS

Searches on Cayley and puzzle graphs are often described through move words.
This creates a tempting but false identification:

```text
do not immediately undo the last move
    ==
do not revisit an already reached state.
```

The left side is a local property of a word. The right side is a global
property of the projected graph search.

## Word tree and state graph are different objects

Let `S` be the move alphabet and let `S*` be the rooted tree of finite words.
An action maps a word to a state:

```text
pi: S* -> X
pi(s_1 ... s_d) = x_0 acted on by s_1 ... s_d.
```

Every word-tree node has one parent and depth equal to word length. The map
`pi` is generally many-to-one:

- inverse cancellation gives `pi(w s s^-1) = pi(w)`;
- generator relations make distinct reduced words equal;
- a group action can have stabilizers, so a nonidentity group element may fix
  the current puzzle state;
- duplicate move labels or state symmetries can add more convergence.

Ordinary state BFS works on the image graph `X` and retains each state at its
first distance. A word search works on path descriptions. It becomes state BFS
only after exact equality and old-ball semantics are imposed on the image.

## What immediate-inverse pruning proves

Assume a symmetric generator alphabet with a known involution
`s -> inverse(s)`. A non-backtracking word obeys

```text
s_(i+1) != inverse(s_i).
```

This removes length-two spurs. In an undirected Cayley graph, a shortest path
cannot contain such a spur: deleting `s inverse(s)` produces the same endpoint
two steps sooner. Therefore immediate-inverse pruning preserves the existence
of a shortest path and the word metric under these assumptions.

That is the full general guarantee. It does not imply:

- one word per state;
- a simple path;
- a first-discovery frontier;
- absence of longer cycles;
- complete BFS visited semantics.

For an involutory generator, `s = inverse(s)`, repeating the same move is the
immediate reversal. A rule that merely forbids a differently numbered inverse
would miss this case. For a positive-only directed generator set, the inverse
may not be an allowed edge at all, so the undirected argument cannot be copied
without changing the graph contract.

## Minimal finite counterexample

Consider the cyclic group

```text
C_4 = <a | a^4 = e>
S = {a, a^-1}.
```

At word depth two, both

```text
a a              -> a^2
a^-1 a^-1        -> a^2
```

are non-backtracking, but denote the same state. At depth four, `a a a a`
returns to the identity without containing an adjacent inverse pair.

So non-backtracking generation has already duplicated a state at depth two and
revisited the root at depth four. Exact BFS instead has state layers

```text
F_0 = {e}
F_1 = {a, a^-1}
F_2 = {a^2}
F_3 = empty.
```

The word-tree layer sizes after immediate-inverse pruning are `1, 2, 2, 2, ...`;
the BFS sphere sizes are `1, 2, 1, 0, ...`. Local cancellation and global
first-discovery semantics visibly diverge.

## Free groups are the special tree case

For a free group with a free symmetric basis, every group element has one
freely reduced word and its Cayley graph is a tree. There, removing immediate
inverse pairs does give one word-tree node per state, and the reduced-word
levels equal BFS spheres.

This is a special consequence of having no nontrivial reduced relations. Once
relations are imposed, distinct reduced words can represent the same element.
For Schreier/puzzle graphs the condition is stricter still: even if two words
are different group elements, their quotient action may reach the same state.

The useful diagnostic question is:

> Is the reduced-word-to-state map injective through the depths being claimed?

Girth can provide a limited local answer. Before a first reduced cycle becomes
possible, a neighborhood can look tree-like. Beyond that boundary, local move
history cannot substitute for state identity.

## Product-state interpretation

Non-backtracking walk generation can itself be expressed as an ordinary graph
search on an expanded state:

```text
(current vertex, previous directed edge)
```

The transition forbids the inverse of the stored edge. This is exact BFS for a
different product graph when equipped with its own visited set. Two records at
the same base vertex but with different previous edges are different product
states because their legal next moves differ.

This explains why deduplicating only by base-state identity is not generally
valid for a true non-backtracking output contract. Conversely, if the history
rule is merely a sound path-normalization shortcut for finding ordinary state
distances, the implementation must prove that at least one shortest ordinary
path survives; it need not claim to enumerate non-backtracking product states.

## What the inspected CayleyPy snapshot does

In the `D:\100XH100` working-tree snapshot described in note 38, the main GPU
generation path uses

```text
candidate_count = parent_count * MOVE_COUNT
parent_local = candidate / MOVE_COUNT
move = candidate % MOVE_COUNT.
```

No last-move or inverse-move exclusion appeared in the inspected Stream 1,
Stream 2, or dispatcher generation path. Every configured move is considered
from every retained parent.

Three nearby mechanisms must not be confused with search-path backtracking
pruning:

1. `apply_inverse_move_flat_host` is used to build the goal-centered
   predecessor neighborhood. It applies inverse transformations; it does not
   forbid moves in the outer search.
2. same-depth `Hash128` deduplication merges generated candidate endpoints; it
   is neither immediate-inverse pruning nor an accumulated visited set;
3. `CpuCandidateHistory::prune_adjacent_depths` removes previous-depth history
   records that no current retained candidate references, then remaps parent
   indices. It compresses reconstruction storage after beam selection and does
   not alter which next moves are generated.

Thus the observed outer search is not a non-backtracking word search. It is a
full-move expansion of a pruned state beam, with within-depth hash merging and
ancestry storage compression.

## Work counts must name their universe

For degree `q`, three superficially similar counts can differ sharply:

```text
all words of length d:                 q^d
non-backtracking words (symmetric):    q (q-1)^(d-1)
new BFS states at distance d:          |F_d|.
```

The second formula assumes every letter has exactly one allowed inverse and no
other history-dependent legality. It counts words, not unique endpoints. The
third depends on all group/action relations and prior discoveries.

Consequently, eliminating inverse moves can reduce generated work without
changing the mathematical BFS frontier, but the saved-work factor is not a
frontier-growth formula. Candidate rate, reduced-word rate, and unique-state
rate must be reported separately.

## Sources

- Clara Löh, [Geometric Group Theory lecture notes](https://loeh.app.uni-regensburg.de/teaching/ggt_ss22/lecture_notes.pdf),
  develops reduced words and proves that Cayley graphs of free groups in a free
  basis are trees.
- Nicholas Touikan, [An introduction to combinatorial and geometric group
  theory](https://ntouikan.ext.unb.ca/MATH6022/IntroCGGT/IntroCGGT.pdf), makes
  explicit that distinct reduced words can become equal after group relations
  are imposed.
- Joel Friedman and David Kohler, [The Non-Backtracking Spectrum of the
  Universal Cover of a Graph](https://arxiv.org/abs/0712.0192), uses the standard
  directed-edge definition: the next edge may not be the inverse of the
  preceding edge.

## Current conclusions

1. Immediate-inverse pruning is a local word normalization, not visited-state
   semantics.
2. It preserves shortest paths only under a declared inverse/action contract.
3. It coincides with unique-state BFS levels in the free-tree case, not in a
   general Cayley or Schreier graph.
4. History storage pruning is yet another operation and must not be inferred to
   prune search paths.
5. Any performance claim must distinguish generated words, reduced words,
   unique current candidates, old-state hits, and accepted BFS states.
