# Reverse BFS goal neighborhoods and suffix certificates

A goal-centered table is one of the cleanest places to see BFS as a proof
mechanism rather than a queue pattern. This note derives its contract and then
applies it to the K1/K2 suffix machinery in the inspected CayleyPy snapshot.
It proposes no implementation changes.

## Why reverse expansion is required

Let the forward graph contain a labeled edge

```text
x --s--> T_s(x).
```

To build the set of states that can reach target `t` in at most `R` forward
moves, a search starting at `t` must traverse predecessor edges:

```text
Pred_s(y) = {x | T_s(x) = y}.
```

For a bijective puzzle move, this predecessor is uniquely
`T_s^-1(y)`. Importantly, `T_s^-1` need not itself be an allowed forward move.
It is an operation used to traverse the reversed graph while the stored witness
still says “from the predecessor, apply forward label `s`.”

If the forward graph is directed, ordinary BFS from `t` over forward edges
answers which states `t` can reach. Reverse BFS answers which states can reach
`t`. These sets coincide only under additional symmetry.

## Reverse-BFS invariant and suffix witness

Let `R_d` be the reverse frontier and `Q_d` the reached reverse ball:

```text
Q_0 = {t}
R_0 = {t}
R_(d+1) = unique(Pred(R_d)) minus Q_d.
```

For every first-discovered predecessor `x` of a state `y`, store

```text
suffix(x) = [s] concatenated with suffix(y),
```

where `T_s(x)=y`. Inductively, replaying `suffix(x)` from `x` reaches `t`, and
its length is `d+1`. With complete predecessor generation, exact identity, and
level-order first discovery, this length equals the forward distance
`dist(x,t)`.

The proof has two halves:

- the stored suffix is an upper-bound witness of length `d+1`;
- if a shorter forward path existed, reversing it would have discovered `x` in
  an earlier reverse layer, contradicting first discovery.

This is a genuine local shortest-path certificate, not merely a convenient
lookup.

## What CayleyPy K1 does

The inspected `build_solved_neighborhood_host` follows this structure:

1. insert the central target with an empty suffix;
2. keep explicit `frontier` and `next` vectors by depth;
3. for every node and every configured move, compute a predecessor with
   `apply_inverse_move_flat_host`;
4. skip a predecessor whose `Hash128` is already in `suffix_by_hash`;
5. store `move` in front of the current node's suffix;
6. continue through the configured radius, at most 12.

The action convention is internally coherent for a permutation generator:

```text
child[p] = parent[generator[p]]
```

and the inverse helper reconstructs it using

```text
parent[generator[p]] = child[p].
```

Therefore the packed move saved at reverse expansion is the forward move that
replays from the predecessor to the current node. `prepend_suffix_move` and
`append_packed_suffix` preserve that order.

When `BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES` would be exceeded, construction
throws before silently truncating the table. The device bucket builder likewise
grows its table until placement succeeds, or throws if a caller mandated a
fixed shape that does not fit. These are fail-closed capacity semantics for the
table build.

## Exact algorithm, probabilistic identity

The control structure is reverse BFS, but its visited identity is
`Hash128`, produced by deterministic Zobrist XOR. The host map does not retain a
collision bucket of full `State128` values. Consequently the mathematical
guarantee is conditional:

```text
if Hash128 is injective over every generated K1 state,
then the stored first suffix has exact minimum forward length;
exact lookup also requires no queried state to alias a different stored state.
```

Without that premise, a collision can suppress a distinct predecessor or
associate a suffix with the wrong state. The GPU table screens with a 32-bit
fingerprint and confirms the full 128-bit hash, which protects against
fingerprint-only false positives but not against two semantic states sharing
the same `Hash128`.

Final CPU replay still has important value: an accepted full solution must
actually transform the initial state into the target. It turns a false lookup
hit into rejection rather than a silently valid artifact. It cannot recover a
real path whose K1 state was suppressed by a collision.

## Unchecked predecessor precondition

`apply_inverse_move_flat_host` is a true inverse only when every generator row
is a permutation of the state positions. The inspected JSON loader reads and
casts generator entries and fills padding positions, but the nearby code does
not visibly prove:

- every logical index is in range;
- no logical source index is repeated;
- every logical source position occurs exactly once.

This does not show that current generator files are malformed. It identifies a
precondition on which the reverse-BFS proof depends and whose validation was not
established in this read-only pass.

## K2 is a word table, not a state BFS table

The optional Stream 2 suffix list has different semantics. It begins with the
empty word and then appends every one of `MOVE_COUNT` moves to every word at the
current length, through radius at most 3. It performs no endpoint
deduplication:

```text
number of stored words = 1 + q + q^2 + ... + q^K2.
```

Thus K2 enumerates a bounded move-word tree. Relations, cancellations, and
stabilizers may make many entries reach the same state. Enumeration order is
nondecreasing word length, so the first exact-target hit has a minimum K2 word
length among the enumerated words.

When K1 is enabled, a K2 word succeeds by landing anywhere in the K1
ball. The code returns the first successful K2 suffix and later appends that
K1 state's stored suffix. The objective actually reported is

```text
K2_word_length + K1_distance(landing_state, target).
```

For an **exact complete reverse ball** of radius `R`, with exact shortest
suffixes, this first-hit rule does minimize the combined residual. The required
K2 contract is enumeration of all allowed words of lengths `0..K`, including
the empty word, in nondecreasing length. Moves have unit cost and both tables
refer to the same forward graph and target.

Let `D=dist(x,t)` for the fixed candidate. If `D<=R`, the empty word hits and
returns `D`. If `D>R`, every hit with prefix length `k` and suffix length
`r<=R` satisfies

```text
D <= k+r <= k+R, hence k >= D-R.
```

A shortest path has a prefix of length `D-R` whose endpoint is at distance
`R`, so this lower bound is attained whenever `D<=K+R`. The first hit therefore
has `k=D-R` and `r=R`, and returns exactly `D`. A hit exists if and only if
`D<=K+R`; a complete miss proves `D>K+R` (including unreachable states).

The earlier purported example of a first `1+4` residual followed by a
length-two path to the target is impossible under these premises: that
candidate already belongs to the radius-four K1 ball, so the empty word hits
first and returns distance at most two.

This is a conditional algorithmic theorem, not a certificate that the inspected
hash-only implementation meets all its premises. With an incomplete table,
an omitted or out-of-order word, an inexact suffix, or a false hash hit, first
success need not have this guarantee. A genuine replay remains an upper-bound
witness. Even when all local premises hold, outer beam pruning is a separate
obstacle to global shortestness from the original source.

## Selection scopes

There are three nested competitions:

1. **Inside K1:** first reverse-BFS discovery chooses a minimum-length K1
   suffix, conditional on exact identity and valid inverse moves.
2. **Inside one candidate's K2 scan:** first successful K2 word is retained;
   it has minimum combined residual under the exact-complete-ball and ordered
   exhaustive-prefix premises above, but does not retain all shortest suffixes.
3. **Across recorded hits in one outer depth:** `select_best_solved_snapshot`
   compares computed total lengths and deterministic tie fields.

Stage 3 can choose the best record it receives. Under the theorem's premises,
stage 2 does not discard a strictly shorter residual for the same candidate;
it can discard equal-length witnesses. Solved-result capacity can omit other
candidates' hits, and earlier beam pruning can omit a globally shortest branch.
Neither current-depth selection nor the local theorem repairs those omissions
or proves the inspected hash-only tables exact.

Keeping these scopes explicit prevents a local “best” from silently becoming a
global shortest-path claim.

## Reverse BFS in explicit, implicit, and distributed form

The same proof obligations appear in every representation:

| Representation | Required predecessor mechanism |
|---|---|
| explicit CSR | incoming adjacency or a transposed graph |
| implicit bijection | exact inverse transition for every forward label |
| nonbijective implicit graph | complete enumeration of all preimages |
| multi-GPU | route equal predecessor states to one exact visited authority |

For nonbijective transitions, writing one procedural “inverse” is insufficient:
a target state can have zero, one, or many predecessors for the same label.
For distributed construction, local first discovery is insufficient unless
all equal states meet at a common owner or an equivalent exact protocol.

## A compact audit checklist for goal tables

```text
forward graph and action convention:
predecessor completeness proof:
generator bijection/range validation:
reverse level invariant:
semantic identity and collision handling:
stored suffix order and replay test:
radius and boundary inclusion:
capacity/overflow behavior:
word enumeration versus state BFS:
complete reverse-ball radius and shortest suffixes:
empty-word check and exhaustive nondecreasing prefix lengths:
one shortest residual versus all/canonical shortest witnesses:
device lookup confirmation strength:
outer-search guarantee kept separate:
```

## Current conclusions

1. Goal-centered BFS must traverse predecessors, not merely run ordinary
   forward BFS from the goal.
2. CayleyPy K1 has the control flow of bounded reverse BFS and stores correctly
   oriented forward suffixes under permutation-generator assumptions.
3. Its shortest-suffix guarantee is conditional on collision-free `Hash128`
   identity over the built neighborhood.
4. K2 is bounded word enumeration, not BFS over unique states.
5. First K2 hit into an exact complete K1 ball returns the exact residual
   distance when all shorter allowed words, including the empty word, have
   been checked and K1 suffixes are shortest; a hit exists iff `D<=K2+K1`.
6. Replay alone certifies an upper-bound path. The local optimality theorem
   requires its stated premises and does not establish outer-beam optimality.
