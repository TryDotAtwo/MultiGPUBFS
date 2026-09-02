# Current CayleyPy Megaminx vertex and equality contract

The words “CayleyPy Megaminx state” hide two independent questions:

1. Is the mathematical state representation a free group action or a quotient?
2. Does the implementation compare those mathematical states exactly?

REF-025 answers them differently for the current local snapshot: the 120-value
representation is injective on generated permutations, while the hot dedup keys
are probabilistic hashes.

## Why this current state is genuinely Cayley

Let the central vector be

```text
c = [0,1,...,119]
```

and let a generated permutation `g` act by position gather. If `c*g=c`, every
destination receives its original unique value, so `g` fixes every position and
is the identity permutation. The stabilizer of `c` is trivial.

Thus the orbit map from the generated permutation subgroup to stored vectors is
injective. The reachable state graph is the Cayley graph of that generated
subgroup under the declared 24-move alphabet. “Subgroup” matters: distinct
stickers prove freeness but not that the moves generate all `120!`
permutations.

Had the central vector stored repeated face colors instead of unique sticker
identities, permutations exchanging equal-colored positions could stabilize it,
and the same move tables would instead induce a Schreier quotient. Representation
alone changes the graph without changing generator arrays.

## Logical equality versus key equality

Mathematically, two states here are equal exactly when all 120 logical entries
match. There are three common implementation contracts:

```text
full equality                     deterministic state identity
hash bucket + full equality       deterministic identity if handled correctly
hash equality only                collision-assuming identity
```

A hash table does not inherently weaken equality: Rust's REF-025 `HashSet`
uses hashes to locate a bucket and then compares full vectors. The weakening
occurs when equal hash values themselves are treated as proof of equal state.

The inspected native CayleyPy BFS follows the third contract with a scalar
`int64` hash for this encoded state. The production beam pipeline follows it
with `Hash128` during candidate dedup. The latter has a much wider key, but key
width changes collision probability, not the logical implication:

```text
same state  => same deterministic hash
same hash   !=> same state.
```

## What a successful replay proves

Full replay of a returned move sequence proves that this sequence reaches the
claimed concrete target. It does not retroactively prove:

- that no unequal candidate collided with a retained hash;
- that no colliding target candidate was discarded;
- completeness of a BFS layer;
- optimality of a beam-search result.

Witness validity and search completeness are different output contracts.

## What REF-025 independently establishes

Using full-vector equality, the current config has:

```text
state length  = 120
move count    = 24
inverse pairs = 12
all move orders = 5
F_0..F_4 = 1, 24, 408, 6208, 90144.
```

The agreement with native CayleyPy's test is useful cross-evidence because the
Rust probe has an independent parser and equality mechanism. It still covers
only four expansions and the current file snapshot.

## Consequences for BFS reasoning

1. The current Megaminx frontier counts may be interpreted as Cayley spheres,
   not merely action-state spheres, within the verified generator contract.
2. A collision between distinct move words is group-element equality here,
   rather than only membership in a nontrivial state stabilizer.
3. Exact mathematical BFS and hash-only implementation BFS must not share one
   undifferentiated “exact” label.
4. GPU or multi-GPU ownership by hash does not itself harm correctness; using
   hash equality as final identity is the relevant proof boundary.
5. Beam search remains a pruned search even if every retained-state comparison
   were exact.

## Practical evidence ladder

For later Cube/Megaminx audits:

1. Verify that the central state is injectively labeled or characterize its
   stabilizer.
2. Verify every generator is a permutation and every claimed inverse composes
   to identity under the actual gather/scatter convention.
3. Compare a few exact full-state frontier layers with an independent oracle.
4. Trace whether visited/dedup uses full equality, hash-plus-equality, or hash
   equality only.
5. Keep replay validity, layer completeness, shortest distance, and heuristic
   beam success as separate claims.

## Evidence boundary

The source snapshot, exact paths, commits, failures, and Docker result are
recorded in REF-025. This note makes no claim that a hash collision occurred,
nor that the current design should be changed. It records which assumptions a
correctness proof would need.
