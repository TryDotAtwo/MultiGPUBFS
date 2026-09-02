# BFS conservation checks, fingerprints, and the verification ladder

Conservation identities are powerful runtime tripwires: if an exact identity
fails, at least one declared accounting assumption is false. The converse is
not valid. A matching scalar total says that the implementation preserved that
total, not that it constructed the correct frontier.

This note separates falsification checks from evidence that can establish exact
BFS semantics. It introduces no implementation, benchmark, or optimizer.

## 1. Necessary is not sufficient

Suppose the expected frontier is `{0,3}`, while an implementation returns
`{1,2}`. Both sets have

```text
cardinality = 2
sum         = 3
xor         = 3
```

Thus even count, sum, and xor together do not prove frontier-set equality. A
lost state can be balanced by a wrong or duplicated state while aggregate
checks and work-waterfall totals still match.

The logical direction is:

```text
failed exact conservation check  => a bug or violated assumption exists
passed exact conservation check  => this check did not expose a bug
```

It is not:

```text
passed conservation check => exact BFS is correct.
```

## 2. What each evidence type can and cannot establish

### Conservation counters

Counts of generated occurrences, loops, aliases, visited hits, convergence,
and accepted states detect loss, duplication, overflow, or category mistakes
when those failures disturb the declared identity. Balanced errors may remain
invisible. Global reductions can also hide a loss on one owner balanced by an
extra record on another; per-owner and per-level matrices localize failures but
still do not prove semantic correctness.

Retries need stable logical message or occurrence identities. Otherwise a
transport retry can be mistaken for a second graph occurrence and make a
physically consistent counter describe the wrong semantic object.

### Fingerprints

Commutative hashes, sums, xor values, or several independently seeded
fingerprints are compact regression evidence. Finite fingerprints collide, so
equality is probabilistic evidence unless the encoding is proved injective on
the entire stated domain. Cryptographic strength can make accidental collision
implausible; it does not turn hash equality into exact set equality.

### Parent and path replay

A replay-valid parent chain proves that a reported state is reachable and gives
an upper bound on its distance. With separately established BFS layer
invariants it can certify the reported distance. It does not prove that every
successor was generated, every frontier state was retained, or no shorter
unreported path exists.

### Exact set equality

At tractable scale, compare canonical sorted frontier states or an injective
domain bitmap, including full collision resolution. Equality of hash tables or
hash values alone is not exact set equality. The reference should use an
independent implementation or specification; two traversals sharing the same
move table, encoder, or omission can agree through a common-mode bug.

### Exhaustive independent oracle

For a tiny exhaustible domain, enumerate every state and every declared
successor with an independently represented rule, then compare successor sets,
frontiers, distances, and component size. This is the strongest finite-domain
evidence in the ladder, but its scope must remain explicit when transferring a
claim to a larger domain.

## 3. A verification ladder

From cheapest tripwire to strongest bounded evidence:

1. Local scalar counts and conservation identities.
2. Per-level, per-owner, and per-category accounting plus overflow checks.
3. Deterministic sums, xor values, and commutative fingerprints.
4. Replay-valid parents/paths and local edge-distance inequalities.
5. Exact canonical frontier-set equality against an independent traversal at
   tractable scale.
6. Exhaustive tiny-domain comparison against an independently specified
   successor oracle, including forced omission and mutation cases.

The levels are complementary. Higher evidence does not make lower runtime
tripwires useless: cheap counters are valuable for every large run, while exact
oracles are usually affordable only on bounded fixtures.

## 4. One GPU and many GPUs

The semantic ladder is unchanged by the device count. One-, two-, and
four-worker parity is useful evidence for partition/routing mistakes, but all
configurations can preserve the same common-mode successor or identity bug.

For distributed BFS, verify both graph semantics and transport conservation:

- every logical `(parent,label)` obligation is generated exactly once;
- routing and retries preserve stable identities;
- authoritative owners resolve full state identity;
- no queue, buffer, kernel, message, or publication remains outside the
  completed cut;
- per-owner exact sets agree with an independent bounded reference after union.

Matching global cardinalities or fingerprints alone cannot identify which
owner lost which state, nor prove that the union is correct.

## 5. Consequences for future experiments

- Treat conservation mismatches as decisive failures.
- Describe matching counters and fingerprints as regression evidence, not
  correctness proofs.
- Keep the canonical state representation and collision-resolution rule in the
  artifact contract.
- Use an independently represented successor specification somewhere in the
  validation chain for implicit and Cayley graphs.
- Include adversarial fixtures where one state is omitted and another inserted,
  and where a logical message is retried.
- Report the largest exhaustively validated domain separately from the largest
  performance run.

## 6. Current conclusion

Accounting answers whether declared quantities balance. Exact comparison
answers whether the same states were produced. Independent successor validation
answers whether the intended graph was traversed. These are three distinct
claims; a reliable BFS study needs all three at appropriate scales.

