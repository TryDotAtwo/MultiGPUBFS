# Validating implicit successor completeness

For an explicit graph, the adjacency structure can be inspected independently
of BFS. For an implicit graph, `successors(state)` is the adjacency structure.
If it omits one legal transition, exact BFS can return an overestimated distance
or false exhaustion while every emitted edge and returned path remains valid.

This note develops a validation ladder for implicit successor oracles. It
distinguishes proof, exhaustive finite validation, metamorphic evidence, and
runtime coverage accounting. It adds no implementation.

## Three separate oracle properties

Let the declared graph specification define the legal labeled transitions

```text
E_spec(x) = multiset/set of (label,y) allowed from x.
```

Let the implementation emit `E_impl(x)`. Correct expansion requires more than
one Boolean notion of correctness.

### Edge soundness

Every emitted transition is legal:

```text
E_impl(x) subseteq E_spec(x).
```

An extra illegal edge can create a path that does not exist and make a distance
too small.

### Endpoint/label correctness

For each emitted label `m`, the endpoint is exactly the specified result:

```text
emit (m,y)  implies  y = apply_spec(x,m).
```

A wrong endpoint may still happen to be some legal neighboring state under a
different move, so unlabeled adjacency or path-length checks can miss a label
semantics error.

### Successor completeness

Every required legal transition is emitted with the requested multiplicity:

```text
E_spec(x) subseteq E_impl(x).
```

Together with soundness this is equality under the declared simple/labeled/
multigraph convention.

These properties can fail independently. Validation must name which one it
supports.

## Why replay is one-sided evidence

Consider the specified graph

```text
s -> a -> t.
```

An implementation that emits no edge from `s` is perfectly sound in the weak
sense that every emitted edge is legal—there are none. BFS returns an empty next
frontier and may report `t` unreachable. There is no returned path to replay,
so replay cannot expose the missing edge.

Conversely, if an implementation adds illegal edge `s->t`, replay against the
same flawed successor code can accept a false length-one path. Independent move
application is needed even for positive witnesses.

Thus:

```text
path replay                 -> positive witness soundness for selected edges
complete successor coverage -> no legal branch was omitted.
```

The second is a universal statement over all expanded states and legal moves.

## The oracle problem

If the only definition of legal successors is the production function itself,
testing it against its own output is circular. A meaningful oracle must come
from at least one independent source:

- a formal move specification;
- a second interpreter using independently represented rules;
- a trusted enumerated adjacency table for a bounded instance;
- algebraic/domain laws that constrain results;
- manually audited fixtures with known endpoints;
- a proof that code generation preserves the move definition.

No finite sample can prove an arbitrary black-box program correct on an
unbounded state domain. Evidence must retain its scope: exhaustive for a named
finite domain, property-tested for generated cases, or proved under a formal
model.

## Total finite generator collections are unusually auditable

For a Cayley graph with ordered total generator collection

```text
S=(s_0,...,s_(q-1)),
```

the required occurrence set from every state is structurally known:

```text
one occurrence (i, x*s_i) for every i in 0..q-1.
```

This reduces completeness to two questions:

1. was every `(parent,i)` work item processed exactly once or at least once
   under the multiplicity contract?
2. did each work item compute the correct `x*s_i`?

The expected raw count for frontier `F` is

```text
q * |F|.
```

But aggregate count equality is insufficient. One missing `(parent,i)` and one
duplicated `(parent,j)` preserve the total. Stronger coverage identifies the
work coordinates themselves.

Possible evidence, from weaker to stronger, includes:

- total generated count;
- per-generator and per-parent counts;
- exact bitmap/set of processed `(parent_index,generator_index)` coordinates;
- deterministic index mapping whose range is proved to cover
  `0..|F|*q-1` exactly;
- exhaustive endpoint comparison with an independent interpreter.

Checksums/fingerprints of work coordinates are compact regression evidence but
remain probabilistic unless injectivity is proved.

## Partial moves need explicit outcome coverage

Some implicit graphs have move labels that may be illegal depending on state.
Then "one emitted successor per label" is wrong. A complete enumeration still
needs one resolved outcome per attempted label:

```text
(parent,label) -> VALID(endpoint)
               | INVALID(reason/spec predicate).
```

Dropping invalid moves silently prevents auditing whether the label was tested
or forgotten. Useful counters include:

```text
labels expected
labels evaluated
valid endpoints emitted
invalid outcomes by reason
exceptions/unknown outcomes.
```

An exception, unsupported rule, or timed-out legality check is `UNKNOWN`, not
`INVALID`. Treating operational failure as illegality removes graph edges.

State-dependent successor lists need an independently specified rule for which
labels are applicable. Merely agreeing with the production list length is not
independent evidence.

## Validating permutation generators

If a move is represented by a permutation row over `n` positions, structural
validation should establish:

```text
row length = n
every entry lies in 0..n-1
every destination/source index occurs exactly once
inverse row composes to identity
CPU and GPU conventions apply the same direction/order.
```

These checks prove the row is a bijection and that the declared inverse matches
it. They do **not** prove that the row corresponds to the intended physical
puzzle move or label. That requires an independent move definition or audited
fixtures.

Likewise, round-trip

```text
apply_inverse(apply_move(x,m),m)=x
```

proves mutual invertibility on tested states. Two consistently wrong inverse
tables can still pass if they implement some unintended permutation pair.

## Metamorphic relations: strong checks, incomplete specifications

When individual expected endpoints are hard to enumerate, group/domain laws
provide metamorphic tests:

- move followed by inverse returns the original state;
- an involution squares to identity;
- known commuting moves satisfy `ab=ba`;
- braid/presentation relations produce equal endpoints;
- declared invariants such as parity, piece multiset, orientation sum, or
  conservation law are preserved;
- left/right action normalization agrees on independently transformed queries.

These checks can expose composition order, row direction, stale tables, and
state-field omissions. They do not prove completeness of the generator manifest:
an implementation can omit one whole legal generator while satisfying every
relation tested among the remaining generators.

The manifest of allowed moves is itself part of the graph specification and
needs versioned validation.

## Differential tests and independence

Comparing GPU and CPU endpoints is valuable only to the degree their failure
modes are independent.

Weak differential test:

```text
CPU and GPU both read the same possibly wrong generated permutation rows.
```

It validates execution parity for those rows, not row semantics.

Stronger differential test:

```text
production GPU applies packed/generated rows
independent CPU interpreter applies declarative piece cycles/orientation rules.
```

Shared fixtures, tables, canonicalization, hash code, or composition helpers
should be listed because they reduce independence. Agreement is still useful,
but its claim must be narrower.

## Exhaustive small domains

When a parameterized family has small enumerable instances, compare for every
valid state:

```text
exact labeled successor multiset
inverse predecessor multiset
per-level frontier sets
distance labels
component size and known invariants.
```

This validates the complete finite corpus, not larger parameter values. Its
strength depends on whether larger instances reuse the same local move logic or
introduce new packing widths, index arithmetic, parity cases, or capacities.

Boundary instances are especially valuable:

- smallest valid size;
- first size crossing a machine word/packing field;
- maximum configured index;
- states at legality/invariant boundaries;
- noncommuting words that expose composition direction.

## Mutation sensitivity

A validation suite should fail when representative defects are deliberately
introduced conceptually or in a controlled test fixture:

- omit one generator label;
- duplicate another label;
- swap two permutation indices;
- reverse composition order;
- classify one valid move as invalid;
- alias two states in identity;
- truncate the final work tile;
- drop one routed candidate.

If removing one generator does not fail any check, the suite has not established
generator-set completeness. Mutation sensitivity does not prove absence of all
bugs, but it tests whether the claimed oracle can detect the failure class it is
supposed to cover.

## GPU work-coordinate coverage

For a total `q`-move frontier of `N` parents, the logical work domain is

```text
W = {(p,m) | 0<=p<N, 0<=m<q}.
```

A GPU launch may map threads, warps, tiles, or persistent queue items to `W`.
Correctness needs every coordinate processed according to the declared
multiplicity, regardless of launch geometry.

Common coverage failures include:

- floor division drops the final partial tile;
- grid limit/32-bit multiplication truncates `N*q`;
- a bounds predicate excludes a valid final coordinate;
- queue overflow loses work after generation;
- early exit/cancellation skips unchecked moves;
- fused output capacity failure suppresses later coordinates;
- stale counters make the host believe a level completed.

Aggregate `generated=N*q` helps but can hide duplicate compensation. Strong
bounded validation can compare an exact coordinate bitmap or independently
reconstruct the expected endpoint multiset for tractable `N`.

Memory/race tools can detect out-of-bounds access, uninitialized data, and some
races. They do not prove that every semantic work coordinate was scheduled or
that the move table describes the intended graph.

## Fusion must preserve observation boundaries

A fused kernel may generate, goal-check, deduplicate, visited-filter, and compact
without materializing raw candidates. This can be exact, but validation must
still distinguish:

```text
move coordinate evaluated
endpoint computed
goal predicate evaluated
identity decision made
candidate accepted/rejected
output slot committed or overflowed.
```

If counters are recorded only after visited rejection, a missing move and a
correctly generated visited hit can look identical. Instrumentation may be
sampled or enabled only in validation builds, but the semantic stages need
independent evidence somewhere.

## Multi-GPU conservation

Distributed successor completeness adds two questions:

1. was every frontier parent assigned to an expander exactly once under the
   schedule?
2. did every produced candidate reach the correct authoritative owner or a
   non-lossy equivalent path?

Per-level conservation can be written as

```text
expected work coordinates
= evaluated valid + evaluated invalid + explicit failures

source-local candidate records
= local-owner records + remote records + explicit drops/failures

sent remote records
= received remote records + in-flight-at-snapshot

owner received records
= owner duplicates + old-visited hits + accepted states
   + explicit failures.
```

These identities catch loss only when categories are exact and the observation
boundary is globally consistent. Equal aggregate sent/received counts can hide
one missing record and one duplicate; per-peer sequence/range IDs or exact
candidate fixtures strengthen the evidence.

Global frontier-set parity across `P=1,2,...` is strong regression evidence for
the tested instances. It still shares the same successor implementation unless
one side uses an independent oracle.

## Distance certificates cannot reveal every missing edge

Given a returned label map, checking

```text
d(v) <= d(u)+1
```

for every **emitted** edge detects some underestimated/inconsistent labels. It
cannot check an edge absent from both the traversal and validator.

A valid parent chain proves that returned paths exist. It does not prove that a
missing edge would not create a shorter path or reach a new component region.

Therefore local distance certificates and successor completeness are
complementary:

```text
label/parent certificate -> result is coherent on checked edges
independent expansion evidence -> checked edges cover the declared graph.
```

Neither substitutes for the other.

## Applying the ladder to CayleyPy evidence

The read-only CayleyPy audits established useful but partial facts:

- production paths generate configured move occurrences from retained beam
  parents;
- K1 host construction uses inverse move application;
- selected CPU/GPU fixtures compare direct child hashes/endpoints and K2
  composition branches;
- final concrete replay validates positive returned solutions;
- capacity paths inspected generally throw rather than silently truncate.

The remaining expansion-evidence gaps include:

- no observed loader proof that every generator row is an in-range permutation
  before inverse construction;
- shared move/hash tables can make CPU/GPU parity nonindependent;
- current unit tests do not exercise the real host K1 builder end to end on a
  nonempty suffix corpus;
- no forced omission/mutation fixture demonstrates that generator completeness
  checks fail as intended;
- outer beam pruning intentionally prevents complete-frontier coverage, separate
  from whether each retained parent's move set is complete.

These statements qualify evidence; they do not assert that an unobserved bug
exists.

## A validation ladder

Increasingly strong evidence for a declared scope:

1. **Positive replay:** selected emitted transitions/paths are real.
2. **Structural state checks:** output fields, invariants, ranges, permutation
   bijectivity.
3. **Per-label fixtures:** audited endpoints for each move and inverse.
4. **Metamorphic laws:** inverse, relations, invariants, action convention.
5. **Independent differential interpreter:** separate rule representation.
6. **Exact work-coordinate coverage:** no missing/duplicated `(parent,label)` on
   bounded batches.
7. **Exhaustive small-domain successor-set equality:** every state and label.
8. **Frontier/distance/component parity:** independent full traversals on
   tractable domains.
9. **Mutation sensitivity:** each claimed failure class is detected.
10. **Formal/code-generation proof:** specification-to-implementation argument
    for the stated general domain.

Higher rungs do not make lower ones useless: replay catches artifact corruption
even after a proof, while exhaustive finite tests expose integration mistakes a
local algebraic proof may omit.

## Minimum artifact schema

```text
graph/move/identity version:
state domain and validity constraints:
generator manifest and multiplicity convention:
independent oracle description and shared dependencies:
states/frontiers covered:
expected and evaluated work coordinates:
valid/invalid/unknown outcome counts:
endpoint/move-label mismatches:
inverse/relation/invariant checks:
GPU launch and overflow/failure flags:
per-rank send/receive/owner conservation:
frontier/distance parity scope:
mutations detected/not detected:
claim status: proved | exhaustive finite | sampled | unknown.
```

## Counterexamples and rejected shortcuts

### Every returned path replays, so successor enumeration is complete

Replay cannot witness a legal edge that was never emitted.

### Generated count equals `N*q`, so every move ran

One duplicate coordinate can compensate for one missing coordinate.

### Inverse round trips prove the move table is intended

A consistently wrong permutation and its inverse can round-trip perfectly.

### CPU/GPU equality is an independent oracle

Not when both consume the same flawed tables or composition helpers; it then
proves execution parity for shared semantics.

### Sanitizer-clean means graph-complete

Memory/race safety does not prove semantic work coverage or move definitions.

### Correct distance labels prove no edge is missing

Checks over emitted edges cannot constrain an edge absent from both search and
validator.

### Multi-GPU sent/received totals prove losslessness

Aggregate equality can hide one missing and one duplicated record without
per-record or stronger scoped evidence.

## Sources

- Elaine Weyuker,
  [On Testing Non-Testable Programs](https://doi.org/10.1093/comjnl/25.4.465),
  provides the classical test-oracle problem framing.
- Koen Claessen and John Hughes,
  [QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs](https://doi.org/10.1145/351240.351266),
  develops property-based generated testing.
- Tsong Yueh Chen et al.,
  [Metamorphic Testing: A Review of Challenges and Opportunities](https://doi.org/10.1145/3143561),
  surveys follow-up relation testing when direct expected outputs are difficult.
- Notes 06, 28, 37, 41, 43, 48, 52 and REF-001/002/010 provide the implicit
  oracle, identity, contract, label-certificate, CayleyPy audit, closure,
  authority, and finite-parity context used here.

## Current conclusions

1. Implicit successor correctness has separate soundness, endpoint/label, and
   completeness obligations.
2. Positive path replay and distance certificates cannot reveal legal edges
   omitted by both traversal and validator.
3. Total finite generator sets make work coordinates enumerable, but aggregate
   counts alone cannot prove coordinate coverage.
4. Inverse, relation, and invariant tests are strong metamorphic evidence but do
   not prove the generator manifest complete or intended.
5. CPU/GPU differential evidence must disclose shared move tables and helpers;
   independence determines the claim strength.
6. GPU and multi-GPU exactness additionally need work-coordinate, capacity, and
   routing conservation with explicit failures.
7. Exhaustive small-domain equality and mutation sensitivity are high-value
   validation rungs, while general proof requires a specification-to-code link.
