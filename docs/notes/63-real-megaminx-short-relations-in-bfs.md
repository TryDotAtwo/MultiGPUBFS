# Real Megaminx short relations in BFS

REF-026 transfers the relation-signature vocabulary from toy groups to the
current independently loaded Megaminx move table. It shows that two mechanisms
can coexist at the same BFS radius and remain exactly distinguishable.

## Three counts at depth two

With 24 inverse-closed moves:

```text
all length-two words             24^2 = 576
non-backtracking length-two words 24*23 = 552
unique depth-two states                    408.
```

The first subtraction removes immediate inverse spurs. The second quotient is
group equality. Conflating them would call 168 occurrences “duplicates” and
erase the split between 24 trivial cancellations and 144 commutation
convergences.

## Commutation produces the whole first convergence layer

Every depth-two endpoint has multiplicity one or two. All 144 multiplicity-two
endpoints have witnesses `ab` and `ba`; there are no other collisions.

After inverse orientations are grouped by face, these are 36 commuting face
pairs with all four sign choices. Each commuting equality gives the relator

```text
a b a^-1 b^-1 = e
```

of length four. The relation becomes visible halfway through: two geodesic
words converge while `F2` is constructed.

This also explains the 144 extra backward occurrences when `F2` is later
expanded. One predecessor can be chosen as the BFS-tree parent; the other is an
equally short alternate predecessor. Forward convergence and later backward
multiplicity are two temporal views of the same diamond.

## Odd power relations use another counter

Each signed face move has order five. At radius two, `g^2` and `g^-2` are
distinct states with unique paths along their respective directions. The last
edge of the 5-cycle joins them inside `F2`:

```text
e --g--> g --g--> g^2 --g--> g^3=g^-2 --g--> g^-1 --g--> e.
```

Across all orientations, REF-026 sees 24 directed same-level occurrences and
proves that every one is of this form. A candidate-only collision counter would
detect the commutators but miss these order-five boundaries.

## Girth and generator order answer different questions

Face turns have order five, but commuting pairs supply 4-cycles. Therefore the
current Cayley graph has girth four, not five.

Generator order describes cycles contained in one cyclic subgroup. Girth asks
for the shortest simple cycle over the whole alphabet. Adding more generators
can introduce shorter mixed-generator relations without changing individual
orders.

## Why deeper duplicate counts are not relation counts

While forming `F3`, REF-026 sees 3,008 extra candidate occurrences. These are
word-pair witnesses, not 3,008 independent relators.

Known relations translate throughout a Cayley graph. Their translated diamonds
can overlap, and compositions of known relations create further equal words.
A presentation seeks generators for all relations; a BFS trace enumerates
local consequences. Moving from the latter to the former requires witness-word
classification and algebraic reduction, not only counting.

## Undirected layer geometry removes one category

For any undirected edge `(u,v)`, BFS distances satisfy

```text
|dist(s,u)-dist(s,v)| <= 1.
```

Hence a transition from `F_d` can reach only `F_(d-1)`, `F_d`, or `F_(d+1)`.
The “older-ball edge” category from a general directed audit is identically
empty here. Its zero count is an invariant check, not an empirical curiosity.

## Hardware boundary

The 144 semantic commutation collisions do not prescribe a GPU primitive.
Their removable location depends on parent order, generator order, batching,
hash partition, and owner routing. REF-016/017 already show this ordering
dependence on symmetric-group layers; REF-026 supplies a real Megaminx relation
mechanism but no locality measurement.

## Current synthesis

At the first non-tree-like Megaminx radius:

- inverse cancellation explains 24 occurrences;
- 36 commuting face pairs explain all 144 candidate convergences;
- the same diamonds later explain 144 alternate shortest predecessors;
- order-five face cycles explain all 24 same-level occurrences;
- 3,008 next-layer convergences remain consequences to classify, not primitive
  relations already understood.

This is the first current-config bridge from abstract relation theory to exact
Megaminx BFS counters. It remains a semantic result, not an optimization plan.
