# Generator-set changes alter visited work

Two BFS runs over the same semantic state universe need not present similar
work to `visited`. In a Cayley graph, the generator set defines the edges and
therefore the metric balls themselves.

REF-030 compares the ordinary 3x3x3 Cube QTM and HTM generator sets on one exact
54-sticker group action.

## Same vertices does not mean same traversal

QTM has 12 unit quarter-turn generators. HTM adds six half turns as unit
generators. Although both generate the same Cube group:

- their distances differ;
- their frontier sizes differ;
- their degrees differ;
- the position of old-ball and same-level hits differs;
- their diameters differ.

Hence "same puzzle" is not a complete BFS workload definition. The metric and
generator manifest are part of the graph identity.

### Four-state hand trace

Take the additive group `Z_4` rooted at zero. With the inverse-closed generator
set

```text
S = {+1, -1} = {1, 3} mod 4,
```

the Cayley graph is the four-cycle and BFS gives

```text
F_0={0},  F_1={1,3},  F_2={2}.
```

Every edge crosses depth parity. Now make the old length-two element `2` a
unit generator:

```text
S' = {1,2,3}.
```

The state universe and generated group are unchanged, but the Cayley graph is
`K_4` and

```text
F'_0={0},  F'_1={1,2,3}.
```

Expanding `F'_1` produces six directed occurrences whose endpoints are other
vertices of `F'_1`. For example `1 + 1 = 2` and `1 + 2 = 3`. The shortcut
`0--2` closes the old two-step path `0--1--2` into a triangle, destroys
bipartiteness, changes the diameter from two to one, and makes current-layer
filtering observable. Nothing about the encoded states changed; only which
group elements count as one BFS step changed.

## Why current-frontier membership matters

QTM is bipartite under quarter-turn parity, so no edge joins two vertices of
one BFS layer. An implementation that accidentally filters against
`B_(d-1)` rather than the complete `B_d` may appear correct on such a graph.

HTM destroys that protection by adding half turns as unit edges. REF-030 sees
same-level occurrence counts 36, 540, and 7,128 while expanding F1 through F3.
Those endpoints must not enter the next frontier.

This is a useful validation pattern: use both bipartite and non-bipartite
generator sets over the same representation to expose whether current-frontier
states are genuinely committed to visited.

## Boundary-edge conservation

In a symmetric labeled Cayley graph, each occurrence crossing from `F_d` to
`F_(d+1)` has its inverse occurrence crossing back. Therefore

```text
forward candidate occurrences from F_d
    = backward occurrences from F_(d+1).
```

This equality concerns occurrences, not unique states. It is stronger than
checking only frontier cardinalities and can detect a missing or duplicated
generator direction in a bounded exact run.

For a distributed execution it becomes a semantic baseline. Per-rank routing
may move and deduplicate records, but the global labeled boundary incidence
must still be conserved before representation-specific filtering.

## Hardware caution

Degree alone cannot predict the ratio of QTM and HTM work. At expanded depth
three, HTM's degree is 1.5 times QTM's, its frontier is about 3.03 times wider,
and its generated occurrence stream is about 4.55 times larger. Equal numeric
depths refer to different metric radii, so this is not a claim that one metric
is intrinsically slower.

It is instead a workload-definition rule:

```text
state representation + generator manifest + metric + output contract
```

must be fixed before a GPU or multi-GPU performance result is interpretable.
