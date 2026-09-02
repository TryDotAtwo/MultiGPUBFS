# k-hop batching: one physical superstep is not one BFS level

Combining several edge expansions can reduce the number of visible control
rounds, but it creates two very different mathematical objects:

1. execute `k` ordinary BFS microlevels inside one physical superstep;
2. replace the graph by a power graph whose macro-edges represent short walks.

The first can preserve the original BFS semantics.  The second changes the
metric unless the internal lengths and witnesses remain part of the state.

This note studies that boundary only.  It proposes no GPU implementation.

## Exact microlevel batching

Let `Post^i` mean `i` successive applications of the successor relation.  If a
superstep begins with the exact ball `B_d`, it may internally compute

```text
F_(d+1) = Post(F_d)       \ B_d
F_(d+2) = Post(F_(d+1))   \ B_(d+1)
...
F_(d+k) = Post(F_(d+k-1)) \ B_(d+k-1).
```

Afterwards it may expose only `B_(d+k)` to the outer controller.  The physical
barrier count changed, but the logical recurrence did not.  Exact distances are
preserved if the computation retains the equivalent of these microlevel
boundaries:

- every accepted state carries its minimum original-edge depth;
- a state reached at several internal depths is assigned the minimum;
- shorter-depth consequences cannot be suppressed by a longer first arrival;
- every required cross-owner transition is processed at the correct internal
  depth;
- target stopping accounts for all still-possible smaller internal depths;
- path metadata expands each macro record into original moves.

Calling this "one level" would be misleading.  It is one physical superstep
containing `k` logical BFS levels.

## Coarse balls without exact strata

Define

```text
Post_[1,k](X) = union_(1 <= i <= k) Post^i(X).
```

Starting from `C_0=S`, the recurrence

```text
C_(r+1) = C_r union Post_[1,k](C_r)
```

computes `C_r=B_(rk)` as sets, provided every walk of length at most `k` is
represented exactly.  Induction gives the result: one macro round extends the
known radius by at most `k`, and every shortest path of length at most `(r+1)k`
can be split after at most `rk` edges with a suffix of length at most `k`.

But the difference `C_r \ C_(r-1)` contains original distances

```text
(r-1)k+1, ..., rk.
```

It is a thick annulus, not one BFS frontier.  The coarse sets preserve bounded
reachability while forgetting exact distance inside each block unless
subdepths are stored or recomputed.

## The at-most-k graph changes the metric

Construct a graph `G^[<=k]` on the same vertices with a macro-edge `u->v`
whenever `G` has a directed walk from `u` to `v` of length between one and `k`.
Then for every reachable `v`,

```text
dist_(G^[<=k])(s,v) = ceil(dist_G(s,v) / k).
```

One direction follows because each macro-edge expands to at most `k` original
edges.  The other follows by splitting an original shortest path into chunks of
at most `k` edges.

Thus BFS in the power graph is exact for the power-graph metric, not the
original move count.  Multiplying the macro distance by `k` gives only an
interval:

```text
(m-1)k < dist_G(s,v) <= mk.
```

## Exactly-k reachability is different again

Boolean `A^k` indicates the existence of a walk of exactly length `k`, not at
most `k`.  Using only exact-k macro-edges can miss reachable vertices whose path
lengths cannot be padded to a multiple of `k`.

For the chain

```text
s -> a -> t
```

with `k=2`, exact-two expansion reaches `t` but not `a` from `s`.  Repeated
exact-two macrosteps still never enumerate `a` as an endpoint.  Adding identity
loops permits padding, but that modifies the transition system unless waiting
or identity moves were already legal.  It also changes labeled-walk
multiplicity even when vertex reachability is unaffected.

The Boolean expression for walks of at most `k` is support of

```text
I OR A OR A^2 OR ... OR A^k,
```

with the usual orientation convention.  It must not be replaced silently by
`A^k`.

## First-wins visited fails when depths mix

Consider

```text
s -> a -> x
s -> b -> c -> x.
```

Inside a `k=3` batch, `x` has candidates at original depths two and three.  If
the depth-three record happens to perform a first-wins visited insertion first,
then suppresses the depth-two record, the stored distance is wrong.

Representative semantic repairs are:

- preserve strict microlevel order so depth two is finalized before depth
  three can claim `x`; or
- collect competing mixed-depth candidates and reduce the exact minimum before
  any deeper consequence is allowed to depend on the winner; or
- treat distance as a tentative value and perform a minimum relaxation with
  reactivation/propagation when it improves.

The relaxation alternative is no longer the simple boolean-visited BFS
invariant; it inherits
the asynchronous relaxation obligations from note 18.  Atomicity of the first
write does not imply minimality of the winning depth.

## Intermediate deduplication is not optional semantics

Suppose a macro expansion enumerates every word/walk of length at most `k` but
deduplicates only final endpoints.  Endpoint reachability may remain correct,
yet:

- intermediate targets are invisible unless explicitly tested;
- original distances are unavailable without internal-depth metadata;
- parent reconstruction lacks the intermediate witness;
- relations/cycles can create enormous multiplicity;
- an intermediate state owned elsewhere may need to generate its successors
  before the macrostep ends.

Conversely, deduplicating intermediate states by identity is safe only with a
depth-aware rule.  Merging a shorter and longer arrival as interchangeable is
safe for future unweighted reachability when the shorter one dominates; letting
the longer record win is not.

For path-dependent move restrictions, note 20 still applies: visible vertex
identity may be too coarse, and the macrostep must operate on the product state.

## Cross-owner paths

Multi-GPU ownership creates another boundary.  If device `P` expands only
locally owned states for all `k` internal hops and sends remote endpoints only
afterwards, it misses a path whose first hop changes owner and whose later hops
must be generated by that new owner.

Correct alternatives are semantic, not performance prescriptions:

- communicate at each logical microlevel;
- replicate enough transition/state information to continue a remote path
  locally under an exact contract; or
- ship a macro request/witness that the authoritative owner can evaluate
  completely.

Whatever execution is chosen must account for in-flight work at every internal
depth.  Local completion of `k` steps does not prove the global ball `B_(d+k)`
complete if a depth-`d+j` message can still create depth-`d+j+1` work elsewhere.

## Target detection inside a batch

The first target record observed in a mixed-depth batch need not have the
smallest depth.  A depth-`d+3` candidate can race ahead of a depth-`d+1`
candidate.

For one exact minimum-hop target path, the batch must reduce the minimum target
depth and prove that every work item capable of producing a smaller depth has
completed.  To claim complete frontiers or all shortest parents, it must also
finish the relevant equality boundary.  Merely finishing the physical
superstep is sufficient only if the superstep protocol itself proves all its
logical sublevels complete.

## Macro-edge path witnesses

A parent pair `(macro_parent, child)` proves only that some short walk was
claimed.  Replay requires one of:

- the complete sequence of original edge/generator labels;
- all intermediate parent records;
- a deterministic, independently validated procedure to regenerate a valid
  witness of the declared length and convention.

For directed graphs the witness must follow original forward edges.  For
Cayley/Schreier graphs it must also preserve left/right action, move order,
inverse convention, legality, and any symmetry frame.

If every length-at-most-`k` generator word is treated as one unit-cost macro
move, the algorithm computes a new word metric.  If macro-edges retain their
original lengths as unequal weights, ordinary BFS is no longer the appropriate
outer shortest-path proof.  Keeping the internal BFS strata avoids both metric
changes.

## Matrix squaring and transitive closure

Boolean matrix powers provide increasingly long-walk reachability, and repeated
squaring can expose a transitive closure in logarithmically many algebraic
stages for a finite matrix.  That does not mean it performs logarithmically many
ordinary BFS levels:

- an algebraic stage represents a range or exact set of path lengths;
- intermediate distance shells and parents are not automatically retained;
- matrix support may densify even when a frontier representation is sparse;
- reachability closure is a weaker output than a distance map or replayable
  shortest-path tree.

The comparison is useful for understanding information flow, not for declaring
one formulation a drop-in BFS replacement.

## Audit questions

Before accepting a k-hop BFS claim, record:

1. Is `k` a physical batching parameter or a new macro-edge definition?
2. Does a macro-edge mean exactly `k` or at most `k` original edges?
3. Are exact original depths retained for every accepted state?
4. Can a longer first arrival suppress a shorter later arrival?
5. At which internal depths are visited and duplicate decisions finalized?
6. Can paths cross ownership boundaries inside the batch?
7. Which in-flight condition proves every logical sublevel complete?
8. How is the minimum target depth reduced before stopping?
9. What original-edge witness reconstructs each macro parent?
10. Is the reported metric the original graph metric or the power-graph metric?

## Sources and failed expert check

- De Schutter and De Moor,
  [Consecutive Powers of a Boolean Matrix](https://citeseerx.ist.psu.edu/document?doi=310b511629a4cc3e02964b1f73ee5735ed88d6e6&repid=rep1&type=pdf),
  states the exact-length-walk interpretation of Boolean matrix powers.
- Fischer and Meyer,
  [Boolean Matrix Multiplication and Transitive Closure](https://doi.org/10.1109/SWAT.1971.4),
  is the classical source connecting Boolean multiplication and transitive
  closure.
- The recurrence, power-graph distance identity, and counterexamples above are
  proved directly rather than inferred from a performance result.
- Two attempts to query the `multigpu_beam` expert returned only `fetch failed`.
  No expert recommendation was available or used for this note; the failure is
  retained as research-process evidence.

## Current conclusion

Several BFS depths may share one physical GPU or distributed superstep without
changing the answer only when their logical ordering, minimum-depth dominance,
cross-owner consequences, target lower bound, and original-edge witnesses are
preserved.  Erasing those internal strata produces a coarse reachability
annulus or a different graph metric, not ordinary exact BFS distances.
