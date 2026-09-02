# BFS proof obligations: independence and counterexample matrix

An exact BFS claim is a conjunction of predicates. Several predicates are often
grouped under the single word “correctness,” but they do not imply one another.
A valid path does not prove a complete search; a complete successor list does
not prove exact identity; correct local queues do not prove global termination.

This note refines the contract stack of note 37 into a formalization-oriented
matrix of independent obligations and minimal counterexamples.

## 1. The predicates

For a fixed graph/search epoch, distinguish:

```text
GS  graph/successor soundness
GC  graph/successor completeness
ID  exact semantic identity
MV  metric and unit-step validity
SC  schedule/finalization validity
CV  traversal coverage and no silent loss
PB  accepted-state publication/durability
OM  output-specific merge correctness
TS  termination safety: never declare too early
TL  termination liveness: eventually declare after true completion
EV  evidence validity and independence
```

The names are not implementation stages. They are propositions that can be
proved by different mechanisms.

## 2. Soundness and completeness are two directions

Let `SpecSucc(x)` be the declared successor set and `Emit(x)` the implementation
result.

```text
GS: Emit(x) subset SpecSucc(x)
GC: SpecSucc(x) subset Emit(x)
```

Together they give equality. Separately:

- emitting only the real edge `s->a` while omitting real `s->t` is sound but
  incomplete;
- emitting every real edge plus a spurious `s->z` is complete but unsound.

Path replay tests selected `GS` witnesses. It cannot establish `GC` for
unemitted transitions.

## 3. Exact identity is independent of transition equality

An oracle can enumerate every correct successor while visited aliases two
different states `a` and `b`. Then `GS` and `GC` hold at the state-transition
producer, but `ID` fails at deduplication and one branch disappears.

Conversely, a perfect injective identity function cannot repair a missing or
spurious transition. Identity answers when endpoints are equal, not which
endpoints should exist.

## 4. Metric validity is independent of reachability

A traversal may reach exactly the correct component while reporting the wrong
metric:

- an incidence representation counts two physical half-steps as distance two
  when the requested hyperedge step is one;
- a k-hop macro-transition is counted as one BFS edge;
- a zero/weighted edge is treated as unit cost.

Thus exact reachable membership does not imply exact requested distances.

## 5. Schedule validity is independent of graph fidelity

Use

```text
s -> a -> x
s -> b -> c -> x.
```

Every edge and identity decision can be correct. Under arbitrary execution, the
length-three proposal may irrevocably claim `x` before the delayed length-two
proposal. `GS`, `GC`, `ID`, and `MV` hold while `SC` fails.

Closed levels, global-minimum settlement, or fair corrective relaxation are
different ways to prove `SC`; atomic first claim alone is not one.

## 6. Coverage differs from successor completeness

Even if `Emit(x)=SpecSucc(x)` whenever `x` is expanded, a traversal can fail to
expand a reached state because:

- its frontier record overflowed;
- routing lost its only occurrence;
- a shard of the frontier was never scheduled;
- a retry was acknowledged before semantic application.

`GC` is a property of the oracle call. `CV` additionally quantifies over every
required parent, shard, message, and accepted future expansion.

## 7. Publication is independent of membership

On `s->a->b`, suppose `b` is durably marked visited and the system fails before
durably publishing `b` as pending work. After recovery, replay of `a` sees
`visited[b]` and suppresses it; `b` is never expanded.

The reached-state set can contain `b` while `PB` fails. A membership
linearization point is not by itself an expansion/publication transaction.

## 8. Output merge is independent of scalar distances

In the diamond

```text
s -> a -> t
s -> b -> t,
```

distance two and reached membership can be exact while:

- one predecessor is omitted from the all-parent DAG;
- a retry counts `(a,t)` twice and reports three shortest paths;
- first arrival chooses a noncanonical parent;
- a move label needed for replay is dropped.

Therefore scalar distance correctness does not imply `OM` for a richer output.

## 9. Termination safety and liveness are independent

`TS` means no incomplete execution is reported complete. `TL` means a genuinely
complete finite execution eventually produces the completion decision.

They are logically independent:

- a protocol that never announces completion is safe but not live;
- a protocol that announces when every local queue is momentarily empty can be
  live but unsafe while a message is in flight.

True global quiescence is a semantic state. Detecting it requires both a sound
decision rule and progress of the detection protocol itself.

## 10. Evidence validity is not another runtime invariant

An execution may be correct while its artifact is too weak to prove the claim.
Conversely, internally consistent telemetry can be produced by two
implementations sharing one common bug.

Examples:

- count/sum/xor parity with swapped frontier members;
- CPU/GPU parity using the same incorrect move table;
- replay of one valid path while another reachable branch was omitted;
- local conservation totals taken from an inconsistent distributed cut.

`EV` asks whether evidence actually covers each predicate and whether its basis
is sufficiently independent. It should not be conflated with the truth of the
runtime result.

## 11. Non-implication matrix

| Observed/proved fact | Still does not prove | Minimal witness |
|---|---|---|
| every emitted edge is real | no real edge omitted | omit `s->t` |
| every real edge emitted | no spurious edge emitted | add `s->z` |
| successor equality | exact visited identity | alias `a,b` |
| exact component membership | requested distance metric | two half-steps versus one logical step |
| exact graph and identity | shortest labels | long branch claims first |
| exact oracle per call | every reached state expanded | lost frontier shard |
| durable visited bit | durable future expansion | orphaned `b` |
| exact distances | parents/counts/labels | diamond retry/omission |
| all local queues empty | global completion | message in flight |
| termination never false | termination eventually reported | never announce |
| matching aggregates | matching frontier set | `{0,3}` versus `{1,2}` |
| matching CPU/GPU output | independent validation | shared faulty table |

Each row rejects one implication only. It does not claim that the listed
counterexample is the only failure mode.

## 12. Requirement matrix by requested output

All columns assume a fixed declared problem/epoch.

| Requested result | GS/GC | ID | MV/SC | CV/PB | OM | TS/TL |
|---|---:|---:|---:|---:|---:|---:|
| one positive reachability witness | selected GS | endpoint | witness length only | witness records | replay | stop after valid witness |
| exact target distance | full relevant scope | exact | required | all shorter-capable work | one path if requested | lower-bound safe stop |
| exact radius-`R` ball | through `F_(R-1)` | exact | required | complete prefix | declared metadata | completed boundary |
| complete reachable set | full reachable scope | exact | reachability schedule | exhaustive/durable | declared metadata | global exhaustion |
| canonical shortest path | relevant full scope | exact | required | equal-depth closure | total-order reduction | metadata-complete stop |
| shortest-path DAG/counts | relevant full scope | exact edge/occurrence | required | every predecessor contribution | set/addition semantics | forward closure plus output closure |

“Selected GS” for one positive witness is intentionally weaker: it proves the
returned path exists, not that the target is unreachable by a shorter path or
that the traversal was complete.

## 13. Formalization skeleton

A proof assistant or executable specification can separate assumptions as:

```text
GraphSpec(State, Label, Edge)
SuccessorEq: emit(x,l)=y iff Edge(x,l,y)
IdentityEq: key(x)=key(y) iff x=y
LayerInvariant: visited=B_d and frontier=F_d
Coverage: every (parent,label) obligation retires
Publication: every accepted state is pending-or-expanded durably
OutputReduce: merge operation matches requested object
TerminationSafe: done -> semantic completion
TerminationLive: semantic completion -> eventually done
```

The BFS theorem should consume only the assumptions needed for its exact
conclusion. A path-witness theorem should not require exhaustive termination;
an unreachable theorem must.

Counterexample fixtures can accompany every dropped assumption. This is often
more informative than one monolithic theorem whose premises hide which failure
caused which lost guarantee.

## 14. One GPU to many GPUs

Moving to many GPUs does not create new mathematical distances, but broadens
the quantified objects:

- `CV` includes every rank, partition, routed occurrence, retry, and device
  queue;
- `PB` includes authoritative owner claim and durable/global frontier
  publication;
- `OM` includes cross-owner equal-depth contenders and contribution identity;
- `TS/TL` include messages, collectives, kernels, failures, and detection
  progress;
- `EV` needs per-owner evidence and a consistent cut, not only reduced totals.

A one-GPU proof transfers only after each local quantifier is lifted to the
distributed execution domain.

## 15. Failed external critique attempt

One `ask_experts` call was made to the multi-GPU beam and theorem-proving
experts with this independence-map question. The tool returned `fetch failed`
before any expert content was received. No claim in this note is attributed to
that call, and no infrastructure repair or repeated request was attempted.

## 16. Current synthesis

Exact BFS correctness is not one invariant but a conjunction whose components
fail independently. The most overloaded word is “complete”: it can refer to a
successor list, scheduled frontier coverage, durable publication, rich output
closure, or safe global termination. These must be named separately.

The practical payoff of the matrix is diagnostic precision. A failed proof or
test can be attached to one predicate without weakening unrelated established
facts, and a successful narrow witness cannot silently inherit a stronger
conclusion.

This note extends notes 03, 09, 18, 30, 37, 41, 55, 57, 162, 163, 164, and 172.

