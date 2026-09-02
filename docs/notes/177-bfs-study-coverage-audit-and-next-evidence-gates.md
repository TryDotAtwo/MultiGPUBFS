# BFS study coverage audit and next evidence gates

## Purpose

This audit prevents the research goal from drifting into duplicate synthesis,
unsolicited optimization, production implementation, or infrastructure repair.
It distinguishes three levels of support:

1. **Conceptual:** definitions, proofs, counterexamples, and applicability
   boundaries are recorded.
2. **Bounded evidence:** a declared finite fixture or measurement checks a
   claim, without implying target-scale behavior.
3. **Real target evidence:** the actual 1-GPU or multi-GPU execution path is
   measured end to end with semantic parity and an environment snapshot.

Counts are inventory facts, not quality scores. At the historical post-note-195
snapshot the corpus has 196 numbered-note files covering 195 unique numeric
identifiers, 2,048 unique `SEM-*` claim IDs, and 43 unique `REF-*` experiment
IDs. Numeric identifier 185 is intentionally occupied by two distinct notes
(Moore/Lee history and simultaneous action conjugacy), so file count and unique
number count differ by one. The initial note-177 snapshot had 176/1,862/42; the
delta is mostly conceptual and source-audit coverage. REF-046 promoted one
counterexample-only cell to bounded finite-model evidence, not runtime
validation.

## Coverage matrix

| Area | Conceptual status | Bounded evidence | Real target evidence | Next evidence gate |
|---|---|---|---|---|
| Exact BFS semantics, schedules, frontier and visited | Strong | Exhaustive small references and counterexamples | Not required for the mathematical claims | Add only for a genuinely new semantic boundary |
| Single-GPU exact primitives | Strong for the studied dense-key paths | REF-012--017 | One RTX 3070 Laptop snapshot; not application scale | Re-measure only for a named transfer question |
| Cayley, Schreier, relations and puzzle actions | Strong through the declared actions and depths; notes 184-185 specify the exact cross-runtime state and tuple conjugacy contract | REF-022, REF-024--030 | Current bounded puzzle audits only; DeepCubeA remains source-audited, not executed | Simultaneous CayleyPy/DeepCubeA position/value and signed-label conjugacy, then an application-scale oracle |
| CayleyPy ordinary BFS implementation and export contract | Notes 193-195 source-audit neighbor layout, batching alignment, duplicate generator entries, inverse-label collapse, distributed owner closure, rolling seen state, truncation, diameter, and explicit export semantics | Source inspection only; no new runtime fixture | Installed/runtime parity, torchrun parity, and directed incomplete-export behavior unverified | Keep as named open questions; do not create a fixture without explicit user authorization |
| Random and structured graph wave behavior | Strong within declared models | REF-031--044 | No target application claim | Extend only for a new graph mechanism, not an arbitrary family |
| Distributed ownership, routing and exactness | Strong protocol model | REF-010/023 one-process simulation | Missing real interconnect/runtime evidence | Exact 1/2/4-GPU parity run with routing and timeline evidence |
| Distributed 1D/2D layout transfer | Note 189 separates expand completeness, fold authority, collectives, memory placement, and implicit generator sharding | Primary-source conceptual audit only | Missing explicit and implicit matched runtime evidence | Compare the same exact frontier/output contract with per-phase bytes, aggregation, topology, and peak memory |
| Distributed bottom-up early exit | Note 190 derives frontier snapshot, systolic completion rotation, publication, and output-contract obligations | Primary-source protocol audit only | Missing bounded/runtime comparison | Compare fresh exact BFS with checks-before-hit, bitmap traffic, substep latency, and richer-output parity |
| Termination and retry obligations | Strong consistent-cut model in notes 56 and 173--174 | Counterexamples; no runtime failure injection | Missing | Inject delayed, duplicated and lost physical messages while preserving logical IDs |
| Discovery-to-publication continuity | State-machine contract in note 178 | REF-046 bounded Rust model: 6 local and 18 path single-stop schedules | Missing runtime/memory-model evidence | Add memory ordering, capacity, delayed recovery, and failure injection only for a named protocol question |
| Scaling and cost coordinates | Vocabulary and regimes are explicit in notes 165--166 | Prior primitive timings only | Missing separated strong, weak/capacity, throughput and latency rows | One fixed-work matrix, then separate capacity/throughput questions |
| Distributed communication meaning | Cut/information/protocol decomposition in note 179 | Finite counterexamples only | Missing topology/runtime measurements | Hold graph/partition fixed while varying knowledge placement and recomputation |
| Distributed exact result reconciliation | Exact verifier-sharding contract in note 180 | Counterexamples only | Missing bounded/runtime comparison | Compare dense-rank bitmap and full-state sort/merge against compensating errors |
| Safe visited forgetting | Undirected/window/boundary contracts in note 181 | Conceptual counterexamples only | Missing bounded/runtime memory evidence | Compare permanent visited with proved and violated backward-span windows |
| BFS-constrained live boundary | Layer/queue/separation/pathwidth distinctions in note 182 | Exact graph-family counterexamples | Missing bounded/runtime width corpus | Enumerate small orders and retain every width coordinate independently |
| Infinite-branching finality | Dovetail/convergence/finality split in note 183 | Exact infinite-family counterexamples | Finite probes cannot close universal claim | Seek domain-specific finite negative certificates, not throughput |
| Uncountable frontier semantics | Note 196 separates natural-number distance and stage-`omega` closure from uncountable layer width and explicit enumerability | Exact cardinality counterexamples; executable enumeration is impossible by premise | Not a record-throughput target | Require a declared symbolic-set language and decision operations before discussing computation |
| Symbolic BFS and reachability | Note 197 separates exact image/frontier layers, accumulated reachable fixed points, distance retention, witness extraction, and representation size | Primary-source conceptual audit plus direct set proofs | No symbolic backend/runtime evidence | Name a symbolic domain and output contract before any implementation question |
| Partial-order reduction versus BFS metric | Note 198 separates path reordering, goal reachability, length-preserving optimality, all-target distance preservation, and shortest-word/frontier loss | Primary-source proof audit plus exact hand counterexample | No runtime evidence needed for the semantic separation | Name the preserved target/property before treating POR as BFS or as a performance comparison |
| Decremental invalidation versus repair | Note 186 gives an exact old-DAG preservation theorem and output-specific change regions | Conceptual diamond and batch counterexamples only | Missing bounded/runtime comparison | Compare DAG support invalidation with fresh BFS after declared deletions |
| Incremental single-edge sensitivity | Note 187 gives exact directed/undirected distance and output cones | Conceptual path-decomposition and batch counterexample only | Missing bounded/runtime comparison | Compare formula cones and fresh BFS for strict, equal, irrelevant, and chained-insertion fixtures |
| Incremental batch closure | Note 188 gives exact inserted-edge-use recurrence and endpoint-metric/counting boundaries | Conceptual chaining and redundant-segmentation counterexamples only | Missing bounded/runtime comparison | Compare recurrence rounds and fresh BFS; test scalar equality separately from path counts |
| Deterministic and canonical output | Forward and bidirectional recurrences in notes 175--176 | Conceptual counterexamples | Missing across rank/GPU counts | Compare frontier, distance, parent and shortlex word independently |
| Shortest-path tree versus first-in BFS-tree realizability | Notes 191-192 separate pointwise geodesic parents from globally realizable FIFO history; Cayley generator autocorrelation explains shared-child coupling | Exact hand proofs including all-transposition `S_3`; no executable fixture | Missing and not required for the theorem | Preserve distance, replay, deterministic reduction, and first-in realizability as separate future gates |
| Algebraic quotient/coset ownership | Strong conceptual model in notes 167--172 | Missing focused Rust fixture | Missing | Small exact coset/orbit routing fixture before system design |

## Corrections to the previous gap list

- “Build a cost vocabulary” is no longer an open conceptual task: note 165
  supplies the coordinates and note 166 separates scaling regimes. The open
  part is measurement on a real execution path.
- “Separate scaling regimes” is conceptually complete. It must not trigger an
  optimizer; it names the rows a future measurement must keep distinct.
- The unfinished-depth condition is no longer only note 56: notes 173--174 add
  independent proof obligations and logical-credit conservation. Runtime
  validation remains absent.
- REF-010 and REF-023 validate a simulation and its artifacts, not a real
  multi-GPU transport, device-resident traversal, or interconnect.
- Notes 191--192 close the conceptual gap between a deterministic shortest-path
  parent reduction and a tree realizable by one serial first-in FIFO history.
  No runtime test is needed for the exact counterexamples.
- Notes 193--195 strengthen the retained CayleyPy source contract but do not
  validate the installed version, torchrun execution, HDF5 round trip, or
  exported directed prefix at runtime.
- Missing bounded or runtime evidence is an open question, not implicit
  authorization to write a probe, test, benchmark, or demonstration program.

## Ordered evidence gates (dormant without explicit authorization)

These are promotion gates, not an implementation backlog. None authorizes code,
Docker execution, a benchmark, or a measurement by itself. A gate becomes active
only after a separate explicit user request naming or clearly authorizing the
experiment.

1. Validate an independently specified application-scale successor and equality
   oracle before using application performance as BFS evidence.
2. The read-only CayleyPy/DeepCubeA audit is complete in note 184: both use
   54-ID sticker states, but simultaneous position and signed-label conjugacy
   remains an executable gate. Do not infer runtime parity from source
   similarity.
3. Docker became naturally available and REF-046 completed. The remaining
   semantic microfixtures are exact result reconciliation and safe/violated
   backward-span windows; do not turn them into infrastructure or production
   work.
4. Treat REF-045 as a preserved random-graph question, not the automatic next
   run merely because its number is earlier.
5. If explicitly moving to real multi-GPU evidence, begin with semantic parity
   on a tiny exact workload, then record routing bytes, synchronization and
   topology. Do not begin by optimizing throughput.
6. Exercise consistent-cut accounting with controlled delay/retry/failure
   cases before trusting termination timings.
7. Compare deterministic output contracts separately across worker counts:
   reached set, distance, parent, move word, path count and connector closure.
8. Test algebraic ownership first on a bounded Rust coset/orbit fixture; only
   then ask whether it reduces real routing.
9. If CayleyPy runtime validation is explicitly requested, separate scalar
   frontier parity from state/hash row alignment, duplicate label identity,
   torchrun flag behavior, incomplete directed edge export, and serialization.

## Current blocked and failed evidence

REF-045 remains `not run` because its readiness check failed. REF-046 preserves
the same earlier access failures but later completed after Docker returned
without repair. An expert consultation returned `fetch failed`; it supplied no
recommendation and therefore no evidence. These failures do not block continued
reading, source audits, proof work, counterexample construction, or coverage
auditing.

## Post-note-183 anti-duplication conclusion

Notes 178--183 filled six distinct conceptual cells: discovery/publication,
communication meaning, exact reconciliation, safe forgetting, BFS-constrained
live boundaries, and infinite-branching finality. Each now has an explicit
evidence boundary. Do not add another synthesis of these topics until a new
counterexample changes a theorem or an evidence gate promotes their status.

The highest-value executable work remains semantic microfixtures, including
the now precisely scoped cross-runtime conjugacy check, not production
optimization or another arbitrary graph family. Further source paraphrase is
not evidence for that check.

## Post-note-195 source-audit conclusion

Notes 191--195 added two conceptual cells rather than performance evidence:

1. shortest-path parent validity does not imply first-in BFS-tree
   realizability, including in a symmetric Cayley graph;
2. retained CayleyPy ordinary BFS has explicit occurrence-order, ownership,
   output, and export boundaries that are narrower than “exact labeled graph.”

The three-card CayleyPy source axis is closed. Do not continue by walking every
remaining library file. Reopen it only for a distinct semantic question or when
runtime evidence is explicitly authorized. In particular, the newly found
state/hash alignment, torchrun flag, directed prefix-export, and matrix-load
boundaries remain recorded unknowns rather than automatic test tasks.

## Decision rule for the next study step

### 2026-08-31 scope and evidence refresh

Recent cycles refined existing proofs rather than adding new runtime evidence:
source-pair identity versus Cayley translation, directed lookup orientation,
bidirectional integer stopping bounds, finite-distance DAG/count certificates,
bounded absence versus infinity, and the zero-cost weighted boundary. Their
corrections are recorded in the research log and relevant notes; this is not a
claim that all remaining corpus statements have been independently audited.

The original study axes remain covered at different strengths:

| Goal axis | Evidence checked in this refresh | What it does not establish |
|---|---|---|
| BFS meaning, guarantees and variations | Existing core notes and explicit corrected hand proofs in notes 08, 11-13, 41, 56 | Universal correctness of every implementation or all stored claims |
| Implicit/Cayley states and CayleyPy context | Note 193's source-action contract; retained REF-026/027/028 counts and matching recorded inputs | Fresh installed-runtime parity or independent full puzzle validation |
| One-GPU understanding | REF-017 report: four S9 traversals, full layer oracles, separate traversal repetitions | Large-puzzle throughput, other hardware, or new current measurements |
| Multiple-GPU understanding | REF-010 report explicitly describes a bulk-synchronous owner-routing simulator | Real interconnect latency, concurrent device execution or scaling |
| Recording failures and insights | Research log retains counterexample, quantifier, attribution and certificate corrections | Automatic permission to resolve unknowns by writing or executing code |

The roadmap still contained weaker old probe permissions. Its phase lists are
now explicitly historical/dormant, and its purpose is aligned with the protocol:
even a tiny new executable probe requires a separate explicit user request.

**Closed read-only question:** REF-017 timing boundaries were checked against
the Rust and CUDA source. Per-step counter resets and Rust count validation
are outside kernel intervals but inside the traversal timer. The residual
does not isolate synchronization as the dominant individual cost. The host
loop is oracle-bounded and validates final emptiness; it is not an
unknown-depth stop-on-empty driver. The report and synthesis now state both
boundaries. Do not schedule this same source check again without new evidence.

### Cross-axis review after the foundational clarification cycle

The review inspected the synthesis recurrence and variant map, note 07's
owner/closure/scaling model, the REF-010 simulated superstep, REF-017's timing
contract, and the recent journal corrections. It is a selective evidence
review, not independent verification of every claim or historical run.

- Core understanding is linked to concrete mechanisms: frontier-only expansion
  follows prior closure; visited follows prefix replacement; history-sensitive
  continuations require sufficient state.
- Representation and resource costs are separated: query initialization versus
  reachable traversal, metric width versus mixed FIFO occupancy, kernel versus
  traversal intervals. These distinctions do not select an optimal design.
- Cayley symmetry explains relative-distance lookup and one-root diameter;
  the three-point Schreier example limits transfer to puzzle-state actions.
- Output guarantees are separated: exact policy descent versus complete layers,
  admissible abstract bounds versus exact concrete distances, certified versus
  requested table radius, and scramble length versus minimum distance.
- REF-010 remains simulation evidence; REF-017 remains a small historical
  one-GPU traversal. No application-scale or real multi-GPU speed claim follows.

No new foundational gap was established by this review. Do not manufacture a
new theorem or another implementation audit merely to populate a next-step
slot. The next admissible study step must expose a concrete unresolved
prediction, contradiction, or source-specific question under the standing
selection rule. Existing runtime gaps remain dormant, not mandatory work for
this understanding-only scope. The open-ended goal is left active; this review
does not assert that the entire corpus or all BFS variations are verified.

Application-scale parity and real multi-GPU timing remain unknown and dormant.
Their absence is not permission to run tests and is not a blocker to this
read-only study. No completion claim for the full open-ended goal follows from
this scoped refresh.

### Completion review of the requested learning goal, 2026-08-31

This review supersedes the earlier decision to leave the learning goal active.
It does not supersede the evidence limitations. The user's corrected objective
is to understand BFS and record the learning, not to deliver an optimized
engine, exhaust all possible research questions, or establish target-runtime
performance. Historical phase lists are not implementation deliverables.

| Requested outcome | Evidence inspected and reasoning established | Assessment |
|---|---|---|
| Mathematical essence and guarantees | Note 03's soundness/completeness proof and FIFO invariant; synthesis's exact-ball recurrence and frontier-only closure argument | Learning outcome met; corrected depth-versus-expansion count wording during this review |
| Frontier and visited intuition | Prefix replacement, mixed-layer queue trace, integer-line word/state example, target discovery versus count finalization in notes 29, 54, 57, 73 | Learning outcome met through hand explanations and counterexamples |
| Variations | Synthesis variant map and worked notes for reverse/bidirectional, multi-source, weighted, product/quotient, beam and PDB boundaries | Learning outcome met; no claim to enumerate every possible variation |
| Explicit, implicit, Cayley and library context | Note 06's graph interface, notes 193-195's action/ownership/export contracts, note 38's production beam distinction, three-point Schreier witness | Learning outcome met; current runtime and checkpoint provenance remain separate unknowns |
| One-GPU understanding | Note 07's expansion/identity/control model and retained REF-017 timing, oracle, noise and scope limits | Learning outcome met; no universal bottleneck or performance-optimality claim |
| Multiple-GPU understanding | Owner union, producer versus owner dedup, global closure, work/skew/scaling distinctions; notes 07/194 and retained REF-010 simulation | Learning outcome met; real transport/scaling is not established |
| Durable sources, observations, failures and open questions | Thematic notes and source references, research log, retained experiment reports including corrected benchmark protocol and unverified/dormant evidence gates | Record exists and distinguishes proof, source inspection, historical measurement and unknown |
| No autonomous optimization/code | Current protocol forbids new executable work without a separate request; final review uses file inspection, hand reasoning and documentation edits only | Current scope respected; old experiments are not renewed authority |

The completed deliverable is the connected mental model and research record,
not certification of every implementation or every sentence in the corpus.
Its central explanation is: exact graph/state semantics plus complete
successor generation and nondecreasing finalization create metric layers;
output requirements decide which duplicate information can be discarded;
GPU placement changes resource costs without changing those obligations.

Remaining questions are explicitly outside this completed learning outcome:
application-scale performance, actual multi-GPU timing, fresh CayleyPy runtime
parity, and linkage of the staged trainer to a production checkpoint. They
remain unknown, not failed proof obligations hidden by a passing test. Further
implementation or measurements require a separately authorized task. Further
study can reopen a specific question without pretending the field is exhausted.

### Selection rule for any subsequently requested study

A new step must answer one precise question and state which cell of the matrix
it promotes. If it merely repeats an existing synthesis, proposes an optimal
implementation without a hypothesis, or requires repairing infrastructure, it
is out of scope until the user explicitly changes the goal. Source reading,
proofs, and hand counterexamples may proceed autonomously; writing or running a
new experimental/test/benchmark program requires separate explicit user
authorization.
