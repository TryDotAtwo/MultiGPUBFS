# Research log

This is a historical journal, not a sorted current-status index. Entries may
appear out of time order. Use `experiment-log.md` for the latest recorded
experiment outcome; earlier failed attempts remain evidence of those attempts.
Corrections dated 2026-08-31 update misleading passages without presenting the
corrected claim as something that was already known in the original run.

## 2026-08-31: repair audited research; plugin deferred

- User requested correction of the audit findings before further plugin work.
- The correction scope covers notes, dependent claim/question indexes,
  historical status wording, and the strength of experimental interpretations.
- The prior audit remains a dated snapshot; completion of this correction pass
  is recorded separately, not asserted by this opening entry.

Resolution: [2026-08-31 correction record](reviews/2026-08-31-bfs-audit-corrections.md).
Known textual findings were repaired in 29 notes and 35 existing evidence rows,
with corresponding report/index updates. The lost first REF-017 raw sweep and
three source-code findings remain explicitly open; no runtime fix or plugin
completion is claimed.

## 2026-08-29: REF-046 bounded discovery-publication interleavings

- Docker became available naturally; no Docker Desktop or WSL repair was made.
- **Question:** Does atomic visited claim preserve expansion coverage if the
  claimant stops before publishing the frontier payload?
- Followed test-first Rust in Docker. The first apparent RED was rejected
  because the Dockerfile omitted integration tests; adding `COPY tests/ tests/`
  exposed the missing library/API, and the empty model then produced four
  expected behavioral failures.
- The final finite model enumerates all six orders of stop, claimant publish,
  and recovery after claim. Blind drop leaves `visited && !published` in three;
  helpable descriptor and logged intent publish in all six.
- With the single interruption placed on each edge of `s->a->b->c`, blind drop
  reaches `c` in `9/18` schedules; both recoverable protocols reach it in
  `18/18`.
- Physical duplicate publication attempts can occur while idempotent set commit
  retains one record. This does not validate additive/count outputs.
- Final Docker builder gate passed one existing test and five REF-046 tests,
  plus formatting and release build. This is bounded sequential-model evidence,
  not a runtime memory-model, GPU, or multi-GPU protocol validation.
- Detailed evidence: `experiments/REF-046-discovery-publication-interleavings.md`.

## 2026-08-29: Moore--Lee history and BFS as a wave plus trace

- **Question:** Did early shortest-maze work present the essential idea as a
  FIFO queue, or as a spreading distance wave?
- **Primary evidence:** Lee's 1961 paper defines separate search and trace
  procedures, calls its minimal-distance use a specialization based on Moore,
  and explicitly compares the search pattern to waves expanding from a source.
- **Correction:** "Lee algorithm equals ordinary FIFO BFS" is too coarse.
  Lee's Algorithm A handles lexicographically ordered vectors of monotone path
  properties; the unit-increment minimum-distance case is the BFS-like
  specialization.
- **New intuition:** first-arrival wave labels compute the canonical distance
  field; a later decreasing-label trace selects one witness. The FIFO queue is
  a scheduling representation of that wave, not its conceptual essence.
- **Evidence limit:** Moore's 1959 publication metadata and Lee's citation were
  confirmed, but Moore's full text was not inspected; no detailed pseudocode is
  attributed to him here.
- Detailed source notes: note 185. No code, Docker action, implementation,
  benchmark, optimization, or GPU work was introduced.
- **Follow-up:** a targeted search still found no inspectable Moore scan.
  Lawler's 1976 unit-edge/two-bit description is retained only as a secondary
  lead; the historical branch stops here rather than converting it into an
  unsupported primary-source claim.

## 2026-08-29: goal correction — prevent lemma-chain drift

- **Observed failure:** the study could obey the ban on optimization and code
  yet still drift into a long sequence of increasingly narrow, self-generated
  lemmas. Correct statements and additional notes did not necessarily improve
  the intuitive understanding of BFS.
- **Goal correction:** one study cycle now answers exactly one plain-language
  BFS question with the smallest useful example and a plain-language
  conclusion. After at most three related cards, work must return to synthesis
  or a different foundational BFS axis.
- **Stop rule:** do not add material for duplicate/minor refinements, questions
  whose main relevance is adjacent theory, or results that increase coverage
  without making ordinary BFS easier to explain, trace, or recognize.
- **Progress metric:** clearer and more connected understanding of BFS; not
  note count, claim count, novelty, token use, or continuous activity.
- No new BFS lemma, implementation, experiment, benchmark, or optimization was
  introduced in this correction step.

## 2026-08-29: why FIFO makes first discovery safe

- **Question:** Why does the FIFO queue prevent ordinary BFS from freezing a
  non-shortest first discovery?
- **Smallest useful trace:** `s -> a -> x` versus `s -> b -> c -> x`. A LIFO
  schedule can reach and freeze `x` at depth three before expanding `a`; FIFO
  must finish both depth-one vertices before it can expand `c` at depth two.
- **Prediction:** FIFO is useful because it preserves nondecreasing depth, not
  because a queue by itself defines BFS.
- **Conclusion:** appending children behind the remaining current-depth records
  keeps at most two consecutive depths in the queue, shallower first. Thus the
  shorter route through `a` is exposed before a longer claim can be finalized.
- **Clarified roles:** `visited` suppresses duplicate commitment; FIFO makes
  first commitment safe. A different scheduler needs either the same depth
  finalization order or a relaxation mechanism that can correct earlier labels.
- This was a hand trace only: no code, Docker run, benchmark, implementation, or
  optimization was needed.

## 2026-08-29: anti-duplication checks and REF-046 readiness recheck

- The proposed enqueue-versus-dequeue question was rejected as a duplicate:
  notes 03, 73, and 74 and prior research-log cards already cover its semantic
  and work boundaries.
- The proposed explicit-versus-implicit and minimal `Z_4` Cayley explanations
  were also rejected as duplicates of notes 06, 16, and 54 and existing hand
  traces. No new conceptual material was added for them.
- Rechecked the next recorded bounded-evidence gate instead. `docker version`
  returned a null server value and both it and `docker ps` reported named-pipe
  `permission denied` for `dockerDesktopLinuxEngine`.
- At this earlier recheck REF-046 remained `not run`; no infrastructure repair,
  Rust/C++ code, protocol result, benchmark, GPU work, or optimization was
  introduced by the recheck. The later completed 2026-08-29 report supersedes
  this as the overall REF-046 status, without erasing the failed attempt.

## 2026-08-29: multi-source balls union, exact frontiers do not

- **Question:** Is the depth-`d` frontier of joint multi-source BFS the union of
  the depth-`d` frontiers from independent single-source traversals?
- **Prediction:** No, because a vertex at depth `d` from one source may already
  be closer to another source.
- **Smallest useful trace without using another source as the witness:** on the
  path `s--x--y--t` with sources `{s,t}`, independent depth-two frontiers union
  to `{x,y}`, while the joint depth-two frontier is empty; both vertices have
  joint distance one.
- **Correction:** balls do union directly,
  `B_S(d)=union_s B_s(d)`, but an exact joint frontier is that union of
  depth-`d` spheres minus every single-source ball of radius `d-1`.
- **Intuition:** a ball asks whether at least one source is near enough; a
  frontier asks whether the nearest source is exactly this far away.
- Removed the now-answered artifact-audit question from `open-questions.md`.
  No code, experiment, implementation, GPU work, or optimization was needed.

## 2026-08-29: frontier, candidate occurrences, and visited in one wave

- **Question:** On one ordinary BFS level, what is the concrete difference
  between the frontier, generated candidates, and visited?
- **Smallest useful trace:** in the undirected graph with edges `s--a`, `s--b`,
  `a--c`, `b--c`, and `b--d`, depth one has `F_1={a,b}` and
  `B_1={s,a,b}`. Expanding it emits `[s,c,s,c,d]`.
- **Observation:** exact identity collapses the two new occurrences of `c`,
  while subtraction against visited removes the old state `s`. The survivors
  are the next frontier `F_2={c,d}`.
- **Intuition gained:** frontier is the current sphere, visited is the whole
  accumulated ball, and generated occurrences are temporary edge-level work.
  Calling all three “the queue” hides both BFS semantics and duplicate work.
- Added the trace to the existing mental-model synthesis. No new thematic note,
  code, experiment, GPU work, or optimization was introduced.

## 2026-08-29: stale multi-GPU open question removed

- **Question checked:** Can one bridge message lie on the causal critical path
  of a much larger remote BFS wave despite tiny cut/traffic volume?
- **Existing answer found:** note 179, section 6, already gives two large
  owner-local subgraphs joined by one bridge. The bridge discovery activates
  every later frontier on the remote owner, separating cut volume from causal
  amplification and time-to-participation.
- Removed the answered bullet from `open-questions.md`; no duplicate proof or
  new note was added.
- The proposed equal-frontier-count multi-GPU question was also rejected as
  already answered by notes 07, 29, and 158: raw generator counts can be equal
  while cost, convergence, routing, and critical time differ.
- No Docker retry, code, experiment, GPU work, implementation, or optimization
  was introduced.

## 2026-08-29: empty frontier versus bounded absence

- **Question:** What does an empty next frontier prove that a radius cutoff does
  not?
- **Smallest useful trace:** on `s--a--b` plus disconnected `x--y`, exhaustive
  BFS from `s` produces `{s}`, `{a}`, `{b}`, `{}`. Complete expansion proves the
  reached ball `{s,a,b}` is successor-closed, hence `x,y` are unreachable from
  `s` in the declared graph.
- **Countercontrast:** stopping after depth one leaves `b` absent even though it
  is reachable at depth two. That status is only `NOT_FOUND_WITHIN_RADIUS`.
- **Intuition gained:** absence becomes a global negative certificate only when
  the final empty frontier follows complete expansion of the entire reached
  boundary. A cutoff records where observation stopped, not graph closure.
- Added this hand trace to the existing mental-model synthesis. No new note,
  code, experiment, Docker action, GPU work, or optimization was introduced.

## 2026-08-29: what a frontier payload must preserve

- **Question:** Which state fields may an implicit-BFS frontier record omit
  without changing BFS semantics or the requested output?
- **Generic answer:** expansion fields may be omitted only when the retained
  payload plus declared immutable context still determines the complete exact
  successor set. Equality, replay, and presentation data may live elsewhere,
  but their respective claim and output obligations must remain provable.
- **Correction:** there is no universally minimal BFS record. Sufficiency is
  relative to four interfaces: successor generation, exact identity, requested
  path evidence, and requested final presentation.
- **Remaining unknown:** the smallest sufficient payload and auxiliary records
  for each actual CayleyPy/puzzle domain still require a domain inventory. The
  corresponding open question was narrowed rather than falsely closed.
- Added this role-specific omission rule to note 06. No implementation, format
  optimization, code, experiment, Docker action, or GPU work was introduced.

## 2026-08-29: visited false positives and false negatives are asymmetric

- **Question:** Why are false `seen` and false `unseen` answers not symmetric
  errors in BFS visited handling?
- **Smallest useful trace:** on `s->a->t`, a false positive for new state `a`
  drops the only gateway and can make reachable `t` appear unreachable.
- **Countercontrast:** a false negative for an old `a` initially adds a duplicate
  occurrence rather than deleting a path.
- **Boundary:** the duplicate remains semantically tolerable only if a later
  exact authority merges it, the requested output is idempotent under retries,
  capacity does not drop records, and termination includes the extra work.
  Otherwise overflow, overcounting, or premature completion can turn it into a
  correctness failure.
- **Intuition gained:** false positives directly subtract semantic search space;
  false negatives first inflate physical work. “False negatives are harmless”
  is too broad.
- Added the trace to the existing mental-model synthesis. No new note, code,
  experiment, Docker action, GPU work, or optimization was introduced.

## 2026-08-29: reverse BFS must follow predecessors

- **Question:** Why does a target-centered BFS in a directed graph need reverse
  edges rather than ordinary outgoing edges from the target?
- **Smallest useful trace:** in `s->a->t`, forward BFS from `t` reaches only
  `{t}`. Predecessor traversal produces reverse frontiers `{t}`, `{a}`, `{s}`
  with depths equal to original forward distances to `t`.
- **Intuition gained:** forward-from-target answers where the target can go;
  reverse BFS answers which states can reach the target. These coincide only
  under an additional edge-reversal symmetry such as an undirected graph.
- **Replay boundary:** backward expansion may apply an inverse operation, while
  the stored suffix label must still be the original forward move from the
  predecessor toward the target.
- Added this trace to the existing mental-model synthesis. No new note, code,
  experiment, Docker action, GPU work, or optimization was introduced.

## 2026-08-29: ordinary BFS minimizes hops, not heterogeneous cost

- **Question:** Why does ordinary BFS stop solving the requested shortest-path
  problem when edge costs differ?
- **Smallest clear contrast:** one route `s-1->a-1->t` has two hops and cost two;
  another `s-0->b-0->c-0->t` has three hops and cost zero. FIFO BFS reaches the
  former first.
- **Correction:** BFS is still exact for the metric it actually computes—the
  number of edges. It is not exact for total edge cost unless every edge has the
  same relevant cost (up to a common positive scale).
- **Intuition gained:** unit cost is part of the semantic problem statement,
  not an implementation convenience. With heterogeneous costs, weighted
  finality needs relaxation and cost-ordered settlement such as 0-1 BFS or
  Dijkstra.
- Added the trace to the existing mental-model synthesis. No new note, code,
  experiment, Docker action, GPU work, or optimization was introduced.

## 2026-08-29: equal traversal work can expose opposite parallelism

- **Question:** Why do equal vertex/edge counts not imply equal GPU-parallel BFS
  behavior?
- **Smallest family contrast:** `P_n` and `K_(1,n-1)` both have `n` vertices,
  `n-1` undirected edges, and `2(n-1)` directed adjacency occurrences in a full
  scan.
- **Observation:** rooted at a path endpoint, frontier sizes are
  `1,1,...,1` across `n` logical levels. Rooted at the star center, they are
  `1,n-1` across two logical levels.
- **Intuition gained:** total work measures how much must be done; the frontier
  profile and level dependencies determine how much is available at once and
  how much causal span remains. Neither `|V|+|E|` nor average work predicts GPU
  occupancy over time.
- Added the comparison to the existing mental-model synthesis. It is a workload
  explanation, not an implementation or optimization proposal; no code,
  experiment, Docker action, or GPU run was introduced.

## 2026-08-29: Cayley words collapse into frontier states

- **Question:** On one minimal Cayley graph, how do generator-word occurrences,
  endpoint elements, visited states, and the next frontier differ?
- **Existing checked trace promoted into the synthesis:** in
  `Z_2 x Z_2=<a,b | a^2=b^2=e, ab=ba>` with `S=[a,b]`, expanding
  `F_1={a,b}` emits `aa=e`, `ab=c`, `ba=c`, and `bb=e`.
- **Observation:** four word occurrences produce two distinct candidates
  `{e,c}`; subtracting visited ball `{e,a,b}` leaves one new state `F_2={c}`.
- **Intuition gained:** `ab` and `ba` are two labeled shortest paths to one
  vertex, while `aa` and `bb` are generated work returning to an old vertex.
  Word count, candidate count, and frontier count answer different questions.
- This was a synthesis of an existing relation witness, not a new theorem or
  thematic note. No code, experiment, Docker action, GPU work, implementation,
  or optimization was introduced.

## 2026-08-29: local empty frontiers do not prove multi-GPU termination

- **Question:** Why are empty local frontier shards insufficient to declare a
  distributed BFS level or traversal complete?
- **Smallest useful timeline:** owner 0 expands `a`, sends new child `b` to
  owner 1, and becomes locally empty. Before delivery, owner 1 is also locally
  empty while `b` remains in flight. After delivery its frontier becomes `{b}`.
- **Counterexample conclusion:** a snapshot of empty owner queues can coexist
  with a nonempty logical next frontier. Stopping at that point loses reachable
  work.
- **Intuition gained:** frontier ownership can move through active, staged,
  transport, claimed, and published forms without being visible in a local
  queue. Termination needs a consistent global cut covering all such
  obligations, not a sum of unrelated local empty flags.
- Added the timeline to the existing multi-GPU mental model. No new protocol,
  implementation, code, experiment, Docker action, GPU run, or optimization was
  introduced.

## 2026-08-29: stable owner gives one global novelty decision

- **Question:** Why is per-GPU local visited insufficient when equal candidate
  states can be generated on different GPUs?
- **Smallest useful trace:** depth-`d` parent `p` on GPU 0 and parent `q` on GPU
  1 both generate child `x`. Independent local authorities can both accept `x`
  and place two physical records for one semantic vertex in the next frontier.
- **Intuition gained:** routing by a stable function of exact state identity
  makes all equal candidates meet at one authority, which commits one frontier
  membership and distance.
- **Output boundary:** committing `x` once does not authorize throwing away every
  losing occurrence. Distinct `p->x` and `q->x` contributions may both be needed
  for all-shortest-parent, labeled-path, or count outputs.
- Added the trace to the existing multi-GPU mental model. No owner-function
  design, protocol implementation, code, experiment, Docker action, GPU run, or
  optimization was introduced.

## 2026-08-29: self-loops and parallel edges separate distance from output

- **Question:** What do a self-loop and parallel labeled edges change in BFS if
  they do not create a new vertex distance?
- **Smallest useful trace:** edges `s-e->s`, `s-p->a`, and `s-q->a` emit three
  occurrences `[s,a,a]`. Exact identity leaves candidates `{s,a}`, visited
  removes `s`, and the next vertex frontier is `{a}`.
- **Intuition gained:** the loop and duplicate endpoint labels do not change
  `dist(s,a)=1`, but they change generated work. The two labels `p,q` also give
  two labeled shortest paths where the simple support graph records one edge.
- **Output boundary:** simplifying a multigraph can preserve reached sets and
  distances while changing occurrence counts, labels, shortest-path counts,
  and required capacity.
- Added the trace to the existing mental-model synthesis. No new note, code,
  experiment, Docker action, GPU work, implementation, or optimization was
  introduced.

## 2026-08-29: undirected edges stay local in depth; directed arcs may jump back

- **Question:** How far apart can BFS depths of the endpoints of one edge be?
- **Undirected answer:** applying the edge in both directions gives
  `d(v)<=d(u)+1` and `d(u)<=d(v)+1`, hence `|d(u)-d(v)|<=1`. A frontier can
  touch only the previous, same, or next depth.
- **Directed boundary:** an arc `u->v` gives only `d(v)<=d(u)+1`. It cannot jump
  far forward, but a chain endpoint at depth `k` can have an arc back to the
  source at depth zero.
- **Intuition gained:** BFS layering itself is asymmetric on a directed graph.
  A short rolling visited window is justified by undirected reversibility or a
  separate bounded-backward-span theorem, not by level order alone.
- Promoted the existing result into the mental-model synthesis. No new theorem,
  implementation, code, experiment, Docker action, GPU work, or optimization
  was introduced.

## 2026-08-29: synthesis audit after the intuition traces

- Returned to synthesis after the recent question cards rather than extending
  another lemma chain.
- Checked the added frontier/visited, output, reverse, weighted, Cayley, GPU,
  multi-GPU, and directed-edge traces against the existing mental-model
  contracts. Found no conflicting distance, identity, output, or completion
  claim requiring correction.
- **Coherence gap fixed:** renamed the fixed-count “Twelve false equalities”
  section and added the now-explicit distinctions for bounded miss versus
  unreachable, hops versus heterogeneous cost, independent source frontiers
  versus a joint multi-source frontier, state commitment versus path
  contribution, and total work versus parallelism/time profile.
- **Provenance gap fixed:** the source map still ended at early notes and
  `REF-017`. Extended it through the later detailed notes used by the synthesis
  and distinguished passing bounded evidence `REF-001..044` from the preserved
  `not run` outcomes `REF-045..046`.
- This step added no new BFS theorem, example, code, experiment, Docker action,
  GPU work, implementation, or optimization.

## 2026-08-29: bidirectional meeting versus stopping proof

- **Question:** When does a meeting of forward and reverse BFS waves prove a
  shortest path rather than merely exhibit a path?
- **Existing theorem promoted into the synthesis:** every replay-valid meeting
  gives an upper bound `mu`. In complete-layer unit-cost search, minimum
  unexpanded depths `a,b` give the stopping certificate `a+b>=mu`.
- **Important correction preserved:** when two exact balls were disjoint and
  one complete next layer is generated, the first intersection in that layer
  is already safe for one shortest distance/path. “Never stop at first meeting”
  is too broad.
- **Unsafe boundary:** partial/asynchronous/weighted or distributed in-flight
  work needs a truthful global lower bound; a local first contact is not enough.
  All-shortest-connector outputs may also require completing equal-boundary
  work after one optimal path is known.
- Added this proof pattern to the existing mental-model synthesis. No new
  theorem, code, experiment, Docker action, GPU work, implementation, or
  optimization was introduced.

## 2026-08-29: arbitrary canonicalization can invent a quotient path

- **Question:** Why is merging “symmetric-looking” states not merely a smaller
  visited representation?
- **Smallest useful trace:** concrete edges are `s->a` and `b->t`, with no path
  between `a` and `b`. Declaring `a~b` gives quotient path
  `[s]->[a,b]->[t]` under existential class-edge construction.
- **Failure:** the first quotient edge reaches representative `a`, while the
  second edge exists only from incompatible representative `b`; the quotient
  path has no concrete lift from `s` to `t`.
- **Intuition gained:** BFS can be perfectly exact on the quotient while the
  quotient itself answers an invented or relaxed problem. Safe merging needs a
  transition/path-lifting proof, not only a canonical byte representation.
- **Boundary:** valid automorphism-orbit quotients avoid this arbitrary
  representative mismatch, but naturally compute distance to an orbit; a fixed
  concrete target still needs an additional alignment guarantee.
- Added the trace to the existing mental-model synthesis. No quotient design,
  code, experiment, Docker action, GPU work, implementation, or optimization
  was introduced.

## 2026-08-29: history-dependent legality makes product states

- **Question:** When must two arrivals at the same base configuration remain
  different BFS vertices?
- **Existing minimal counterexample promoted into the synthesis:** edges are
  `s-a->x`, `s-b->x`, and `x-a->t`, while only label word `ba` is accepted.
  Base-only visited can accept arrival `a` first, discard arrival `b`, and erase
  the only accepted path.
- **Intuition gained:** `(x,after_a)` and `(x,after_b)` have the same base state
  and depth but different future languages. When history affects legality or
  acceptance, the exact vertex is a product `(base,memory)` rather than `base`.
- **Boundary:** a previous-move rule used only as proved-safe pruning for
  ordinary unconstrained distance need not change semantic identity. Identical
  filter syntax can describe either a different product graph or an
  optimization; the declared problem and proof decide.
- Added the trace to the existing mental-model synthesis. No new theorem, code,
  experiment, Docker action, GPU work, implementation, or optimization was
  introduced.

## 2026-08-28: post-note-183 coverage and formula audit

- Audited notes 178--183 as six distinct conceptual areas and prohibited another
  synthesis until a theorem changes or evidence status is promoted.
- Corrected backward radial span from a raw maximum to an extended-natural
  supremum including zero and possible infinity.
- Corrected the rolling-window lower index to `max(0,d-L)`.
- Updated the coverage inventory to 183 notes, 1,966 semantic rows, and 43
  experiment references without treating counts as quality evidence.
- Added REF-046 to the blocked-evidence account and retained its Docker
  readiness correction.
- Reordered future Docker work toward semantic publication/reconciliation/
  reclamation fixtures rather than numerical order or arbitrary graph families.
- Selected read-only independent CayleyPy/DeepCubeA action audit as the next
  available high-value step while execution remains unavailable.
- Added no code, Docker retry, benchmark, optimizer, or GPU implementation.

## 2026-08-28: BFS dovetailing and infinite-branching finality

- Split witness discovery, pointwise exact-distance convergence, and certified
  finality for countably branching effective graphs.
- Constructed a fair successor-index dovetailer that eventually visits every
  finite indexed path without completing infinite metric layers.
- Used a late direct root edge versus early length-two path to reject first-hit
  shortestness under fair dovetailing.
- Proved pointwise convergence of fair label-correcting relaxation along a fixed
  finite shortest witness.
- Showed that convergence need not expose a computable finalization event.
- Used indistinguishable finite enumeration prefixes to prove the need for
  extra negative/lower-bound evidence.
- Separated finite semantic state count from terminating successor-occurrence
  presentation.
- Applied the distinction to countably infinite Cayley alphabets while excluding
  ordinary finite-move puzzle graphs from that failure mode.
- Kept finite GPU batches classified as enumerator prefixes, not completed
  infinite levels.
- Added no code, Docker retry, benchmark, optimizer, or GPU implementation.

## 2026-08-28: BFS orders, live boundaries, and pathwidth

- Separated metric layer, physical queue/Open, processed-side live vertices,
  unprocessed-side live vertices, crossing edges, and owner boundaries.
- Defined BFS-constrained vertex separation and bounded it below by ordinary
  pathwidth through order-set inclusion.
- Used a center-rooted star to show layer/Open width `n-1` with left boundary
  one.
- Used complete binary trees to show exponential BFS-constrained boundaries
  despite an `O(h)` unrestricted pathwidth upper bound.
- Separated cutwidth, left/right vertex separation, queue capacity, and note
  179's communication coordinates.
- Clarified that note 181's rolling layers are sufficient certificates, not
  universal minimal boundaries.
- Added output-metadata liveness beyond graph-boundary liveness.
- Split multi-GPU temporal boundaries from static ownership cuts.
- Applied the distinctions to Cayley spheres and Schreier occurrence metadata.
- Added no code, Docker retry, benchmark, optimizer, or GPU implementation.

## 2026-08-28: BFS safe forgetting and rolling visited windows

- Corrected note 45: undirected cycles do not invalidate a strict three-layer
  scalar BFS window because every edge changes source distance by at most one.
- Proved that previous, current, and building-next exact layers suffice for
  undirected duplicate rejection after completed-level publication.
- Separated safe scalar novelty reclamation from preservation of reached sets,
  distances, parents, DAGs, counts, canonical words, and restart artifacts.
- Built directed cycle and arbitrary-depth DAG counterexamples to naive bounded
  forgetting.
- Defined backward radial span and proved the corresponding `L`-window
  sufficiency theorem.
- Derived a Cayley bound from positive-alphabet word lengths of generator
  inverses; symmetric generator sets give span at most one.
- Interpreted frontier search as moving information from Closed membership into
  solid-boundary used-transition metadata.
- Separated delayed duplicate detection from actual old-state reclamation.
- Required a global consistent cut before multi-GPU layer-bit reuse.
- Added no code, Docker retry, benchmark, optimizer, or GPU implementation.

## 2026-08-28: distributed exact BFS set reconciliation

- Defined distributed equality as emptiness of an exact semantic symmetric
  difference, relative to the requested BFS output.
- Applied deterministic equality communication complexity to reject universal
  fixed-size exact digests for arbitrary frontier membership vectors.
- Showed how identically sharded injective-rank bitmaps allow wordwise exact
  comparison plus a one-bit global mismatch reduction without central gather.
- Extended exact comparison to canonical full-state sort/merge and
  collision-resolving verifier maps for wide implicit states.
- Required a validation-specific semantic sharding epoch when runtime GPU/owner
  counts differ.
- Separated local comparison exactness from normalization-shuffle completeness;
  an omitted whole shard can evade every local comparator.
- Classified Merkle roots and IBLT reconciliation as conditional/probabilistic
  unless finite injectivity, exact leaf validation, or fallback closes them.
- Added replayable symmetric-difference witnesses and common-mode oracle bugs to
  the validation ladder.
- Added no code, Docker retry, benchmark, optimizer, or GPU implementation.

## 2026-08-28: BFS cuts, information, and protocol communication

- Separated active cross-owner edge occurrences, receiver information deficit,
  and physical protocol traffic.
- Proved that no graph-only nonzero traversal-communication lower bound exists
  without constraints on initial knowledge, replication, redundant work,
  output placement, uncertainty, or rounds.
- Derived the conditional `ceil(log2 binomial(N,k))` lower bound for an arbitrary
  exact `k`-subset of a known `N`-state universe.
- Used one cross-owner star with two adjacency placements to show identical
  graph cuts can induce different information obligations.
- Used two subgraphs joined by one bridge to separate cut volume from causal
  amplification and owner activation.
- Introduced per-level coordinates for occurrences, unique states, owner pairs,
  payload bits, actual bytes, and rounds.
- Located producer filtering, owner dedup, replication, output metadata, and
  compression at different traffic stages.
- Interpreted 1D/2D distributions as rearrangements of initial graph knowledge
  and collective structure rather than universal winners.
- Applied the distinction to implicit and Cayley graphs, where shared generator
  actions permit recomputation but do not make frontier/visited state common.
- Added no code, Docker retry, benchmark, optimizer, or production design.

## 2026-08-28: REF-046 publication-interleaving gate did not run

- Tried to open note 178's bounded Rust/Docker evidence gate.
- Corrected an empty `docker ps` result: it did not positively establish a
  reachable Docker server.
- Confirmed the blocker with `docker info --format "{{.ServerVersion}}"`, which
  returned exit code 1 and named-pipe permission denied.
- Stopped before writing tests or implementation because the mandatory Docker
  RED step could not run.
- Preserved the intended finite model and unknowns in REF-046.
- Per scope, performed no Docker repair, host execution, optimizer work, or GPU
  implementation.

## 2026-08-28: discovery, publication, and helpable commit

- Separated exact novelty claim, authoritative frontier publication, and
  expansion retirement into one explicit state machine.
- Strengthened the invariant from `x in visited` to publication/expansion or a
  live recoverable publication obligation for every accepted state.
- Used `s->a->b->c` to show how an exact visited claim plus blind retry drop can
  make `c` falsely unreachable.
- Classified four sufficient coupling patterns: atomic joint commit, helpable
  descriptor, log-before-claim, and conserved publication credit.
- Separated detection of an orphaned duty from liveness/recovery of that duty.
- Required payload-before-status visibility across host/device/network
  boundaries without prescribing a concrete memory model.
- Kept schedule depth and richer output merge obligations separate from mere
  set publication.
- Grounded consistent-cut and diffusing-computation interpretations in the
  primary Chandy--Lamport and Dijkstra--Scholten works.
- Defined a future bounded Rust interleaving gate but ran no code, Docker repair,
  optimizer, benchmark, or GPU implementation.

## 2026-08-28: coverage audit and anti-drift correction

- Audited 176 prior notes, 1,862 unique semantic claim IDs, and 42 experiment
  IDs as inventory rather than proof of completeness.
- Separated conceptual coverage, bounded evidence, and real target-runtime
  evidence.
- Corrected stale gaps: cost coordinates and scaling regimes are conceptually
  covered by notes 165--166, while their real measurements remain missing.
- Reframed termination work around notes 56 and 173--174 and preserved the
  absence of failure-injection runtime evidence.
- Kept REF-010/023 classified as one-process distributed simulation rather than
  real multi-GPU evidence.
- Added explicit anti-drift rules against duplicate synthesis, unsolicited
  optimization, production implementation, and Docker repair.
- Preserved Rust for research/host work, C++ only for explicitly requested GPU
  translation units, and Docker-only executable work.
- Turned the remaining work into narrow evidence-promotion gates rather than an
  implementation backlog.

## 2026-08-28: BFS semantics versus schedule

- Reframed BFS as iteration of metric balls `B_(d+1)=B_d union N(B_d)` and
  frontiers as consecutive differences, separating the computed object from
  the FIFO mechanism commonly used to schedule it.
- Derived soundness and completeness as separate proof obligations. Exact
  visited membership alone is insufficient: first discovery is shortest only
  under a nondecreasing-layer schedule or a relaxation scheme that permits
  correction.
- Catalogued counterexamples for stack scheduling, weighted edges, hash
  collisions, omitted current-layer visited state and capacity-consuming races.
- Clarified that enqueue-time, dequeue-time and level-batched visited policies
  can all preserve distances under different conditions, while having radically
  different work/capacity behavior.
- Connected Cayley BFS to the word metric: generator relations are the semantic
  source of duplicate words representing one group element.
- Corrected the repository roadmap per user direction: GPU code is now limited
  to minimal explanatory probes unless substantial implementation is explicitly
  requested.

## 2026-08-28: frontier, candidates, and visited

- Separated the mathematical frontier set, physical candidate occurrence bag,
  unique candidate set, next frontier, and accumulated visited ball. Treating
  them as one “queue” hides both correctness and work.
- Interpreted visited as an exact certificate of the completed metric ball, not
  a cache. False positives can delete reachability; false negatives primarily
  create recoverable duplicate work until capacity is affected.
- Classified duplicates by generator redundancy, convergent parents,
  current-layer hits, earlier-layer hits, cross-partition convergence, and
  representation collisions. Only the first five are graph/work duplicates.
- Clarified which observations are invariant under permutation of a complete
  layer (distances and sets) and which are not (parents, early-stop work,
  locality, capacity pressure, routing and next-level order).
- Distinguished frontier width, edge volume, candidate occurrences, unique
  candidates and accepted states as separate workload dimensions.
- Connected distributed visited to authority: local source dedup is optional
  work reduction, while owner-local membership is the exact global decision.

## 2026-08-27: initial orientation

- Separated general BFS from the small solved-neighborhood BFS used by a beam
  search implementation.
- Established level-synchronous frontier and visited invariants.
- Identified the main portability hazard: classical GPU BFS literature usually
  assumes explicit CSR graphs and compact vertex IDs.
- Collected primary starting sources: direction-optimizing BFS, Merrill GPU
  traversal, Graph500, and Gunrock.
- Defined a staged route from CPU reference correctness to implicit and
  multi-GPU experiments.

No implementation or benchmark has been completed yet.

## 2026-08-27: former optimization objective (superseded)

- Historical record: the objective was temporarily expanded from algorithm
  orientation toward maximally efficient exact BFS on one and many GPUs.  This
  direction is superseded by the current study objective above: understand the
  algorithm, preserve observations, and use only bounded explanatory probes
  unless substantial implementation is separately requested.
- Added an evidence taxonomy so facts, hypotheses, observations, decisions, and
  failures remain distinguishable.
- Defined correctness and performance gates for future comparisons.
- Required negative and inconclusive experiments to remain in the record.
- The performance-specialization idea remains a conceptual hypothesis, not an
  implementation backlog.

## 2026-08-27: REF-001 CPU oracle

- Added a deterministic level-synchronous CPU BFS over an injected neighbor
  function, so it supports explicit and small implicit graphs.
- Added an independent semantic validator rather than comparing an
  implementation only with its own generated output.
- Confirmed handling of self-loops, duplicate edges, duplicate sources, cycles,
  and multiple shortest-parent opportunities.
- Added negative fixtures for a wrong-level parent, a silently dropped reachable
  vertex, a connected but non-shortest tree, and frontier/distance disagreement.
- Observation: parent validity alone does not prove shortest distances. Checking
  each explored edge for `dist[child] <= dist[parent] + 1` is necessary to catch
  a longer-but-internally-consistent tree.
- The oracle currently validates only complete traversal. Bounded-depth and
  early-target termination need explicitly different validation semantics.

## 2026-08-27: REF-002 implicit symmetric groups

- Added move-aware implicit traversal, parent generator metadata, path
  reconstruction, replay, and a labeled-result validator.
- Enumerated the full Cayley graphs of `S3` through `S8` using adjacent
  transpositions and verified every result with the semantic validator.
- Observed the expected vertex counts `n!` and maximum depths
  `n(n-1)/2`, providing two independent structural checks for this generator
  family.
- The fraction of generated transitions that were not tree discoveries rose
  from `0.5833` for `S3` to `0.8571` for `S8`.
- Inference: as traversal approaches exhaustive coverage of a regular finite
  graph, visited/dedup work necessarily dominates successful insertions. This
  does not by itself select sort versus hash visited, but it rules out treating
  duplicate handling as a rare slow path.
- Failure recorded: WMI CPU-name collection through `Get-CimInstance` returned
  access denied. The fallback captured processor family/model, OS, Python
  version, and logical CPU count, but not the marketing model name.

## 2026-08-27: REF-003 level-wise rejection decomposition

- Split every `S8` level into generated transitions, unique candidate states,
  batch duplicate occurrences, exact visited hits, and accepted states.
- At the peak frontier (depth 14), 26,852 transitions collapsed to 7,472 unique
  candidate states before consulting earlier visited levels. Thus 19,380
  transition occurrences, about 72.17%, were removable by exact same-batch
  deduplication.
- Of the 7,472 unique candidates at depth 14, exactly 3,736 were already visited
  and 3,736 formed the next frontier.
- Observation: candidate-batch dedup and authoritative visited filtering remove
  different kinds of work and should remain separate metrics and design phases.
- Inference: for this generator family, pre-communication dedup has a large
  theoretical byte-reduction opportunity near the widest levels. Whether GPU
  sort/unique costs less than the avoided exchange remains unmeasured.

## 2026-08-27: REF-004 generator-set effects

- Compared four exact `S8` Cayley traversals: adjacent transpositions, identity
  added, a duplicate adjacent transposition added, and a 3-cycle/inverse pair
  added.
- Adding identity created exactly one unique same-level visited hit per state and
  did not increase exact batch-duplicate occurrences.
- Adding a duplicate generator created exactly one extra batch-duplicate
  occurrence per state and did not create same-level hits.
- Adding a 3-cycle/inverse pair made the graph non-bipartite, reduced diameter
  from 28 to 22, increased peak frontier from 3,836 to 4,420, and created 36,897
  unique same-level hits.
- Insight: total rejection ratio hides where work can be removed. Generator
  relations determine whether overhead is best attacked during generation,
  batch dedup, or authoritative visited lookup.

## 2026-08-27: REF-005 owner-routing simulation

- Simulated exact owner-computes BFS over the recorded `S8` levels for 1, 2, 4,
  and 8 logical ranks.
- Used exact Lehmer rank for state identity and compared direct `rank % P` with
  SplitMix-style avalanche mixing before modulo.
- As ranks increased, more duplicates crossed source-rank boundaries and became
  removable only after owner routing: 53,704 at 2 ranks, 116,402 at 4, and
  133,886 at 8 for direct rank modulo.
- Direct rank modulo preserved useful ownership locality and reduced remote
  payload, but had worse frontier imbalance away from the peak. Mixed rank was
  substantially more balanced on large levels but approached the random-routing
  remote fraction `1 - 1/P`.
- Insight: ownership is a three-way trade among balance, communication locality,
  and the amount of duplicate elimination available locally before exchange.
  Uniform final visited counts do not prove balanced per-level work.
- Failure recorded: the first simulator launch had a PowerShell/Python quoting
  error and produced no data. The corrected run used `str.format`.
- Reconciliation caught one inconsistent derived remote fraction. Independent
  integer-count recomputation established the direct-modulo value as
  `53,760 / 134,342`; the correction and both raw ratios are retained in REF-005.

## 2026-08-27: REF-006 partition Pareto study

- Compared 35 deterministic 8-rank owner functions on the same exact levels:
  direct modulo, contiguous rank ranges, multiplicative high bits, and 32 salted
  avalanche mappings.
- Rejected the hypothesis that one tested mapping dominates the others across
  frontier balance, receive balance, remote fraction, cross-rank duplicates,
  and final visited capacity.
- Contiguous ranges minimized remote fraction (`0.3334`) and cross-rank
  duplicates (`40,306`) but reached `4.51x` frontier and `4.00x` receive skew.
- Multiplicative high bits minimized both measured large-level imbalance metrics
  (`1.20x` frontier, `1.12x` receive) but raised remote fraction to `0.9117`.
- Under an explicit example constraint of frontier imbalance `<=1.30` and
  receive imbalance `<=1.50`, only three tested strategies were feasible; the
  lowest-traffic one was salted mixer 29 at remote fraction `0.8745`.
- Insight: partition selection must state operational constraints or a workload
  objective. Calling a hash "balanced" is not enough to choose it for BFS.

## 2026-08-27: REF-007 exact bidirectional BFS

- Added a deterministic bidirectional BFS reference that expands the smaller
  frontier and stops only when the sum of minimum unexpanded depths reaches the
  best known meeting distance.
- Defined reverse transitions precisely: `(forward_move, predecessor)` must
  mean that applying `forward_move` to `predecessor` reaches the current reverse
  state. This makes the suffix directly replayable.
- Exhaustively checked all 49,152 nontrivial ordered pairs across every directed
  loop-free graph on four vertices: zero distance/found mismatches and zero
  replay failures, including 12,288 unreachable pairs.
- Checked all 576 ordered pairs in the adjacent-transposition `S4` Cayley graph:
  zero distance mismatches and replay failures.
- In an `S8` one-target-per-depth sweep, bidirectional work saved about 86.5% of
  level-complete unidirectional generated transitions at depth 14, but only 9.5%
  at the diameter endpoint depth 28.
- Insight: bidirectional BFS is not automatically exponentially better at the
  hardest diameter endpoints of a finite graph. Benefit depends on frontier
  growth and how much of both search balls must be constructed before meeting.

## 2026-08-27: REF-008 target-stop granularity

- Added exact unidirectional target BFS with candidate, fixed parent-batch, and
  full-level stopping semantics.
- Exhaustively validated all three modes over 49,152 ordered pairs from every
  directed loop-free graph on four vertices, with zero distance mismatches or
  replay failures.
- On the selected S8 depth-14 target, level completion generated 127,694
  transitions versus 101,544 for immediate candidate stop; a 32-parent batch
  was within about 0.22% of candidate stop.
- On the diameter-28 target, only 48 transitions separated candidate and level
  stop, while bidirectional BFS retained its roughly 9.5% saving.
- At depth 2, candidate-stop unidirectional BFS beat bidirectional BFS, 9 versus
  14 transitions. The algorithm choice therefore depends on target depth and
  stop latency, not only graph branching factor.
- Insight: a GPU target-search result must name its cancellation granularity.
  Candidate-stop work is an order-dependent lower reference, while batches or
  already-issued kernels better describe realizable parallel work.

## 2026-08-27: REF-009 bidirectional expansion policies

- Generalized the exact reference to smaller-frontier, strict-alternating, and
  supplied estimated-work side selection while preserving complete levels and
  the same lower-bound stopping proof.
- Across all 49,152 nontrivial ordered pairs in the four-vertex directed corpus,
  every policy had zero distance and replay errors.
- Exact degree-work selection generated 76,416 transitions versus 95,232 for
  alternation and 102,912 for smaller frontier. It was never worse on this tiny
  corpus and uniquely best on 13,440 pairs, but estimator and reduction costs
  were deliberately excluded.
- On the regular symmetric adjacent-transposition S8 graph, all policies were
  identical at every sampled depth. Fixed degree makes work proportional to
  frontier cardinality, and the two frontier profiles alternate symmetrically.
- Insight: a sophisticated side selector is workload-dependent overhead. For
  multi-GPU regular Cayley BFS, strict alternation can eliminate a policy
  all-reduce without changing search work; irregular graphs need end-to-end
  timing before making that choice.
- Failure recorded: direct file execution could not import the repository
  package. Re-running the reproducible driver as a module succeeded.

## 2026-08-27: REF-010 distributed bidirectional owner routing

- Added a bulk-synchronous owner-computes simulator with per-round lossless
  accounting of source pre-dedup, remote routing, owner convergence,
  authoritative visited lookup, acceptance, and intersection.
- Ran 294,912 searches across every nontrivial pair of every four-vertex
  directed loop-free graph, three world sizes, and two side policies. Distance,
  replay, and accounting failures were all zero.
- For S8 at depth 28 and eight ranks, direct Lehmer ownership sent 123,388
  remote candidates after source pre-dedup versus 188,615 for mixed ownership,
  a 34.58% reduction.
- Increasing rank count moved duplicate convergence downstream: in the same
  direct depth-28 run, source pre-dedup fell from 182,222 removed occurrences at
  P=1 to 61,218 at P=8, while owner-side duplicate removal rose from zero to
  121,004.
- Mixed ownership improved balance on earlier substantial frontiers. Measuring
  only the widest frontier hid direct ownership's worse intermediate skew.
- Insight: identical owner functions for both directions turn intersection
  detection into a local lookup at the authoritative owner. Different mappings
  would require another distributed join or replicated opposite-side metadata.
- Boundary: this is a correctness and traffic-volume model, not evidence of GPU
  throughput, collective latency, or scalability.

## 2026-08-27: REF-011 wire-record strategies

- Added byte-exact payload accounting for eager full records and two-phase
  key/acceptance-bitmap/deferred-parent exchange across four illustrative wire
  widths and 120 S8 routing configurations.
- Derived the exact byte criterion `bitmap < rejected_remote * metadata_width`.
  This separates the benefit from intuition about "smaller messages."
- Rejected the hypothesis that two-phase transfer is universally better. At
  depth 2 every remote candidate was accepted, so all 24 configurations paid
  bitmap overhead without suppressing metadata.
- At depth 28/P8, two-phase reduced modeled bytes by 56.39% for direct/rank16
  and 46.63% for mixed/state128. Direct ownership also sharply reduced the
  fraction of accepted states needing a remote parent record.
- An oracle per-round hybrid used eager for the first two rounds and two-phase
  later in every deeper P8 search, eliminating shallow regressions. A realizable
  hybrid still needs an online predictor and latency measurements.
- Insight: parent tie-breaking is a communication policy. Preferring an owner-
  local producer preserves shortest distance while eliminating a remote parent
  fetch for that state.
- Boundary: byte savings do not prove speedup; the second phase, retained source
  buffers, packing kernels, and transport headers remain unmeasured.

## 2026-08-27: REF-012 Docker Rust-to-CUDA smoke

- Adopted the user-required boundary: all build/run/profiling in Docker, Rust
  for host orchestration and validation, and C++ only inside CUDA translation
  units behind a narrow C ABI.
- Reused the existing CUDA 12.8.1/CUTLASS/Nsight image and added a cached Rust
  1.75 builder layer instead of duplicating the large CUDA toolchain.
- Built a native `sm_86` shared CUDA library and a Rust executable, ran it on
  the RTX 3070 Laptop GPU, and validated 1,048,576 results in Rust.
- Verified the native cubin and kernel symbol with `cuobjdump`, excluding a
  PTX-only accidental build.
- Failure recorded: the first container build declared an unnecessary CMake
  3.24 minimum against image version 3.22.1. Correcting the true minimum made
  the rebuild pass.
- Host CuPy was found unusable due to a missing CUDA 11.2 NVRTC DLL; it is not
  part of the Docker baseline. Host PyTorch worked but is likewise not the
  implementation path.
- Defined the first performance contract around exact visited/dedup backends,
  fixed capacity, device-resident timing, CPU-oracle validation, and explicit
  failure rows.

## 2026-08-28: REF-013 exact bitmap visited baseline

- Added a persistent fixed-capacity CUDA bitmap context behind the Rust-owned C
  ABI, with explicit invalid-key and output-overflow semantics.
- Validated full accepted sets for reduced edge cases and a 4.19-million
  candidate workload, then passed memcheck, racecheck, initcheck, and synccheck
  on reduced fixtures with zero reported errors.
- Swept four batch sizes and four rejection/concentration profiles. At 16.78
  million candidates, medians ranged from 30.62 billion candidates/s for
  distributed already-visited keys to 1.65 billion/s when every thread
  contended for one key.
- Insight: duplicate ratio and accepted fraction omit spatial concentration.
  Atomic hotspot structure must be a first-class BFS workload metric.
- `all-new` reached 7.94 billion candidates/s but paid a global output
  reservation and full output writes. The baseline needs block compaction before
  attributing a definitive cost to bitmap membership itself.
- Isolated kernel and Rust iteration measurements differed by roughly 2-18x in
  the large rows because the latter resets state and transfers pageable host
  buffers. Neither is yet an end-to-end GPU-resident BFS timing.
- Decision: preserve this exact workload and validation contract for the warp-
  aggregated bitmap, CUB sort/unique, and 64-bit hash comparisons.

## 2026-08-28: REF-014 conditional bitmap optimizations

- Added four exact CUDA paths behind the same Rust-owned context: baseline,
  warp equal-key aggregation, CUB block compaction, and their combination.
- A Rust/Docker artifact validator established complete `4 x 4 x 4` sweep
  coverage and identical accepted count/fingerprints for every comparable row.
  All paths also passed four Compute Sanitizer tools.
- Warp aggregation reduced the 16.78-million single-key kernel from 10.154 ms
  to 0.373 ms (27.2x), but was 2.5% slower for spaced fourfold duplicates and
  4.3% slower for distributed already-seen keys.
- Insight: global multiplicity is not the right dispatch metric. Equal-key
  co-location within warps is what predicts whether `match_any` removes bitmap
  atomics. Candidate ordering is therefore part of the BFS performance model.
- Block compaction was 4.0% slower on the largest all-new case. Correction
  2026-08-31: this shows a net regression, not that the output-counter bottleneck
  was absent. Scan/barrier cost outweighing savings is an interpretation;
  retained aggregate timings do not establish that causal decomposition.
- Rejected universal warp, block, and combined defaults. Retain baseline for
  broad batches and treat warp aggregation as a measured conditional path.
- Recorded orchestration failures: two wrong Dockerfile paths and one missing
  sanitizer entrypoint override. Corrected Docker-only commands passed.
- Next: charge the complete CUB radix-sort/unique pipeline and test whether its
  duplicate grouping can repay sorting and temporary-storage costs.

## 2026-08-28: REF-015 complete sort/unique pipeline

- Added a persistent CUB radix-sort/unique backend with Rust lifecycle, exact
  bitmap claims, phase/total CUDA events, explicit capacity errors, and exact
  device-memory accounting. No host synchronization separates its GPU phases.
- The Rust oracle checked full sets, persistent visited state, empty input,
  overflow and invalid keys. All 16 rows matched REF-014 counts/fingerprints;
  four Compute Sanitizer tools were clean.
- At `2^24`, sort/unique took 3.966 ms all-new, 2.216 ms fourfold, 1.793 ms
  all-seen, and 1.662 ms single-key. An independent repeat agreed within 1.17%
  on all four large cases.
- Rejected it as the dense 32-bit default: it was 1.88-3.66x slower than the
  best direct bitmap path on three broad profiles and used 324.04 MiB device
  memory. It beat naive single-key atomics 6.11x but lost to warp aggregation
  4.46x.
- Insight: duplicate convergence has a locality hierarchy. Warp aggregation is
  cheap/local, sorting is costly/global, and visited atomics are direct but can
  serialize. Dispatch must model where duplicates meet, not just their count.
- Boundary: sort may be justified for non-rankable wide keys or if sorted order
  is reused for owner routing. Neither is established by this experiment.

## 2026-08-28: REF-016 Cayley successor locality

- Reimplemented complete adjacent-transposition S8 BFS and Lehmer ranking in
  Rust. The rank bijection is exhaustively tested in the Docker build; traversal
  reproduces 40,320 states, diameter 28, peak frontier 3,836 and 282,240 edges.
- Measured identical exact frontier sets in parent/generator-major layouts under
  rank-sorted, discovery, and deterministic hash-shuffled frontier orders. A
  Rust validator checked all 174 rows and level-wise conservation identities.
- Parent-major discovery order exposed 48,713 equal-key warp savings (17.26% of
  candidates); rank order exposed 28,290 and shuffled order only 861. The same
  exact traversal has 201,602 global duplicate occurrences in every case.
- Generator-major layout exposed only 31-57 warp-local duplicates overall.
  Because each generator is a bijection, equal children occur across generator
  slices and are normally separated by an entire frontier.
- Insight: frontier ordering is performance-relevant state carried between BFS
  levels. Nondeterministic atomic compaction can change the next level's
  locality even when the current kernel and exact frontier set are correct.
- Corrected near-miss: the initial rank-only artifact hid its sorted-frontier
  assumption. It was replaced with v2 controls before drawing conclusions.
- Next: run consecutive GPU generation/filter levels at S9/S10 scale so layout,
  output ordering, visited cost, and downstream locality are timed together.

## 2026-08-28: REF-017 fused exact GPU S9 BFS

- Added a fixed-capacity fused CUDA traversal: packed permutation generation,
  Lehmer rank, optional warp aggregation, exact bitmap visited and next-frontier
  append occur without materializing candidates. Rust owns all host behavior.
- Every S9 frontier in four configurations matched Mahonian counts, unique ranks,
  inversion depth and cross-configuration fingerprints. Full S8 traversals also
  passed four Compute Sanitizer tools.
- Actual parent-major atomic-output frontiers retained about 334k warp-local
  duplicate claims (11.5% of 2.90m transitions); generator-major retained only
  31-40. The layout result from REF-016 survives concurrent GPU compaction.
- Ten warmed repetitions observed parent-major warp medians of 0.515 ms kernel
  sum / 3.462 ms traversal versus baseline 0.541 / 3.909 ms. Generator-major
  warp regressed to 0.606 / 4.361 ms versus baseline 0.519 / 3.812 ms.
- Insight: the same `match_any` instruction is optimization or overhead based on
  layout. Dispatching solely by graph/global duplicate ratio is invalid.
- Host-observed traversal was 6.7-7.3x the summed kernel intervals on S9.
  Correction 2026-08-31: resets, copies, synchronization, launches and host
  bookkeeping are mixed in that residual; synchronization cost was not isolated.
  Device-driven control was a historical investigation idea, not a measured
  solution or a currently authorized optimization task.
- Failure description retained: the first no-warmup sweep mixed CUDA cold-start
  into one configuration. It was rejected and overwritten, then replaced by
  oracle + 2 warmups + 10 runs. Correction 2026-08-31: the first raw sweep is
  not retained as a separately identifiable artifact; prose is not raw evidence.
- Boundary: timing ranges overlap and clocks are unlocked; S10 must establish
  whether the modest parent-major win survives larger saturated levels.

## 2026-08-28: taxonomy of BFS variants and boundaries

- Classified BFS-related algorithms by the object that changes: source set,
  termination contract, search direction, neighbour-enumeration direction,
  execution schedule, edge-cost model, or requested output.
- The queue is not the defining property.  Exact ordinary BFS is identified by
  the hop-distance layers it certifies; an alternative schedule must still
  prove those layers or supply a different shortest-path proof.
- Multi-source BFS computes distance to a source set.  Source ownership is a
  separate, tie-sensitive output even when all distances are deterministic.
- Bidirectional search needs reverse edges on directed graphs and a stopping
  lower bound.  Alternation, smaller-frontier selection, and first contact are
  policies rather than standalone proofs.
- Push/pull changes enumeration of the same next-frontier predicate.  Pull's
  need for an enumerable unvisited universe explains why it transfers poorly
  to many implicit state spaces.
- 0-1 BFS is a relaxation-based weighted shortest-path method; LexBFS computes
  an ordering; beam search prunes away the exact BFS guarantee.  Similar names
  or frontier-shaped control flow do not make their semantics equivalent.
- No implementation was added.  This entry deliberately redirects work from
  unsolicited optimization toward conceptual distinctions and proof duties.

## 2026-08-28: explicit, implicit, and Cayley graph model

- Unified the three presentations behind a graph-oracle contract: exact state
  identity, sources, and complete outgoing-transition enumeration.  The BFS
  layer theorem does not depend on how the oracle is represented.
- Identified dense vertex IDs simultaneously serving as state, visited index,
  frontier payload, adjacency key and owner key as an explicit-graph
  convenience—not a BFS invariant.
- In implicit graphs the expansion procedure is part of the graph
  specification.  State counts and plausible paths are weak evidence; small
  exhaustive models, replay, inverse properties, and deliberate collision
  tests address different failure modes.
- Interpreted Cayley BFS as word-metric computation.  Generator choice,
  inverse closure, left/right action, labels, multiplicity, and relations are
  semantic parts of the graph rather than mere kernel parameters.
- Separated full state, equality key, dense rank, frontier payload, parent
  record, wire record, and ownership key.  A bijective rank is exact identity
  but may still be insufficient or expensive for successor generation.
- Recorded counterexamples for hash-as-identity, inverse-generators-as-pull,
  duplicate labeled-generator removal, quotienting without path lifting, and
  rank spaces that include unreachable states or alias reachable ones.
- Used adjacent transpositions as a proof-bearing example: inversion count is
  exactly word length, while the local S8 enumeration independently validates
  the resulting 40,320-state, diameter-28 traversal.
- No new implementation or performance optimization was introduced.

## 2026-08-28: evidence map and scope audit

- Reorganized accumulated knowledge into definitions, proved facts, validated
  finite-scope facts, observations, inferences, rejected universal claims, and
  unknowns.  Chronology remains in this log; claim status now lives in
  `docs/evidence-map.md`.
- Attached an explicit scope to every experimental conclusion.  In particular,
  synthetic dense-key probes, adjacent-transposition S8/S9, CPU routing models,
  and RTX 3070 Laptop timings are not generalized to all BFS workloads.
- Preserved five orchestration/measurement failures and explained which claim
  each could otherwise have contaminated.
- Consolidated the strongest cross-cutting lesson: correctness, mathematical
  work, physical ordering, representation, and hardware cost are separate axes.
- Recorded six highest-value gaps without converting them into an implementation
  backlog.
- Corrected the research protocol so GPU/multi-GPU work is framed as conceptual
  and measurement-based study rather than a high-performance implementation
  objective.

## 2026-08-28: conceptual GPU and multi-GPU cost model

- Mapped the exact BFS transformation into six physical cost layers: expansion,
  identity/visited, duplicate convergence, compaction/capacity, parent metadata,
  and level control.  Fusion may combine these layers but cannot remove their
  logical obligations.
- Replaced one-dimensional TEPS reasoning with a per-level work vector covering
  frontier size, generated occurrences, unique candidates, visited hits,
  accepted states, record bytes, and synchronization.
- Distinguished explicit irregular edge scheduling from regular-count implicit
  generation.  Constant Cayley degree removes one form of imbalance but not
  identity locality, ranking, legality, or relation-driven convergence.
- Clarified that a race is benign only relative to the output and capacity
  contract.  A nondeterministic parent can preserve one shortest tree while
  violating deterministic or all-shortest-path requirements.
- Decomposed multi-GPU ownership across adjacency, full state, visited, frontier,
  parent metadata, routing buffers, and termination state.  "Vertex ownership"
  without these distinctions is underspecified.
- Separated 1D/2D stored-graph partitioning from owner-computes implicit search;
  the questions transfer, but a sparse-matrix protocol cannot be copied
  literally into a graph whose edges are generated.
- Defined strong, weak, and capacity scaling as different claims and recorded
  counterexamples to more-candidates, fewer-atomics, sort-wins, regular-degree,
  linear-scaling, and API-implies-overlap intuitions.
- Added a measurement ladder that prevents isolated primitives or simulated
  wire bytes from being reported as traversal or multi-GPU speed evidence.
- No code, kernel, backend, or new optimization was introduced.

## 2026-08-28: bidirectional meeting and stopping proof

- Reframed bidirectional BFS as an upper/lower-bound algorithm: every valid
  meeting gives a feasible-path upper bound `mu`, while the unfinished search
  state must provide a lower bound before termination is justified.
- Proved the complete-level rule used by the local references.  If the current
  exact discovered balls have radii `a` and `b`, a known path is optimal once
  `a+b>=mu`.
- Corrected an overbroad warning about "first intersection."  When two exact
  balls are initially disjoint and one complete next layer is generated, every
  first intersection in that layer has the same depth sum and is sufficient for
  one shortest distance/path.  The unsafe cases are partial/asynchronous
  layers without a bound, weighted labels, the wrong reverse graph, or
  distributed notification before global convergence.
- Separated shared-vertex and crossing-edge candidates, and explained why they
  coincide on unit first discovery but differ for weighted or partially settled
  searches.
- Distinguished distance-optimal termination from enumeration-complete
  termination: equality-boundary work may still contain additional shortest
  parents or meeting edges.
- Audited the local code variables against the proof: each distance map is an
  exact ball at loop boundaries, each frontier is its minimum unexpanded layer,
  and side policies advance only complete levels.
- Extended the proof obligations to implicit reverse moves and distributed
  owner/epoch completion without designing a new implementation.

## 2026-08-28: completeness and termination boundaries

- Split the vague claim "BFS is complete" into solution completeness,
  enumeration completeness, and decision termination.  They coincide for a
  finite effective reachable graph but diverge on infinite graphs.
- Proved by finite-ball induction that local finiteness is enough to find every
  reachable finite-depth target after finite work; a uniform maximum branching
  factor is needed for the familiar geometric work bound, not for eventual
  discovery itself.
- Recorded the infinite-ray counterexample: BFS can be complete for reachable
  targets yet run forever rather than certify an unreachable target.
- Distinguished infinite depth from infinite branching.  An infinite successor
  enumeration can prevent a strict level-synchronous BFS from ever reaching
  depth two, even when a target there exists.
- Separated graph-search and tree-search BFS: finite branching preserves
  shallow-target discovery for the search tree, while exact visited is what
  permits finite cyclic graph exhaustion.
- Made fairness an explicit liveness obligation independent of correct distance
  assignment, including parallel/distributed starvation cases.
- Connected finitely generated Cayley graphs to finite metric balls, while
  noting that an abstract finite group presentation need not provide decidable
  equality; effective state identity remains a required graph-oracle operation.
- Introduced four noninterchangeable run outcomes: `FOUND`, `EXHAUSTED`,
  `BOUNDED`, and `INCOMPLETE`.
- No implementation or performance optimization was added.

## 2026-08-28: frontier growth and metric-ball geometry

- Reframed frontier width as the outer vertex boundary of the accumulated
  metric ball, distinct from generated edge/move occurrences and unique
  neighbor identities.
- Defined per-level width, volume, transition work, unique candidates, visited
  hits, growth ratio, and discovery yield with exact conservation equations.
- Contrasted tree-like exponential growth, polynomial lattice/Cayley growth,
  bottleneck-and-burst profiles, and finite saturation.  Branching factor is a
  word-tree quantity unless convergence is accounted for.
- Derived the adjacent-transposition `S_n` frontier polynomial as the Mahonian
  q-factorial and explained the exact `S_8` layer symmetry around diameter 28.
- Used bipartiteness and involutive adjacent swaps to strengthen the local
  accounting to `7*w_d = duplicates + w_(d-1) + w_(d+1)`; a read-only Docker
  check passed this equality for all 29 retained REF-003 levels.
- Interpreted REF-004 geometrically: identity and duplicate generators preserve
  spheres while adding occurrences; the 3-cycle pair changes the metric,
  reduces diameter, broadens the peak, and creates same-level edges.
- Separated frontier, raw-candidate, cumulative-visited, parent, and scratch
  capacity peaks, and connected bidirectional work to two-ball volume rather
  than the tree heuristic alone.
- Recorded counterexamples to degree-predicts-width, duplicate-ratio-predicts-
  growth, midpoint-peak, lower-diameter-means-less-work, and vertex-transitive-
  means-tree-like intuitions.
- No new experiment, implementation, or optimization was introduced.

## 2026-08-28: shortest-path trees, DAGs, and path counts

- Separated canonical distance labels, one selected BFS tree/forest, the full
  shortest-path predecessor DAG, and explicit all-path enumeration as four
  increasingly rich output contracts.
- Defined shortest predecessor edges by `d(u)+1=d(v)` and proved that depth
  increase makes their graph acyclic.  One BFS tree selects one predecessor per
  non-source vertex.
- Recorded the path-count recurrence `sigma(v)=sum sigma(u)` over predecessor
  edges and a layered counterexample with `O(k)` DAG size but `2^k` shortest
  paths.
- Connected Cayley parents to geodesic words/reduced decompositions.  Duplicate
  labeled generators can change path multiplicity without changing distances
  or predecessor vertices.
- Explained why a locally valid parent chain only proves an upper bound on true
  distance; edge inequalities and reachability closure supply the missing
  shortestness direction used by REF-001.
- Made deterministic parent choice an explicit reduction/tie-break algorithm,
  not an automatic consequence of exact distances or parallel first discovery.
- Reclassified same-depth duplicate candidates: they may be discarded for next-
  frontier membership but are required predecessor contributions for a
  shortest-path DAG or count.
- Compared GPU metadata contracts and multi-GPU eager/deferred/distributed
  parent strategies without selecting or implementing one.
- Distinguished early distance/path stopping from completing every target
  predecessor and every bidirectional equality connector.
- No code, experiment, or optimization was added.

## 2026-08-28: ordinary BFS versus relaxation-based shortest paths

- Isolated the fact that makes first discovery final in ordinary BFS: every
  later unit-edge proposal is no smaller than the first `d+1` proposal.
- Added a three-edge 0-1 counterexample where visited-on-first-discovery freezes
  distance 1 although a later zero-cost path has distance 0.
- Interpreted the 0-1 deque as two moving monotone distance buckets, not as
  ordinary FIFO BFS with a cosmetic insertion policy.
- Separated undiscovered, tentative, active, and settled states; ordinary BFS
  collapses tentative/final only because its unit-layer proof permits it.
- Related ordinary BFS, 0-1 BFS, Dial buckets, Dijkstra, delta-stepping, and
  Bellman-Ford by their label-finalization and relaxation contracts rather than
  by container names.
- Explained how zero-cost edges create same-distance closure and can turn the
  shortest-predecessor DAG into a cyclic graph with infinitely many shortest
  walks.
- Clarified multi-source super-source equivalence: virtual edges are zero cost,
  while direct depth-zero initialization precomputes that known closure.
- Extended the distinction to weighted Cayley metrics, where longer words can
  be cheaper and zero-cost generators collapse entire cost layers.
- Listed GPU/multi-GPU obligations—atomic/owner minima, reactivation, stale
  work, bucket agreement, and in-flight lower proposals—without implementing a
  weighted backend.
- No code, experiment, or optimization was added.

## 2026-08-28: multi-source distance fields and Voronoi ties

- Distinguished one joint nearest-source wavefront from batching many
  independent BFS traversals.  The first returns a pointwise minimum; the second
  retains one distance dimension per source.
- Defined multi-source balls as unions of single-source balls and clarified the
  zero-cost virtual-super-source equivalence.
- Separated canonical scalar distance from nearest-source labels.  Arbitrary,
  canonical, and set-valued tie contracts return different metadata while
  preserving the same distances.
- Identified equal-distance label improvement as a propagation problem: a
  smaller canonical source arriving later may need to recolor descendants even
  though no distance changes.
- Separated source label from parent selection and showed that coherent
  connected cells require same-label parent chains, not merely pointwise valid
  nearest labels.
- Distinguished `min_s dist(s,v)` from `min_s dist(v,s)` on directed graphs; the
  latter requires reverse expansion from facilities.
- Connected Cayley multi-source waves to unions of translated balls and warned
  that seeding a symmetry orbit changes fixed-goal distance into distance to the
  orbit unless quotient/lifting semantics justify it.
- Separated semantic Voronoi ownership from physical GPU/rank ownership and
  extended tie convergence/termination obligations conceptually to multi-GPU.
- Generalized bidirectional BFS to source and target sets while preserving the
  distinction between optimal scalar distance and all minimizing endpoint
  pairs.
- No code, experiment, or optimization was added.

## 2026-08-28: push/pull equivalence and transfer boundaries

- Derived push and pull from the same exact predicate:
  `v notin B_d` and `exists u in F_d: u->v`.  Mode changes enumeration, not the
  mathematical next frontier.
- Listed the full equivalence conditions: matching edge orientation, complete
  outgoing/incoming enumeration, exact frontier/visited membership, immutable
  level snapshot, and lossless capacity.
- Distinguished push edge occurrences from pull predecessor checks.  Pull work
  depends on unvisited-universe size, predecessor order, hit probability, and
  membership cost rather than frontier size alone.
- Separated pull from backward bidirectional BFS.  Pull reads incoming witnesses
  while advancing the original source distance field; backward BFS owns a
  second target-rooted distance field.
- Made snapshot semantics explicit: exposing newly accepted next-layer states
  as current frontier can cascade multiple hops and corrupt depths.
- Explained why pull's first-parent early exit is valid only for distance/one
  arbitrary parent, not canonical parents, all predecessor edges, path counts,
  or canonical multi-source labels.
- Established the main implicit/Cayley boundary: inverse generators provide
  predecessors of a known state but not an outer enumeration of every unvisited
  state.  Dense rank makes pull possible in principle, not necessarily useful.
- Mapped push/pull to different GPU work signatures and multi-GPU frontier-
  membership obligations without choosing thresholds or implementing a hybrid.
- Recorded counterexamples to pull-is-backward, invertible-implies-pull,
  pull-always-cheaper, dense-rank-implies-efficient, and first-parent-is-enough.
- No code, experiment, or optimization was added.

## 2026-08-28: external-memory BFS as an exact set transaction

- Read primary starting material on external-memory BFS and delayed duplicate
  detection, then asked the `multigpu_beam` expert for conceptual failure modes.
- Separated same-layer candidate deduplication from subtraction against old
  visited states. Either physical ordering can be exact if the final set is
  `unique(expand(F_d)) \\ B_d`.
- Corrected an over-strong proposed invariant: physical visited need not remain
  immutable `B_d` during generation. Exact first claims may be committed early;
  expanding them before the next level is the semantic failure.
- Distinguished frontier spill, semi-external BFS, and fully external duplicate
  detection. Only the latter must organize oversized visited/state identity as
  bulk partitions and merges.
- Recorded why hash routing is not state equality, why Bloom-filter false
  positives cannot be final in exact BFS, and why a dense rank is useful but
  not logically required for implicit/Cayley external search.
- Kept this block conceptual: no implementation or performance tuning was
  attempted.

## 2026-08-28: Cayley versus Schreier state semantics

- Derived right- and left-action replay order separately and identified which
  side of multiplication is a graph automorphism in each convention.
- Proved the ordinary right-Cayley normalization
  `dist(x,y)=length_S(x^-1*y)` and recorded why right invariance does not hold
  without additional conjugation/commutation structure.
- Distinguished algebraic invertibility from availability of a reverse move.
  Also separated finite group generation from positive-monoid reachability in
  infinite groups using directed `Z` as a counterexample.
- Modeled a right puzzle action with stabilizer `H` as the Schreier graph on
  right cosets `Hg`, forming `H\G` (terminology corrected 2026-08-31).
  This separates group elements, concrete configurations,
  and state identities.
- Refined arbitrary-start normalization for a non-free action: if
  `start=x0*a` and `goal=x0*b`, valid solution words satisfy
  `w in a^-1*H*b`, not generally `w=a^-1*b`.
- Distinguished an intrinsic stabilizer collision from an optional puzzle
  symmetry quotient; both reduce counts but have different semantics.
- Derived when Cayley word parity descends to coset states:
  `H subset ker(chi)` for the generator-parity homomorphism `chi`.
- Asked the `multigpu_beam` expert for practical convention failures, retained
  its replay/stabilizer test ideas, and independently narrowed its single-target
  normalization to the coset-valued condition above.
- Added no code, performance probe, or optimization.

## 2026-08-28: symmetry quotients and path lifting

- Separated three map strengths: a homomorphism projects paths, an
  automorphism-orbit quotient also permits path-lift existence, and a graph
  covering provides a unique lift after fixing the concrete start.
- Proved that automorphism-orbit BFS computes
  `min_(u in [t]) dist(s,u)`, not automatically `dist(s,t)` for one fixed
  representative.
- Added a four-vertex reflected-path counterexample where quotient distance is
  one but concrete fixed-target distance is two.
- Stated the transition-congruence obligation for arbitrary canonicalization:
  equivalent representatives must expose the same neighbor-class set.
- Distinguished preservation of unlabeled distance from preservation of move
  labels. Symmetries may conjugate/rename generators, requiring an accumulated
  frame for replay.
- Identified a bidirectional quotient hazard: equal canonical meeting keys may
  represent incompatible concrete endpoints until forward/backward frames are
  aligned.
- Connected target-orbit quotient BFS with concrete multi-source BFS: scalar
  distances can agree while source labels, ties, and reconstruction metadata do
  not.
- Added no code, benchmark, or optimization.

## 2026-08-28: asynchronous BFS as fair relaxation

- Separated arbitrary-schedule first claim, which can freeze a non-shortest
  discovery, from asynchronous distance relaxation with corrections.
- Proved correctness at quiescence in two directions: every finite proposal is
  a real path witness and cannot undershoot distance; complete propagation
  leaves every edge inequality satisfied and cannot overshoot a shortest path.
- Added the missing-reactivation counterexample: correcting `D[x]` with
  `atomicMin` does not correct descendant `y` unless `x` propagates again.
- Classified stale work as potentially harmless extra work, provided it cannot
  overwrite a better label or suppress the only propagation of an improvement.
- Replaced state-only asynchronous dedup semantics with minimum-by-state plus
  activation/version semantics.
- Distinguished local idleness from global quiescence: passive workers and
  empty local queues do not account for in-flight device/network messages.
- Treated first target discovery as an upper bound; early finality requires a
  global lower bound over active and in-flight work.
- Identified parent/version consistency, repeated expansion, and distributed
  termination traffic as separate output/work obligations.
- Added no implementation or optimization probe.

## 2026-08-28: BFS order, shortlex, and LexBFS

- Split "deterministic BFS" into distance/frontier determinism, vertex order,
  deterministic parent, lexicographically least shortest path, and shortlex
  move-word contracts.
- Proved by layer induction that ordered FIFO expansion yields shortlex-minimal
  representatives only when complete parent path prefixes—not merely state
  keys—and outgoing labels are processed in lexicographic order.
- Added a `za` versus `az` counterexample showing that state-sorted frontier
  order or minimum parent ID need not select the least move word.
- Demonstrated that valid shortest-parent choices can be globally incompatible
  with any first-winner FIFO BFS ordering using two parents sharing two children.
- Corrected an initially proposed false LexBFS counterexample during review.
  LexBFS remains consistent with BFS distance layers; it differs from
  sorted-adjacency BFS by refining ties with full selected-neighbor histories.
- Interpreted ordered Cayley BFS as selecting a generator-order-dependent
  shortlex normal form, distinct from arbitrary algebraic rewriting normal
  forms and from fixed-target paths in a symmetry quotient.
- Listed owner/global reduction obligations for reproducible parents across GPU
  and rank counts without proposing an implementation.
- Added no code, benchmark, or optimization.

## 2026-08-28: product-state BFS and history constraints

- Reframed history-dependent search as ordinary BFS on
  `(base_state,memory_state)` rather than as base BFS with optional metadata.
- Added a three-edge labeled counterexample where base-only visited discards the
  only path accepted by the word automaton.
- Distinguished a simple product path from its base projection, which may
  revisit one base vertex under different automaton states.
- Separated semantic path constraints from search-only pruning. Proved exact
  immediate parent-edge reversal removable from every unconstrained unit
  geodesic, while rejecting broader last-move rules without a proof.
- Mapped regular-language constraints, periodic time phases, non-backtracking
  directed-edge states, and Cayley word automata into one product construction.
- Recorded epsilon-transition boundary: zero-consumption automaton moves need
  closure/elimination or zero-cost shortest-path semantics, not blind unit BFS.
- Derived the Cayley product target: the shortest word must both represent the
  target element/orbit and be accepted by the automaton.
- Identified reverse-automaton compatibility as an extra bidirectional meeting
  condition and product memory as part of exact multi-GPU ownership/visited.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: BFS as a least fixed point

- Reframed BFS as distance-stratified least-fixed-point iteration rather than
  identifying it with a FIFO queue implementation.
- Proved by induction that accumulated visited sets are metric balls and their
  successive differences are exact distance shells.
- Derived frontier-only expansion as a semi-naive delta rule from the
  union-distributivity of relational image.
- Distinguished Knaster--Tarski existence from omega-stage construction:
  monotonicity alone does not justify the BFS delta recurrence or countable
  convergence for an arbitrary operator.
- Added an explicit monotone powerset operator whose finite approximants cover
  every natural number but whose `infinity` element appears only at stage
  `omega+1`, isolating the union-continuity property used by graph reachability.
- Separated set-theoretic correctness on infinite graphs from finite
  materializability, per-level completion, and whole-search termination.
- Connected the same recurrence to Boolean-semiring matrix expansion with a
  complemented visited mask and to recursive reachability evaluation.
- Characterized a completed empty frontier as a closure/fixed-point certificate
  and explained why local emptiness, fairness, or a transient absence of
  messages cannot certify distributed quiescence.
- Asked the `autolean` expert to check the induction and fixed-point boundaries,
  then retained only claims supported by the set proof and primary sources.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: exact checkpoint/restart semantics

- Made a completed logical level the simplest recoverable BFS cut: a complete
  ball, its unexpanded newest frontier, metadata, and no earlier work in flight.
- Added an orphaned-visited counterexample where a crash between durable visited
  claim and frontier enqueue permanently suppresses a reachable suffix.
- Separated rollback to a clean boundary from preserving a partial distributed
  cut with process, channel, pending-work, and contribution state.
- Qualified at-least-once delivery by requiring idempotence of the whole
  application transition, not only the visited bit.
- Classified retry algebra for reached sets, minimum distances, arbitrary and
  deterministic parents, all-parent sets, path counts, and source labels.
- Bound checkpoints to graph/generator/state/owner epochs and made repartition
  a migration operation rather than a modulus change.
- Connected durable termination to distributed quiescence and stable-property
  snapshots rather than persisted local emptiness.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: bipartite witnesses versus shortest odd cycles

- Separated non-bipartiteness detection, one replayable witness, the shortest
  witness exposed by one BFS tree, and global odd girth.
- Added a six-vertex counterexample where one exact BFS exposes a length-five
  odd cycle although the graph contains a triangle.
- Proved that all-root BFS recovers odd girth: rooting on a shortest odd cycle
  forces its opposite edge into one level, because any external shortcut would
  create a shorter odd closed walk and hence a shorter simple odd cycle.
- Observed that same-parity and equal-depth edge tests coincide for exact
  unweighted undirected BFS because adjacent depths differ by at most one.
- Used Cayley vertex transitivity to explain when identity-rooted BFS replaces
  all roots, while retaining the Schreier/puzzle qualification.
- Asked the `autolean` expert to review the theorem and counterexample; the
  channel returned `fetch failed`, so no conclusion depends on expert advice.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: distance regularity and BFS intersection profiles

- Split each state at depth `i` into backward, same-layer, and forward neighbor
  counts and derived the consecutive-layer edge-balance identity.
- Distinguished its general sum form from the scalar intersection-number form
  available in distance-regular graphs.
- Showed that frontier sizes alone omit lateral edges and the distribution of
  convergence multiplicity using a rooted `C_6` chord counterexample.
- Derived by hand that `Cay(Z_8,{+1,-1,4})` is vertex-transitive but not
  distance-regular: last-layer vertices have one or two backward neighbors.
- Recovered hypercube binomial layers from `c_i=i` and `b_i=n-i` as the clean
  uniform example.
- Connected per-layer intersection histograms to candidate convergence and
  owner skew without proposing a GPU implementation or optimization.

## 2026-08-28: adjacency powers, Cayley word mass, and BFS layers

- Separated arithmetic `A^d` walk counts, Boolean exact-length support,
  cumulative balls, and first-discovery BFS spheres.
- Used a three-vertex path to show that exact-length support can omit an earlier
  layer while including a backtracking return to the source.
- Identified the first positive arithmetic coefficient with distance and its
  value with shortest-path multiplicity.
- Expressed length-`d` Cayley generator words as convolution mass totaling
  `q^d`, distinct from both support size and unique minimal-length states.
- Distinguished normalized random-walk mixing and spectral walk constraints
  from exact visited/exhaustion semantics.
- Reconciled exponential word-tree work with the `q|R|` labeled occurrences of
  a complete finite exact Cayley BFS.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: hypergraph BFS and incidence semantics

- Defined unweighted Berge vertex distance and proved its equality with clique-
  expansion distance.
- Proved that the lossless bipartite incidence graph doubles distances between
  original vertices and made its BFS a typed two-phase process.
- Separated distance preservation from hyperedge-label, multiplicity, and
  multiway-structure preservation.
- Qualified when first settlement of an undirected hyperedge is safe for
  distance-only output and why all-parent/count outputs need more contributions.
- Added an AND-tail directed-hyperarc counterexample where ordinary incidence
  reachability activates a head from only one of two required tails.
- Kept binary Cayley moves distinct from physical batching and genuine
  hypergraph arity.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: Cayley growth series and frontier extrapolation

- Encoded BFS sphere and cumulative ball sizes as formal generating series and
  derived their prefix-sum relation.
- Connected rational series to eventual constant-coefficient recurrences while
  rejecting recurrence fitting from a finite trace as a proof.
- Collected exact free-group, `Z^n`, hypercube, and adjacent-transposition
  `S_n` examples under explicit generator conventions.
- Used `Cay(Z,{+/-1})` versus sufficiently large `Cay(Z_N,{+/-1})` to show
  arbitrarily long identical frontier prefixes with infinite versus finite
  tails.
- Separated a proved coefficient oracle from state identities, intersection
  profiles, owner balance, device capacity, and runtime.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: frontier representation information bounds

- Derived the `log2 binomial(N,k)` lossless lower bound for a size-`k` frontier
  in an exact ranked universe.
- Compared idealized sparse-ID and dense-bitmap payloads while explicitly
  excluding operation, allocation, conversion, and compression costs.
- Separated mathematical frontier sets from occurrence bags, parent labels,
  order, and shortest-path contribution metadata.
- Distinguished push enumeration, pull membership, candidate, frontier, and
  monotone visited representation lifecycles.
- Explained why clustering and rank order affect compressed bitmap size even at
  fixed global density.
- Extended the distinction to owner-local density, bitmap OR, replication, and
  multi-GPU communication semantics without selecting an implementation.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: explicit GPU BFS papers versus implicit Cayley search

- Rejected a duplicate work-depth study after finding the core already covered
  in notes 7 and 29, then shifted to a source-oriented transfer audit.
- Read the representation, frontier, visited, direction, measurement, and
  distributed contracts in Merrill et al., Beamer et al., Enterprise, and the
  Kepler distributed BFS paper.
- Separated transferable mechanisms—frontier regimes, duplicate locality,
  data movement, barriers, partitioning, communication—from nontransferable
  numeric throughput and representation-specific thresholds.
- Made dense IDs, resident CSR/CSC, bitmaps, enumerable unvisited universes,
  compact communication IDs, and graph-volume TEPS explicit assumptions.
- Defined a cross-representation passport and a transition funnel from generated
  occurrences through accepted exact states.
- Asked the `multigpu_beam` expert one combined question about transfer
  boundaries; accepted only recommendations independently supported by the
  primary papers and existing project semantics.
- Kept CayleyPy outer beam results separate from exact explicit BFS results.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: CayleyPy K1/K2 test evidence audit

- Audited registered Stream2, contract, history, and stitched CUDA test sources
  plus retained K1/K2 verification reports without running them.
- Limited direct Stream2 evidence to one direct hit, one manually inserted K1
  hash-table hit, and one manually supplied one-move K2 suffix under each
  backend.
- Found that the host K1 reverse-BFS builder, nonempty K1 suffix replay,
  production K2 list/composition builders, negative bounds, collision behavior,
  and result overflow are not established by those fixtures. Correction
  2026-08-31: a later-shorter combined residual cannot occur under the exact
  full-ball/length-order theorem and is not a missing mandatory fixture for
  that contract; actual premise violations require separate checks.
- Reclassified the stitched CUDA executable as a component smoke: it replaces
  intermediate candidate data before Stream4 and final materialization.
- Distinguished the standalone history library test from the production
  runner's separate `CpuCandidateHistory` paths.
- Bound historical PASS reports to their May implementation commits and did
  not present them as current dirty-working-tree results.
- A current `docker info` probe failed with `permission denied` on the Docker
  Desktop Linux-engine API pipe. This was recorded as session-level engine
  inaccessibility, not overinterpreted as proof that Docker Desktop was stopped.
- Recorded shared-oracle risks from involutory swaps and common action/hash
  helpers, and created an evidence ladder from test registration through
  independent oracle and rank/backend parity.
- Added no tests, implementation, benchmark, or optimization.

## 2026-08-28: bounded BFS negative results and three-valued lookup

- Initially selected enqueue-versus-dequeue marking for the next study pass,
  then rejected that route after finding it already covered in note 3; retained
  the correction instead of duplicating material.
- Distinguished exact `WITHIN_RADIUS`, exact `NOT_WITHIN_RADIUS`, incomplete
  `UNKNOWN`, and exhaustively proved `UNREACHABLE` outcomes.
- Separated positive replay witnesses from negative coverage certificates: a
  partial table may validate a hit but cannot validate a miss.
- Proved by contraposition that exhaustive words of lengths `0..K` missing an
  exact reverse ball of radius `R` imply distance greater than `K+R`.
- Applied the theorem to CayleyPy K1/K2 and kept its scope local to generated
  children and retained outer-beam parents.
- Explained how bare-hash collision can create a direct false hit and an
  indirect false miss by preventing expansion of a collided K1 state.
- Separated positive result-buffer overflow, scan cancellation, fail-closed
  table capacity, and ordinary completed misses.
- Added distributed negative-result obligations beyond an all-reduced
  `found=0` flag.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: local certificates for BFS distance labels

- Derived a schedule-independent certificate for complete directed BFS output:
  root uniqueness, one real predecessor at label minus one, and edge
  feasibility `L(v)<=L(u)+1` for every edge from a finite-labeled vertex.
- Proved exactness by combining predecessor-chain upper bounds with edge-scan
  lower bounds and reachability closure.
- Rejected the universal use of `abs(L(u)-L(v))<=1`; it is an undirected
  specialization and fails on ordinary directed back edges.
- Added counterexamples showing that valid parents alone allow overestimated
  distances and edge inequalities alone allow underestimated labels.
- Adapted the certificate to a bounded radius: closure is required below the
  boundary, while layer-`R` edges may leave the materialized ball.
- Connected the proof to Graph500's undirected tree/edge/component validator
  while preserving its narrower graph contract.
- Identified independent successor completeness and semantic state identity as
  the difficult parts for implicit/hash-indexed validation.
- Described distributed validation obligations and concrete retained failure
  witnesses without implementing a validator.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: reverse BFS goal neighborhoods and suffix certificates

- Derived reverse BFS as forward shortest-path search to a target over the
  predecessor relation, including the suffix-witness induction and lower-bound
  proof.
- Traced CayleyPy K1 construction: explicit reverse frontiers, inverse
  permutation application, first-`Hash128` discovery, and prepended forward
  suffix moves.
- Confirmed fail-closed configured entry and device-table placement limits;
  neither path silently returns a partially packed table.
- Limited K1 exactness to the unproved premises that generator rows are valid
  permutations and Zobrist keys are injective over the generated neighborhood.
- Distinguished K2 as exhaustive bounded word enumeration rather than
  unique-state BFS.
- Retracted 2026-08-31: the originally claimed K2/K1 objective mismatch and
  residual-length counterexample were invalid under the exact-full-ball
  premises. With unit costs, shortest K1 suffixes, and all K2 lengths including
  zero in length order, the first hit gives the exact residual distance:
  its prefix length is `max(0,D-R)`. Actual hash-only/completeness premises
  and outer beam optimality remain separate questions (corrected notes 40–43).
- Separated best K1 suffix, first per-candidate K2 hit, best recorded hit in one
  outer depth, and globally shortest original-graph path.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: non-backtracking words versus state BFS

- Separated local immediate-inverse cancellation from global first-discovery
  and visited-state semantics.
- Proved that deleting a two-edge inverse spur preserves shortest paths under a
  symmetric reversible unit-edge contract, while explicitly limiting the
  conclusion to existence of a shortest path.
- Added the `C_4` counterexample: two non-backtracking depth-two words converge
  to one state, and a length-four non-backtracking word revisits the root.
- Identified the free-group Cayley tree as the special case where freely reduced
  words and unique graph states coincide.
- Recast genuine non-backtracking traversal as BFS on
  `(vertex, previous directed edge)` product states.
- Inspected CayleyPy's current generation path and confirmed it considers all
  `MOVE_COUNT` moves from each retained parent; no inverse-move exclusion was
  found in that path.
- Distinguished goal-neighborhood inverse construction, same-depth hash
  merging, and ancestry-history compaction from search-path pruning.
- The first documentation validator incorrectly required the literal phrase
  `Added no implementation` inside note 39 even though that scope statement
  belongs to this log; the content checks were narrowed to the note's actual
  required concepts and the failed check was not treated as a content failure.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: read-only CayleyPy beam contract audit

- Inspected the dirty `D:\100XH100` working tree at commit `b5fcf6b` without
  changing or running it.
- Traced the production depth loop through generation, goal inspection,
  thresholding, `Hash128` deduplication, global beam selection, materialization,
  history, reconstruction, and CPU target replay.
- Confirmed that breadth-like depth scheduling does not supply exact BFS
  semantics: the next frontier is globally width-bounded and the traced path
  contains no accumulated old-ball subtraction.
- Distinguished complete generation from the retained beam from complete graph
  exploration: all moves of retained parents are inspected, while parents
  discarded at earlier depths are gone.
- Confirmed that multi-rank finalization computes one global keep count before
  load balancing, rather than using independent local beam quotas.
- Classified the exact goal-neighborhood as a suffix component whose replay can
  certify a valid path but cannot recover pruned prefixes or prove global
  shortest distance.
- Recorded the identity limitation: candidate equality is bare deterministic
  128-bit Zobrist equality; full-state CPU replay protects accepted path
  artifacts, not completeness of hash-pruned search.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: exact BFS contract-map synthesis

- Audited all 36 prior notes, the evidence map, roadmap, and research protocol
  across graph, metric, identity, transaction, schedule, output, completion,
  recovery, and physical-evidence layers.
- Consolidated the independent obligations into a minimal exact-BFS passport
  and a claim-strength validation ladder.
- Classified variants into semantics-preserving evaluations, different exact
  mathematical problems, and explicitly approximate/incomplete executions.
- Attached common hardware metrics to what they explain and what they cannot
  prove.
- Collected cross-layer counterexamples showing why one locally correct artifact
  cannot certify the whole traversal.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: REF-022 Cayley girth probe blocked before execution

- Planned a minimal educational Rust/Docker comparison of `Z_31`, `Z_8 x Z_8`,
  and adjacent-transposition `S_3` to expose inverse returns, relation
  convergence, and sharp girth boundaries in actual BFS layer counts.
- Refused to compile or run the probe on the Windows host because all project
  builds and executions must occur in Docker.
- Docker Desktop frontend processes started, but the Linux engine remained
  absent and the `docker-desktop` WSL distribution stayed stopped.
- Read-only logs identified a backend crash while initializing the Inference
  manager at the inaccessible Docker runtime path `dockerInference`.
- Preserved exact commands/errors and explicitly did not delete runtime files,
  reset Docker/WSL, or change out-of-scope host settings.
- Recorded REF-022 as not executed; no algorithmic observation or performance
  result is claimed.

## 2026-08-28: what BFS complexity means

- Scoped `Theta(|R|+A_R)` to full traversal of a reachable component in an
  explicit adjacency-list access model and gave the matching worst-case
  inspection argument.
- Separated dense-matrix input size from sparse adjacency storage and implicit
  successor generation.
- Replaced a single `V+E` label for Cayley search by generated occurrences,
  exact identity, unique accepted states, output, and control costs.
- Rejected `O(b^d)` as an equality for graph BFS; it is an unmerged path-tree
  model whose relation to state work depends on convergence and saturation.
- Made output size and persistent visited storage distinct from peak frontier
  queue size.
- Separated total parallel work, causal depth, communication, synchronization,
  capacity, and throughput.
- Clarified from the Graph500 specification that TEPS uses normalized input
  edges in the traversed component rather than necessarily counting actual
  adjacency inspections.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: exact state identity and hashing

- Consolidated the exact visited predicate around semantic state equality rather
  than compact-code equality.
- Added a three-edge witness where one hash collision rejects the only gateway
  to a reachable target and destroys BFS completeness.
- Qualified false negatives: they are merely extra work only if a later exact
  stage repairs them before distance, capacity, counting, or termination output
  is affected.
- Separated injective ranks, collision-resolving hash tables, fingerprints,
  perfect hashing over a fixed set, and Bloom-filter approximate membership.
- Clarified that existing aggregate experiment fingerprints are compact
  regression evidence, not deterministic arbitrary-set equality proofs.
- Distinguished hash roles for table addressing, owner routing, visited
  equality, and artifact validation.
- Made consistent canonical keys and ownership epochs necessary for all equal
  distributed states to meet at one authoritative exact comparison.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: girth, relations, and tree-like BFS

- Proved the distinct sharp thresholds `2r<girth` for unique root geodesics and
  `2r+1<girth` for the entire induced radius-r ball to be a tree.
- Connected collision-free non-backtracking BFS spheres to the regular-tree
  counts and derived the odd/even Moore lower bounds by vertex-rooted and
  edge-rooted BFS trees.
- Separated immediate inverse backtracking from genuine relation convergence
  and earlier-ball closure; an infinite tree already has the first phenomenon.
- Related Cayley cycles to cyclically reduced identity words under an explicit
  simple undirected generator contract.
- Rejected the shortcut from shortest written presentation relator to girth:
  reductions, repeated vertices, derived relations, and generator changes all
  break it.
- Replaced identity words by stabilizer words for Schreier/puzzle action graphs.
- Added free-group, square-lattice, and adjacent-transposition sanity examples.
- An `autolean` expert request returned `fetch failed`; the off-by-one results
  were instead checked by direct cycle proofs and sharp cycle examples.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: k-hop batching and graph powers

- Separated one physical superstep containing `k` exact logical microlevels
  from BFS on a graph whose edges represent multi-edge walks.
- Proved that an at-most-`k` power graph has distance
  `ceil(original_distance/k)` and therefore changes the reported metric.
- Distinguished Boolean `A^k` exact-length reachability from the union of
  lengths zero through `k`; a length-two chain shows that endpoint-only
  exact-two expansion omits the depth-one vertex.
- Added a mixed-depth counterexample where a depth-three candidate wins a
  boolean visited race and suppresses the true depth-two arrival.
- Derived conditions under which coarse rounds recover exact balls at radii
  `rk` while losing the individual distance strata inside each annulus.
- Made cross-owner intermediate hops, in-flight logical depths, minimum target
  reduction, and macro-edge path witnesses explicit multi-GPU obligations.
- Applied the same distinctions to generator words and relations in
  Cayley/Schreier graphs without proposing an implementation.
- Tried twice to ask the `multigpu_beam` expert for a correctness review; both
  calls returned `fetch failed`, so no expert recommendation was used.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: exact BFS versus beam search

- Made complete retention of every eligible newly reached state part of the
  exact BFS contract, rather than treating level-synchronous execution alone as
  sufficient.
- Separated heuristic ordering from pruning: ordering a complete layer can
  preserve BFS distances, while top-k, thresholds, overflow drops, and
  unfinished timeouts remove the original-graph guarantee.
- Added a width-one graph counterexample where beam pruning discards the unique
  shortest branch and returns a longer path or false failure.
- Added a record-versus-state example showing that top-k before dedup can spend
  the whole width on duplicate transitions and retain fewer unique states.
- Added a two-partition counterexample showing that equal local top-k quotas do
  not in general equal one global top-k selection.
- Restricted an exact BFS lookup/table claim to its covered subproblem; an
  exact suffix does not restore globally pruned beam prefixes or prove a global
  shortest path.
- Defined minimum audit fields needed to distinguish exact frontiers from beam,
  capacity-truncated, and hybrid runs.
- Asked the `multigpu_beam` expert for a compact conceptual review and checked
  the accepted statements against recurrences, counterexamples, and primary
  literature.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: BFS certificates and REF-021 diameter counterexample

- Separated exhaustive one-source reachability from undirected connected
  components, directed weak connectivity, and strong connectivity.
- Derived the bipartite certificate from BFS depth parity and the explicit odd
  cycle obtained from a same-level edge plus two tree paths to their LCA.
- Distinguished source eccentricity from graph radius and diameter; proved the
  generic bound `ecc(s) <= diameter <= 2*ecc(s)`.
- Used a bounded educational Rust/Docker probe to reject the claim that a unique
  first farthest vertex makes two-sweep exact on general connected graphs.
- REF-021 exhaustively found a seven-vertex witness: start `4` has unique
  farthest `3`, but `ecc(3)=3` while diameter is `4` via `5-1-2-0-6`.
- Proved why one complete identity-rooted BFS does give finite connected Cayley
  diameter: left translations make all vertices' eccentricities equal.
- Refused to transfer that conclusion automatically to a fixed-generator
  Schreier/puzzle graph; transitivity of the underlying state action need not
  preserve the exact generator edge relation.
- Added only the minimal hypothesis probe, not an optimized or production
  diameter implementation.

## 2026-08-28: static snapshots, dynamic maintenance, and temporal BFS

- Made the fixed-edge-relation assumption of ordinary BFS explicit and split
  mutable-graph questions into snapshot BFS, dynamic distance maintenance, and
  temporal journeys.
- Added a two-update counterexample where BFS observes `s->a` before its
  deletion and `a->t` after its insertion, producing a path that belongs to no
  static snapshot.
- Observed that the same edge-time sequence can be a valid temporal journey,
  so correctness depends on the declared time semantics rather than individual
  edge validity alone.
- Derived update monotonicity: insertions preserve old paths and can only
  decrease distances; deletions can only increase distances but may invalidate
  parent witnesses and cascade.
- Distinguished one stored parent from the full shortest-path DAG when testing
  whether deletion removes every shortest witness.
- Treated Cayley generator changes as structured global edge-family updates.
  Added/removed/reordered generators have different effects on metric, labels,
  direction, and shortlex outputs.
- Separated temporal objectives (foremost, fastest, minimum hops/cost) and
  explained when time-expanded BFS versus 0-1/weighted search is appropriate.
- Added graph-version epochs, parent replay versions, and update quiescence as
  conceptual multi-GPU obligations.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: BFS versus iterative deepening tree search

- Separated BFS over unique graph states from IDDFS over bounded path/word tree
  nodes; equal minimum target depth does not imply equal explored objects.
- Derived the shallowest-solution proof from complete iterations at every
  smaller limit and made finite branching plus FOUND/CUTOFF/EXHAUSTED semantics
  explicit.
- Qualified the familiar constant-overhead intuition: it applies to regular
  exponential trees, while chains, irregular successor costs, and graph
  transpositions can behave very differently.
- Added a depth-3 counterexample where boolean global visited records `x` at the
  cutoff depth and suppresses a later shallower arrival with enough remaining
  budget to reach the goal.
- Replaced state-only transposition dominance by
  `(state, maximum searched remaining budget, semantic context/version)`.
- Explained why retaining an unqualified visited bit across larger IDDFS limits
  can turn an old cutoff into false exhaustion.
- Mapped raw Cayley IDDFS to the infinite word tree `S*`; a finite group still
  has infinitely many words because relations/cycles repeat elements.
- Distinguished one optimal target result from complete BFS balls, unique-state
  metrics, component exhaustion, and diameter certificates.
- Clarified that frontier search and delayed duplicate detection add exact
  mechanisms; neither is obtained by simply deleting Closed.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: exact implicit GPU BFS representations

- Separated a proved dense state rank from an ordinary hash and from a static
  minimal perfect hash over a previously known subset.
- Identified three incompatible uses of the phrase "one-bit BFS": permanent
  visited plus a separate frontier, final reachable-set output, and recycled
  layer/parity bookkeeping.
- Made requested output part of the storage contract: one visited bit alone
  cannot encode distances, parents, layer boundaries, counts, or paths.
- Rejected reversibility or undirectedness as a standalone proof for recycling
  one/two bits; move scheduling and rediscovery coverage remain explicit proof
  obligations.
- Distinguished exact on-the-fly GPU hashing, which retains semantic keys and
  resolves collisions, from fingerprint-only duplicate suppression.
- Explained how a finite ranked puzzle abstraction can make pull meaningful
  without making pull available for an arbitrary implicit Cayley graph.
- Applied the boundary to the inspected CayleyPy path: bare `Hash128` and beam
  pruning do not satisfy exact dense-rank/complete-frontier premises.
- Recorded source-access limits: current PDF downloads failed with Schannel
  `SEC_E_NO_CREDENTIALS`, so no unchecked numeric or detailed paper-specific
  claims were imported.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: expansion, diameter, and BFS memory pressure

- Identified the exact next frontier with the external vertex boundary of the
  completed metric ball, separating it from boundary-edge occurrences.
- Derived geometric ball growth while a ball is below half the graph under a
  proved global vertex-expansion bound.
- Proved that bounded-degree constant-expansion families force at least one
  `Omega(n)` BFS frontier from every source, exposing a depth-versus-width
  trade-off rather than treating low diameter as memory relief.
- Bounded the conversion from edge boundary to vertex boundary by maximum
  degree and interpreted the gap as duplicate endpoint convergence.
- Kept spectral gap and conductance as indirect, hypothesis-bearing evidence
  rather than exact rooted-frontier predictors.
- Distinguished Cayley vertex transitivity from expansion using cycles,
  hypercubes, generator-dependent symmetric groups, and expander families.
- Separated graph expansion cuts from multi-GPU owner-crossing candidate and
  state-identity traffic.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: work, span, and frontier parallelism

- Applied the work-span lower bound `T_P >= max(W/P,S)` to exact BFS and
  distinguished average from time-varying per-level parallelism.
- Made the `D`-edge causal chain explicitly relative to on-demand local
  successor expansion rather than claiming an unconditional lower bound for
  every representation or preprocessing model.
- Distinguished logical levels from physical stage span: scans, sorts, routing,
  reductions, and collectives can add critical depth inside one level.
- Explained why persistent kernels and k-hop supersteps can remove visible
  launches without removing internal layer dependencies.
- Connected narrow frontiers to an occupancy/strong-scaling ceiling and wide
  frontiers to simultaneous parallelism and memory pressure.
- Separated generated-work throughput from accepted-state progress near
  duplicate-heavy or saturated levels.
- Modeled multi-GPU level time by the slowest owner and communication/collective
  critical path rather than aggregate work divided by GPU count.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: frontiers as separators and exhaustion certificates

- Proved by the first-exit argument that every directed source path to a vertex
  deeper than `d` crosses the exact sphere `S_d`.
- Distinguished this metric separator from a minimum vertex cut and supplied a
  bottleneck counterexample where a later sphere is much larger than the cut.
- Rejected the claim that equal-distance layers are reachability antichains by
  adding a same-level directed-edge witness.
- Derived `EXHAUSTED` from successor closure of the completed ball rather than
  from a momentarily empty physical queue.
- Showed geometrically why a beam subset is not generally a separator and why
  exact suffix lookup cannot restore a discarded prefix cut.
- Treated a level-boundary checkpoint as a consistent cut across visited,
  frontier publication, graph version, output metadata, and in-flight work.
- Made distributed exhaustion a property of the settled global frontier union,
  not any rank-local empty shard.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: pattern databases and abstract BFS heuristics

- Defined a PDB as exact reverse shortest-path distances in an abstract graph,
  separated from exact distances in the concrete puzzle graph.
- Proved admissibility by projecting every concrete goal path to an abstract
  walk of no greater cost and consistency by the per-edge triangle inequality.
- Made exact PDB construction and collision-free identity correctness
  obligations because an overestimated heuristic can invalidate optimal
  pruning.
- Proved that maxima preserve admissibility/consistency and supplied a one-move
  counterexample to blindly summing two admissible pattern heuristics.
- Derived additive PDB correctness from nonnegative per-operator cost partitions
  whose sum does not exceed concrete edge cost.
- Showed that a completed unit-cost abstract radius table supports capped miss
  value `R+1`, whereas incomplete construction must return `UNKNOWN`.
- Distinguished abstract lower bounds from concrete reverse-ball suffix upper
  bounds and applied the distinction to CayleyPy K1.
- Kept GPU replication/sharding as a lookup cost/capacity question under one
  immutable abstraction/table contract.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: BFS, A*, and bound-certified pruning

- Separated heuristic ordering of a complete BFS layer from cross-layer
  best-first scheduling and from actual state deletion.
- Derived `g+h` as a lower bound for every concrete solution extending one
  prefix record and combined it with a replay-valid incumbent upper bound.
- Made equality pruning output-dependent: `g+h=U` may still contain alternative
  optimal paths, parent/count contributions, or better secondary ties.
- Expressed optimality as closure of the global bound gap: minimum open/in-flight
  lower bound at least the validated incumbent cost.
- Distinguished first goal generation from final goal selection under the
  relevant lower-bound ordering.
- Recorded consistency/reopen and best-`g` state-dominance obligations,
  including the history-sensitive product-state exception.
- Positioned PDB as lower-bound evidence and concrete K1 suffix as a possible
  incumbent upper-bound witness without treating either as beam exactness.
- Extended the bound certificate to multi-GPU active, staged, in-flight, and
  owner-pending records.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: owner hashing, load balance, and routing

- Defined unique state-stable authority as the correctness contract and hash
  uniformity as a separate performance property.
- Derived multinomial/binomial owner loads, variance/covariance, and a Chernoff
  plus union-bound baseline under independent uniform assignment.
- Proved unavoidable small-frontier skew: for `0<w<P`, at least `P-w` ranks are
  idle and max/mean is at least `P/w`.
- Distinguished final visited, per-level frontier, receive, accepted, scratch,
  byte, and wall-time balance.
- Derived ideal remote fraction `1-1/P` only under parent/child owner
  independence and connected mixing improvements to locality loss.
- Reconciled the model with retained REF-005/006/010 observations: mixed owners
  improved balance while raising remote traffic and moving duplicate convergence
  from source pre-dedup to destination owners.
- Made mean provisioning insufficient for exact capacity and required explicit
  non-lossy overflow behavior.
- Introduced ownership epochs for changes in world size, seed, rank,
  canonicalization, or graph version and explained the two-authority failure.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: authoritative visited and advisory replicas

- Defined authoritative visited as one monotone linearized set within a fixed
  graph/identity/ownership epoch.
- Proved that a delayed exact replica sourced only from authority accepts is a
  sound subset: positive is true while negative is globally unknown.
- Made early dropping from a sound positive conditional on the output contract,
  since all-parent/count/labeled histories may require duplicate metadata.
- Distinguished exact dense-rank bitmap replicas from fingerprint/Bloom
  filters; Bloom positives cannot final-drop exact candidates.
- Kept stale-negative work safe only through authoritative fallback and showed
  why replicas cannot independently claim novelty.
- Applied the one-sided logic to source pre-dedup and bidirectional meeting
  caches without transferring termination authority to them.
- Scoped cache soundness to an immutable search epoch and required invalidation,
  namespacing, or reconstruction after restart/repartition.
- Separated authoritative termination traffic from advisory replica-update
  convergence.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: uniform sampling from shortest-path DAGs

- Derived exact backward sampling: choose predecessor edge `u->v` with weight
  `sigma(u)` and proved every full path has telescoping probability
  `1/sigma(t)`.
- Added a two-predecessor counterexample showing that uniform local predecessor
  choice biases global paths when prefix multiplicities differ.
- Separated vertex sequences, labeled multiedges, Cayley generator words,
  multiple sources, and concrete quotient lifts as different sample spaces.
- Specified unbiased integer interval sampling and rejected modulo, overflow,
  saturation, modular counts, and floating conversion as exact-uniform evidence.
- Showed that correct distances and one parent tree are insufficient without
  complete predecessor-edge coverage and exact count recurrence.
- Derived prefix-times-suffix connector weights at one fixed bidirectional cut
  and rejected uniform meeting selection or mixing several cuts.
- Made distributed count addition retry-deduplicated and layer-finalized because
  addition is not idempotent.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: integrated BFS mental model

- Synthesized BFS as proof-producing metric-ball growth rather than a queue or
  frontier-kernel shape.
- Organized correctness around five contracts: graph/identity, expansion,
  schedule/finalization, output, and completion/failure.
- Mapped the semantic candidate funnel to distinct failure modes and retained
  counts from transition occurrences through accepted states.
- Classified BFS variants and neighboring searches by which graph, metric,
  completeness, finalization proof, and output they preserve or change.
- Connected explicit, implicit, Cayley/Schreier, single-GPU, and multi-GPU
  presentations through one performance stack from graph geometry to measured
  hardware time.
- Unified BFS, bidirectional search, PDB, A*, bounded lookup, and replay as
  upper/lower-bound closure certificates.
- Recorded twelve common false equivalences and a source/code/run reading
  checklist.
- Applied the map to the inspected CayleyPy beam, K1, K2, and replay components
  without changing their distinct guarantees.
- Summarized strong current understanding and application-scale/multi-GPU/
  identity/integration evidence gaps.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: validating implicit successor completeness

- Separated edge soundness, endpoint/label correctness, and completeness for an
  implicit successor oracle.
- Added a missing-edge counterexample showing why positive replay and checks on
  emitted edges cannot prove coverage.
- Exploited total finite generator structure to define exact
  `(parent,generator)` work coordinates while rejecting aggregate `N*q` as a
  sufficient certificate.
- Required explicit `VALID/INVALID/UNKNOWN` outcomes for state-dependent partial
  moves so operational failures cannot masquerade as illegality.
- Distinguished permutation bijectivity/inverse laws from independent evidence
  that a row implements the intended named puzzle move.
- Ranked differential, metamorphic, exhaustive-small-domain, mutation, and
  specification-to-code evidence by claim strength and independence.
- Added GPU work-coordinate/fusion coverage and multi-GPU conservation limits
  without designing a backend.
- Applied the ladder to current CayleyPy K1/K2/replay evidence and stated only
  unverified gaps, not inferred bugs.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: partial-layer bidirectional stopping

- Generalized the complete-layer bidirectional proof to partial and asymmetric
  schedules using global minimum unfinished depths rather than loop counters.
- Proved `a+b>=mu` sufficient for one optimal distance/path when both metric
  balls through the minima are complete and connector detection is persistent.
- Added a length-four-versus-length-two counterexample showing why completing
  one chunk cannot advance a depth while same-depth work remains.
- Included vertex, edge-range, GPU execution, host staging, routed,
  owner-pending, retry, and spill work in the semantic minima.
- Distinguished conservative stale-smaller minima from unsafe stale-larger
  minima and required the incumbent and minima to share a consistent epoch.
- Kept global closure separate from local emptiness and scheduling fairness
  separate from distance safety.
- Made equality-boundary work dependent on the requested output instead of
  treating distance-optimal termination as enumeration-complete termination.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: BFS output contracts and finalization boundaries

- Reframed exact BFS validity as a vector indexed by requested output rather
  than one success flag.
- Distinguished distance, one arbitrary replayable path, one canonical path,
  complete predecessor DAG, exact counts, explicit enumeration, uniform
  sampling, and multi-source ownership contracts.
- Made graph/path identity precede dedup semantics, including vertex sequences,
  labeled edges, generator occurrences, sources, and quotient lifts.
- Showed why a canonical winner needs all potentially better equality proposals
  while one arbitrary winner does not.
- Separated predecessor-DAG completeness, count aggregation, and exponentially
  large explicit path enumeration.
- Classified equal-child records as retries, older routes, alternative
  same-depth parents, or parallel labeled edges depending on output semantics.
- Distinguished multi-source scalar distance, arbitrary/canonical/all owners,
  and coherent same-label parent forests.
- Added a per-output failure matrix for capacity, retry, tie-finalization, and
  partial enumeration outcomes.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: CayleyPy output-contract audit

- Applied note 57's matrix to the current dirty `D:\100XH100` production source
  and one retained two-GPU Cube4 solve-bucket log.
- Identified one CPU-replayed move word as the strongest intended output,
  separate from any exact original-graph shortest-distance claim.
- Traced prefix history reconstruction, K2/K1 suffix append, full CPU move
  replay, target equality, and failure-before-output behavior.
- Classified `solved_count` and per-length bucket counts as operational hit
  record counts rather than shortest-path counts.
- Separated deterministic selection over stored implementation records from a
  semantic canonical generator-word contract.
- Recorded that the historical log contains eight length-10 records across two
  ranks but lacks immutable linkage to current source/container/generators.
- Treated a path-only submission CSV as a candidate witness rather than a
  self-contained replay or optimality certificate.
- Marked DAG, all-path, exact-count, uniform-sampling, and unreachable outputs
  as not provided, not as inferred implementation bugs.
- Added no implementation, benchmark, or optimization.

## 2026-08-28: REF-010 exact distributed BFS output audit

- Applied notes 56-57 to the bulk-synchronous owner-computes bidirectional
  simulator and separated logical distribution from a real multi-GPU runtime.
- Confirmed the model's strongest output as exact target distance plus one
  replayable shortest path under complete-level assumptions.
- Distinguished deterministic implementation parent selection from a semantic
  canonical path and recorded absent DAG/count/all-path/sampling contracts.
- Audited exhaustive four-vertex evidence, S8 coverage, aggregate conservation,
  and the missing per-case historical witness data.
- Reran REF-010 inside existing Docker image
  `sha256:55f9efc3c2d82a3110e23f9fdc194026d6f55197105d10dfd6f48a4d0240bf0f`
  with the workspace mounted read-only.
- Reproduced 294,912 zero-failure distributed searches and 40 S8 routing rows.
- Passed both focused distributed-bidirectional unit tests in the same
  read-only Docker environment.
- Preserved the first failed raw comparison: CRLF/LF made JSON byte-different;
  normalized diff then showed both retained artifacts semantically identical.
- Added a ten-item transfer checklist for future real multi-GPU evidence.
- Added no implementation or optimization.

## 2026-08-28: REF-022 Cayley relation-onset retry

- Retried only after Docker became healthy and preserved the earlier backend
  failure in the experiment history.
- Used test-first development: the initial Docker compile failed on the four
  intentionally absent probe symbols, then all three fixture tests passed after
  the minimal educational implementation.
- Ran tests, formatting, compilation, and all calculations in the read-only
  workspace mount of `multigpubfs-rust-toolchain:dev`, image
  `sha256:764a443c2ddc39b28b8fbb0b1495656984ea5ee8dd82f7f435f2069a6574ce69`.
- Retained 29 exact rows covering C31, Z8xZ8, and S3.
- Observed C31's odd boundary as a same-level visited edge without candidate
  convergence, Z8xZ8 commutation as depth-two candidate convergence, and the
  S3 braid as depth-three candidate convergence.
- Separated parent returns, visited non-parent occurrences, candidate first
  occurrences, and candidate convergence instead of using one duplicate ratio.
- Added no reusable BFS library, GPU path, benchmark, or optimization.

## 2026-08-28: REF-024 Cayley versus Schreier action probe

- Compared the same adjacent-swap alphabet on all six S3 group elements and on
  the three positions of one marked point.
- Used test-first Rust in Docker: retained the missing-symbol RED compile, then
  passed all three semantic tests; the first full gate stopped only on three
  rustfmt line wraps and passed after the formatting-only correction.
- Observed Cayley frontiers `1,2,2,1` with braid convergence while forming
  depth three, versus Schreier frontiers `1,1,1` with a nonidentity stabilizer
  loop already at the root.
- Verified state distance two to point `2` while one group representative of
  the same target has Cayley distance three.
- Derived state distance as minimum word length over the target coset and
  separated group identity relations from stabilizer words.
- Recorded why deleting loops or merging labeled destinations changes word and
  occurrence contracts even when distinct-state distances survive.
- Added no reusable implementation, benchmark, GPU path, or optimization.

## 2026-08-28: REF-025 current Megaminx vertex/equality audit

- Read the current `D:\100XH100` Megaminx config and relevant state, hash, and
  Stream4 sources without modifying its dirty worktree.
- Verified that the central state is the unique identity vector over 120
  positions, so the generated permutation action is free and the represented
  orbit is a genuine Cayley graph of the generated subgroup.
- Added a standalone test-first Rust parser/auditor and exact full-state BFS;
  retained the intended missing-symbol RED, an explicit-`TryFrom` compile
  correction, and a formatting-only failed gate before the final pass.
- In read-only Docker, validated 24 permutation moves, 12 inverse pairs, no
  depth-one loops, 24 unique children, order five for every generator, and
  frontiers `1,24,408,6208,90144`.
- Matched native CayleyPy's retained depth-four expectation using an independent
  parser and full-vector equality rather than scalar-hash identity.
- Audited native CayleyPy's int64 hash-only dedup and production Stream4's
  Hash128-only dedup as collision-assuming identity contracts; observed no
  collision and proposed no redesign.
- Separated valid returned-path replay from layer completeness and separated
  the production beam's inherent pruning from its hash-equality assumption.
- Added no GPU run, performance measurement, production code, or optimization.

## 2026-08-28: REF-026 real Megaminx relation signatures

- Predicted and then independently checked the first relation signatures in the
  current checksum-pinned 24-move Megaminx configuration.
- Used two test-first RED cycles: absent depth-two analyzers first, then absent
  face-pair classification fields; retained one formatting-only failed gate.
- Passed five Rust tests in read-only Docker, including both inherited REF-025
  config/full-state contracts.
- Partitioned 576 depth-one transition occurrences into 24 inverse returns and
  552 reduced words reaching 408 exact `F2` states.
- Proved by complete word-pair classification that all 144 convergence extras
  are `ab=ba`, arranged as 36 face pairs with all four sign combinations; no
  other collision or endpoint multiplicity above two occurs.
- Partitioned all 9,792 transitions from `F2` into 552 backward, 24 same-level,
  zero older-ball, and 9,216 forward occurrences producing 6,208 `F3` states.
- Verified every same-level occurrence as `g^2 -> g^3=g^-2`, accounting exactly
  for the 12 undirected order-five face-cycle boundaries.
- Recorded girth four from commutation squares and left the 3,008 convergence
  extras forming `F3` unclassified rather than calling them independent relators.
- Added no timing, GPU run, production implementation, or optimization.

## 2026-08-28: REF-027 Megaminx F3 word classes

- Revisited the 3,008 convergent candidate records forming `F3` after noting
  that unique `F2` states had already erased alternate shortest-word histories.
- Used test-first Rust to validate an adjacent-commutation normal form and
  conservation among word occurrences, candidate records, classes, and states;
  retained one formatting-only failed gate.
- Enumerated all 12,696 non-backtracking length-three words, retained 12,384
  whose endpoints lie in exact `F3`, and used full 120-entry state equality.
- Found 12,384 shortest words, 9,216 candidate records, and 6,208 states:
  3,168 word extras disappear at parent-state dedup and 3,008 remain visible as
  candidate convergence.
- Computed complete adjacent-swap closures from the actual commuting move
  matrix; every endpoint has exactly one commutation class and zero cross-class
  remainder.
- Recorded shortest-word multiplicities: 2,880 states with one word, 960 with
  two, 2,208 with three, and 160 with six.
- Narrowed the conclusion to geodesic length-three equality; did not claim a
  full presentation or absence of other length-six relators.
- Added no production implementation, timing, GPU run, or optimization.

## 2026-08-28: REF-028 first non-trace Megaminx geodesics

- Used one final tractable raw-word audit at length four rather than inventing
  a compact canonicalizer before evidence required one.
- Enumerated all 292,008 non-backtracking length-four words and retained
  274,224 whose full 120-entry endpoints lie in exact `F4`.
- Separated 274,224 shortest words, 139,248 records generated from unique `F3`
  parents, and 90,144 states; these projections lose different multiplicities.
- Found 90,624 static commutation classes: 89,664 states have one class and 480
  states have two, so `F4` is the first cross-class geodesic layer.
- Classified every one of the 480 class pairs as
  `g(h k h^-1)=(h k h^-1)g`, with no unclassified remainder at this layer.
- Interpreted the result as conjugated or transported independence: a fixed
  alphabet-level commutation matrix becomes incomplete even though state-only
  BFS distance and visited semantics remain exact.
- Recorded the failed evidence honestly: one Docker Engine `unexpected EOF`,
  one inherited fixture-variable omission, and two formatting-only gates; the
  final read-only Docker gate passed five tests, formatting, compile, and run.
- Added no production implementation, GPU run, performance claim, or
  optimization.

## 2026-08-28: trace quotient and group-equality literature pass

- Connected REF-027's adjacent-swap closure to the Cartier-Foata partially
  commutative monoid and Mazurkiewicz trace interpretation.
- Formalized the sound map `words -> static traces -> evaluated group element`:
  trace equality certifies endpoint equality, while REF-028 rejects the
  converse even among same-length geodesics.
- Used Green's graph-product normal-form theorem to mark the hypothetical
  commutator-only world in which reduced representatives are unique up to
  syllable shuffling; explicitly did not identify the Megaminx group with a
  graph group.
- Distinguished global noninjectivity from the bounded geodesic question:
  inverse and order-five relations already collapse words globally, whereas
  F4 is the first same-depth geodesic fiber with two static trace classes.
- Corrected an imprecise interpretation from note 65/REF-028: conjugation
  creates a composite group element in a centralizer; it does not make a fixed
  right-Cayley generator state-dependent.
- Recorded that relation classification is richer than the state equality BFS
  needs, so this theory is explanatory rather than an implied hot-path design.

## 2026-08-28: REF-029 Cube QTM order-four signatures

- Audited clean external CayleyPy and DeepCubeA Cube3 sources, recording their
  commits, source hashes, move counts, state layouts, and action construction.
- Transcribed CayleyPy's explicit six 54-sticker face cycles into a small Rust
  fixture, generated the 12 signed QTM moves, and verified order four and every
  inverse pair.
- Matched the published exact QTM sphere prefix `1,12,114,1068,10011` through
  F4 with full equality and no hashing.
- Compared the unique-sticker Cayley action against CayleyPy's repeated-color
  orbit action; equal balls through radius four exclude any nonidentity color
  stabilizer word of length at most eight, but do not prove global freeness.
- Found Cube's first non-trace geodesic equalities at F2: exactly the six
  order-four witnesses `g^2=g^-2`, unlike Megaminx's order-five F2 same-level
  edges.
- Closed words under static commutation plus that half-turn rewrite and reduced
  the F2/F3/F4 class remainders from 6/108/1,521 to zero.
- Recorded the conceptual lesson that translated and composed occurrences of a
  short relation can dominate later duplicate counts without constituting new
  primitive relation families.
- Retained failures: lost Docker-cell output, absent NumPy for external Python
  smoke, intentional RED compiles, and a formatting-only gate.
- Added no production implementation, GPU run, timing, or optimization.

## 2026-08-28: REF-030 QTM/HTM visited-boundary work

- Reused the exact REF-029 Cube action but changed the generator manifest from
  12 QTM quarter turns to 18 HTM quarter/half turns, treating them as two graph
  definitions rather than two execution modes.
- Reproduced the published QTM and HTM exact sphere prefixes through F4.
- Partitioned every labeled expansion occurrence through F3 into backward,
  same-level, older-ball, and forward categories, then split forward records
  into unique next states and duplicate extras.
- Observed zero older-ball occurrences in both metrics, matching the general
  undirected layer-distance bound.
- Observed no QTM same-level work but HTM counts 36, 540, and 7,128 due to unit
  half turns destroying quarter-turn bipartiteness.
- Verified occurrence-level conservation: backward records from each layer
  equal forward candidate records across the preceding layer boundary.
- Recorded that HTM/QTM generated-work ratios cannot be inferred from the 1.5
  degree ratio because frontier geometry also changes; equal depths are also
  different metric radii and not a speed comparison.
- Passed six Docker/Rust tests after intentional RED and one formatting-only
  gate; added no timing, GPU code, production implementation, or optimization.

## 2026-08-28: Cartesian-product BFS and initial REF-031 preparation

- Avoided repeating the already developed BFS-order and directed-graph notes;
  selected Cartesian products as an uncovered foundational topic.
- Proved that Cartesian-product distance is the sum of coordinate distances
  and that its sphere sequence is the discrete convolution of factor spheres.
- Derived the binomial shuffle factor for shortest-path counts, separating one
  product state from its potentially many coordinate-interleaved histories.
- Used the strong product as a contract counterexample: allowing one diagonal
  transition changes the distance law rather than merely execution cost.
- Prepared a small Rust oracle with four semantic tests and no performance or
  GPU work, but did not run it natively when Docker Desktop was unavailable.
- Recorded permission, missing-engine, stalled-start, and stopped-service
  evidence; REF-031 was left explicitly pending rather than prematurely passed.

## 2026-08-28: REF-031 Docker completion

- Rechecked rather than assuming the old infrastructure state; Docker Engine
  `29.3.1` had recovered.
- Retained two new failures: login-shell PATH hid `rustc`, and the base image
  lacked the `rustfmt` component after four semantic tests had already passed.
- Installed `rustfmt` only inside the disposable container and completed the
  read-only-mounted formatting, four-test, compile, and executable gate.
- Observed exact convolution `[1,3,4,3,1]`, far endpoint distance four with 12
  shortest paths, and Cartesian/strong diagonal distances two/one.
- Preserved the raw output and exact image digest; made no timing, GPU, puzzle
  factorization, or optimization claim.

## 2026-08-28: direct-product Cayley BFS

- Specialized Cartesian-product geometry to the exact tagged generator set
  `(S x {e}) union ({e} x T)` under consistent Cayley action conventions.
- Extended additive distance and sphere convolution to directed positive-monoid
  reachability, retaining infinity for an unreachable factor coordinate.
- Derived cross-factor commutation and the binomial shuffle contribution to
  shortest labeled-word counts.
- Identified the depth-two BFS signature: two distinct factor-parent records
  converge at the state reached by `st=ts`.
- Rejected three tempting converses: commutation alone need not give an internal
  direct product, a diagonal generator changes the product metric, and a
  Schreier quotient can introduce coupled visible-state equality.
- Kept semantic equality separate from physical adjacency in a warp, batch, or
  owner partition; no optimization or implementation was proposed.
- Attempted the user-authorized expert workflow, but the external Telegram call
  was rejected by the data-protection boundary before any expert received it.

## 2026-08-28: arbitrary BFS frontier profiles

- Strengthened the earlier non-unimodality warning to a constructive theorem:
  every finite positive sequence beginning with one is a rooted sphere profile
  of a finite connected simple tree when degree is unrestricted.
- Isolated the only unrestricted shape obstruction: after the first empty exact
  frontier, every later frontier must remain empty.
- Derived the sharp maximum-degree tree conditions `a_1<=Delta` and
  `a_(i+1)<=(Delta-1)a_i`, distinguishing undirected degree from directed
  out-degree.
- Noted that oscillation needs neither cycles nor duplicate convergence; an
  irregular tree already suffices.
- Showed why even the complete frontier-size sequence cannot recover edge work,
  same-level edges, convergence, or the shortest-path DAG.
- Kept the construction separate from regular and Cayley realizability, which
  remains a genuinely narrower open problem.

## 2026-08-28: Cayley dead ends and radial progress

- Rejected a weakly sourced web claim of universal finite vertex-transitive
  frontier unimodality instead of promoting it without proof.
- Used primary papers by Cleary-Taback, corrected Cleary-Riley, and Lehnert to
  establish dead ends as generator-dependent Cayley geometry with potentially
  unbounded depth in stated group families.
- Adopted an explicit escape-depth convention to avoid the literature's
  off-by-one and strong/retreat-depth ambiguity.
- Verified algebraically that `1` is a depth-two interior dead end in
  `Cay(Z,{+/-2,+/-3})`, while standard `+/-1` has none.
- Separated full regular generator work from zero outward and accepted work at
  a dead-end parent.
- Proved that dead ends cannot lie internally on root geodesics and that one
  parent-local zero yield is not a global BFS termination certificate.
- Made no claim that the inspected puzzle metrics contain dead ends and added
  that as a bounded future question rather than an implementation task.

## 2026-08-28: dead ends inside intersection profiles

- Integrated dead ends into note 32 rather than duplicating its existing
  per-vertex `c/a/b` framework: dead ends are precisely the `b(v)=0` mass.
- For layer size `m`, outward incidence `E`, maximum forward degree `B`, and
  dead count `z`, derived `m-z<=E<=B(m-z)` and the corresponding fraction
  bounds.
- Constructed two four-parent boundary profiles, `[1,1,1,1]` and
  `[4,0,0,0]`, with identical frontier sizes, total forward work, unique next
  states, and child backward multiplicities but zero versus three dead ends.
- Strengthened the measurement lesson: aggregate boundary work does not reveal
  whether outward progress is distributed or concentrated through gateways.
- Added no code, hardware claim, or scheduling recommendation.

## 2026-08-28: leaf and dead-end semantics

- Avoided duplicating the existing eccentricity/two-sweep analysis and instead
  isolated a terminology problem in BFS logs.
- Distinguished graph leaves, leaves of one selected BFS parent tree, radial
  dead ends, completed terminal-layer vertices, and execution-policy zeros.
- Showed why parent tie-breaking can create a BFS-tree leaf even when the same
  vertex has outward graph neighbors.
- Required complete successor coverage before interpreting zero emitted output
  as geometric `b(v)=0`; target stopping, pruning, overflow, and loss remain
  alternative explanations.
- Clarified that every vertex of a finite connected Cayley graph is peripheral
  as a possible root, while radial dead-end status remains relative to the
  fixed identity root.

## 2026-08-28: FIFO occupancy versus frontier width

- Proved the strict mark-on-enqueue FIFO invariant: the queue contains an
  unprocessed suffix of `F_d` followed by a discovered prefix of `F_(d+1)`.
- Derived exact occupancy `Q_k=|F_d|-k+D_k`, exposing the order-sensitive
  prefix union of next-layer endpoints.
- Established the sharp nonempty-layer bound from `max(m,n)` through `m+n-1`.
- Constructed one fixed tree with frontier profile `[1,100,100]` whose queue
  peak is 199 with its productive parent first and 100 with it last.
- Separated semantic frontier width, FIFO records, candidate occurrences, bulk
  current/next buffers, persistent visited state, and distributed local peaks.
- Scoped the theorem away from mark-on-dequeue duplicate queues, asynchronous
  reactivation, pruning, and incomplete traversal.
- Added no executable implementation, timing, or ordering recommendation.

## 2026-08-28: discovery, settlement, and duplicate queues

- Formalized exact claim-before-enqueue and duplicate-tolerant
  settle-on-dequeue as two distinct physical contracts.
- Proved that strict FIFO first settlement still gives shortest unweighted
  distances when stale copies are suppressed and first-settled states expand
  complete successors.
- Used a complete bipartite layer boundary to separate `mn` queued occurrences
  from `n` semantic next states; equal-width layers yield quadratic record
  amplification without changing BFS distances.
- Distinguished generated, materialized, enqueued/routed, claimed, settled,
  stale-pop, and accepted counters.
- Reconnected visited claim to recoverable work publication: a claimed but
  unpublished state is an exactness failure, not successful deduplication.
- Kept distance agreement separate from arbitrary-parent, canonical-parent,
  shortest-DAG, and path-count outputs.
- Added no queue implementation, GPU kernel, or performance recommendation.

## 2026-08-28: duplicate queue complexity boundary

- Proved that settle-on-dequeue with stale suppression retains explicit-graph
  `O(|R|+A_R)` work: every semantic vertex expands once and every generated
  adjacency occurrence creates at most one record and pop.
- Separated that work bound from an `O(A_R)` physical queue-memory bound; the
  complete-bipartite boundary attains linear-in-adjacency live occurrences.
- Reconciled quadratic-in-layer-width queue memory with linear-in-input-edge
  work rather than treating them as contradictory claims.
- Constructed a two-vertex-per-layer directed DAG where expanding stale copies
  produces `2^D` records despite only `O(D)` unique vertices and edges.
- Identified duplicate re-expansion as walk-tree enumeration, not ordinary
  state-graph BFS; without settled-target filtering, cycles can make it
  nonterminating without a depth bound.
- Added `stale pops` versus `stale expansions` to the counter vocabulary and
  required the latter to remain zero under this graph-BFS contract.

## 2026-08-28: REF-032 duplicate queue oracle

- Wrote three semantic tests before schedule implementations and retained the
  expected three-test `unimplemented!` RED run.
- Corrected the layered stale-suppressed peak expectation from four to six
  before implementation after noticing mixed current-stale and next records.
- On one 201-state graph, observed unique-claim versus delayed-settlement queue
  peaks 199 versus 10,000 and enqueues 201 versus 10,101, with identical
  distances and unique expansion count.
- On the 25-state layered DAG, stale suppression retained 25 expansions and 22
  stale pops, while path-prefix expansion doubled to 4,096 records at depth 12.
- Retained a formatting-only failed gate, final three-test pass, raw output,
  exact Docker image/toolchain, and source hash.
- Added no timings, GPU code, optimized data structure, or schedule
  recommendation.

## 2026-08-28: GraphBLAS matrix orientation and directed BFS

- Used the official GraphBLAS C specification, API design paper, and LAGraph
  source to distinguish `vxm` (`v^T A`) from `mxv` (`A v`).
- Under source-row adjacency, proved row `fA` gives successors while column
  `Af` gives predecessors; forward column BFS therefore needs `A^T f`.
- Added the directed path `0->1->2` as a minimal orientation witness and showed
  why a complemented visited mask cannot repair reversed support.
- Identified undirected and inverse-closed Cayley fixtures as false-comfort
  tests because `A=A^T` hides a transpose error.
- Connected explicit transpose storage to implicit predecessor generation via
  inverse right-action transformations without adding inverses to the forward
  metric.
- Kept algebraic orientation separate from physical transpose representation
  and made no implementation or performance recommendation.

## 2026-08-28: GraphBLAS masks, replace, and sparse support

- Expanded note 25's one-line mask caution into explicit structural-versus-
  valued behavior for stored false tuples under complement.
- Used `0->1` to show that reusing a frontier output without replace can retain
  old `{0}` beside new `{1}`, while replace yields the exact delta `{1}`.
- Separated output replacement, accumulator, initialization, and logical input
  snapshot semantics; an immediate in-place cascade would cross several hops.
- Distinguished stored tuple count from true-valued Boolean support, so `nvals`
  is an emptiness certificate only under a no-explicit-false invariant.
- Defined the full GraphBLAS BFS operation contract across orientation,
  semiring, mask, replace, accumulator, stored-zero, alias, and termination
  fields.
- Added no GraphBLAS implementation, benchmark, or optimization choice.

## 2026-08-28: multi-source superposition boundary

- Proved exact superposition of balls:
  `B_d(S union T)=B_d(S) union B_d(T)`.
- Derived the non-superposing frontier formula that removes every separate
  depth-`d` state already reached earlier by the other source set.
- Used path `0--1--2` to reject layerwise union: separate depth-two union is
  `{0,2}`, while combined multi-source depth two is empty.
- Located the nonlinearity in shared visited/delta subtraction rather than the
  Boolean neighbor operator, which does distribute over union.
- Restricted combined path counts to nearest-source contributions and kept
  scalar distance, one tie label, all tied labels, and per-source fields as
  distinct output contracts.
- Made source-set composition part of workload identity and added no parallel
  implementation or performance recommendation.

## 2026-08-28: source-set updates and layer migration

- Proved distance decrease and ball inclusion under source insertion, then
  separated those monotone quantities from nonmonotone exact frontiers.
- Used path `0--1--2--3--4` to move depth-two membership from `{2}` to disjoint
  `{4}` at unchanged cardinality after adding source `2`.
- Recorded source deletion as the reverse scalar monotonicity with the same
  layer-migration and membership-validation problem.
- Distinguished unchanged-distance tied-path additions from decreased-distance
  replacement of the entire shortest-path sample space.
- Rejected old Boolean visited as sufficient incremental state: decreased
  labels may need reactivation, while deletion may need alternatives absent
  from one stored parent tree.
- Added the exact source set to checkpoint/search epoch identity and separated
  it from unchanged physical ownership.
- Added no incremental implementation, GPU code, or scaling claim.

## 2026-08-28: BFS landmarks and Cayley distance coordinates

- Reinterpreted a complete BFS distance array as one exact metric coordinate
  field rather than only one source-to-target answer.
- Proved the undirected landmark lower and upper bounds and their monotone
  combination across several landmarks.
- Used the four-cycle to show that equal landmark coordinates can leave a zero
  lower bound for a pair at distance two.
- Separated directed distances from and to a landmark; the latter requires a
  reverse/transposed traversal rather than reuse of one forward field.
- Preserved bounded-BFS `UNKNOWN` semantics instead of treating a missing
  coordinate as an exact finite value.
- Derived the stronger Cayley identity `d(g,h)=d(e,g^-1 h)` from left
  translation and explained why a complete identity table is exact all-pairs
  data in that specific model.
- Marked the Schreier boundary: non-free actions may turn the relative target
  into a stabilizer coset, so the Cayley lookup formula cannot be copied by
  vertex-transitivity alone.
- Added no implementation, benchmark, landmark-selection policy, or GPU
  optimization.

## 2026-08-28: resolving sets and BFS coordinate injectivity

- Distinguished exact landmark coordinates from injectivity of their joint
  distance-vector map.
- Used `P4` to show that joint multi-source distance is a lossy minimum even
  when two independent landmark fields resolve every vertex.
- Proved from BFS layers that a single landmark resolves a connected graph if
  and only if the graph is a path rooted at an endpoint.
- Derived the Cayley corollary that a finite connected undirected Cayley graph
  of order greater than two needs at least two resolving landmarks.
- Calibrated metric dimension on paths, cycles, and complete graphs and added
  the elementary `(D+1)^q` counting lower bound.
- Separated trivial pointwise automorphism stabilizer as a necessary, not
  sufficient, resolving condition.
- Clarified that a complete Cayley identity table can synthesize arbitrary
  landmark coordinates for known group elements but its scalar values do not
  themselves identify elements on the same sphere.
- Added no search implementation, landmark optimizer, benchmark, or GPU code.

## 2026-08-28: BFS distance embeddings and strong resolution

- Recast the maximum landmark-coordinate difference as a nonexpanding
  `l_infinity` metric lower bound.
- Proved the full distance-vector Fréchet embedding is isometric by selecting
  one queried endpoint as a coordinate, and noted that one arbitrary coordinate
  can be omitted.
- Used two landmarks on `C5` as a resolving but non-isometric map: pair `2,4`
  contracts from graph distance two to coordinate distance one.
- Derived equality in one coordinate from triangle equality and connected it
  exactly to strong resolution by a shortest path containing the other vertex.
- Distinguished ordinary metric dimension from strong metric dimension.
- Generalized singleton coordinates to subset-distance coordinates produced by
  independent multi-source BFS runs, while preserving the union/minimum loss.
- Kept directed distances outside the symmetric-norm theorem instead of
  silently symmetrizing them.
- Added no implementation, performance experiment, storage design, or GPU
  optimization.

## 2026-08-28: BFS-tree root exactness versus pairwise stretch

- Separated exact root distances in a BFS parent tree from arbitrary-pair
  distances in that tree.
- Derived the root-detour bound `d_T(u,v)<=2 ecc(s)` and expressed pair distance
  through the selected lowest common ancestor.
- Used odd cycle `C_(2r+1)` to attain the bound on one original edge: its graph
  distance is one and its unavoidable BFS-tree distance is `2r`.
- Built a five-vertex tie witness where valid parent choices change one pair's
  tree distance from two to four while every BFS label stays fixed.
- Distinguished one parent tree, the shortest-root-path predecessor DAG, and
  original adjacency as three different retained-information contracts.
- Interpreted a Cayley parent tree as selected geodesic normal forms whose
  common-prefix metric need not equal the relative-element word metric.
- Limited distributed parent replay evidence to root-path correctness rather
  than silently claiming all-pairs geometry.
- Added no tree optimizer, spanner construction, benchmark, or GPU code.

## 2026-08-28: BFS fundamental cycles and Cayley relation witnesses

- Turned every non-tree edge into its unique tree-path fundamental cycle and
  expressed its length through endpoint depths and LCA depth.
- Proved that same-layer chords yield odd fundamental cycles and adjacent-layer
  non-parent edges yield even ones.
- Derived the `m-n+1` cycle-space dimension and proved fundamental-cycle
  independence/spanning over binary symmetric difference.
- Kept cycle-space dimension invariant while identifying parent-dependent
  basis shapes and lengths.
- Translated a labeled Cayley non-tree transition into the identity word
  `p(u)x p(v)^-1` and separated the LCA-based cycle word from its conjugate
  identity-based form.
- Rejected binary cycle-space data as a group presentation: it loses labels,
  order, direction, multiplicity, and stabilizer meaning.
- Scoped bounded/distributed cycle evidence to observed endpoints, parent
  chains, closing transitions, and one consistent epoch.
- Added no cycle-basis optimizer, group-presentation inference code,
  benchmark, or GPU implementation.

## 2026-08-28: BFS fundamental cuts and bridge evidence

- Defined the fundamental cut of each BFS-tree edge as the full original-edge
  boundary of its selected descendant subtree.
- Proved the exact bridge criterion: the tree edge is a bridge iff it is the
  only member of that cut, equivalently iff no fundamental cycle contains it.
- Derived the `n-1` fundamental-cut basis of the binary cut space.
- Proved cycle-cut even-intersection orthogonality and specialized it to zero or
  two intersections between fundamental cycles and cuts.
- Separated parent-subtree, BFS-ball/frontier, and distributed-owner boundaries
  by the state that defines each partition.
- Rejected bounded absence of an alternate crossing as a global bridge
  certificate and carried consistent-epoch requirements into distributed
  evidence.
- Interpreted a Cayley subtree as a selected normal-form prefix class rather
  than silently treating it as a subgroup coset.
- Added no bridge algorithm, cut optimizer, benchmark, or GPU code.

## 2026-08-28: directed BFS, SCCs, and condensation distance

- Proved that forward and transpose reachability intersect to exactly the SCC
  containing one root, while two BFS runs do not provide a full decomposition.
- Used `0->1->2` to reject forward completion as evidence of mutual
  reachability.
- Proved SCC condensation is acyclic and that quotient paths lift through
  internal strong-component paths.
- Identified condensation BFS distance with minimum cross-SCC arc count, or a
  0-1 metric that makes every internal SCC arc free.
- Used a directed cycle to show arbitrarily large original distances collapse
  to zero after SCC contraction.
- Kept condensation BFS depth distinct from topological order and from the
  undirected adjacent-layer rule.
- Derived positive-alphabet Cayley SCCs as cosets of
  `H=<S>_+ intersect <S>_+^-1`.
- Separated finite groups, where positive powers recover generator inverses,
  from infinite `Z` with `{+1}`, whose SCCs are singletons.
- Added no SCC implementation, benchmark, or GPU optimization.

## 2026-08-28: BFS depth slack and directed period

- Introduced strongly connected digraph period as the GCD of directed cycle or
  closed-walk lengths and separated it from strong connectivity itself.
- Proved all root-to-vertex walk lengths share one residue modulo the period,
  giving cyclic classes advanced by every arc.
- Derived the exact BFS certificate
  `p=gcd_(u->v)(d(u)+1-d(v))` by walk congruence in one direction and
  telescoping around every cycle in the other.
- Calibrated the formula on a directed cycle, where only the closing arc has
  nonzero slack.
- Recovered undirected bipartiteness as period two for the symmetric digraph,
  versus period one when an odd cycle exists.
- Connected cyclic classes to adjacency-power support without turning BFS
  exhaustion into a mixing or eventual-walk claim.
- Derived the Cayley word-length homomorphism to `Z_p` and its inverse-closed
  consequence `p|2`.
- Separated easy distributed GCD reduction from hard evidence obligations:
  complete arc coverage, finalized labels, and one ownership epoch.
- Added no period implementation, performance measurement, or GPU code.

## 2026-08-28: eventual walk lengths and BFS first arrival

- Proved pairwise eventual completeness in the one period-compatible residue by
  concatenating a fixed path with a finite GCD-generating set of closed walks.
- Defined primitive digraphs and separated their exact-length exponent from BFS
  diameter; recorded Wielandt's sharp `(n-1)^2+1` general bound.
- Explained why periodic cyclic classes prevent any single adjacency power from
  having full support while compatible blocks become eventually complete.
- Built two three-vertex period-one graphs with identical root BFS depths but
  different exponents, rejecting distances plus period as sufficient transient
  information.
- Separated exact-length recurrence `W_(k+1)=Post(W_k)` from BFS delta
  recurrence with visited subtraction.
- Distinguished immediate even inverse padding from eventual all-length
  availability in an aperiodic graph.
- Applied the result to positive Cayley word-length spectra without confusing
  longer relation-padded words with new states or shortest paths.
- Marked periodic non-quiescence as expected for walk support, not a failed BFS
  termination detector.
- Added no propagation implementation, exponent calculator, benchmark, or GPU
  optimization.

## 2026-08-28: BFS walks versus exact-length simple paths

- Proved shortest unit-edge walks are simple by deleting any repeated-vertex
  closed segment, explaining why ordinary BFS needs no path-history key.
- Separated endpoint-mergeable exact-length walk support from history-sensitive
  exact-length simple paths.
- Built a five-vertex witness with two equal-depth arrivals at `v`, only one of
  which may legally continue through `b->t`.
- Made `(v,U)` the direct product-state contract for vertex-simple histories
  and recorded its `n 2^(n-1)` raw state bound.
- Located Hamiltonian Path at requested length `n-1`, while preserving the
  fixed/parameterized-`k` qualification and color-coding literature.
- Kept nonbacktracking, freely reduced, state-simple, and geodesic paths
  distinct in Cayley graphs.
- Limited endpoint owner hashing to routing rather than semantic deduplication
  when used-set history changes legality.
- Added no simple-path solver, subset representation, benchmark, or GPU code.

## 2026-08-28: trails, edge history, and line digraphs

- Positioned trails between walks and vertex-simple paths and kept shortest
  trail semantics equal to ordinary shortest-path semantics.
- Made `(v,F)` with a per-history used-edge set the direct exact trail state.
- Built a six-vertex equal-length-arrival witness where only one history may
  reuse continuation `v->x->t` without repeating an arc.
- Proved directed trails correspond to simple paths in the directed line graph,
  while noting that the transformation moves rather than removes history.
- Used a three-edge star to reject the same lifting claim for an ordinary
  undirected line-graph path without orientation state.
- Separated generic trail history from the special degree/connectivity structure
  of Eulerian traversal.
- Defined Cayley trail edge identity across source, generator label, parallel
  occurrence, and inverse-paired undirected orientation.
- Derived Euler-circuit existence for finite strongly connected labeled directed
  Cayley multigraphs from in/out balance.
- Added no trail solver, Euler implementation, benchmark, or GPU code.

## 2026-08-28: BFS trees, shortest gateways, and dominators

- Separated one selected BFS parent path, all shortest paths, and all graph
  paths as three different evidence universes.
- Proved that a dominator is an ancestor in every BFS tree, then rejected the
  converse with a diamond whose parent tie chooses only one branch.
- Used a branch-and-rejoin graph to show that an immediate-dominator edge need
  not be a graph edge or connect adjacent BFS layers.
- Built a longer-bypass witness where a vertex is mandatory on every shortest
  path but is not a full dominator.
- Recorded the all-predecessor dominator fixed point and explained why a
  previous-layer pass computes only shortest-path unavoidability.
- Added the deletion characterization and a qualified exact shortest-path-count
  certificate.
- Identified incomplete incoming-edge views as a source of false dominators,
  even when reachability and shortest distance appear unchanged.
- Rejected the idea that strong connectivity or Cayley transitivity alone
  removes nontrivial directed dominators, using a one-way Cayley cycle.
- Added no dominator implementation, benchmark, GPU kernel, or optimization.

## 2026-08-28: BFS separators, dominators, and Menger paths

- Placed dominators inside Menger's min-max relation between internal vertex
  separators and internally vertex-disjoint directed paths.
- Identified each non-endpoint dominator with a singleton source-target
  separator and distinguished a chain of such separators from one cut value.
- Rejected path-count multiplicity as a proxy for route independence using two
  branches that rejoin at an unavoidable vertex.
- Derived `kappa(s,t) <= |S_i|` for every intermediate BFS sphere and built an
  arbitrarily wide frontier followed by a size-one bottleneck.
- Applied the same theorem separately to the shortest-path DAG and full graph,
  preserving the longer-bypass counterexample.
- Kept vertex-disjoint and edge/arc-disjoint resilience distinct under Cayley
  labels and parallel occurrences.
- Contrasted the one-way directed Cayley cycle with the two internally disjoint
  routes of an undirected cycle.
- Separated physical distributed redundancy from semantic path disjointness.
- Added no flow solver, disjoint-path implementation, benchmark, or GPU code.

## 2026-08-28: reverse BFS, postdominators, and inevitable targets

- Defined postdominance only under an explicit single-exit and exit-reachable
  universe, avoiding vacuous claims for states with no exit path.
- Derived postdominance exactly as dominance in the reversed graph.
- Rejected reverse BFS parents as postdominator evidence with a reversed
  diamond and separated shortest-suffix gateways using a longer bypass.
- Used `v->v` plus `v->z` to show that exit postdominance does not imply
  inevitable termination.
- Characterized adversarial finite-graph inevitability for an absorbing target
  by the absence of avoiding cycles and non-target dead ends.
- Distinguished per-exit analysis from a virtual exit spanning all terminating
  choices.
- Recorded a five-level contract ladder from reverse reachability through
  maximal-path liveness.
- Preserved asymmetric Cayley reverse-transition and history-product semantics.
- Separated distributed search quiescence from termination of every graph path.
- Added no postdominator implementation, liveness checker, benchmark, or GPU
  code.

## 2026-08-28: reachability-preserving graphs and BFS metric

- Separated transitive closure/reduction's reachability contract from BFS's
  unit-distance and layer contract.
- Used a directed chain and its closure to preserve every reachable pair while
  changing diameter from `n-1` to one and the first frontier from one to `n-1`.
- Recorded distance monotonicity under arc addition/deletion while rejecting
  equality of intermediate BFS evidence.
- Showed that one reachability-redundant shortcut can destroy dominators and
  raise path independence.
- Preserved SCC partition and component reachability order while rejecting a
  preserved condensation metric.
- Distinguished unit graph powers/closure from weighted replayable macros.
- Proved the bounded-distortion relation between nested Cayley generator sets
  and used the all-elements alphabet as an extreme diameter-one example.
- Kept positive-semigroup reachability distinct from group generation with
  formal inverses.
- Marked generator-set changes as graph/workload changes in GPU and multi-GPU
  comparisons.
- Added no closure, reduction, macro-generator, benchmark, or GPU code.

## 2026-08-28: Cayley word metrics and generator changes

- Generalized the nested-generator inequality to two arbitrary finite symmetric
  generating sets using maximum word-substitution lengths in both directions.
- Derived mutual BFS-ball inclusions after linear radius rescaling.
- Separated invariant coarse growth class from generator-dependent exact
  spheres, growth series, girth, relations, and parents.
- Used `Z` with step-one versus step-one-and-two alphabets as an immediate
  layer and girth counterexample.
- Identified existential quasi-isometry as nearly vacuous for one finite graph
  and required uniform constants for puzzle-family scaling claims.
- Transferred substitution bounds to Schreier action distances while preserving
  stabilizer and state-identity qualifications.
- Required bounded mutual positive-word simulation for directed alphabets and
  rejected formal inverse generation as sufficient on infinite `Z`.
- Separated bounded macro-word replay from geodesic or canonical replay.
- Kept geometric distortion bounds distinct from GPU edge work, frontier peak,
  synchronization, and communication.
- Added no generator selector, Cayley implementation, benchmark, or GPU code.

## 2026-08-28: BFS boundaries, Følner sets, and amenability

- Identified the exact BFS next frontier with the external boundary of one
  special set, the completed metric ball.
- Defined the Følner condition and emphasized its existential quantifier over
  arbitrary finite subsets rather than BFS balls.
- Rejected amenability as sufficient evidence that ordinary BFS balls have
  vanishing frontier-to-visited ratio.
- Derived proportional frontier width and exponential ball growth from a
  positive external-vertex isoperimetric constant.
- Preserved the failed converse using amenable groups of exponential growth.
- Calibrated the ratio exactly on the integer line and a standard free-group
  Cayley tree.
- Separated finite saturation from uniform pre-saturation expansion across a
  graph family.
- Kept Cayley-group amenability distinct from one Schreier action's boundary
  geometry.
- Separated semantic vertex/edge boundaries from owner cuts, duplicate work,
  and elapsed GPU performance.
- Added no Følner constructor, expansion estimator, benchmark, or GPU code.

## 2026-08-28: BFS versus random-walk hitting and cover time

- Separated deterministic BFS distance from expected hitting, mixing, and cover
  times under a declared Markov transition rule.
- Used path hitting time `(n-1)^2` versus distance `n-1` to expose repeated
  backtracking cost.
- Rejected a long no-new-visit interval as a closure or unreachable
  certificate.
- Derived the complete-graph coupon-collector cover time despite depth-one BFS
  discovery, separating stationary mass from historical coverage.
- Recorded the tight cubic worst-case cover-time scale and lollipop boundary.
- Kept uniform Cayley stationarity distinct from coverage, geodesic discovery,
  and periodicity.
- Rejected universal linear speedup from multiple independent walkers and
  separated global visit dedup from minimum-depth scheduling.
- Preserved directed positive-alphabet, parallel-label, and nonreversible chain
  qualifications.
- Classified walker GPU throughput as sampling work rather than exact BFS
  progress.
- Added no random-walk engine, sampler, benchmark, or GPU code.

## 2026-08-28: BFS flooding, rumor spreading, and message time

- Proved first-receipt round equals hop distance under reliable synchronous
  all-neighbor unit-delay flooding.
- Identified informed-set deltas with BFS spheres and first senders with valid
  tied BFS parents.
- Equated last delivery round with source eccentricity while separating semantic
  finish time from knowledge of completion.
- Changed fixed heterogeneous delay into a weighted earliest-arrival metric and
  kept queue/temporal effects outside that static model.
- Rejected one-shot completeness under message loss and timeout silence as an
  unreachable certificate.
- Separated randomized sparse-contact rumor spreading from all-edge BFS even on
  a complete communication graph.
- Restricted first-receipt finalization to schedules whose timing metric makes
  it safe; arbitrary asynchronous proposals need improvements/reactivation.
- Translated the contract to multi-GPU Cayley routing without identifying
  logical states with physical processes.
- Classified sampled generators/peers as partial or random exploration under
  the original graph contract.
- Added no flooding protocol, rumor simulator, benchmark, or GPU code.

## 2026-08-28: BFS balls, separators, and graph ends

- Defined ends as ray classes inseparable by finite vertex deletion in connected
  locally finite undirected graphs.
- Used complements of nested BFS balls to expose progressively refined infinite
  components and proved their count cannot decrease.
- Bounded currently visible infinite components by the next frontier size while
  rejecting frontier width as an end count.
- Calibrated rays, the integer line, the square grid, and regular trees.
- Constructed identical bounded BFS prefixes with one, infinitely many, or zero
  eventual ends, rejecting finite-prefix certification.
- Recorded the Freudenthal-Hopf `0,1,2,infinity` classification and Stallings
  structural context without attempting a splitting algorithm.
- Made group end count generator-independent through quasi-isometry while
  preserving generator-dependent sphere traces.
- Separated Cayley, Schreier, finite-orbit, and directed-end semantics.
- Kept end topology distinct from frontier memory and physical owner routing.
- Added no end detector, group-splitting procedure, benchmark, or GPU code.

## 2026-08-28: BFS on percolated Cayley graphs

- Conditioned on one bond/site realization so that BFS remained an ordinary
  exact traversal of a random input graph.
- Separated quenched frontiers and clusters from annealed expectations and
  survival probabilities.
- Derived the exact regular-tree frontier expectation and critical threshold
  through a Galton-Watson process.
- Used critical almost-sure extinction despite constant expected generations to
  reject expectation as per-run survival evidence.
- Kept general graph cycles, duplicates, bottlenecks, and dependence outside the
  exact tree offspring model.
- Distinguished infinite clusters, finite-family giant components, and survival
  through one tested radius.
- Preserved distributional Cayley transitivity while rejecting symmetry of an
  individual open cluster.
- Separated static edge percolation from retriable message loss and temporal
  resampling.
- Required random edge identity and statistical tails to remain separate from
  capacity/communication failures in GPU evidence.
- Added no percolation simulator, trial harness, benchmark, or GPU code.

## 2026-08-28: BFS, geodesic languages, and automatic Cayley graphs

- Separated regularity, coverage, uniqueness, geodesicity, and prefix closure
  instead of treating "finite automaton" as an exact BFS certificate.
- Proved that accepted words at length `r` equal a Cayley BFS sphere only under
  coverage, uniqueness, and geodesicity.
- Identified prefix closure as the extra condition that turns those words into
  a canonical online geodesic prefix tree.
- Derived the DFA matrix count and rational accepted-word generating series,
  while keeping it distinct from spherical growth without the semantic proof.
- Added minimal counterexamples: padded unique representatives of `Z`, parallel
  geodesic labels on an integer line, and a freely reduced but nongeodesic word
  in `C_3`.
- Kept automatic structures distinct from arbitrary regular languages and from
  necessarily unique or geodesic normal forms.
- Recorded why finite-instance regularity says nothing about family scaling.
- Applied the stabilizer distinction: unique group normal forms may collide as
  CayleyPy puzzle states in a Schreier action.
- Preserved exact state equality as the bidirectional meeting predicate rather
  than confusing it with automaton-state equality.
- Made no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS level graphs, blocking flows, and matching phases

- Treated BFS as a temporary metric-skeleton builder inside Dinitz and
  Hopcroft-Karp rather than as the complete optimization algorithm.
- Proved that level-graph `s-t` paths are exactly the current shortest residual
  paths.
- Distinguished a blocking flow from one augmentation and from a maximum flow
  inside the level graph.
- Proved strict growth of the next residual shortest-path length using the old
  BFS levels as a potential after every old admissible path is blocked.
- Recorded residual unreachability as a max-flow certificate under the proper
  residual and max-flow/min-cut contract.
- Interpreted Hopcroft-Karp as multi-source, multi-target BFS over a graph whose
  alternating orientation changes with the matching.
- Separated phase-local levels and dead branches from permanent `visited`
  semantics in a static graph.
- Rejected first-target stopping when a phase requires all tied shortest
  improvement routes.
- Kept frontier width, retained edges, augmentation count, and objective gain
  as different measurements.
- Added no flow/matching implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS profiles, color refinement, and isomorphism limits

- Separated the identity-preserving root distance map, distance partition, and
  layer-size histogram.
- Proved rooted isomorphism preserves a BFS profile while rejecting the
  converse with the triangular prism and `K_(3,3)`.
- Used the same pair to show that equal complete profiles can coexist with
  different triangle and bipartiteness structure.
- Used long cycles to reject global conclusions from any fixed-radius rooted
  neighborhood, not only from its histogram.
- Distinguished one row or histograms from the complete all-pairs distance
  matrix, which recovers simple-graph adjacency through distance one.
- Compared BFS with 1-WL/color refinement and showed how root individualization
  can split one BFS layer.
- Recorded uniform 1-WL's blind spot on same-order regular graphs.
- Applied Cayley translation symmetry to reject BFS profiles as state keys.
- Preserved layer statistics as useful one-way validation and diagnostic
  evidence rather than exact frontier-set proof.
- Added no isomorphism solver, refinement implementation, optimizer,
  benchmark, or GPU code.

## 2026-08-28: BFS, local message passing, and neural receptive fields

- Proved the radius-`r` locality bound for `r` rounds of strictly local message
  passing while rejecting receptive radius as faithful ball storage.
- Expressed exact BFS as synchronous `min` relaxation and Boolean fixed-point
  propagation under explicit exact-state assumptions.
- Identified the source marker as indispensable on homogeneous regular and
  Cayley graphs.
- Qualified standard MPNN expressiveness by the 1-WL upper bound and separated
  global, higher-order, positional, and individualized architectures.
- Reused the prism/`K_(3,3)` witness to expose uniform-feature local
  message-passing indistinguishability.
- Separated walk aggregation from BFS first arrival and `visited` subtraction.
- Related rapid frontier growth and bottlenecks to over-squashing without
  claiming universal exponential Cayley growth.
- Kept learned scores, embeddings, and distance estimates advisory unless an
  independent exactness or pruning proof exists.
- Distinguished feature-message throughput from exact unique-state BFS work on
  one or many GPUs.
- Added no neural model, BFS implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS, succinct graphs, and description size

- Separated compact description length, state bits, expanded vertices, and
  expanded arcs as independent complexity parameters.
- Distinguished circuit adjacency membership from a direct named-successor
  oracle.
- Recorded succinct directed reachability as PSPACE-complete by compact input
  size through configuration-graph hardness and Savitch-style membership.
- Explained why bounded branching avoids adjacency-row scanning but not
  exponential reachable volume or depth.
- Separated BFS's stored shortest-layer semantics from polynomial-space
  decision procedures that recompute subproblems.
- Identified explicit component, distance-table, parent-tree, and replay-path
  output as possible exponential lower bounds.
- Kept symbolic set compression conditional rather than universal.
- Preserved special algebraic algorithms as family-specific alternatives, not
  generic BFS consequences.
- Derived the logarithmic increase in feasible state bits from multiplicative
  multi-GPU memory growth under a full `2^n` universe model.
- Added no succinct solver, symbolic engine, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS versus topological waves and critical paths

- Separated multi-source BFS's minimum path recurrence from synchronous Kahn
  levelization's maximum predecessor recurrence.
- Proved Kahn wave rank equals longest source-to-vertex path length in a DAG.
- Used `s->v` and `s->u->v` as the minimal witness separating distance one from
  readiness wave two.
- Characterized graded DAGs as the clean equality case and identified BFS
  shortest-path DAGs as one such construction.
- Kept wave partitions distinct from chosen linear topological orders.
- Rejected BFS-depth sorting as a general topological-order procedure.
- Separated Kahn's residual-cycle certificate from BFS reachable-set
  exhaustion.
- Extended unit waves to weighted max-plus critical-path completion while
  preserving resource contention as an additional runtime constraint.
- Distinguished exact predecessor retirement from BFS first-discovery and
  duplicate-parent semantics.
- Added no topological scheduler, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS distance transforms and discrete wavefronts

- Interpreted multi-source BFS as a unit-edge Bellman arrival-time field whose
  frontiers are distance level sets.
- Equated repeated graph dilation with BFS balls under the same adjacency.
- Derived exact `L1` distance for four-neighbor grids and `L-infinity` distance
  for unit-cost eight-neighbor grids.
- Rejected both stencils as exact Euclidean distance and used `(2,1)` to reject
  exactness even after assigning diagonal cost `sqrt(2)`.
- Kept grid refinement distinct from changing the induced local norm.
- Separated obstacle-aware graph geodesics from straight-line geometry.
- Preserved source labels and Voronoi ties as outputs beyond a scalar distance
  transform.
- Distinguished graph BFS/Dijkstra from numerical Eikonal fast marching while
  retaining their common causal-front intuition.
- Translated the wavefront view to Cayley word metrics and generator relations.
- Added no distance-transform, PDE, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS versus union-find connectivity state

- Proved complete undirected BFS labeling and unioning every edge produce the
  same static component partition.
- Separated DSU representative pointers from graph edges, replay paths, and
  shortest-path parents.
- Used a path-plus-shortcut insertion to show unchanged connectivity with
  radically changed BFS distance.
- Distinguished insertion-only component merging from deletion-induced splits
  and bidirectional distance changes.
- Rejected endpoint union as directed reachability or SCC computation.
- Kept repeated connectivity-query workloads distinct from distance/path
  workloads.
- Required complete implicit edge discovery before treating DSU separation as
  a graph-level disconnection certificate.
- Applied the distinction to Cayley/Schreier orbit components versus word
  metric spheres.
- Separated parallel hooking/compression rounds from BFS depth barriers.
- Preserved exact state identity as a prerequisite to every union operation.
- Added no DSU, dynamic-connectivity, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS trees versus minimum spanning trees

- Separated root-relative distance preservation from root-free minimum total
  selected-edge weight.
- Proved every unit-weight spanning tree is an MST and used a path in `K_n` to
  reject MST optimality as BFS depth optimality.
- Added a weighted triangle where the unique MST is not an SPT and the SPT is
  not an MST.
- Constructed root-star/leaf-chain families exposing growing root-distance and
  total-weight objective gaps.
- Preserved the MST minimax-edge path property while separating it from
  additive distance and hop count.
- Kept BFS predecessor/distance certificates distinct from MST cut/cycle
  certificates.
- Applied unit-weight triviality to Cayley/Schreier spanning trees while
  retaining BFS geodesic normal forms as nontrivial information.
- Distinguished shared parallel primitives from end-to-end BFS versus MST
  semantics.
- Added no tree algorithm, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS distance versus effective resistance

- Separated one shortest-route length from minimum-energy flow over every
  undirected route.
- Proved equality on trees and `R_eff<=d` for connected unit-resistance graphs.
- Used parallel length-`L` paths to make resistance arbitrarily smaller than a
  fixed BFS distance.
- Calibrated direct-plus-two-edge alternatives and the complete graph.
- Rejected BFS profiles as resistance certificates because they omit internal
  and cross-layer route structure.
- Recorded the exact `2mR_eff` commute-time identity while keeping hitting and
  BFS first-arrival semantics separate.
- Distinguished electrical series/parallel algebra from minimum additive path
  weight.
- Added the Laplacian-pseudoinverse viewpoint without selecting a solver.
- Applied translation invariance to finite undirected Cayley graphs while
  requiring separate symmetry or calculation before claiming resistance is a
  function of word length alone.
- Added no resistance solver, random-walk engine, optimizer, benchmark, or GPU
  code.

## 2026-08-28: BFS geodesics and graph hyperbolicity

- Treated BFS as a one-source distance oracle and hyperbolicity as a global
  comparison of several shortest paths.
- Recorded both thin-triangle and four-point viewpoints while refusing to mix
  their numerical constants without an explicit convention conversion.
- Proved the tree calibration and rejected transfer of a BFS tree's zero
  hyperbolicity back to the original graph.
- Used paths, regular trees, complete graphs, grids, and cycles to separate
  hyperbolicity from frontier width, growth, and girth.
- Identified finite-instance vacuity: every finite puzzle graph has some finite
  `delta`, so the constant, diameter scale, and generator choice matter.
- Separated the general shortest-path membership equality from additional
  thin-corridor geometry.
- Recorded why thin geodesics alone do not make bidirectional BFS fast or
  justify pruning.
- Connected Gromov products to multiple BFS distance rows rather than one root
  profile.
- Applied the distinction to Cayley, Schreier, and directed-alphabet models.
- Classified sampled quadruples as lower-bound witnesses only.
- Added no estimator, search implementation, optimizer, benchmark, or GPU
  code.

## 2026-08-28: BFS balls, convexity, gates, and Helly intersections

- Separated BFS balls, graph intervals, all-geodesics convexity, gated
  projection, and the ball-Helly property.
- Used `C6` to reject the assumption that an arbitrary BFS ball is even weakly
  geodesically convex.
- Recorded gated-set uniqueness, convexity, and finite Helly behavior without
  transferring those properties to arbitrary balls.
- Used four unit balls in `C4` as a pairwise-but-not-total intersection
  witness.
- Proved pairwise ball intersection from center-distance inequalities.
- Derived `radius=ceil(diameter/2)` for finite Helly graphs directly from
  pairwise intersections of radius balls.
- Kept that existential identity separate from two-sweep BFS correctness.
- Interpreted ball intersections as simultaneous multi-source distance
  constraints rather than frontier unions.
- Used strong grids to separate Helly geometry from uniformly bounded
  hyperbolicity.
- Rejected Cayley vertex transitivity as a Helly certificate and retained the
  directed-alphabet boundary.
- Added no recognizer, estimator, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS intervals, triple medians, and partial cubes

- Defined intervals through exact distance equalities and distinguished them
  from one selected BFS-tree path.
- Introduced median graphs through unique triple-interval intersections.
- Calibrated the class with trees, hypercubes, Cartesian grids, triangles, and
  the partial-cube counterexample `C6`.
- Connected isometric cube coordinates to BFS depth while rejecting binomial
  frontier counts for arbitrary partial cubes.
- Separated coordinatewise triple majority from the weighted total-distance
  facility-location median objective.
- Used an equal-weight edge to show that a median graph can have a non-singleton
  weighted graph-median set.
- Used `C4` to separate median graphs from ball-Helly graphs.
- Used hypercubes and grids to reject frontier-size and hyperbolicity conclusions
  from median structure.
- Recorded a generator-sensitive Cayley example on `Z2^2`: `C4` versus `K4`.
- Kept validated partial-cube coordinates separate from puzzle encodings and
  hash bits.
- Added no recognizer, embedding algorithm, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS layers and weak modularity

- Expressed the triangle and quadrangle conditions as exact diagrams across
  adjacent BFS layers.
- Kept their adjacency, distance-two, and common-upper-neighbor premises
  explicit instead of reducing them to generic triangles or four-cycles.
- Defined weakly modular graphs by TC/QC over every root and modular graphs by
  nonempty triple-interval intersections plus bipartiteness.
- Used `K_(2,3)` to separate modular from median and `K3` to separate weakly
  modular from modular.
- Rejected a single BFS parent tree and a single arbitrary root as universal
  verification evidence.
- Treated local-to-global results as theorem-dependent rather than as permission
  to extrapolate from bounded-radius samples.
- Separated weak modularity from Helly balls, unique medians, hyperbolicity, and
  frontier width.
- Used Cayley translation to reduce roots only under complete identity-relative
  coverage.
- Reused `Z2^2` to show generator-sensitive movement between median/modular and
  merely weakly modular graphs.
- Added no recognizer, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS frontiers versus treewidth

- Defined tree decompositions through vertex, edge, and running-intersection
  conditions rather than informal tree-likeness.
- Separated root-free bag width from root-relative BFS layer cardinality.
- Used stars and complete binary trees to reject bounded frontiers from
  treewidth one.
- Distinguished whole BFS layers from overlapping tree-decomposition bags.
- Defined layered width as bag-layer intersection size, not layer size.
- Used the star again to show layered width one with frontier `n-1`.
- Recorded planar layered treewidth and local-treewidth results without turning
  them into BFS-memory bounds.
- Kept existential layering/decomposition choices separate from the requested
  operational BFS root.
- Used `Z2^2` to show generator-dependent treewidth through `C4` versus `K4`.
- Separated bag dynamic programming from frontier/visited GPU dataflow.
- Added no decomposition algorithm, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS, LexBFS, and chordal elimination

- Defined chordality through induced cycles and separated it from acyclicity and
  unique geodesics.
- Defined simplicial vertices and perfect elimination orderings.
- Connected chordal clique trees to treewidth without treating bags as
  frontiers.
- Extended note 19's LexBFS distinction to the exact reversed-order PEO theorem.
- Built a four-vertex diamond counterexample where a valid ordinary BFS order
  reverses to a non-PEO.
- Kept LexBFS's returned order separate from the subsequent chordality check.
- Rejected chordal completion as a metric-preserving substitute for original
  BFS.
- Used complete graphs and stars to reject frontier bounds from chordality.
- Derived that every finite connected simple vertex-transitive chordal graph is
  complete, then applied it to finite Cayley graphs.
- Preserved the infinite and directed graph boundaries.
- Added no recognizer, triangulation algorithm, optimizer, benchmark, or GPU
  code.

## 2026-08-28: BFS and distance-hereditary subgraphs

- Started from the monotonic fact that deletion can only increase retained-pair
  distances or disconnect them.
- Separated one isometric subgraph from the hereditary requirement over every
  connected induced subgraph.
- Recorded the equivalence between distance heredity and every induced path
  being geodesic.
- Used `C5` deletion as a minimal hidden-shortcut counterexample.
- Derived exact restriction of BFS layers inside connected induced subgraphs of
  a distance-hereditary graph.
- Documented pendant, true-twin, and false-twin pruning sequences.
- Recorded hole/house/gem/domino forbidden obstructions and the universal
  coverage boundary.
- Used `C4` and the gem to separate distance heredity from chordality.
- Rejected merging twins as exact BFS duplicates without an output-specific
  quotient/lifting proof.
- Used two generating sets of `Z6` and an induced six-cycle in `Q3` as Cayley
  counterexamples.
- Added no recognizer, pruning implementation, optimizer, benchmark, or GPU
  code.

## 2026-08-28: BFS versus spanners, emulators, and hopsets

- Extended note 81 without duplicating its BFS-tree stretch analysis.
- Separated multiplicative/additive spanners from emulators and hopsets.
- Treated exact BFS on a spanner as an upper-bound path computation in the
  original graph, not as exact original distance.
- Used a star two-spanner of `K_n` to separate edge sparsity from frontier width
  and depth.
- Required unpacking witnesses for virtual emulator and shortcut edges.
- Separated hop count from metric weight and original BFS level count.
- Proved the Cayley generator-substitution bound from per-generator retained
  words of maximum length `L`.
- Kept generator connectivity, worst-case stretch, actual stretch distribution,
  and frontier evolution distinct.
- Distinguished algebraic group-word substitutions from stabilizer-specific
  Schreier paths.
- Preserved the exact-search requirement for an independent matching lower
  bound.
- Added no spanner, emulator, hopset, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS balls and doubling dimension

- Defined doubling constant and dimension through universal half-radius covers.
- Derived an explicit iterated-cover cardinality bound for unweighted BFS balls.
- Distinguished the genuine polynomial-growth consequence from the false
  treewidth/frontier implication in note 113.
- Proved the smallest-scale lower bound `lambda>=maximum_degree+1`.
- Explained why stars, complete graphs, and hypercubes do not contradict the
  theory: their dimension parameter grows with branching.
- Separated worst-case ball envelopes from exact frontier profiles, duplicates,
  paths, and edge layout.
- Introduced packing bounds and metric nets without identifying them with exact
  visited-state quotients.
- Recorded finite-instance vacuity and the need for quantitative family-scale
  claims.
- Used Cayley translation to remove center redundancy while retaining every
  scale requirement.
- Separated sampled lower-bound witnesses from universal cover certificates.
- Added no net construction, nearest-neighbor structure, optimizer, benchmark,
  or GPU code.

## 2026-08-28: replacement paths and fault-tolerant BFS

- Separated graph edge, graph vertex, local labeled-edge, global generator,
  temporal-availability, and compute-worker failures.
- Defined replacement distance in the surviving graph and distinguished the
  classical one-path problem from broader sensitivity contracts.
- Used path-set inclusion to recover deletion monotonicity.
- Rejected the ordinary BFS tree as a one-edge fault-tolerant structure with a
  four-vertex diamond.
- Distinguished equal-length replacement, longer finite detour, and genuine
  bridge disconnection using diamond, cycle, path, and complete-graph cases.
- Showed exactly what the old predecessor DAG can and cannot certify.
- Defined exact FT-BFS as scenario-universal preservation of surviving source
  distances and recorded the worst-case `Omega(n^(3/2))` sparsity barrier.
- Required surviving-graph replay plus a surviving-graph lower bound for exact
  replacement-distance certificates.
- Separated local Cayley-edge failure from global generator-family failure and
  used `Z_6` cosets as a connectivity calibration.
- Kept GPU/rank recovery as an execution problem rather than graph-state
  deletion, with separate reporting contracts.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: degree--diameter Moore capacity

- Detected and avoided duplicating note 27's degree--girth Moore lower bounds.
- Derived the degree--diameter Moore upper bound directly from BFS layers.
- Separated collision-free tree capacity from expected or measured frontier
  growth.
- Introduced per-layer Moore deficits and proved that they sum to total order
  defect without identifying them with duplicate-candidate counts.
- Distinguished one-root eccentricity, graph radius, and all-pairs diameter.
- Used complete graphs, odd cycles, Petersen, and the three-cube as calibration
  cases.
- Explained why equality is rigid and why the same odd-girth expression points
  in the opposite inequality direction.
- Derived the directed `1+d+...+d^D` bound without importing the undirected
  inverse-edge assumption.
- Bound finite Cayley group order using true simple degree rather than raw move
  labels, while treating Cayley structure as an extra extremal restriction.
- Kept combinatorial capacity separate from frontier, memory, routing, and
  throughput measurements.
- Added no construction, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS distance sums and centrality aggregates

- Avoided duplicating note 21 and REF-021's existing eccentricity, diameter,
  and double-sweep analysis.
- Reinterpreted the complete BFS layer histogram as a source-distance
  distribution and generating polynomial.
- Derived farness, mean source distance, closeness, and harmonic centrality from
  layer counts.
- Used a triangle with one leaf to separate equal eccentricity from equal
  farness.
- Derived the Wiener index by double-counting all vertex transmissions.
- Separated connected closeness, component-restricted variants, and harmonic
  treatment of unreachable vertices.
- Distinguished outgoing and incoming centrality in directed graphs.
- Proved that one complete identity BFS gives global distance sums in a finite
  Cayley graph, but rejected automatic transfer to arbitrary Schreier graphs.
- Treated bounded histograms and sampled sources as lower-bound or estimation
  evidence rather than exact global metrics.
- Separated exact integer accumulation from floating normalization and GPU
  reductions from traversal correctness.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS shortest-path DAG and betweenness

- Extended note 11's brief Brandes mention without repeating basic shortest-path
  counting.
- Defined vertex and edge betweenness with explicit pair, endpoint, path, and
  normalization conventions.
- Used the diamond to reject betweenness computed from one deterministic parent
  tree.
- Interpreted Brandes accumulation as reverse-depth dynamic programming on a
  completed source shortest-path DAG.
- Calibrated complete graphs, paths, stars, cycles, and diamonds.
- Proved that total raw unordered vertex betweenness is
  `W-binom(n,2)` and total edge betweenness is `W`.
- Distinguished edge betweenness from the edge-Wiener index.
- Combined the aggregate identity with Cayley translation symmetry to derive
  common vertex score `(T(e)-n+1)/2` from one complete identity BFS.
- Separated vertex transitivity from edge transitivity and Cayley graphs from
  unproved Schreier symmetry.
- Recorded arithmetic, retry, reverse-finalization, and distributed-consistent
  epoch boundaries.
- The first journal-reordering patch failed because its expected neighboring
  heading did not match the actual file order; the relevant region was reread
  and a narrower context patch succeeded without changing note content.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS orderings and Cuthill--McKee

- Defined bandwidth and profile as different objectives on a symmetric sparsity
  graph.
- Interpreted Cuthill--McKee as ordinary BFS plus nondecreasing-degree neighbor
  order and an explicit tie-break.
- Derived a two-consecutive-layer upper bound for any contiguous BFS-layer
  numbering.
- Used a star to reject global bandwidth optimality of BFS layer orderings.
- Proved that exact reversal preserves bandwidth, while directional profile can
  change.
- Separated pseudo-peripheral sweep termination from a peripheral-vertex or
  exact-diameter certificate.
- Distinguished traversal correctness from ordering quality and original
  pattern recomputation.
- Observed that Cayley regularity removes the degree heuristic and Cayley
  transitivity removes the root-periphery search, leaving tie order unresolved.
- Separated graph invariants from representation locality and ID-based owner
  changes under permutation.
- Recorded the preprocessing and global-consistency boundaries for implicit and
  multi-GPU graphs.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS graph coverings and universal trees

- Read note 17 first and restricted the new scope to BFS balls, layers, and
  collision onset rather than repeating quotient path lifting.
- Proved distance nonexpansion and exact distance-to-fiber under a cover.
- Proved that every lifted BFS ball projects exactly onto the base ball while
  its cardinality may be larger.
- Separated exact ball projection from the fact that one lifted sphere can mix
  several base distances.
- Interpreted universal-cover BFS as non-backtracking history traversal rather
  than base-state BFS.
- Derived the `2r<girth` vertex-injectivity and `2r+1<girth` induced-ball
  thresholds in covering language.
- Used `C5`, `C6`, `C3`, and the finite cover `C6 -> C3` as calibrations.
- Rejected layer-size division by sheet count.
- Connected free-group Cayley trees to relation fibers and Schreier fibers to
  stabilizer words under explicit local-bijection conditions.
- Separated lifted-history ownership from authoritative base-state visited
  semantics in multi-GPU execution.
- Corrected the Fiala--Paulusma source title and DOI after primary metadata
  showed that the initially inserted chapter number was wrong.
- The first research-log append matched an earlier repeated anchor and placed
  this section before notes 119--122; an exact-section move restored chronology.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS deletion, contraction, and minor metrics

- Extended note 115's deletion monotonicity with the opposite contraction and
  supergraph inequalities.
- Proved that one unit edge contraction decreases fixed-pair distance by at
  most one.
- Derived a cluster-diameter lifting bound for coarse quotient paths.
- Distinguished exact cover-ball projection from contraction-ball overreach.
- Used `C4` edge deletion and path contraction to reject any universal metric
  monotonicity for arbitrary graph minors.
- Required cross-edge and intra-branch-set witnesses for replaying minor paths.
- Separated simple vertex distance from loop, parallel-label, and path-count
  semantics after contraction.
- Interpreted quotient-group Cayley distance as minimum distance to a coset,
  not a fixed representative.
- Paired coarse lower bounds with replayable original upper bounds as the only
  exactness route considered here.
- Separated coarsening and owner changes from original multi-GPU BFS throughput.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS edge subdivision and topological-minor metrics

- Filled the metric counterpart to note 124's contractions: subdivision turns
  positive integer edge length into unit BFS depth.
- Proved exact weighted-distance equivalence on original branch vertices and
  exact scaling under uniform subdivision.
- Separated branch-only spheres from the transit vertices inserted between
  them.
- Used unequal lengths `1+1` versus `3` to reject global rescaling for a
  nonuniform subdivision.
- Used once-subdivided `K_n` to show that exact branch-distance scaling does not
  preserve frontier width or traversal work.
- Distinguished abstract topological-minor distance, selected subdivision-path
  length, and shortest distance in the whole host graph.
- Separated original Cayley/Schreier states from artificial generator-phase
  states.
- Recorded Dial's integer-weight method as evidence that the semantic reduction
  does not require literal graph materialization.
- The first Docker validation of note 124 used textual `comm` ordering on
  zero-padded SEM identifiers and falsely reported SEM-001--089 missing; the
  corrected validator normalizes identifiers numerically.
- A second validator revision passed the ID list to an `awk` program that was
  itself read from standard input, so the program saw no IDs and printed a
  meaningless concatenated missing list; the final check uses explicit
  minimum, count, maximum, uniqueness, and adjacent-increment invariants.
- The first research-log insertion used a repeated generic anchor and placed
  this section before notes 119--124; an exact-section move restored chronology.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS sweeps and diameter certificates

- Deepened note 21's double-sweep boundary into lower-bound and upper-bound
  certificate mechanics.
- Proved that eccentricities along an undirected farthest-vertex sweep chain
  are nondecreasing.
- Rejected plateau, mutual-farthest, repeated-restart, and degree-based stopping
  as general exactness certificates.
- Derived per-vertex and global multi-pivot eccentricity bounds.
- Derived the outer-fringe bound `D<=max(M,2(i-1))` and identified complete
  processing of every outer vertex as its critical premise.
- Separated four-sweep root quality from iFUB correctness and recorded its
  `Theta(nm)` worst-case boundary.
- Distinguished BFS-tree diameter upper bounds from graph-distance lower
  bounds and exact equality.
- Restricted monotone sweep reasoning to symmetric undirected distance.
- Explained why finite connected Cayley graphs need one identity sweep rather
  than a multi-root heuristic.
- Separated concurrent root sweeps from parallelizing one BFS across GPUs.
- The first journal insertion again matched the repeated generic final bullet
  and landed before notes 119--125; moving by the unique note-125 tail restored
  chronology. Future journal patches must use a unique heading or exact tail.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS on complement graphs

- Defined complement BFS strictly for finite simple graphs before discussing
  directed, labeled, or action-graph variants.
- Classified complement distance one and two using original adjacency and
  common nonneighbors.
- Proved that a disconnected graph has connected complement of diameter at
  most two and that original diameter at least four forces complement diameter
  two.
- Derived the next-frontier identity as unvisited vertices minus the
  intersection of original frontier neighborhoods.
- Used a two-vertex frontier to reject the tempting union-subtraction rule.
- Separated complement visited evidence from original-graph visited evidence.
- Recorded implicit complement traversal as an existence result without
  designing or implementing a data structure.
- Derived the exact complementary generator set for finite simple Cayley
  graphs while rejecting transfer of complement paths to original moves.
- Identified authoritative nonedge evidence as the multi-GPU correctness
  boundary under sharded original adjacency.
- Used the unique note-126 journal tail for insertion; no chronology repair was
  required.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS bisimulation and safe state merging

- Expanded note 17's transition congruence into strong labeled bisimulation,
  one-way simulation, and observation-preserving quotient contracts.
- Proved path projection and representative-by-representative path lifting.
- Made quotient self-loop retention explicit for same-length trace projection.
- Proved exact BFS distance preservation for goals saturated by whole
  bisimulation classes.
- Reused the reflected-path counterexample to reject fixed-target preservation
  inside a nonsingleton class.
- Distinguished one-way over-approximation from exact two-sided quotienting.
- Used universal bisimulation on `P3` to show that bisimulation ignores degree
  and is weaker than equitable partition/count refinement.
- Narrowed the universal-bisimulation observation to unlabeled/one-label
  systems after self-review caught that differing enabled labels invalidate it.
- Separated unit-step, weighted, probabilistic, and stutter semantics.
- Required incoming as well as outgoing safety for directed bidirectional BFS.
- Derived the deterministic Cayley generator-congruence condition and retained
  note 17's label-frame boundary.
- Identified global partition consistency as a multi-GPU correctness premise.
- The initial context read used an obsolete guessed filename for note 17;
  repository search located the authoritative file before analysis continued.
- The first correction patch mixed a research-log anchor into note-file
  context and was rejected atomically; separate exact-file patches succeeded.
- Used the unique note-127 journal tail, so no chronology repair was needed.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS residual languages and DFA minimization

- Connected product-state BFS memory to residual suffix languages rather than
  current depth or prefix identity.
- Defined Myhill--Nerode equivalence and its symbol-congruence property.
- Explained why state-level visited is exact for one shortest accepted word but
  not an enumerator of every accepted prefix or word.
- Recorded which language outputs minimal DFA preserves and which original
  state/provenance outputs it discards.
- Used equal one-step acceptance distance with different accepted labels to
  reject distance-based memory merging.
- Related deterministic residual equivalence to observation-respecting strong
  labeled bisimulation.
- Separated partial-DFA dead sinks, NFA subset states, and epsilon-cost
  semantics.
- Proved that minimizing the DFA factor preserves constrained product-BFS
  distance without merging base vertices.
- Kept Cayley language minimization separate from geodesic/unique normal-form
  evidence.
- Required globally identical residual classes across GPU owners.
- Used the unique note-128 journal tail; no chronology repair was needed.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS NFA subset states and dominance

- Expanded note 129's one-line subset-construction boundary into exact
  deterministic NFA configuration semantics.
- Proved that a BFS frontier is a set of active-state subsets, not their union.
- Used two prefixes with different one-letter suffixes to show how frontier
  union invents accepted words.
- Separated exponential subset capacity from actually reachable subsets.
- Derived transition and existential suffix-language monotonicity under subset
  inclusion.
- Proved a qualified shortest-path dominance rule requiring the same base
  state and a no-later superset arrival.
- Rejected transfer of that rule to shortlex, word enumeration, run provenance,
  universal acceptance, or different base vertices.
- Distinguished subset equality, inclusion dominance, and residual-language
  minimization.
- Separated epsilon word length and accepted-word versus accepting-run counts.
- Required complete cross-owner successor union before subset finalization.
- Used the unique note-129 journal tail; no chronology repair was needed.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS AND/OR reachability games

- Identified ordinary reverse BFS as the all-existential special case of a
  reachability-game attractor.
- Defined mixed existential/universal predecessors and the attractor least
  fixed point.
- Proved layer rank as a worst-case forced-target bound with `min` and `max`
  successor recurrences.
- Used a target edge plus adversarial self-loop trap to separate path distance
  from winning status.
- Used a universal self-loop to expose why least rather than self-supporting
  fixed-point reasoning matters.
- Derived positional attractor and counter-attractor strategies from ranks.
- Made dead-end ownership and vacuous-universal semantics explicit.
- Clarified that dead-end extensions rank time to a winning terminal; literal
  target-arrival rank requires a total arena or explicit target totalization.
- Classified missing and spurious successor errors separately for existential
  and universal vertices.
- Separated one path from a strategy covering all opponent responses.
- Reframed adversarial Cayley moves as an expanded game state and a different
  minimax metric.
- Identified global all-successor evidence as the multi-GPU universal
  finalization boundary.
- Used the unique note-130 journal tail; no chronology repair was needed.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS support graphs and probabilistic reachability

- Extended note 95's fixed-random-walk boundary to qualitative and quantitative
  target reachability in finite Markov chains and MDPs.
- Proved that finite support reachability is exactly positive-probability
  finite-time reachability and that BFS distance is the earliest nonzero-hit
  time.
- Used an absorbing trap to separate positive probability from probability one.
- Used a probabilistic self-loop to separate almost-sure reachability from
  adversarial sure reachability.
- Identified reachable target-free BSCCs as the finite-chain obstruction to
  almost-sure target reachability.
- Separated support distance, bounded reachability values, eventual hitting
  probability, unconditional/conditional expectation, and MDP policy value.
- Recorded the finite-state assumption behind almost-sure finite expected
  hitting time.
- Related MDP Bellman equations to graph preprocessing without treating them as
  BFS level recurrences.
- Preserved parallel-label probability mass in the Cayley/Schreier contract.
- Flagged tiny omitted trap edges as capable of reversing qualitative results.
- Required complete global probability rows and support epochs in sharded
  analysis.
- Used the unique note-131 journal tail; no chronology repair was needed.
- The first read-only Docker validator invocation failed before validation with
  `unexpected EOF while looking for matching backtick`: a code-fence regex was
  unsafe across PowerShell and nested Bash quoting. Replaced it with a
  backtick-free executable-marker check; no corpus content was implicated.
- The second invocation passed semantic and note checks but rejected an assumed
  README count of 132: the legacy `sed` parser counted only bullet links and
  omitted the inline note-02 benchmark-contract link. A direct target extractor
  found 132 unique note targets matching all 132 note files; the final validator
  uses that exact-link set instead of layout-dependent bullets.
- Added no implementation, optimizer, benchmark, or GPU code.

## 2026-08-28: BFS on de Bruijn and Kautz overlap digraphs

- Audited the topic index and found no existing de Bruijn or Kautz coverage.
- Defined both directed word-shift graphs with explicit alphabet, adjacency,
  loop, degree, and orientation conventions.
- Proved distance as word length minus maximum source-suffix/target-prefix
  overlap.
- Derived diameter `n` for nontrivial alphabets without treating small diameter
  as small traversal work.
- Separated append histories, unique candidates, and new BFS states.
- Used source border/periodicity to explain root-dependent frontier profiles.
- Extended the overlap proof to Kautz's adjacent-symbol restriction.
- Connected both families to iterated line digraphs while keeping fixed-window
  state distinct from trail history.
- Kept de Bruijn Hamiltonian/cyclic enumeration separate from BFS ordering.
- Rejected a direct Cayley interpretation because fixed-symbol shift maps are
  non-bijective.
- Added a deliberately small transparent Rust exhaustive probe, run only in
  Docker, with zero overlap-distance mismatches on `B(2,3)` and `K(2,3)`.
- Recorded two failed `bash -lc` runs: the local image reset `PATH` and hid its
  installed rustup links; absolute toolchain invocation under `bash -c`
  succeeded without changing the probe.
- Separated logical de Bruijn/Kautz state graphs from physical interconnect
  topologies in the GPU/multi-GPU interpretation.
- The first journal insertion used a repeated generic tail and landed before
  note 119; moved this section after the unique note-132 tail before validation.
- The first combined final validator stopped before compilation because the
  image toolchain lacks the optional `rustfmt` component. No dependency was
  installed; retained manual source inspection plus successful `rustc`
  compilation and exhaustive output assertions.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on lamplighter Cayley graphs

- Expanded the corpus's brief lamplighter dead-end references into an exact
  configuration-position state and word-metric study.
- Declared separate toggle and left/right movement generators before making any
  distance claim.
- Proved word distance as symmetric-difference toggle cost plus a shortest base
  walk visiting every changed lamp.
- Derived the two-direction interval formula for the infinite-line base and
  rejected its unqualified transfer to general base Cayley graphs.
- Rejected Cartesian-product frontier convolution because movement changes
  which lamp coordinate the toggle generator acts upon.
- Explained exponential wreath-product growth over a line without inferring
  nonamenability.
- Connected regular local degree to globally induced radial dead ends rather
  than missing successors.
- Added a transparent Rust exhaustive probe for `C_2 wr C_4` and `C_2 wr C_5`,
  run only in Docker.
- Observed zero decomposition mismatches over all 64 and 160 states,
  respectively.
- Found 5 and 18 interior dead ends after excluding diameter-layer states.
- Separated complete-state visited identity from cursor-only and mask-only
  projections.
- Kept shortest words, routes, histories, parents, and distinct states as
  separate counts.
- Distinguished logical lamplighter edges, owner routing, and physical GPU
  interconnect paths.
- Used the unique note-133 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on Tower-of-Hanoi Schreier graphs

- Confirmed that Tower of Hanoi had no prior dedicated coverage in the corpus.
- Fixed the classical three-peg, distinguishable-disk, one-legal-move contract.
- Proved that every ternary disk-to-peg word is a legal state, giving `3^n`
  vertices.
- Distinguished partial puzzle moves from total involutory peg-pair actions with
  fixed-point loops.
- Derived three corner loops, three degree-two simple vertices, all remaining
  vertices degree three, and `(3/2)(3^n-1)` simple edges.
- Classified the finite puzzle graph as a level Schreier graph rather than a
  Cayley graph on configurations.
- Derived the three-copy recursion and largest-disk connector structure
  (terminology corrected 2026-08-31: these edges are not individual bridges).
- Reproved corner distance and diameter `2^n-1` for the classical model.
- Derived the complete corner frontier recurrence and closed form
  `f_n(k)=2^popcount(k)`.
- Checked that summing the layers gives exactly `3^n` states.
- Added a transparent all-state Rust probe for `n=1..6`, run only in Docker.
- Exhaustively confirmed state counts, corner eccentricities, all-pairs
  diameters, loop counts, degree counts, and frontier profiles through 729
  states.
- Separated one recursive solution path, move histories, shortest paths, and
  BFS frontier states.
- Kept three logical bridge edges distinct from owner routing and physical GPU
  links.
- Used the unique note-134 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on pancake Cayley graphs

- Confirmed that prefix-reversal pancake graphs had no dedicated corpus note.
- Fixed the unsigned `S_n` state and `r_2,...,r_n` generator convention.
- Distinguished the genuine Cayley graph from note 135's configuration Schreier
  graph.
- Established factorial state count, exact degree, connectivity, involution,
  and absence of fixed-point/parallel generator endpoints.
- Used Cayley translation to justify one-root frontier and eccentricity transfer
  without inferring edge transitivity or distance regularity.
- Separated nonbacktracking generator histories from distinct permutation
  states.
- Derived the first two sphere sizes and connected the depth-three deficit to
  `r_2 r_3 r_2 = r_3 r_2 r_3`.
- Recorded the classical exact `F_3` formula and girth-six boundary.
- Added a transparent Rust exhaustive probe for `P_2,...,P_8`, run only in
  Docker.
- Confirmed exact `n!` exhaustion, diameters, complete frontier profiles, and
  all three early-layer formulas through 40,320 states.
- Used `P_8` to show frontier contraction after a depth-seven peak rather than
  extrapolating early branching.
- Kept one sorting path, minimum distance, lower bound, upper bound, and exact
  diameter as separate evidence claims.
- Separated unsigned, burnt/signed, and repeated-symbol variants.
- Distinguished logical prefix-reversal edges, owner routing, and physical GPU
  communication.
- Used the unique note-135 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on star-transposition Cayley graphs

- Confirmed that star-transposition graphs had no dedicated corpus coverage.
- Fixed `ST_n=Cay(S_n,{(1 i)})` and contrasted it with pancake BFS on the same
  `n!` vertices and degree `n-1`.
- Derived exact word length from nontrivial cycle support, cycle count, and
  whether symbol `1` lies in a nontrivial cycle.
- Explained the extra enter/leave cost for cycles avoiding the center symbol.
- Derived the closed diameter `floor(3(n-1)/2)` by maximizing short disjoint
  cycle contributions.
- Proved bipartiteness and exact depth parity from the common odd sign of every
  star transposition.
- Derived the first three frontier sizes by enumerating center-containing cycle
  types and leaf transpositions.
- Added a transparent Rust exhaustive probe for `ST_2,...,ST_8`, run only in
  Docker.
- Confirmed exact `n!` exhaustion, layer profiles, diameters, and zero metric or
  parity mismatches through 40,320 states.
- Compared star and pancake profiles under identical vertex-count, degree, and
  root-symmetry controls without asserting a universal ranking.
- Rejected cycle type as a visited key despite its sufficiency for scalar star
  distance.
- Kept the star graph as computed state graph separate from its use as a
  processor interconnection topology.
- Distinguished transformation, oracle, visited, owner-routing, and physical
  communication costs.
- Used the unique note-136 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS with all transpositions

- Avoided duplicating the already complete adjacent-transposition/Mahonian
  treatment and selected the absent all-transposition metric.
- Defined the complete transposition Cayley graph with `n!` states and degree
  `C(n,2)`.
- Proved that every move merges or splits permutation cycles and changes cycle
  count by exactly one.
- Derived exact distance `n-c(pi)`, diameter `n-1`, and the `(n-1)!` farthest
  `n`-cycles.
- Identified complete BFS layers with unsigned Stirling numbers of the first
  kind and recorded their recurrence and generating polynomial.
- Used permutation sign/cycle-count parity to orient every edge between adjacent
  BFS layers.
- Derived exact inward/outward degrees from cycle lengths.
- Used 3-cycles versus two disjoint transpositions at depth two to reject
  distance regularity despite class-function distance.
- Explained immediate duplicate pressure from minimal 3-cycle factorizations
  and commuting disjoint transpositions.
- Added a transparent Rust exhaustive probe for `n=2..8`, run only in Docker.
- Confirmed exact `n!` exhaustion, all Stirling profiles, diameters, and zero
  metric/parity mismatches through 40,320 states.
- Compared four generator sets on the same `S_8` vertex universe without
  ranking their algorithmic or hardware performance.
- Separated normal-Cayley conjugacy symmetry from complete state identity.
- Distinguished candidate degree, oracle/rank work, duplicates, visited,
  routing, synchronization, and total time.
- Used the unique note-137 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS coverage, domination, and k-center

- Confirmed that the corpus had multi-source distance, metric nets, and
  eccentricity material but no direct treatment of center-set evaluation versus
  center selection.
- Defined covering radius as the maximum distance to a fixed source set and
  connected radius-one coverage to ordinary closed-neighborhood domination.
- Separated fixed-set evaluation, minimum distance domination, and graph
  k-center as three different contracts.
- Identified complete multi-source BFS as an exact coverage evaluator and
  distance certificate, not a center-selection algorithm.
- Added upper certificates from proposed centers and lower certificates from
  separated packing witnesses.
- Rejected maximum degree as an exact one-center rule with an eight-vertex
  graph whose unique maximum-degree vertex has radius four versus optimum three.
- Rejected farthest-first as exact with endpoint-seeded `P6`: greedy radius two
  versus exact radius one.
- Preserved its metric factor-two guarantee and distinguished that theorem from
  exactness.
- Corrected the disconnected-graph reduction: a missing component makes the
  radius infinite rather than the maximum among reached vertices.
- Added a transparent Rust probe and exhaustive tiny subset oracle, executed
  only in Docker.
- Recorded one infrastructure-only failed gate: `rust:1.85-bookworm` lacks
  `rustfmt`; formatting passed in the project's toolchain image and execution
  passed in the minimal image.
- Kept Voronoi owner ties separate from scalar covering radius.
- Distinguished outward and inward directed coverage conventions.
- Used the unique note-138 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: shortest-hop BFS with a secondary cost

- Audited two candidate topics first and found bipartite witnesses already
  complete in notes 21/31 and temporal BFS already complete in note 22.
- Selected the absent boundary between minimum hops, secondary cost among
  minimum-hop paths, cost-first paths, and Pareto paths.
- Proved that depth-increasing edges form an acyclic shortest-path graph and
  support a layer-ordered secondary-cost dynamic program.
- Rejected first-discovery parent as a secondary optimum while preserving its
  exact minimum-hop guarantee.
- Used a six-vertex fixture where first-parent cost is 200 and the best cost
  among two-hop paths is 2.
- Added a three-hop zero-cost alternative to separate hop-first `(2,2)` from
  cost-first `(3,0)`.
- Exhaustively retained both incomparable pairs and rejected one visited/label
  field as a Pareto representation.
- Explained why equal-depth secondary improvements must finalize before
  descendant expansion or be repropagated.
- Separated additive secondary metadata from history-dependent costs requiring
  product-state augmentation.
- Added a transparent Rust probe, run only in Docker.
- Retained the first formatting-only failed gate and passed the corrected gate.
- Used the unique note-139 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on Hamming graphs

- Checked the proposed semiring topic and found it already covered in notes 25
  and 33 rather than duplicating it.
- Selected the absent Hamming-graph family as a controlled coordinate-product
  BFS model.
- Derived exact Hamming distance, `q^d` state count, degree `d(q-1)`, diameter
  `d`, and frontier sizes `C(d,i)(q-1)^i`.
- Derived the exact frontier ratio and binomial mode shift toward outer layers
  as `q` increases.
- Derived distance-regular intersection counts `c_i=i`, `a_i=i(q-2)`, and
  `b_i=(d-i)(q-1)`.
- Used `a_i` to separate binary bipartiteness from nonbinary same-layer cliques
  and triangles.
- Derived exact outward candidate multiplicity `i+1` per next-layer state.
- Derived `i!` shortest paths to every depth-`i` word and kept histories
  separate from unique states.
- Connected Cartesian-product, Cayley, Hamming-ball/code, and spectral views
  without treating their objects as interchangeable.
- Added a transparent full-state Rust probe for 15 small fixtures, run only in
  Docker.
- Confirmed zero distance, layer, intersection, and path-count mismatches
  through `H(5,4)` with 1,024 states.
- Used the unique note-140 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on Johnson graphs

- Confirmed that Johnson graphs and fixed-weight exchange spaces had no
  dedicated corpus coverage.
- Defined `k`-subset state identity, membership-exchange adjacency, complement
  isomorphism, `C(n,k)` state count, and degree `k(n-k)`.
- Derived distance as half symmetric difference and diameter `min(k,n-k)`.
- Derived exact layers `C(k,i)C(n-k,i)` and their consecutive ratio.
- Derived intersection counts `c_i=i^2`, `a_i=i(n-2i)`, and
  `b_i=(k-i)(n-k-i)`.
- Used positive same-layer degree to reject a fixed-weight/bipartite inference.
- Derived exact outward occurrence multiplicity `(i+1)^2` per next state and
  shortest-path multiplicity `(i!)^2` per target.
- Distinguished the Johnson exchange graph from the edgeless fixed-weight
  induced subgraph of the binary hypercube.
- Identified the all-transposition presentation as a non-free Schreier action:
  cross-membership moves are neighbors while within-status moves are
  stabilizer self-loops.
- Kept intersection-size distance oracles separate from complete subset
  identity.
- Added a transparent full-state Rust probe for 36 fixtures, run only in
  Docker.
- Confirmed zero distance, layer, intersection, and path-count mismatches
  through `J(12,6)` with 924 states.
- Used the unique note-141 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on Grassmann graphs

- Confirmed that Grassmann graphs and subspace-state BFS had no dedicated
  corpus coverage.
- Defined vertices as `k`-subspaces, adjacency by `(k-1)`-intersection, and
  state count by a Gaussian binomial coefficient.
- Derived graph distance from intersection deficit and distinguished it from
  the factor-two constant-dimension coding metric.
- Derived degree `q[k]_q[n-k]_q`, q-binomial layers, and their exact ratio.
- Recorded intersection counts `c_i=[i]_q^2` and
  `b_i=q^(2i+1)[k-i]_q[n-k-i]_q`.
- Derived q-factorial squared shortest-path multiplicity.
- Recovered every Johnson formula under the formal `q->1` limit without
  treating that limit as a traversal algorithm.
- Made row-space canonicalization part of exact visited semantics: bases are
  representations, not vertices.
- Separated matrix operations, candidate bases, distinct subspace neighbors,
  and new BFS states.
- Identified the `GL(n,q)` action as transitive with a large stabilizer rather
  than a regular Cayley action.
- Added a transparent exact-membership Rust probe for nine tiny binary
  fixtures, run only in Docker.
- Retained one formatting-only failed gate and passed the corrected run.
- Confirmed every formula through `J_2(6,3)` with 1,395 states.
- Used the unique note-142 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on sparse Erdos-Renyi random graphs

- Distinguished a random original graph from note 98's percolated fixed Cayley
  graph.
- Fixed `G(n,p)`, sparse `p=c/n`, frozen-edge, RNG-seed, and root-selection
  contracts.
- Treated Poisson Galton-Watson growth as a local probabilistic approximation,
  not a deterministic frontier recurrence.
- Connected extinction probability to the giant equation
  `rho=1-exp(-c rho)` and separated finite samples from asymptotic claims.
- Distinguished fixed-root, random-root, and largest-component-conditioned BFS.
- Derived the approximate `rho^2` unconditioned-root component fraction in the
  supercritical regime.
- Separated previous-ball hits, same-layer edges, repeated next-layer parents,
  and new states as the tree approximation breaks.
- Distinguished subcritical, critical-window, and supercritical frontier
  behavior.
- Recorded excess degree rather than average degree as the configuration-model
  branching parameter.
- Added stable-pair decision requirements for implicit and distributed random
  graph generation.
- Added a transparent Rust probe over 80 frozen graphs, run only in Docker.
- Observed finite largest-component means 0.0207, 0.0689, 0.2996, and 0.9797
  for `c=0.8,1.0,1.2,4.0` respectively.
- Observed a representative `c=4` wave peaking at 765 vertices and outward
  occurrence multiplicity near 1.88 during contraction.
- Preserved raw finite-sample ranges and did not label agreement with an
  asymptotic fraction as validation of BFS correctness.
- Used the unique note-143 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on random regular graphs

- Confirmed that random-regular/configuration-model BFS had no dedicated corpus
  coverage.
- Separated degree regularity, distance regularity, and regular group actions.
- Defined the pairing model, simplicity conditioning, and generator retry work.
- Derived the exact regular-tree layer envelope
  `1,r,r(r-1),r(r-1)^2,...` for every finite `r`-regular graph.
- Explained why nonroot branching is `r-1`, contrasting it with Poisson
  Erdos-Renyi excess degree.
- Scoped the local weak tree limit to bounded neighborhoods rather than whole
  finite traversals.
- Added depth-wise inward/same/outward ranges and rejected equal degree as
  distance regularity.
- Distinguished early tree equality, first collision deficit, frontier peak,
  and depleted tail.
- Preserved low-degree connectivity exceptions and kept 20/20 observed
  connectivity separate from the `r>=3` asymptotic theorem.
- Added a transparent pairing-rejection Rust probe over forty samples, run only
  in Docker.
- Retained the Rust 1.85 `repeat_n` compile failure and a later formatting-only
  instrumentation failure.
- Verified exact degrees and final runs after both corrections.
- Observed representative profiles beginning `[1,3,6,12,24,46]` and
  `[1,4,12,36,106]`, below tree bounds at the first collision layers.
- Used the unique note-144 journal tail; no chronology repair was needed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on stochastic block models

- Defined a frozen sparse SBM graph separately from an annealed/resampled
  adjacency process.
- Derived the multitype branching mean matrix `M_ab=alpha_b c_ab` and recorded
  row/column frontier-vector conventions.
- Separated the Perron total-growth mode from the signed type-contrast mode.
- Preserved the irreducibility condition: `across=0` has spectral radius four
  but two disconnected giant classes rather than one global giant.
- Added a transparent Rust probe over eighty finite two-block samples, run only
  in Docker.
- Observed confinement, slow assortative mixing, near-immediate neutral mixing,
  and alternating disassortative type dominance at equal expected degree.
- Measured block-owner remote fractions from 0.0000 to 0.9376 while striped
  ownership remained near one half.
- Rejected a universal community-partition advantage and recorded the opposing
  zero-cut versus owner-utilization boundary.
- Retained one sandbox Docker-access failure and one formatting-only gate; the
  corrected Docker format/compile/run chain passed.
- Corrected an initial chronology insertion that matched an older repeated
  journal tail; this section now follows note 145.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on heterogeneous configuration models

- Expanded the earlier one-line excess-degree observation into a dedicated
  root-versus-edge degree-law study.
- Distinguished a direct pairing multigraph, a simple-conditioned graph, and a
  post-hoc collapsed support graph.
- Derived the size-biased endpoint law and the identity
  `E[D*]=E[D]+Var(D)/E[D]`.
- Separated root offspring `D` from later approximate offspring `D*-1`.
- Connected the Molloy-Reed criterion to excess mean `nu>1` with theorem scope
  retained.
- Added a transparent Rust stub-pairing probe over sixty finite samples, run
  only in Docker.
- Compared three exact degree multisets with mean four but excess means 3.00,
  4.00, and 5.25.
- Observed that increasing excess growth did not increase giant coverage: the
  half-1/half-7 case had mean largest fraction 0.9380.
- Retained the unconditioned-root result: 18/20 roots entered that giant and
  mean root component fraction was 0.8435.
- Recorded depth-wise hub depletion: representative frontier mean degree moved
  from 7.00 in early layers to 1.10 and 1.00 in the tail.
- Retained one formatting-only failed gate; the corrected Docker run passed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on directed random graphs

- Extended the existing deterministic SCC semantics to directed random-graph
  frontier and root-conditioning behavior.
- Defined independent ordered arcs and kept weak, forward, reverse, and strong
  reachability separate.
- Recorded that transposition preserves the ensemble law but not a realized
  BFS frontier.
- Derived the symmetric directed-ER bow tie: GIN/GOUT fractions `rho` and GSCC
  fraction `rho^2` above threshold.
- Separated conditional giant traversal size from the unconditioned root
  mixture, including approximate `rho^2` forward and `rho^4` SCC expectations.
- Added a transparent Rust probe over eighty finite digraphs, run only in
  Docker.
- At `c=4`, observed largest SCC 0.9611 and core reverse/forward reach 0.9804.
- Retained root conditioning: 18/20 roots reached the core, 20/20 were reachable
  from it, and 18/20 belonged to it.
- Observed distinct representative forward/reverse layer peaks despite similar
  final reach.
- Explicitly rejected GIN/GOUT terminology for reachability around the largest
  finite SCC at `c<=1`.
- Retained one formatting-only failed gate.
- Rejected the first SCC-derived measurements after finding that eager sibling
  marking did not preserve Kosaraju finish order; replaced it with an indexed
  DFS stack and recomputed every value.
- Exhaustively cross-checked the corrected SCC labels against mutual
  forward/reverse reachability on four 24-vertex fixtures.
- The corrected Docker format/compile/run gate passed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on random geometric graphs

- Distinguished unit-square and flat-torus graphs built on identical random
  points and radii.
- Connected the leading radius scale `sqrt(log n/(pi n))` to connectivity while
  retaining finite-window and boundary qualifications.
- Proved and asserted the deterministic hop lower bound
  `d_G>=ceil(d_E/r)`.
- Kept that lower bound separate from density-dependent upper bounds and finite
  detours around holes.
- Added a transparent all-pairs Rust probe over paired square/torus samples and
  four radii, run only in Docker.
- Observed boundary-lowered mean degree and connectivity counts rising from
  0/20 below scale to 20/20 at twice the scale.
- Observed pair-weighted stretch decline toward one as radius increased, without
  promoting finite measurements to an asymptotic theorem.
- Separated component eccentricity from diameter in disconnected samples.
- Measured low spatial edge cuts versus near-half striped cuts and retained the
  opposing temporal imbalance of a corner-root wave.
- Retained one formatting-only gate and rejected the first `NaN` stretch
  aggregation before recomputing with explicit pair counts.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on additive small-world graphs

- Distinguished Watts-Strogatz rewiring, Newman-Watts addition, and the probe's
  fixed-count additive ensemble.
- Derived the exact `C_n^2` distance and constant-width baseline frontier.
- Used edge-addition monotonicity as an all-vertex correctness oracle.
- Described shortcut endpoints as seeds of new local waves followed by visited
  collision and depletion.
- Added a transparent Rust probe over 120 fixed-count samples, run only in
  Docker.
- Observed 64 shortcuts change mean degree from 4.000 to only 4.031 while
  reducing mean distance from 512.25 to 46.47.
- Observed the simultaneous frontier-peak increase from 4.00 to 130.40 and
  eccentricity decrease from 1024 to 85.75.
- Separated total remote-edge fraction from first depth with useful work on the
  second owner.
- Recorded the trade-off: shortcuts increased contiguous-cut traffic but moved
  mixed-owner work from depth 512 to depth 12.25 at 64 shortcuts.
- Kept unit-shortcut metric change separate from old-metric weighted hopsets.
- Retained one formatting-only failed gate; the corrected Docker run passed.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on preferential-attachment graphs

- Separated a power-law degree description from a fully specified graph-growth
  process.
- Connected ordinary endpoint size bias to the additional age-degree
  correlations produced by preferential attachment.
- Formed the core-entry mental model: peripheral escape, movement toward older
  hubs, rapid core spread, then outward filling of younger branches.
- Rejected degree histogram as sufficient to determine BFS layers, distances,
  candidate collisions, or owner routing.
- Defined a degree-preserving randomized null model as the relevant comparison,
  with degree sequence, semantics, and root conditioning held explicit.
- Kept typical distance, eccentricity, diameter, level count, scan work, and
  peak memory as different observables.
- Identified birth-contiguous ownership as a potential age/hub hotspot rather
  than a neutral ID partition.
- Performed one Docker readiness check; it failed with named-pipe permission
  denial, so REF-045 is explicitly retained as not run.
- Did not repair Docker and made no numerical or performance claim.
- Added no optimizer, production implementation, benchmark, or GPU code.

## 2026-08-28: BFS on growing trees and rerooting

- Isolated the one-edge growing-tree case from cyclic preferential-attachment
  models.
- Proved that a query-root BFS parent is the unique neighbor toward that root,
  not necessarily the vertex's birth parent.
- Proved that rerooting reverses exactly the seed-to-query-root path and exactly
  `dist(q,r)` parent edges.
- Recorded the LCA rerooting identity for exact distances.
- Derived the exact tree frontier recurrence from frontier excess degree.
- Proved zero same-layer occurrences and zero repeated-next-parent candidates in
  a simple tree.
- Corrected note 151: collision-heavy hub expansion requires cyclic structure
  and is absent from the simple `m=1` tree.
- Kept trusted parent exclusion separate from unsafe visited removal on a
  general implicit graph.
- Retained one failed documentation patch caused by stale context; no file was
  partially changed before the corrected patch.
- The first chronology insertion matched an older generic journal tail and was
  moved here after note 151.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS conservation checks and verification ladder

- Separated conservation-law falsification from exact frontier correctness.
- Recorded a two-element count/sum/xor collision counterexample.
- Explained how balanced losses and extras can survive global reductions.
- Kept finite fingerprints as probabilistic regression evidence unless
  injectivity is proved.
- Separated positive path replay from successor and frontier completeness.
- Required canonical collision-resolving set equality for exact comparison.
- Kept shared CPU/GPU and multi-worker dependencies as common-mode risks.
- Organized counters, matrices, fingerprints, replay, exact comparison, and
  exhaustive independent oracles into a verification ladder.
- Required stable logical identities to distinguish retries from graph work.
- Separated exhaustive validation scale from performance-run scale.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS schedule contracts

- Separated completed-layer first claim, nondecreasing-key settlement, and
  arbitrary-order label correction into three correctness contracts.
- Identified strict FIFO unit-edge BFS as a compact label-setting schedule.
- Kept physical within-layer order separate from semantic layer closure.
- Recorded schedule confluence for final labels but not parents, transient
  frontiers, messages, or work.
- Rejected arbitrary-order first claim and atomic-min-without-reactivation
  hybrids.
- Distinguished generation, claim, activation, settlement, and finalization.
- Derived contract-specific target-finalization boundaries.
- Mapped distributed proof obligations to layer closure, global minimum, or
  global quiescence.
- Separated measurement vocabularies for layer-setting, label-setting, and
  label-correcting executions.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS work coordinates and hardware amplification

- Separated frontier states, logical occurrences, support arcs, unique
  candidates, and accepted states as semantic work coordinates.
- Derived the ordering `n_d <= c_d <= p_d <= g_d` under the declared waterfall.
- Separated semantic conservation identities from representation-dependent
  conversion into records, probes, atomics, bytes, messages, and time.
- Defined named amplification and yield ratios without treating them as speed
  predictions.
- Recorded the undefined cost-per-accepted-state boundary at terminal levels.
- Separated sum work, critical-owner work, dependency depth, and elapsed time.
- Refined communication into remote occurrences, support arcs, candidate
  states, routed records, payload bytes, and protocol bytes.
- Rejected universal implications from duplicate ratio, record count, balance,
  barrier count, TEPS, or final-frontier parity to performance.
- Defined a common semantic vector for later one- and multi-GPU probes.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS scaling regimes

- Separated fixed-work latency/strong scaling, weak scaling, capacity scaling,
  and independent-query throughput.
- Required a matching feasible one-GPU baseline for one-to-many speedup ratios.
- Kept aggregate nominal memory separate from usable exact BFS capacity.
- Distinguished independent BFS batches from multisource BFS semantics.
- Explained why fixed vertices per GPU need not preserve BFS work per GPU.
- Kept whole-run scaling subordinate to the changing per-level frontier profile.
- Treated superlinear measurements as physical-regime diagnoses requiring
  workload-equivalence checks.
- Separated latency, scaling efficiency, throughput, and maximum feasible
  workload as different outcomes.
- Defined a minimum experiment matrix without running or prescribing an
  implementation.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: Cayley and Schreier ownership

- Proved that left-coset blocks `gH` under right Cayley multiplication make an
  occurrence local exactly for generator labels in the subgroup (coset names
  corrected 2026-08-31; the locality equation is unchanged).
- Derived exact local and remote labeled-occurrence fractions.
- Kept equal total coset capacity separate from per-level frontier balance.
- Separated normality-dependent quotient structure from basic subgroup
  locality.
- Covered the coalescing of many algebraic cosets onto fewer GPUs.
- Replaced Cayley cosets by `H`-orbits/double cosets for right Schreier actions.
- Derived stabilizer-dependent Schreier orbit sizes and their capacity skew.
- Rejected direct transfer of the Cayley outside-label crossing law to
  nonfree Schreier actions.
- Contrasted algebraic ownership with the idealized hash-balance model.
- Required typed per-level routing matrices and retained authoritative visited
  semantics.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: Cayley quotient BFS and owner activation

- Proved normal Cayley quotient distance equals minimum concrete distance to
  the corresponding coset fiber.
- Interpreted that distance as exact first activation depth of a coset owner.
- Kept quotient shells separate from later fiber-resolved frontier occupancy.
- Proved generic existential block graphs provide only projected lower bounds.
- Added a four-state incompatible-representative counterexample with spurious
  abstract reachability.
- Identified transition congruence/path lifting as the missing exactness
  obligation.
- Kept nonnormal subgroup locality separate from quotient correctness.
- Extended the same boundary to Schreier `H`-orbit blocks.
- Separated owner-activation lower bounds from load, routing, byte, and timing
  predictions.
- Defined validation fields for future algebraic multi-GPU partitions.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS fibers and re-entry

- Separated first quotient arrival from the full distance multiset inside a
  fiber.
- Added a `Z_6` fixture where one owner block is active at depths zero and three.
- Showed that subgroup-local generator labels need not generate the fiber.
- Added a `Z_20` fixture where locally generated fiber distance is beaten by an
  outside leave-and-re-enter path.
- Rejected universal additive quotient-plus-local distance decomposition.
- Treated owner blocks as persistent state-authority shards rather than
  one-time search phases.
- Kept quotient parents separate from concrete shortest-path DAG metadata.
- Rejected owner-block coincidence as a bidirectional state intersection.
- Listed structural conditions that can make hierarchical reasoning exact.
- Defined per-owner temporal measurements for repeated activation and re-entry.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: Cayley quotient routing matrices

- Defined quotient generator-image multiplicities and their conservation law.
- Proved the quotient-image collision condition for generator labels.
- Separated routing aliases from concrete endpoint equality.
- Derived `M_d(C,D)=f_d(C)mu(C^-1D)` and its row/global conservation laws.
- Added a `Z_8` fixture with two distinct states routed to one quotient block.
- Aggregated the coset matrix through an explicit coset-to-GPU map.
- Kept logical occurrences separate from records, messages, retries, and bytes.
- Identified quotient convolution as a normal-Cayley property with a concrete
  visited/acceptance boundary.
- Derived the inverse-image profile for reverse traversal.
- Rejected automatic transfer of the convolution formula to Schreier orbits.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: Cayley convolution and nonlinear frontiers

- Derived the quotient convolution for raw destination occurrence histograms.
- Expressed exact next-frontier counts through concrete endpoint deduplication
  and visited subtraction.
- Added a `Z_8` multisource counterexample with equal coarse histograms but
  next-frontier sizes two and four.
- Separated linear count transport from nonlinear/idempotent set semantics.
- Characterized the exact collision-free, unvisited equality regime.
- Derived occurrence and remaining-block-capacity upper bounds.
- Kept routing sufficiency separate from identity and novelty insufficiency.
- Defined per-block raw, unique, visited, accepted, record, and byte telemetry.
- Rejected universally exact block-count-only frontier models.
- Kept histogram parity below canonical frontier-set equality in the validation
  ladder.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS level union and output merge algebra

- Proved next-frontier set independence under arbitrary exact frontier
  partition and arrival order.
- Identified candidate-set union as associative, commutative, and idempotent.
- Scoped at-least-once tolerance to a complete membership/publication
  transaction rather than only a visited bit.
- Separated set determinism from parent, order, retry, and work determinism.
- Classified reached, distance, parent, all-parent, count, multiplicity, and
  sequence outputs by their merge algebra.
- Added a diamond retry counterexample that preserves distances but changes
  path count from two to three.
- Treated canonical parent as a total-order reduction requiring complete
  equal-depth closure.
- Kept same-layer idempotence separate from arbitrary cross-level first claim.
- Defined typed multi-GPU protocol objects and fault/reorder validation cases.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS proof-obligation independence matrix

- Refined the existing contract map into independent soundness, completeness,
  identity, metric, schedule, coverage, publication, output, termination, and
  evidence predicates.
- Added minimal non-implication witnesses for every adjacent-looking claim.
- Separated successor completeness from traversal coverage and publication.
- Separated termination safety from termination-detection liveness.
- Kept runtime truth separate from evidence adequacy and independence.
- Built a requirement matrix for witness, target, bounded, exhaustive,
  canonical-path, and shortest-DAG/count outputs.
- Added a proof-assistant-oriented assumption skeleton.
- Described which predicate quantifiers expand from one GPU to many GPUs.
- Attempted one combined `ask_experts` consultation with `multigpu_beam` and
  `autolean`; it returned `fetch failed`, so no expert claim was used and no
  infrastructure repair was attempted.
- Added no implementation, optimizer, benchmark, Docker run, or GPU code.

## 2026-08-28: BFS obligation conservation and termination cuts

- Defined stable logical successor obligations separately from physical retry
  copies.
- Introduced a semantic pending/active/transport/owner/publication/retired
  lifecycle conservation equation.
- Delayed retirement until endpoint, metadata, and accepted-state publication
  effects are complete.
- Added sender/receiver transfer-gap and inconsistent-cut false-zero failures.
- Connected destroyed versus leaked credit to termination safety versus
  liveness.
- Included kernels, device buffers, asynchronous transfers, collectives, and
  spills as possible causal-work locations.
- Defined completed-level conditions over root and dynamically created child
  obligations.
- Separated total outstanding work, minimum unfinished key, and output closure.
- Extended the accounting to label-correcting reactivation and checkpoint
  epochs.
- Defined consistent-cut telemetry and four minimal failure fixtures.
- Added no implementation, optimizer, benchmark, Docker run, or GPU code.

## 2026-08-28: BFS shortlex-rank recurrence

- Added an explicit `ab` versus `ba` counterexample separating minimum parent
  state ID from shortlex path choice.
- Defined recursive candidate keys from canonical parent-word rank and label
  rank.
- Proved per-child minimum selects the least equal-length candidate word.
- Defined next-frontier dense ranks by ordering selected minimum keys.
- Kept path-order rank separate from semantic state identity and visited.
- Separated per-child reduction from global frontier ranking.
- Derived distributed closure and partition-count invariance requirements.
- Scoped generator/source/action/path orders as canonical-output epochs.
- Kept scalar target discovery separate from shortlex target closure.
- Extended the boundary to multi-source and quotient/Schreier outputs.
- Added no implementation, optimizer, benchmark, Docker run, or GPU code.

## 2026-08-28: Bidirectional BFS shortlex connector closure

- Derived the reverse prepend recurrence `(label rank,suffix rank)` for
  forward-oriented suffixes.
- Added an `ab` versus `ba` counterexample to ordinary reverse traversal order.
- Rejected universal repair by merely reversing the inverse alphabet order.
- Defined canonical words through fixed vertex and crossing-edge connectors.
- Kept frontier-local ranks from different split depths out of direct global
  comparison.
- Separated shorter-path exclusion from equal-length lexical exclusion.
- Kept distance-optimal first intersection separate from shortlex optimality.
- Defined the global multi-owner reduction over every optimal connector.
- Separated inverse traversal operations, stored forward labels, and lexical
  alphabet semantics.
- Distinguished one canonical connector from all-path/count outputs.
- Added no implementation, optimizer, benchmark, Docker run, or GPU code.

## 2026-08-28: BFS on unicyclic graphs

- Used the unique cycle with attached trees as the minimal exact step beyond a
  tree.
- Derived all distances from the source-to-cycle gate, cyclic distance, and
  attached-tree depth.
- Proved that odd parity produces one same-layer cycle edge and unique shortest
  parents.
- Proved that even parity produces one antipode with two shortest predecessors
  and no same-layer cycle edge.
- Separated localized candidate convergence at the even antipode from
  shortest-path multiplicity propagated through its attached subtree.
- Localized all non-tree candidate convergence to that single parity event.
- Showed why one cycle already invalidates generic parent-only revisit
  suppression despite the tiny duplicate count.
- Distinguished the odd two-direction same-layer scans from the even repeated
  next-state proposal.
- Identified the even cycle as a minimal owner-authority and parent-selection
  fixture.
- Proved that a finite connected simple undirected unicyclic Cayley graph must
  itself be a cycle by regularity and average degree two.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS on cactus graphs

- Extended the one-cycle analysis to edge-disjoint cycles arranged by a
  block-cut tree.
- Derived exact source-target distance as a sum of bridge and shortest cyclic
  arc contributions.
- Proved the shortest-path formula `sigma(s,t)=2^a` for antipodal even-cycle
  crossings.
- Separated immediate predecessor multiplicity from complete path multiplicity.
- Showed that each cactus cycle contributes one odd same-layer or even
  double-parent BFS signature.
- Related the number of cycle blocks to cycle rank `m-n+1`.
- Recorded a `C4--C5--C6` mental fixture with four target geodesics but only two
  local double-parent meetings.
- Rejected a simple product rule for global frontier widths despite exact
  pairwise block decomposition.
- Applied output-contract distinctions to one-tree, DAG, path-count, and
  all-path results.
- Kept low articulation cut separate from per-level owner balance and exact
  metadata combination.
- Marked theta graphs and overlapping Cayley relators outside the cactus product
  theorem.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS on theta graphs

- Used three internally disjoint branch-to-branch paths as the first exact
  obstruction to cactus independence.
- Derived branch distance and one/two/three-way geodesic multiplicity from the
  minimum path lengths.
- Derived each internal path vertex's distance from direct and globally
  shortest via-branch routes.
- Showed that one path's BFS layers can depend on another path's length.
- Classified long-path wave meetings by parity of `d+L_i`.
- Separated cycle rank two from the graph's three simple cycles.
- Used `Theta(3,3,3)` to obtain three candidate proposals and path count three
  at one unique state.
- Used `Theta(2,3,3)` and `Theta(2,4,4)` to move equal-rank duplicate work
  between same-layer and double-parent events.
- Defined a three-producer owner-authority fixture for frontier, parent, retry,
  and path-count semantics.
- Rejected cycle-basis dimension as a counter of Cayley geodesic alternatives
  or BFS duplicate records.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS layer-edge and duplicate conservation

- Partitioned every undirected edge into same-layer or consecutive-layer
  classes `A_d` and `B_d`.
- Derived exact per-level adjacency-scan conservation.
- Proved that outward candidate excess is
  `B_d-|F_(d+1)|=sum(p(v)-1)`.
- Proved the radial cycle-rank identity: same-layer edges plus excess shortest
  predecessors equal `m-n+1`.
- Defined root-dependent cyclomatic charge `q_d` whose total remains invariant.
- Derived the complete scan rejection identity `(n-1)+2 beta` beyond the
  `n-1` accepted nonroot states.
- Recovered tree, odd/even unicyclic, cactus, and theta results as instances of
  one conservation law.
- Separated bipartite predecessor excess from nonbipartite same-layer witnesses.
- Kept structural surplus edges separate from shortest-path count, which can be
  exponentially larger.
- Interpreted outward excess as semantic GPU dedup opportunity rather than a
  promised speedup.
- Restricted the formulas explicitly to complete finite simple undirected BFS
  with symmetric adjacency scans.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS successor occurrences and Cayley label multiplicity

- Generalized simple-support edge accounting to finite directed labeled
  successor multisets.
- Separated occurrence records, support arcs, and endpoint states.
- Partitioned every level's occurrences into visited-ball and next-layer
  destinations.
- Decomposed next-state excess into same-parent label multiplicity and
  cross-parent structural convergence.
- Derived complete occurrence rejection without incorrectly reducing it to
  simple-support cycle rank.
- Applied output contracts to vertex, one-path, predecessor-DAG, labeled-DAG,
  and labeled-count results.
- Kept delivery retries outside semantic graph multiplicity.
- Proved that a free Cayley action with distinct nonidentity generators has no
  same-parent length-one aliases.
- Derived Schreier aliases and loops from the state stabilizer condition.
- Connected state-dependent aliases to different generation and owner-side
  combination scopes.
- Required `P_d` instrumentation to distinguish representation multiplicity
  from structural convergence.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS on Schreier stabilizer cosets

- Converted same-parent Schreier aliases into exact left-coset equivalence
  classes of the current state stabilizer.
- Derived endpoint multiplicity, loop-label count, distinct support endpoints,
  and total alias excess from intersections with `S`.
- Separated conjugate stabilizer order from generator-coset intersection
  profile.
- Proved conjugation invariance of `S` sufficient for uniform alias histograms
  across the orbit.
- Added a hand-checkable transitive `S_3` point action with constant labeled
  degree three but support endpoint counts `2,3,2`.
- Refined the fixture: equal endpoint counts at states one and three hide alias
  without loop versus alias with loop.
- Recovered the free Cayley action as the trivial-stabilizer singleton-coset
  case.
- Showed how loops and new-endpoint aliases enter different occurrence counters.
- Explained how a symmetry quotient can introduce aliases absent in its free
  Cayley cover.
- Separated generator-regular raw work from support, visited, and routing work.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: reverse BFS on Schreier graphs

- Derived exact reverse successor generation from the inverse collection
  `S^-1`.
- Separated backward distance-to-target semantics from forward-from-target
  traversal with an incorrect generator oracle.
- Expressed forward aliases through right stabilizer cosets `Ks` and reverse
  aliases through left-coset `sK` intersections after inversion (names corrected
  2026-08-31; right-action equations unchanged).
- Proved forward/reverse loop-label counts equal even when support profiles
  differ.
- Recorded inverse-closed generators and normal stabilizers as sufficient
  profile-symmetry conditions.
- Extended the three-point `S_3` fixture from forward counts `2,3,2` to reverse
  counts `2,2,3`.
- Kept exact bidirectional correctness separate from smaller-side scheduling.
- Rejected state-frontier cardinality as a universal proxy for occurrence,
  support, or routing work.
- Added reverse-label conversion and rich-output preservation to path-stitching
  obligations.
- Separated direction-independent state ownership from direction-dependent
  routing traffic.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: stabilizer-aware BFS work waterfall

- Derived per-parent loop, distinct support-endpoint, nonloop support, and
  same-parent alias counts from stabilizer cosets.
- Aggregated them into raw occurrence, loop, alias, and distinct support-arc
  level counts.
- Split support arcs into visited-ball and next-layer destinations.
- Defined cross-parent support convergence separately from same-parent aliases.
- Proved the exact waterfall
  `G_d=L_d+R_d+V_d+D_d+|F_(d+1)|`.
- Refined the `S_3` fixture to show equal total support endpoint counts hiding
  different nonloop support volumes.
- Recovered the free Cayley simplification with zero loop and same-parent alias
  terms but retained visited and cross-parent work.
- Mapped waterfall classes to stabilizer, relation, predecessor, and output
  meanings.
- Kept semantic elimination order separate from physical GPU pipeline order.
- Required owner matrices at occurrence, support, unvisited, and accepted
  boundaries.
- Rejected composition ratios as speedup claims without measured removal and
  overhead.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: directed BFS arc-surplus accounting

- Partitioned every reachable directed support arc into next-layer or
  visited-ball destinations.
- Derived exact arc surplus over a one-parent BFS arborescence.
- Refined visited arcs by endpoint layer and back-depth lag.
- Separated support-arc rejection from labeled occurrence multiplicity.
- Used the ordered complete DAG to reject `m-n+1` as directed-cycle count.
- Contrasted a directed diamond and directed cycle with equal scalar surplus but
  different structural meaning.
- Kept BFS depth position separate from return reachability and SCC membership.
- Recorded that SCC condensation may retain deep-to-shallow BFS arcs despite
  being acyclic.
- Applied output-contract distinctions to predecessor versus visited-ball arcs.
- Separated producer-side visited rejection from owner-side new-parent
  convergence.
- Kept forward and reverse radial surplus profiles distinct.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: BFS prefix conservation and early stopping

- Defined a completed-layer prefix through retired logical successor
  obligations rather than physical queue appearance.
- Derived exact nonaccepting-occurrence accounting for the constructed ball.
- Located the exact radius-`R` construction boundary at expansion through
  `F_(R-1)`.
- Separated partial-parent arithmetic from complete next-frontier closure.
- Proved scalar target distance can finalize on mid-layer discovery after exact
  shallower-ball completion.
- Kept canonical parent, complete DAG, path count, connectors, and full frontier
  unfinalized until their equal-depth obligations close.
- Separated positive discovery from bounded negative and unreachable
  certificates.
- Recorded order, batching, check, cancellation, and capacity dependence of
  early-stop work.
- Required global owner/message/kernel/publication closure for a distributed
  completed layer.
- Separated completed-prefix waterfall totals from partial/cancelled latency
  totals.
- Kept partial bidirectional stopping tied to global unfinished-depth bounds.
- Added no experiment, optimizer, production implementation, benchmark, or GPU
  code.

## 2026-08-28: CayleyPy and DeepCubeA Cube action audit

- Re-read the checksum-pinned CayleyPy fixture and the official DeepCubeA
  `cube3.py` source at the commit already named by REF-029.
- Corrected a representation ambiguity: DeepCubeA search identity is a vector
  of 54 unique sticker IDs; six face classes are derived only for neural input.
- Reduced cross-runtime equivalence to one simultaneous 54-position conjugacy
  plus an explicit bijection of the 12 signed move labels.
- Separated unlabeled distance transfer from labeled word/replay transfer.
- Kept REF-029's DeepCubeA status unverified because neither the runtime oracle
  nor the conjugacy equation has been executed.
- Specified a bounded Rust/Docker semantic gate without implementing or
  optimizing it while Docker is unavailable.
- Added no host calculation, dependency installation, Docker repair, GPU code,
  solver, benchmark, or production implementation.

## 2026-08-28: simultaneous conjugacy of labeled permutation actions

- Corrected the cross-runtime map for unique-sticker states: the same
  coordinate bijection must rename array positions and sticker identities.
- Derived the simultaneous tuple conjugacy criterion under the pull-action
  convention used by REF-029.
- Added a three-position counterexample showing that individually conjugate
  generators need not admit one common conjugacy.
- Separated abstract generated-group isomorphism, simple support-graph
  isomorphism, fingerprints, and exact labeled-action equivalence.
- Derived orbit propagation, the anchored stabilizer condition, and
  centralizer-controlled nonuniqueness.
- Separated translated shortest-word validity from shortlex preservation under
  the target alphabet order.
- Kept semantic equivalence separate from layout, routing, hash, and GPU
  performance equivalence.
- Added no runtime experiment, optimizer, solver, Docker repair, or GPU code.

## 2026-08-28: decremental BFS invalidation versus repair

- Rechecked Docker once; authoritative server access still failed with
  permission denied on `dockerDesktopLinuxEngine`, so no executable probe ran.
- Proved that reachability in the surviving old complete shortest-path DAG is
  exactly the set of vertices retaining their old scalar distances.
- Separated exact old-label invalidation from computation of longer replacement
  distances outside the old DAG.
- Derived the depth-ordered surviving-support recurrence.
- Proved that a selected parent subtree overapproximates damage after its tree
  edge fails and that a non-tree edge deletion preserves scalar distances.
- Characterized single-edge invalidation by dominance inside the old shortest
  DAG and explained why batches require combined reachability.
- Separated scalar, parent, complete-DAG, path-count, and canonical-output
  change regions.
- Extended the semantics to global Cayley/Schreier generator-label deletion and
  parallel invalidation reporting.
- Added no runtime experiment, implementation, optimizer, Docker repair, or GPU
  code.

## 2026-08-28: incremental BFS single-edge sensitivity

- Proved the exact all-target formula for one directed unit-edge insertion from
  an old source prefix and old head-rooted suffix.
- Partitioned targets into strict distance-decrease, equal-distance richer
  output, and irrelevant regions.
- Proved that strict head improvement is necessary and sufficient for any
  scalar distance decrease anywhere.
- Derived the equal-head shortest-cone and through-edge path-count product.
- Explained why the old source predecessor DAG need not contain head-rooted
  suffixes used after insertion.
- Added the two-orientation undirected formula.
- Used a two-edge path on isolated vertices to reject independent single-edge
  composition for batches.
- Kept a global Cayley generator-family insertion separate from one local edge.
- Added no executable probe, implementation, optimizer, Docker repair, or GPU
  code.

## 2026-08-28: incremental BFS batch endpoint closure

- Decomposed every updated path into old-graph metric segments separated by a
  unique sequence of inserted-edge occurrences.
- Derived the at-most-`r` min-plus recurrence and finite `|F|` closure bound.
- Built the conceptual endpoint metric graph and stated its exact scalar
  distance contract together with preprocessing cost boundaries.
- Distinguished insertion-use rounds from original BFS depth.
- Showed why redundant terminal subdivisions are harmless for minimum distance
  but overcount additive paths.
- Separated scalar convergence from equal-label DAG, count, and canonical
  closure.
- Kept atomic batch versions separate from sequentially visible updates.
- Clarified undirected semantic edge identity and repeated Cayley generator
  labels at different translated edge occurrences.
- Added no executable probe, implementation, optimizer, Docker repair, or GPU
  code.

## 2026-08-28: distributed BFS 1D/2D expand-fold semantics

- Read the primary Buluç--Madduri 1D/2D BFS and graph-partitioning sources,
  Beamer et al.'s distributed direction-optimizing report, and the Graph500
  reference/specification boundary.
- Fixed the source-row adjacency convention before interpreting processor-grid
  directions.
- Separated 2D expand completeness from fold authority and global level
  closure.
- Distinguished frontier replication, candidate fold, static cut, actual
  records, collective participants, bytes, and elapsed collective time.
- Kept Graph500 one-tree validation separate from DAG, count, and canonical
  output contracts.
- Split adjacency, frontier, visited metadata, and candidate-buffer placement
  for peak-memory reasoning.
- Explained why an implicit Cayley graph needs a proved generator or
  transformation shard axis rather than inheriting an explicit checkerboard.
- Added topology mapping and cross-shard alias obligations for multi-GPU study.
- Added no executable probe, implementation, optimizer, Docker repair, or GPU
  code.

## 2026-08-28: distributed bottom-up systolic early exit

- Read Algorithm 4 of Beamer et al.'s primary distributed bottom-up report.
- Separated exact frontier bitmap replication from full global frontier copies.
- Reconstructed the `p_c` substep schedule that rotates completed candidate
  responsibility along processor rows.
- Derived snapshot, shard coverage, witness, completion persistence,
  publication, exhaustion, and epoch invariants.
- Classified false-completed as potentially lossy and false-uncompleted as
  extra work requiring exact duplicate handling.
- Kept substep-delayed early exit separate from instantaneous cancellation and
  recorded its latency/work tradeoff.
- Separated arbitrary-parent support from canonical, DAG, count, and
  multi-source output requirements.
- Mapped frontier/completed/parent traffic and closure to GPU, multi-GPU,
  implicit, and Cayley boundaries without proposing an implementation.
- Added no executable probe, implementation, optimizer, Docker repair, or GPU
  code.

## 2026-08-28: research-direction correction

- Recorded a failure in the study process: breadth of coverage, note counts,
  and adjacent implementation audits had begun to substitute for the user's
  intended gradual understanding of BFS.
- Made a single plain-language BFS question and a changed mental model the
  entry condition for future study steps.
- Demoted note, claim, source, test, and coverage counts to bookkeeping rather
  than progress metrics.
- Made hand traces, minimal graphs, counterexamples, and proof sketches the
  default instruments; code is exceptional and deliberately small.
- Restricted library, benchmark, validator, and distributed-system audits to
  cases where a previously stated BFS question actually requires them.
- Deferred GPU and multi-GPU engineering until the semantics are understood
  and the user separately requests implementation work.
- Added no BFS theory claim, source audit, executable probe, implementation,
  optimization, Docker action, or GPU code in this correction step.

## 2026-08-28: question card — is `visited` the semantic core?

- **Question:** Is the initial statement that `visited` is the semantic core
  of BFS and the frontier is only a schedule literally correct?
- **Small examples:** In a rooted tree, excluding the incoming parent edge
  prevents every revisit without a global visited set. In a cycle, the same
  omission unfolds the finite state graph into an unbounded walk/path tree.
  In an implicit Cayley or Schreier graph, different move words can converge
  through relations or stabilizers even after immediate inverse moves are
  removed.
- **Prediction checked:** `visited` should be essential whenever BFS is exact.
- **Correction:** What is essential for ordinary graph BFS is exact state
  identity plus a rule preventing an already reached state from becoming a new
  deeper state. A global visited structure is the usual mechanism, but trusted
  tree structure can supply the same fact locally.
- **Frontier correction:** A queue/frontier is a scheduling representation for
  distance computation, but a completed frontier may also be the requested
  metric layer and evidence that all shorter successor obligations closed.
- **Remaining uncertainty:** Memory-reduced schemes that replace global
  visited need a graph-specific proof of which earlier states can reappear;
  absence of duplicates in a finite prefix is not such a proof.
- Updated the overstrong sentence in note 1. Added no new thematic note, code,
  source audit, Docker action, performance work, or GPU design.

## 2026-08-28: question card — when may global `visited` be omitted?

- **Question:** What exact graph property permits BFS without global visited?
- **Correction to the question:** The required condition depends on the output;
  there is no single graph-only answer.
- **First shallowest target:** Breadth-first enumeration of a finitely branching
  path tree reaches every finite depth after finite work. It can therefore find
  a reachable shallowest target without state deduplication, even when the base
  graph has cycles.
- **Unique graph frontiers/distances:** If every state must appear once, paths
  that converge to one state need exact reconciliation. A trusted rooted tree
  supplies uniqueness structurally; more generally an injective canonical
  generator can replace global visited. Merely being a DAG is insufficient:
  `s -> a -> x` and `s -> b -> x` is acyclic but generates `x` twice.
- **Exhaustion/unreachability:** A finite cyclic state graph unfolds to an
  infinite path tree, so frontier emptiness is lost without duplicate/cycle
  control. A finite DAG's path tree is finite and eventually exhausts, but it
  may contain exponentially many duplicate endpoint occurrences.
- **Work bound:** The ordinary `O(V+E)` graph-traversal bound requires one
  authoritative expansion per reached state or an equivalent exact mechanism;
  shortest-target correctness alone does not provide that bound.
- **New intuition:** `visited` simultaneously supports several contracts that
  must not be conflated: unique-state output, finite closure on cyclic finite
  graphs, and graph-sized work. It is not required merely for ordering path
  occurrences by length.
- This reconciles note 1 with the already-correct graph-search/tree-search
  distinction in notes 9 and 23. Added no thematic note, code, source audit,
  Docker action, performance work, or GPU design.

## 2026-08-28: question card — are three rolling layer roles necessary?

- **Question:** Note 181 proves that `previous`, `current`, and `building next`
  suffice for scalar BFS on an undirected graph. Are all three roles real, or
  is one redundant?
- **Previous-layer witness:** In the diamond
  `s--a, s--b, a--x, b--x`, a single selected parent of `x` blocks only one of
  its two depth-one predecessors. Expanding `x` needs the other predecessor's
  old-layer identity to avoid rediscovering it at depth three.
- **Current-layer witness:** In the triangle `s--a--b--s`, expanding `a` sees
  the already-current vertex `b`. Without equivalent `F_1` membership, it can
  be misclassified as next-layer work.
- **Next-layer witness:** In the same diamond, `a` and `b` both generate `x`.
  Building-next reconciliation is what turns two occurrences into one state.
- **Correction:** The three names describe information roles, not a requirement
  for three physical containers. Epoch bits, queues with exact membership,
  structural tree guarantees, or richer predecessor records may encode or
  replace a role, but the corresponding decision still needs a proof.
- **New intuition:** Safe forgetting is a sliding local classification problem:
  every generated endpoint must be recognized as inward, lateral, or newly
  outward. Undirected distance geometry limits those possibilities to three
  adjacent layers.
- Added the three counterexamples to note 181. Added no code, source audit,
  Docker action, performance work, or GPU design.

## 2026-08-28: question card — when is `current` not a novelty filter?

- **Question:** The triangle proves that generated candidates may need testing
  against the current layer. When can that test be omitted?
- **Reasoning:** In an undirected graph every edge spans depths differing by at
  most one. Bipartiteness forbids equal-parity endpoints, hence forbids the
  depth difference zero. A generated neighbor of `F_d` is then only inward in
  `F_(d-1)` or outward in `F_(d+1)`.
- **Converse:** If one complete rooted BFS of a connected component has no
  same-layer edge anywhere, depth parity colors every edge oppositely, so that
  component is bipartite.
- **Correction:** `current` has two roles that should not be conflated. It is
  the work frontier that must be expanded, but on a bipartite component it is
  not needed as a candidate-membership filter. `previous` and `building next`
  still reject inward returns and merge outward convergence.
- **Cayley intuition:** A generator-parity homomorphism to `Z_2` makes the
  distinction structural. Changing the unit generator set can break parity and
  create same-layer hits without changing the underlying state universe.
- Linked this specialization into note 181 using the existing bipartiteness
  and generator-set conclusions of notes 21, 31, and 68. Added no new thematic
  note, code, source audit, Docker action, performance work, or GPU design.

## 2026-08-28: question card — how can one generator change `visited` work?

- **Question:** How can current-layer rejection appear when the state universe
  and generated group do not change?
- **Hand trace:** For `Z_4` with generators `{1,3}`, the Cayley graph is `C_4`
  and the layers from zero are `{0}`, `{1,3}`, `{2}`. Every edge crosses layer
  parity.
- **Generator change:** Adding the old length-two element `2` as a unit
  generator produces `{1,2,3}`. The Cayley graph becomes `K_4`, with every
  nonzero state in `F'_1`.
- **Observed consequence:** Expansion of `F'_1` has six directed occurrences
  ending inside `F'_1`; for example `1+1=2` and `1+2=3`. These are neither new
  states nor inward returns to the root, so current-layer filtering is now an
  independent decision.
- **New intuition:** `visited` work is induced by the chosen unit metric, not by
  state representation alone. Promoting an even old word to one step creates
  an odd cycle, reshapes the layers, and can turn previously impossible lateral
  hits into ordinary candidates.
- Added the four-state trace to note 68 beside the larger Cube QTM/HTM example.
  Added no code, source audit, Docker action, performance work, or GPU design.

## 2026-08-28: question card — may a BFS continue across a generator change?

- **Question:** If exact visited identity is preserved, can an in-progress BFS
  simply begin using a newly added generator?
- **Hand trace:** In `Z_4` with `{1,3}`, expanding root zero first produces
  `F_1={1,3}`. Add generator `2` only after the root retires. Continuing from
  `F_1` discovers state `2` through an old two-step route and can label it two.
- **Contradiction:** In the new generator metric, `0+2=2`, so state `2` belongs
  to the new first layer. The required improving proposal originates in an
  expansion already declared complete and will never appear from merely
  continuing the old frontier.
- **Correction:** Exact visited identity certifies that the record denotes the
  same state; it does not certify that its depth is final across graph epochs.
  After an insertion, old finite labels remain path-length upper bounds, while
  the reached set may already be complete and still hide stale distances.
- **New intuition:** A completed frontier is a certificate relative to one
  transition relation. Changing generators reopens retired expansion
  obligations; it is dynamic BFS repair, not ordinary queue continuation.
- Added the trace to note 22's insertion section. Added no new thematic note,
  code, source audit, Docker action, performance work, or GPU design.

## 2026-08-28: question card — why is deletion repair asymmetric?

- **Question:** Can removing a generator be repaired by the same improving
  relaxation used after insertion?
- **Reverse hand trace:** Begin with `Z_4` and generators `{1,2,3}`. Every
  nonzero state has old depth one. Remove generator `2`; the graph becomes the
  four-cycle generated by `{1,3}`, and state `2` moves to depth two.
- **Invalidation evidence:** The only old depth-one witness `0--2` disappears,
  so the old shortest-path DAG correctly marks `D(2)=1` invalid.
- **Repair boundary:** New shortest paths `0--1--2` and `0--3--2` use edges
  that were same-layer under the old metric. They were not arcs of the old
  shortest DAG, so that DAG can classify the stale label but cannot compute
  its replacement.
- **Correction:** Insertion preserves old paths and can only lower labels;
  deletion destroys witnesses and can only raise labels or reach infinity.
  An `atomicMin`-style improvement cannot express the required increase, and a
  boolean reached set changes not at all in this example.
- **New intuition:** Invalidation asks whether an old optimum still has a
  witness. Repair asks which formerly nonoptimal edges become part of the new
  optimum. They are different graph questions even on four states.
- Added the reverse trace to note 22, consistent with note 186's old-DAG
  preservation theorem. Added no new thematic note, code, source audit, Docker
  action, performance work, or GPU design.

## 2026-08-28: question card — does an edge have an intrinsic BFS role?

- **Question:** Is a predecessor, same-layer, or inward edge an intrinsic kind
  of edge, or only a relation to the current BFS layering?
- **Definition:** For surviving arc `u -> v`, its radial difference is
  `r=D(v)-D(u)`. A shortest-predecessor arc has `r=1`; in an undirected graph,
  `r=0` is lateral and `r=-1` is inward.
- **Epoch change:** After deletion let `D'=D+Delta`. The same arc then has
  `r'=r+Delta(v)-Delta(u)`.
- **Consequences:** An old predecessor remains a predecessor only when both
  endpoints rise equally. An old lateral edge becomes a new predecessor when
  its destination rises exactly one layer more than its source.
- **Concrete witness:** In the reverse `Z_4` trace, state `1` stays at depth one
  while state `2` rises from one to two. Edge `1 -> 2` therefore changes from
  `r=0` to `r'=1` and enters the new shortest-path DAG.
- **New intuition:** BFS does not permanently classify edges; it classifies
  them radially relative to a source, a unit metric, and a graph epoch. The
  formula explains a known role change but cannot determine `Delta`; computing
  those increases is exactly the repair problem.
- Added the radial-change identity to note 22. Added no new thematic note,
  code, source audit, Docker action, performance work, or GPU design.

## 2026-08-28: question card — does a duplicate generator change BFS?

- **Question:** If two generator labels perform the same state transition, has
  the BFS graph changed or only its successor stream?
- **Hand trace:** In `Z_4`, declare `a -> +1`, `b -> +1`, and `c -> -1`. From
  zero the occurrence stream is `(a,1),(b,1),(c,3)`, but the unique support
  neighbors are `{1,3}`.
- **Vertex result:** The support graph remains `C_4`, with layers
  `{0}`, `{1,3}`, `{2}`. Reachability and hop distance do not change.
- **Labeled result:** If `a` and `b` are genuinely distinct moves, state `1`
  has two length-one witnesses and state `2` has five shortest labeled words:
  `aa, ab, ba, bb, cc`.
- **Contract boundary:** If the repeated entry is a retry or an accidental copy
  of one semantic move, it must not increase path counts. Equal endpoints say
  nothing about whether equal-action records are distinct graph occurrences.
- **New intuition:** An implicit successor function naturally emits a multiset.
  Vertex BFS runs on its support graph, while generated work and labeled-path
  outputs may live on the occurrence graph. Deduplication is therefore relative
  to the requested output, not a universal deletion of repeated endpoints.
- Added the four-state trace to note 157. Added no new thematic note, code,
  source audit, Docker action, performance work, or GPU design.

## 2026-08-28: question card — duplicate action versus identity move

- **Question:** Two generator changes can preserve every vertex distance: add
  another label for `+1`, or add an identity label. Are their richer effects
  the same?
- **Identity trace:** Adding `e -> +0` to `Z_4` makes every expanded state `x`
  emit a self-loop occurrence `(e,x)`. Vertex layers remain
  `{0}`, `{1,3}`, `{2}`, but old/current-hit work increases once per state.
- **Shortest-word proof:** Any word containing `e` reaches the same endpoint
  after deleting `e`, with length reduced by one. Therefore no shortest word,
  including the empty shortest word for the root, contains the identity label.
- **Contrast:** A distinct label for the existing nonidentity action `+1` can
  replace a genuine geodesic step and multiply labeled shortest words, as in
  `aa, ab, ba, bb`. Identity adds work but no shortest-path multiplicity.
- **New intuition:** “Distances unchanged” says only that the support metric is
  stable. It does not determine occurrence traffic or richer outputs; those
  depend on whether the extra record can participate in a geodesic and whether
  its label is semantically distinct.
- Added the identity contrast to note 157 beside the duplicate-action trace.
  Added no new thematic note, code, source audit, Docker action, performance
  work, or GPU design.

## 2026-08-28: question card — when may successors collapse by endpoint?

- **Question:** May a producer merge equal-endpoint occurrences before they
  reach authoritative visited or cross a GPU/rank boundary?
- **Common fact:** Within one completed BFS layer, all such occurrences propose
  the same scalar depth `d+1`. Therefore one exact state record preserves the
  next-frontier set, reached membership, and scalar distance proposal.
- **Output-dependent summaries:** One arbitrary path needs one valid witness;
  canonical output needs the minimum complete key; all-parent output needs the
  distinct parent set; labeled DAG/count output needs the declared label or
  contribution identities and their correct aggregation.
- **Four-state instance:** Root occurrences `[(a,1),(b,1)]` may reduce to state
  `{1}`, arbitrary `a` or `b`, canonical `min(a,b)`, label set `{a,b}`, or count
  two. These summaries are not interchangeable.
- **Retry boundary:** Re-delivery of `(a,1)` is operational duplication, not a
  third labeled path contribution. Non-idempotent counts require stable
  contribution identity.
- **Authority boundary:** Local collapse proves only that a local occurrence
  survives. It cannot prove global novelty or complete canonical/all-parent
  closure across other producers; those decisions remain owner/global facts.
- **New intuition:** “Deduplicate by endpoint” is shorthand for an output-
  specific reduction algebra, not a universal BFS operation.
- Added the concrete summary table to note 157. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — what must a distributed owner own?

- **Question:** Can equal endpoint occurrences route to different owners when
  the routing rule is deterministic for each parent/label record?
- **Counterexample:** In `Z_4`, labels `a` and `b` both send root zero to state
  `1`. Route `a` to rank zero and `b` to rank one. Each local visited shard sees
  state `1` as absent and both can accept it, producing two physical frontier
  records for one semantic vertex.
- **Correction:** Determinism per occurrence is insufficient. For owner-computes
  vertex BFS, ownership must be stable over the endpoint's exact equality
  class: equal states route to one authority in one epoch.
- **Parent variant:** Routing by parent fails similarly when two frontier
  parents converge on one child. It partitions histories, not vertices.
- **Richer-output boundary:** Label/parent contributions may travel separately
  when the output needs them, but they still require an explicit exact
  endpoint-keyed reduction before unique acceptance or layer closure.
- **New intuition:** A distributed `visited` table is a partition of the vertex
  identity space. If the partition key includes accidental history, the system
  no longer has one global Boolean fact “was this vertex reached?”
- Added the four-state ownership counterexample to note 51. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-28: question card — must an owner hash be collision-free?

- **Question:** If unequal states share an owner hash, is exact distributed BFS
  already broken?
- **Hand trace:** Let `Z_4` states `1` and `3` satisfy `h(1)=h(3)=0`. Routing
  both to rank zero is safe when that rank retains full keys and compares exact
  state identity; both vertices are accepted.
- **Failure variant:** If the owner equates hash equality with state equality,
  accepting `1` makes it reject `3` as already visited, removing a genuine
  frontier state.
- **Correction:** Owner hashing is allowed to be many-to-one. It partitions
  work locations, not semantic identity. Collisions affect balance until an
  exact comparison incorrectly turns co-location into conflation.
- **New intuition:** The routing hash answers “where should these records meet?”
  The equality check answers “are they the same vertex?” Reusing the first
  answer as the second is the correctness error.
- Added the two-state collision trace to note 51, consistent with note 28's
  collision-resolving table contract. Added no new thematic note, code, source
  audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — which advisory-filter answer may be final?

- **Question:** Can an approximate or stale local membership answer directly
  accept or reject a BFS candidate before authoritative visited?
- **False-positive trace:** In `Z_4`, root expansion must produce
  `F_1={1,3}`. If a bit set for state `1` also matches unequal state `3`, using
  that positive as final `seen` deletes a real depth-one vertex.
- **Stale-negative trace:** If the authority already accepted state `1` but a
  delayed exact replica still reports absent, forwarding the duplicate causes
  only extra work. Accepting it locally creates split authority and a duplicate
  frontier record.
- **One-sided rule:** An approximate positive cannot final-drop because it may
  be false. A stale exact negative cannot final-accept because it may be old.
  A sound exact positive may reject compatible state-only output; exact novelty
  still requires a linearized claim.
- **New intuition:** Filters do not merely have an “accuracy.” Each error
  direction maps to a different BFS failure: false rejection loses a vertex;
  false acceptance duplicates authority. Exactness depends on which action an
  answer is permitted to trigger.
- Added the four-state trace to note 52. Added no new thematic note, code,
  source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — may a true `seen` drop every record?

- **Question:** If an exact replica soundly proves that endpoint `t` is already
  visited, is every later record for `t` semantically redundant?
- **Diamond trace:** In `s->a->t` and `s->b->t`, let `(a,t)` win the vertex
  claim. When `(b,t)` arrives, `t` is truly seen and needs no second insertion.
- **Output split:** Dropping `(b,t)` preserves reached membership, distance,
  and one arbitrary shortest path. It loses parent `b`, changes the shortest-
  path count from two to one, and violates canonical parent output if `b` is
  ordered before `a`.
- **Correction:** A sound positive proves idempotence only of vertex insertion.
  It says nothing by itself about parent, label, count, or canonical-order
  contributions carried by the occurrence.
- **New intuition:** `visited` answers a vertex question. Rich BFS outputs attach
  additional merge state to that vertex, and their closure can remain open
  after the Boolean novelty decision is final.
- Added the diamond trace to note 52's sound-positive section. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-28: question card — is accepted the same as publishable?

- **Question:** For distance and one arbitrary path, is a sound Boolean
  `already accepted` positive sufficient to discard every losing occurrence?
- **Counterexample:** In the diamond, let `(a,t)` win the exact visited claim
  but stop or fail before its frontier and parent payload becomes recoverably
  published. If `(b,t)` observes `seen` and drops, state `t` can remain in
  visited with no expansion duty and no replayable path.
- **Correction:** Safe early rejection requires more than exact novelty. The
  winner must be `PUBLISHED/EXPANDED` or retain a live helpable/replayable
  publication obligation carrying the required output payload.
- **Two independent closure axes:** Even after publication is safe, rich output
  may still require delayed parents, labels, counts, or canonical contenders.
  Publication completeness and contender completeness are separate facts.
- **New intuition:** `visited` says who won the right to represent a vertex;
  publication says whether someone still bears the duty to make that vertex
  usable. Losing candidates may disappear only after responsibility continuity
  is proved.
- Corrected note 52's overbroad sound-positive wording and linked it to note
  178's claim/publication state machine. Added no new thematic note, code,
  source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — may a losing claimant substitute its witness?

- **Question:** If the novelty winner loses its payload, must a helper recover
  that exact parent record, or may another equal-depth candidate publish its
  own witness?
- **Diamond trace:** Let `(a,t)` win and disappear before publication, while a
  loser still holds `(b,t)`. Both parents lie at depth one, so `(b,t)` is itself
  a valid shortest witness of depth two.
- **Safe substitution:** For reached membership, distance, future expansion,
  and one arbitrary replayable path, the loser may publish state `t` with
  parent `b`. The mathematical result is valid though the original winner is
  not reproduced.
- **Insufficient substitution:** One record from `b` cannot prove canonical
  minimum, restore the lost all-parent contribution from `a`, recover the path
  count, or reproduce the exact physical execution.
- **New intuition:** Helpability is output-relative semantic replacement, not
  necessarily byte recovery. A live frontier can be repaired with less
  information than a complete rich output, so the descriptor must declare
  which substitutions close which obligations.
- Added the substitution boundary to note 178. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — when is the endpoint not the whole state?

- **Question:** May implicit BFS merge two records with the same Cayley group
  element when legal successors depend on the last generator?
- **Hand trace:** In `Z_2 x Z_2`, commuting involutions satisfy `ab=ba`.
  Under semantic rule “do not repeat the previous generator,” words `ab` and
  `ba` reach the same group element but product states `(ab,last=b)` and
  `(ab,last=a)`.
- **Different futures:** The first permits `a` and reaches `(b,last=a)`; the
  second permits `b` and reaches `(a,last=b)`. Merging by base element erases a
  residual continuation language.
- **Target witness:** Product target `(a,last=b)` is reached by legal word
  `bab`; the earlier base visit `(a,last=a)` does not dominate it.
- **Constraint/pruning boundary:** For ordinary unconstrained vertex distance,
  forbidding repeated involutions only removes `aa`/`bb` spurs and can be safe
  pruning. When last label affects legality or acceptance, it is part of exact
  vertex identity and the search graph is the product `(g,last)`.
- **New intuition:** Whether history belongs in visited is decided by future
  equivalence, not by record shape. Equal visible endpoints may merge exactly
  only when they have the same relevant continuation language.
- Added the Cayley product-state trace to note 20. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — is one-step future equality enough to merge?

- **Question:** If two history/memory states look identical now and through one
  successor step, may visited merge them safely?
- **Counterexample:** Nonaccepting DFA states `p` and `q` both need two symbols
  to acceptance. On `a` they enter nonaccepting `r` and `u`; on `b` both enter
  a rejecting sink. From `r`, `a` accepts; from `u`, `b` accepts.
- **Delayed divergence:** Immediate acceptance profiles and nearest-goal
  distances agree, but `aa` is accepted only from `p` and `ab` only from `q`.
- **Correction:** Equality of current observations, degree, one-step visible
  successors, or scalar distance-to-goal is not the merge criterion. Matching
  must continue recursively through every suffix relevant to acceptance.
- **New intuition:** Exact history compression is a statement about residual
  continuation languages, not a finite visual resemblance of the next layer.
  Myhill--Nerode equivalence is precisely equality of those futures for a DFA.
- Added the delayed-divergence counterexample to note 129. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-28: question card — does behavioral merging preserve path counts?

- **Question:** If two states are bisimilar and can be merged without changing
  goal reachability or distance, is the number of shortest paths also exact?
- **Diamond witness:** In `s->p->t` and `s->q->t`, states `p` and `q` have the
  same observation and identical future transition to goal `t`, so strong
  bisimulation may merge them into one class `C`.
- **Result split:** Original and quotient both have distance two to `t`. The
  original has two shortest vertex paths; the support quotient
  `s->C->t` has one.
- **Multiplicity boundary:** Preserving two parallel occurrence identities into
  `C` could retain this count, but that is additional quantitative structure,
  not a consequence of existence-based bisimulation.
- **DFA distinction:** Deterministic minimization preserves accepted input words
  because each word has one run. General graph path multiplicity can count
  several concrete runs of the same trace and is not preserved automatically.
- **New intuition:** “Same possible future” is enough for an existential BFS
  question, not for counting how many histories realize that future.
- Added the diamond witness to note 128. Added no new thematic note, code,
  source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — does one Schreier BFS give diameter?

- **Question:** If a group acts transitively on puzzle states, does every state
  have the same BFS eccentricity under a fixed generator set?
- **Three-point witness:** Let `S_3` act on `{1,2,3}` with generators
  `{(12),(23)}`. The successors form support path `1--2--3`, with a loop at
  point `1` from `(23)` and at point `3` from `(12)`.
- **BFS result:** From middle point `2`, all states are reached at depth at most
  one, so `ecc(2)=1`. Endpoints `1` and `3` are distance two apart, hence the
  graph diameter is two.
- **Correction:** Transitivity of the abstract action does not imply that the
  fixed-generator Schreier graph is vertex-transitive. Moving the root can
  conjugate generators outside the declared move set.
- **Cayley contrast:** In the regular Cayley action, left multiplication maps
  every right edge `g->g*s` to `ag->ag*s` with the same generator. That actual
  graph automorphism makes all eccentricities equal and one exhaustive BFS
  sufficient for diameter.
- **New intuition:** The last BFS layer is always source eccentricity. Calling
  it diameter requires symmetry of the precise edge metric, not merely a
  transitive group acting somewhere in the model.
- Added the three-point diameter witness to note 21. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-28: question card — can equal total work hide different waves?

- **Question:** On one fixed Schreier graph, does equal complete generator work
  imply similar BFS frontier behavior from different roots?
- **Middle root:** In the three-point `S_3` fixture, root `2` gives frontier
  sizes `1,2` and generated-occurrence batches `2,4`.
- **Endpoint root:** Root `1` gives frontier sizes `1,1,1` and batches `2,2,2`.
- **Conservation:** Both runs expand all three states, generate six labeled
  occurrences, accept two nonroot states, and reject four occurrences.
- **Different composition:** The middle-root run places all outward discoveries
  first and finishes with inward returns plus self-loops. The endpoint-root run
  spreads outward, inward, and loop occurrences across three levels.
- **New intuition:** Total work is an integral over the traversal; frontier
  shape is its time profile. Equal totals can have different depth, peak
  parallel width, batch sizes, rejection mix, and synchronization opportunities.
  No hardware speed relation follows without measurement.
- Added the root-profile trace to note 21. Added no new thematic note, code,
  source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — is a Cayley wave root-dependent?

- **Question:** Can changing the source in one fixed Cayley graph alter the
  semantic frontier profile as it did in the three-point Schreier graph?
- **Translation proof:** For right edges `g->g*s`, left multiplication by
  `r^(-1)` sends root `r` to identity and preserves the same label `s` on every
  transition. Therefore `F_d(r)=r F_d(e)`.
- **Preserved wave:** Every root has equal frontier sizes, depth/eccentricity,
  labeled occurrence totals, radial rejection classes, and translated shortest
  structures at every level.
- **Directed boundary:** The same holds for positive-generator directed Cayley
  transitions and their translated reachable components; inverse closure is
  not needed for the automorphism itself.
- **Physical boundary:** Rank encodings, hashes, owner maps, partitions, and
  memory layouts need not commute with group translation. Per-rank balance,
  locality, and traffic can differ even when the global semantic wave is an
  exact translate.
- **New intuition:** Genuine Cayley root symmetry freezes BFS geometry, not the
  accidental placement of state records on hardware.
- Added the root-translation boundary to note 21. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can multi-source BFS compute diameter sooner?

- **Question:** Can one merge several roots into a wider initial frontier and
  use the shallower resulting BFS to obtain the same diameter faster?
- **Single-source trace:** In Cayley `Z_4` with `{+1,-1}`, root zero gives
  layers `{0}`, `{1,3}`, `{2}` and maximum depth two, equal to the diameter.
- **Joint-source trace:** Sources `{0,2}` give layers `{0,2}`, `{1,3}` and
  maximum depth one while the graph diameter remains two.
- **Correction:** The joint result is `min` distance to the source set; its
  maximum is that set's covering radius, not an all-pairs maximum or diameter.
- **Execution/semantic boundary:** A wider initial wave and fewer levels may
  look computationally attractive, but they describe a different output.
  Sharing machinery among independent BFS runs is compatible only if source
  identity remains a separate state/output dimension.
- **New intuition:** More sources make the nearest-source landscape flatter;
  they do not reveal long distances between the sources or other vertices.
- Added the `Z_4` counterexample to note 21. Added no new thematic note, code,
  source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — is a same-layer edge useless for shortest paths?

- **Question:** If an edge stays inside one BFS layer, is it absent from all
  shortest paths, or only from shortest paths rooted at the current source?
- **Triangle trace:** In `s--a--b--s`, BFS from `s` gives
  `F_0={s}, F_1={a,b}`. The edge `a--b` is therefore same-layer.
- **Root-relative exclusion:** Taking `a--b` after reaching either endpoint
  gives a length-two route from `s` to a vertex whose distance is already one,
  so the edge is absent from the shortest-path DAG rooted at `s`.
- **Global counterexample:** For the pair `(a,b)`, that same edge is the unique
  shortest path of length one. From root `a`, it is radial and tree-eligible.
- **New intuition:** `same-layer` describes an edge's radial role relative to
  one distance function. It does not declare a source-independent property of
  the edge or make it globally irrelevant to graph geodesics.
- Added the triangle clarification to note 156. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — do directed BFS layers certify bipartiteness?

- **Question:** If a directed BFS sees no arc whose endpoints have equal
  depth, must the graph obtained by forgetting arc directions be bipartite?
- **Prediction transferred from undirected BFS:** One might try to color every
  vertex by depth parity and interpret the absence of equal-depth arcs as the
  absence of parity conflicts.
- **Counterexample:** In `s->a->b->s`, directed BFS from `s` assigns depths
  `0,1,2`. No arc stays inside one layer, but the underlying undirected graph is
  a triangle and is not bipartite. Arc `b->s` connects equal parities across a
  depth gap of two.
- **Correction:** An undirected edge constrains depths in both directions and
  hence gives `|d(u)-d(v)|<=1`. A directed arc only constrains forward progress
  by `d(v)<=d(u)+1`; its backward depth gap is unbounded.
- **New intuition:** The undirected same-layer bipartiteness test is powered by
  a symmetric metric inequality, not merely by BFS depth labels. Directed
  layers cannot inherit that certificate after directions are erased.
- Added the directed three-cycle counterexample to note 31. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — must Cayley parity check every derived relation?

- **Question:** To prove that generator-word parity is well-defined on group
  elements, must one inspect all consequences of a presentation, or are its
  defining relators enough?
- **Prediction:** Even defining relators might conceivably combine into an odd
  derived relation, making a local presentation check too weak.
- **Free-group argument:** Sending every generator to `1 in Z_2` is a
  homomorphism on the free group. If every defining relator maps to zero, then
  so do every inverse, conjugate, and product of those relators. Their normal
  closure stays inside the kernel, so parity descends to the presented group.
- **Minimal contrast:** `<a | a^3=e>` for `Z_3` fails immediately, while
  `<a | a^4=e>` for `Z_4` supports the parity character.
- **Alphabet boundary:** Adding a generator is not harmless. If `b=a^2`, the
  new relator `b a^-2` has odd length when both symbols count as one move, so
  word parity fails for the enlarged Cayley alphabet.
- **Correction:** A complete presentation's defining relators are sufficient;
  an incomplete relation sample is not. The proof is closure of a homomorphism
  kernel, not an empirical search for short odd relations.
- Added the presentation-kernel argument to note 16. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — how does odd stabilizer parity fail in a Schreier graph?

- **Question:** When a Schreier stabilizer contains an odd group element, must
  the resulting state graph be nonbipartite under every graph convention?
- **Existing fixture:** Let `S_3` act on `{1,2,3}` with generators
  `{(12),(23)}`. At state `1`, both `e` and odd `(23)` represent the same point,
  so group sign is not a well-defined state color.
- **Labeled consequence:** `(23)` fixes `1` and yields a labeled loop; `(12)`
  similarly fixes `3`. The loop-retaining Schreier graph is not bipartite.
- **Simple-support contrast:** Suppressing those loops leaves the path
  `1--2--3`, which is bipartite. Its coloring exists on the simplified support
  but is not inherited from permutation sign on group representatives.
- **New intuition:** Stabilizer compatibility answers whether a particular
  group character descends to states. Bipartiteness answers a predicate about
  a precisely declared graph representation. Removing loops can make the
  latter true without repairing the former.
- Added this parity-contract distinction to the existing three-point witness
  in note 21. Added no new thematic note, code, source audit, Docker action,
  benchmark, or GPU implementation.

## 2026-08-29: question card — when is visited really a metric ball?

- **Question:** Is it enough that every vertex currently in `visited` has a
  correct depth, or must all vertices through that depth already be present
  before its external boundary can be called the next BFS layer?
- **Counterexample:** Use edges `s--a`, `s--b`, and `a--x`, but take the
  incomplete depth-one set `B'_1={s,a}`. Its stored labels are individually
  correct while true depth-one vertex `b` is absent.
- **Mixed boundary:** `N(B'_1)\B'_1={b,x}` contains true depths one and two.
  Calling it one frontier conflates an overdue vertex with a next-layer vertex.
- **Partial-frontier failure:** Expanding only `{a}` produces `{x}` and misses
  `b`, so the current-frontier form does not rescue an incomplete prior layer.
- **Correction:** `N(B_d)\B_d=N(F_d)\B_d=F_(d+1)` requires the completeness
  invariant `B_d={v:dist(s,v)<=d}`, not merely sound labels on stored records.
- **New intuition:** Layer closure is the moment a discovered region becomes a
  genuine metric ball. Before closure, its boundary is a scheduling boundary,
  not a single distance sphere.
- Added the incomplete-ball counterexample to note 10. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can an incomplete-ball boundary be an exact frontier?

- **Question:** Although the boundary of incomplete single-source `visited`
  mixes old source depths, can it still be an exact BFS frontier under another
  source contract?
- **Reinterpretation:** In the graph `s--a, s--b, a--x`, declare
  `A={s,a}` itself to be the multi-source layer zero. Then
  `N(A)\A={b,x}` is exactly `F_1(A)` because both vertices are one hop from the
  new source set.
- **Old-metric conflict:** Relative to original source `s`, `b` has depth one
  and `x` depth two, while `a` is not depth zero. The same boundary cannot be
  used as the next layer of that original run.
- **New intuition:** A set recurrence does not carry its metric meaning by
  itself. Reclassifying accumulated visited states as simultaneous sources
  makes the recurrence exact by changing the distance field, not by completing
  the interrupted BFS.
- Added the source-contract reinterpretation beside the incomplete-ball
  counterexample in note 10. Added no new thematic note, code, source audit,
  Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can correct stored depths simply reseed a FIFO?

- **Question:** Can an interrupted BFS preserve correct labels on several old
  depths, sort those records once, and safely resume irrevocable first-claim
  traversal with one ordinary FIFO?
- **Fixture:** Use edges `s--p--a--v` and `s--b--v`. True labels include
  `D(s)=0`, `D(a)=2`, and `D(v)=2`. Seed the FIFO as `[s:0,a:2]`.
- **Trace:** Expanding `s` appends `p:1,b:1` behind the old depth-two record,
  producing `[a:2,p:1,b:1]`. Then `a` can claim `v` at three before `b`
  proposes its true distance two.
- **Failure:** A visited bit that makes the first claim final rejects the later
  improvement, although every initial seed label was individually correct and
  the initial seed list was sorted.
- **Correction:** FIFO shortestness depends on a continuously maintained
  nondecreasing-distance queue history, not merely a one-time sort. Recovery
  must reconstruct that ordering/closure or permit corrective relaxation and
  reactivation.
- **New intuition:** Correct values are not a complete executable BFS state.
  Pending-work order and remaining adjacency work are part of the proof that
  first discovery is final.
- Added the multi-depth reseeding counterexample to note 18. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — when do exact multi-depth seeds recover the old metric?

- **Question:** If FIFO reseeding is unsafe, under what precise conditions can
  exact stored labels still reconstruct the original single-source distances?
- **Offset field:** For seeds `A` with old exact labels, global min-key
  propagation computes `H(v)=min_(a in A)[delta(s,a)+dist(a,v)]`.
- **Lower bound:** Every term is the length of a real route through `a`, so
  `delta(s,v)<=H(v)`.
- **Matching upper bound:** If the original source `s` belongs to `A`, its term
  is `0+dist(s,v)=delta(s,v)`, hence `H(v)<=delta(s,v)` and equality follows.
- **Boundary:** The argument also needs every relevant seed/work item expanded
  and a nondecreasing settlement discipline or corrective relaxation. Exact
  labels without `s` need not preserve the old field on routes not captured by
  the remaining seeds.
- **New intuition:** Old depths can serve as additive source offsets. They are
  sufficient when their lower envelope is proved equal to the old metric; they
  are not a license to mix depth buckets in an ordinary FIFO.
- Added the offset-seed recovery theorem to note 18. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — why is the last frontier sufficient to continue BFS?

- **Question:** If an arbitrary exact seed subset may fail without the source,
  why may ordinary BFS continue from `F_d` without re-expanding `s` or earlier
  layers?
- **Shortest-prefix witness:** Every shortest path to a vertex `v` of depth
  `k>=d` passes at step `d` through some `a in F_d`; its remaining suffix has
  length `k-d`.
- **Offset identity:** Therefore
  `delta(s,v)=d+min_(a in F_d)dist(a,v)` for the exterior. Triangle inequality
  gives one direction and the shortest-path prefix gives the other.
- **Visited boundary:** `F_d` alone does not say which inward destinations are
  already closed. Without `B_d`, reverse or return edges can reintroduce old
  layers as apparently new work.
- **New intuition:** A completed frontier is simultaneously a metric sphere,
  the pending work boundary, and an exact shortest-path cut for the exterior.
  The visited ball supplies the complementary exclusion certificate.
- Added the frontier offset-cut identity to note 18. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — when does a frontier subset preserve one target distance?

- **Question:** If a retained subset of `F_d` can still reach target `t`, is
  that enough to preserve `dist(s,t)`?
- **Counterexample:** In `s->a->t` and `s->b->c->t`, `F_1={a,b}`. Subset
  `{b}` reaches `t`, but its offset route has length three while the true
  distance through `a` is two.
- **Positive half:** Subset `{a}` preserves the scalar distance because it lies
  on one shortest path. It is nevertheless not a separator or dominator,
  because the longer `s->b->c->t` path avoids it.
- **Exact condition:** For `P subseteq F_d`, target distance survives iff
  `min_(p in P)dist(p,t)=dist(s,t)-d`, equivalently some shortest path crosses
  `P` at depth `d`.
- **New intuition:** Four contracts form a strict ladder: some continuation,
  one shortest continuation, every shortest continuation, and every
  continuation. Reachability, scalar distance, shortest-path structure, and
  dominance stop at different rungs.
- Added the partial-frontier distance criterion to note 89. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — how do shortest-path counts cross a BFS layer?

- **Question:** Why may one retained frontier route preserve target distance
  while failing to preserve the number of shortest paths?
- **Unique cut:** Every shortest `s`-to-`t` path crosses an intermediate layer
  `F_j` at exactly one vertex `a` and splits uniquely into a shortest prefix
  and a shortest suffix.
- **Factorization:** With `tau(a,t)` equal to the number of length-`k-j`
  shortest suffixes, `sigma_s(t)=sum_(a in F_j)sigma_s(a)*tau(a,t)` for
  `k=dist(s,t)`.
- **Diamond calibration:** At depth one, the two diamond branches contribute
  one each. Keeping either branch preserves distance two but reduces the count
  from two to one.
- **Output boundary:** Scalar distance needs one nonzero contribution. Exact
  multiplicity needs every contribution or an equivalent aggregate; labeled
  graph semantics must count edge labels inside both factors.
- **New intuition:** A frontier is not only a set cut. For rich outputs it is
  an information cut whose per-state prefix mass combines with suffix mass.
- Added the intermediate-layer count factorization to note 11. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — does frontier sigma replace the old shortest DAG?

- **Question:** At a completed layer, are per-state shortest-prefix counts
  sufficient to discard all earlier predecessor structure?
- **Count-only answer:** For later scalar path counts, yes under the fixed path
  identity convention: `sigma(a)` is the total prefix mass that every future
  shortest suffix must multiply, so the old DAG need not be traversed again.
- **Information loss:** `sigma(a)=2` does not identify the two parents, labels,
  or vertex sequences. The same scalar cannot reconstruct, enumerate, choose a
  canonical member of, or backward-sample those prefixes by itself.
- **Qualification:** Richer outputs remain possible only if the old immutable
  graph and sufficient labels can regenerate the missing structure, or another
  equivalent summary was retained.
- **New intuition:** A frontier record can be a sufficient statistic for one
  continuation algebra and an irreversible projection for another. Checkpoint
  completeness is output-relative.
- Added the count-only sufficiency boundary to note 11. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — when can path counts collapse to Boolean support?

- **Question:** Can shortest-prefix counts be replaced by a Boolean
  exists/absent value while preserving exact reachability continuation?
- **Homomorphism:** For exact nonnegative counts, `phi(0)=false` and
  `phi(n>0)=true` satisfies `phi(x+y)=phi(x) OR phi(y)` and
  `phi(x*y)=phi(x) AND phi(y)`. Count composition therefore projects exactly
  to existence composition.
- **Irreversibility:** Boolean support cannot recover multiplicity; all
  positive counts collapse to the same value.
- **Modulo counterexample:** The diamond has two shortest paths, so modulo two
  its residue is zero. Interpreting modular nonzero as support would call a
  reachable target absent.
- **BFS qualification:** Positive walk support must still be masked by prior
  visited/distance layers to identify first arrival rather than an old vertex
  reached again by a longer walk.
- **New intuition:** Distance/reachability is a lawful projection of exact
  nonnegative path algebra, not merely a count with fewer bits. Other numeric
  quotients may preserve their own output while destroying support.
- Added the support-homomorphism boundary to note 25. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — does min-plus distance project exactly to reachability?

- **Question:** Is the predicate "distance is finite" algebraically compatible
  with shortest-path composition, or only a post-hoc loss of information?
- **Projection:** Set `psi(infinity)=false` and `psi(x)=true` for finite
  nonnegative `x`. Then `psi(min(x,y))=psi(x) OR psi(y)` and
  `psi(x+y)=psi(x) AND psi(y)`.
- **Meaning:** Alternative routes use `min/OR`; concatenated route pieces use
  `plus/AND`. Finiteness therefore preserves exact reachability through the
  min-plus recurrence.
- **Information loss:** A two-edge path and a direct edge both project their
  target to `true` despite distances two and one. Reachability cannot recover
  layers, eccentricity, or which arrival was shortest.
- **New intuition:** Boolean reachability is a lawful quotient of both exact
  count support and finite shortest-distance support. Lawful projection does
  not mean reversible encoding.
- Added the min-plus support projection to note 25. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can distance and shortest count share one algebra?

- **Question:** Can one local merge rule explain why shorter, equal, and longer
  path candidates affect BFS distance/count output differently?
- **Pair state:** Represent a path family by `(d,c)`: minimum length and number
  of paths attaining it, with `(infinity,0)` for absence.
- **Alternative merge:** Keep the pair with smaller `d`; on equal `d`, retain
  that distance and add counts. Thus shorter replaces, equal contributes, and
  longer disappears.
- **Concatenation:** `(d1,c1)*(d2,c2)=(d1+d2,c1*c2)` because every shortest
  prefix can combine with every shortest suffix.
- **Diamond trace:** Two length-two branches merge as
  `(2,1)+(2,1)=(2,2)`; a length-three alternative cannot change either field.
- **Retry boundary:** Addition counts distinct semantic alternatives, not two
  deliveries of the same predecessor-edge contribution. Physical duplicates
  require identity/dedup semantics before this merge.
- **New intuition:** Distance and multiplicity are not two unrelated passes;
  they are two coordinates of one shortest-family summary with tie-sensitive
  addition.
- Added the distance-count pair algebra to note 11. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — may distance-count candidates reduce hierarchically?

- **Question:** Does reducing candidate `(distance,count)` pairs locally and
  then globally give the same result as one global merge?
- **Reason:** Pair alternative merge is associative and commutative. A
  candidate longer than its local minimum cannot equal the global minimum,
  which is no larger than every local minimum.
- **Count rule:** A partition's local count survives exactly when its local
  minimum equals the global minimum; surviving counts add across partitions.
- **Trace:** Local results `(2,1)` and `(2,2)` merge to `(2,3)`, while local
  longer candidates disappear exactly as they would in a flat reduction.
- **Boundary:** The theorem needs complete candidate coverage for one
  target/epoch and distinct semantic contributions. Cross-partition retries of
  one contribution remain non-idempotent and overcount.
- **New intuition:** Hierarchical grouping may change where a reduction occurs
  without changing its mathematical answer. It cannot repair missing inputs or
  invent exactly-once semantics.
- Added the local-global reduction proof to note 11. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — may validity checking follow local minimum reduction?

- **Question:** Can candidates reduce locally by `(distance,count)` first and
  be checked against the graph/epoch only after the losing records are gone?
- **Counterexample:** A local group contains invalid `(1,1)` and valid `(2,1)`.
  Minimum reduction retains the first and discards the second; later removing
  the invalid winner yields absence instead of the correct `(2,1)`.
- **Non-commutation:** `filter(reduce(C))` need not equal
  `reduce(filter(C))`. A shorter invalid witness can suppress the best valid
  witness before validation observes it.
- **Boundary:** Hierarchical pair reduction is exact only over already valid
  contributions for the same graph/source epoch, unless the retained summary
  also contains fallback candidates.
- **New intuition:** Associativity permits regrouping valid evidence; it does
  not permit semantic validation to move arbitrarily across a lossy reduction.
- Added the filter-reduction counterexample to note 11. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — does one valid path certify an aggregate distance-count pair?

- **Question:** After equal-distance contributions are aggregated, does
  replaying one real shortest path validate both reported distance and count?
- **Counterexample:** Valid `(2,1)` and invalid `(2,1)` merge to reported
  `(2,2)`. The real path proves that length two is attainable, but supplies no
  evidence for the second counted alternative.
- **Coordinate split:** A valid summand can make the distance coordinate look
  sound while an invalid equal-distance summand silently inflates multiplicity.
- **Validation boundary:** Exact count needs validated complete contributions,
  sufficient provenance, or independent recomputation. One replayable path and
  scalar range checks certify neither completeness nor distinctness.
- **New intuition:** Pair aggregation combines coordinates with different
  evidence needs. A witness for the selected minimum is not automatically a
  witness for the mass accumulated at that minimum.
- Added the aggregate-witness counterexample to note 11. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can local recurrences certify every shortest-path count?

- **Question:** Once distance labels are exact, can locally checking the count
  recurrence certify all `sigma(v)`, or may wrong counts form a self-consistent
  global solution?
- **Acyclic induction:** Every shortest predecessor of depth-`d` vertex lies at
  depth `d-1`. With `sigma(s)=1`, the full incoming-edge sums uniquely determine
  counts layer by layer, so cyclic self-support is impossible.
- **Completeness counterexample:** In the diamond, a reported parent list that
  omits `b->t` supports the self-consistent report `sigma(t)=1`; scanning the
  complete graph predecessor set yields the correct two.
- **Boundary:** The validator must enumerate every eligible edge under the
  declared vertex/labeled-path identity and use the declared exact, modular, or
  saturated arithmetic semantics.
- **New intuition:** Count recurrence is locally checkable because distance
  gives a well-founded order. The difficult proof obligation is not solving the
  recurrence but establishing that its input predecessor set is complete.
- Added the count-certificate induction to note 41. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — does one identity-rooted Cayley BFS determine pairwise path counts?

- **Question:** Under right Cayley edges `g->g*s`, does identity-rooted BFS
  determine only pairwise distances or also shortest labeled-path counts?
- **Bijection:** A label word takes `x` to `y` iff its product is `x^-1*y`.
  Left translation by `x^-1` preserves every edge label, giving a one-to-one
  correspondence with identity-to-`x^-1*y` paths.
- **Count formula:** Consequently
  `sigma_x(y)=sigma_e(x^-1*y)` under the same path identity convention; the
  correspondence holds for directed positive alphabets without inverse
  closure.
- **Convention boundary:** For left edges `g->s*g`, the label-preserving
  normalization is instead to `y*x^-1`. These differences need not agree in a
  noncommutative group.
- **New intuition:** Cayley translation freezes not only wave geometry but the
  complete labeled geodesic fiber between translated endpoints. Schreier
  quotienting can destroy this single-difference normalization.
- Added the path-count translation formula to note 16. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — what does a Schreier shortest-path count target?

- **Question:** For start `x0*a` and goal state `x0*b`, may shortest labeled
  paths be counted by choosing one group representative of the goal?
- **Set formula:** If `D` is the minimum word length in `a^-1*H*b`, the labeled
  count is the number of length-`D` label words lying anywhere in that whole
  coset-like target set.
- **Minimal witness:** In additive `Z_4` with `H={0,2}` and moves `{+1,-1}`,
  goal state `H+1={1,3}` has two one-move labeled paths: `+1` reaches `1` and
  `-1` reaches `3`.
- **Representative failure:** Searching only for representative `1` preserves
  distance one in this example but undercounts multiplicity as one instead of
  two.
- **Output boundary:** Collapsing the two labels to one simple-support edge can
  make the vertex-path count one. Labeled-word and simple vertex-path counts
  are different declared objects.
- **New intuition:** A Schreier state is a fiber. Distance asks when the wave
  first touches the fiber; labeled multiplicity asks how many minimal words
  touch all of its representatives collectively.
- Added the Schreier target-set count formula to note 16. Added no new thematic
  note, code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can Cayley cover counts sum to a Schreier count?

- **Question:** Can identity-rooted Cayley results compute the labeled shortest
  count for a Schreier query by summing over its target fiber?
- **Nearest-fiber formula:** Set
  `D=min_(h in H)ell(a^-1*h*b)`. Then the Schreier labeled count is the sum of
  `sigma_C(e,a^-1*h*b)` over exactly those `h` attaining `D`.
- **Why disjoint:** Every label word evaluates to one group element, so nearest
  representatives partition the minimal solution words without overlap.
- **Z4 calibration:** Target representatives `1` and `3` each receive one
  one-letter Cayley word, and their sum gives the Schreier labeled count two.
- **Output boundary:** The formula counts labeled words. If generator aliases
  are collapsed into simple-support edges, several summands may represent one
  vertex-sequence path and cannot be summed under that different convention.
- **New intuition:** Cayley BFS resolves a Schreier fiber into group endpoints;
  distance takes their minimum depth, while labeled multiplicity sums the mass
  on all endpoints tied at that minimum.
- Added the Cayley-cover sum formula to note 16. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — how do support paths lift to labeled Schreier paths?

- **Question:** Can labeled shortest multiplicity be reconstructed from the
  simple state-support graph plus per-edge generator alias counts?
- **Local multiplicity:** Define `m(u,v)` as the number of declared generator
  labels taking state `u` to state `v`.
- **Path lift:** A fixed support path `(v_0,...,v_k)` has
  `product_i m(v_i,v_(i+1))` labeled lifts because each step's realizing label
  can be chosen independently.
- **Global formula:** Sum that product over all shortest support vertex paths to
  obtain the labeled shortest count.
- **Z4 calibration:** The quotient witness has one one-edge support path with
  edge multiplicity two, hence one vertex path and two labeled paths.
- **Boundary:** `m` counts semantic label occurrences, not delivery retries.
  Positive-cost loops cannot occur on a shortest path between distinct states,
  although they remain generated transition work.
- **New intuition:** Collapsing labels changes an unweighted labeled graph into
  a support graph carrying integer edge weights for multiplicity. Distance uses
  support; labeled count uses products and sums of those multiplicities.
- Added the support-path lift formula to note 16. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.

## 2026-08-29: question card — can support-edge multiplicity replace parallel labels for counting?

- **Question:** For labeled shortest-path count only, may parallel generator
  edges with one endpoint pair be replaced by their integer multiplicity?
- **Weighted recurrence:** On the support shortest DAG,
  `sigma(v)=sum_u sigma(u)*m(u,v)` over predecessors one level earlier.
- **Reason:** Each shortest prefix to `u` independently extends through each of
  the `m(u,v)` labels realizing `u->v`, yielding the product contribution.
- **Preserved output:** The recurrence produces the same labeled scalar count
  as explicit parallel-edge enumeration.
- **Lost output:** Multiplicity alone cannot identify moves, enumerate words,
  choose a canonical label, or preserve label-specific metadata.
- **New intuition:** Parallel labels can collapse from separate edges to an
  integer coefficient only after the requested output has collapsed their
  identities to addition.
- Added the multiplicity-weighted count recurrence to note 16. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — does constant labeled outdegree determine labeled path counts?

- **Question:** If every Schreier state emits exactly `|S|` generator
  occurrences, is that total enough to infer labeled shortest multiplicity to
  an adjacent target?
- **S3 witness:** With the existing three-point generator set, state `1` has
  two labels reaching state `2`, while state `2` has one label reaching state
  `1`. Both sources emit three total occurrences and both distances are one.
- **Count contrast:** `sigma_1(2)=2` but `sigma_2(1)=1` under labeled-path
  semantics.
- **Conservation only:** `sum_v m(u,v)=|S|` fixes total outgoing label mass but
  not its distribution among endpoints or shortest-DAG edges.
- **New intuition:** Constant generation work can coexist with different
  semantic path mass. Labeled outdegree is a budget; endpoint multiplicities
  decide where that budget contributes.
- Added the outdegree-versus-count distinction to note 158. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — do multiplicity histograms determine the next path count?

- **Question:** If two layers have the same prefix-count histogram and the same
  support-edge multiplicity histogram, must they produce the same next count?
- **Fixture:** Let `sigma(a)=2`, `sigma(b)=1`, and let the two edges into target
  `t` have multiplicities `{2,1}` in both instances.
- **Aligned instance:** Pairing larger values gives
  `sigma(t)=2*2+1*1=5`.
- **Crossed instance:** Swapping which parent owns which edge multiplicity gives
  `sigma(t)=2*1+1*2=4`.
- **Preserved marginals:** Support diamond, frontier sizes, total outward
  multiplicity, and both separate histograms can remain identical.
- **New intuition:** The weighted recurrence depends on joint parent-endpoint
  alignment. Aggregate marginals erase the correlation needed for the dot
  product and therefore cannot certify semantic path mass.
- Added the same-marginals count counterexample to note 16. Added no new
  thematic note, code, source audit, Docker action, benchmark, or GPU
  implementation.

## 2026-08-29: question card — what count bounds survive after parent-edge pairing is lost?

- **Question:** If only the prefix-mass and edge-multiplicity multisets remain,
  can they bound the missing weighted count even though they do not determine
  it?
- **Rearrangement bounds:** For sorted nonnegative sequences
  `x_1<=...<=x_n` and `y_1<=...<=y_n`, every pairing lies between the
  opposite-order dot product and the same-order dot product.
- **Swap proof:** Replacing a crossed pairing by an aligned one changes the sum
  by `(x_j-x_i)*(y_q-y_p)>=0`, so repeated swaps reach the extrema.
- **Calibration:** Marginals `{1,2}` and `{1,2}` give exact pairing envelope
  `[1*2+2*1, 1*1+2*2]=[4,5]`.
- **Graph boundary:** The interval is sharp over abstract pairings. Stabilizer,
  topology, or generator constraints may forbid some pairings, so it need not
  be a sharp attainable interval for one fixed Schreier graph.
- **New intuition:** Losing correlation need not erase all information:
  marginals retain an extremal envelope while losing the exact semantic count.
- Added the rearrangement envelope to note 16. Added no new thematic note,
  code, source audit, Docker action, benchmark, or GPU implementation.
## 2026-08-29: question card — when is a rolling visited window exact?

- **Question:** Is `beta<=L` merely sufficient, or also necessary for exact
  duplicate rejection by an `L`-backward-layer window?
- **Interface:** Every outgoing edge is enumerated; oldness is decided only by
  membership in retained BFS layers; every other endpoint enters next-layer
  reconciliation.
- **Necessity:** If an edge `(u,v)` has
  `dist(s,u)-dist(s,v)>L`, then `v` has already fallen outside the retained
  window when `u` is expanded and is necessarily reaccepted as apparently new.
- **Equivalence:** For this fixed rooted graph and this pure interface, the
  rolling filter is exact iff `beta<=L`.
- **Boundary:** Alternative certificates can still forget layers safely by
  retaining the missing distinction in operator masks, topology, or another
  exact summary.
- **Quantifiers:** Post-hoc `beta` characterizes one rooted instance; a window
  chosen before traversal requires an independent family-level bound. Samples
  do not establish that bound.
- Added the necessity and quantifier distinction to note 181. Added no code,
  test, benchmark, Docker action, or new thematic note.
## 2026-08-29: question card — is the Cayley inverse-length window tight?

- **Question:** Is `max_g d_S(e,g^(-1))` only an upper bound on backward radial
  span in a directed Cayley graph?
- **Witness:** For each generator `g`, choose `x=g^(-1)`. The edge
  `g^(-1)->e` has radial drop exactly `d_S(e,g^(-1))`.
- **Identity:** For a directed right Cayley graph rooted at identity,
  `beta=max_g d_S(e,g^(-1))`; this is the minimum exact window length for the
  pure retained-layer interface.
- **Schreier contraction:** `Z_6` with `S={+1}` has `beta=5`, while its action on
  cosets of `{0,3}` is a directed three-cycle with `beta=2`. The five-step
  group inverse remains safe but is no longer tight at state level.
- **New distinction:** Full Cayley vertices preserve the inverse witness;
  quotient/action states can identify it with a closer representative.
- **Open question:** Find a structural action-level certificate that is exact
  before traversal when stabilizers are non-normal.
- Added the exact Cayley identity and Schreier counterexample to note 181.
  Added no code, test, benchmark, Docker action, or new thematic note.
## 2026-08-29: question card — what lies between Schreier beta and group inverses?

- **Question:** What action-level certificate can sharpen the group inverse
  bound without first computing all root distances?
- **Local return:** For edge `x->xs`, define
  `rho(x,s)=dist_S(xs,x)` and `R_action=max_(x,s) rho(x,s)`.
- **Hierarchy:** Directed triangle inequality and the universal inverse word
  give `beta_root<=R_action<=L_group`.
- **Stabilizer form:** At `x=x_0 a`, with `K=a^(-1)Ha`,
  `rho(x,s)=min{|w|_S : w in s^(-1)K}`. Local reversibility can therefore vary
  across conjugate stabilizers.
- **First strictness witness:** The `Z_6/{0,3}` directed action has
  `beta_root=R_action=2<L_group=5`.
- **Second strictness witness:** In note 158's three-point `S_3` action rooted at
  `1`, all states have depth at most one, so `beta_root=1`; edge `2->3` requires
  the two-step return `3->1->2`, hence `R_action=L_group=2`.
- **New intuition:** Return length bounds how far an edge could fall relative to
  any arrival history; radial span measures how far it actually falls relative
  to one root. Those are different geometries.
- Added the certificate hierarchy and two strictness witnesses to note 181.
  Added no code, test, benchmark, Docker action, or new thematic note.
## 2026-08-29: question card — when does local return equal radial span?

- **Question:** When is `R_action` the exact rolling-window length rather than
  only a root-independent upper bound?
- **Theorem:** In a strongly connected vertex-transitive directed support
  graph, `beta_root=R_action` for every root.
- **Witness transfer:** Map the endpoint `v` of an edge `u->v` attaining
  `dist(v,u)=R_action` to the root. The image edge falls into the root from
  depth exactly `R_action`.
- **Normal quotient:** If the Schreier stabilizer is normal, the orbit graph is
  a quotient Cayley graph and the exact value is the maximum inverse distance
  of the generator images in that quotient.
- **Conjugation condition:** A conjugation-invariant move set makes the action
  maps support-graph automorphisms, possibly permuting labels, and therefore
  also makes the local-return certificate exact for scalar BFS.
- **Rejected implication:** A transitive group action alone does not make the
  graph for one fixed generator set vertex-transitive. The existing `S_3`
  witness has `beta_root<R_action`.
- **New intuition:** Symmetry makes the worst local return visible as a fall
  into every chosen root; without symmetry, the worst return edge may sit
  sideways between equally deep or differently arranged regions.
- Added the vertex-transitivity theorem and Schreier corollaries to note 181.
  Added no code, test, benchmark, Docker action, or new thematic note.
## 2026-08-29: question card — when does one BFS layer become dead state?

- **Question:** What is the exact last moment at which membership in one old
  layer `F_j` can affect scalar novelty?
- **Last-reference depth:** Define `tau_j` as the maximum of `j` and the source
  depths of all reachable edges whose endpoint lies in `F_j`.
- **Exact lifetime:** Under pure layer membership, reclaim `F_j` only after
  `F_(tau_j)` is fully expanded and its publications are quiescent. Before then
  an incoming old edge remains possible; afterwards no future candidate can
  target the layer.
- **Window relation:** `beta=sup_j(tau_j-j)`. A fixed window is therefore a
  uniform upper envelope of individual layer lifetimes, not the fundamental
  object.
- **Calibration:** Undirected graphs have `tau_j<=j+1`; the directed four-cycle
  has `tau_0=3`; the acyclic long-path shortcut makes `tau_1` arbitrarily late.
- **Online boundary:** Seeing no reference from the current frontier cannot
  exclude a reference from a later frontier. Safe early reclamation needs a
  future structural bound, predecessor exhaustion, or another certificate.
- **Distributed boundary:** Logical last use at depth `tau_j` precedes physical
  reclamation; delayed messages, retries, and publications must also retire.
- Added the per-layer liveness formulation to note 181. Added no code, test,
  benchmark, Docker action, or new thematic note.
## 2026-08-29: synthesis card — safe forgetting returns to ordinary BFS

- **Purpose:** End the safe-forgetting lemma chain by integrating only its
  mental-model consequences into the central synthesis.
- **Cayley correction:** The maximum exact positive inverse-word length is the
  actual backward span of the full directed Cayley graph, witnessed by
  `g^-1->e`, not merely a sufficient upper bound.
- **Schreier compression:** Root radial drop, worst action-local return, and
  group inverse length form a hierarchy; directed support transitivity makes
  the first two equal.
- **Layer liveness:** A layer remains relevant through its last incoming-edge
  source depth `tau_j`; the uniform backward span is the worst excess
  `sup_j(tau_j-j)`.
- **Plain-language model:** Old membership can be deleted only after proving
  that no future legal expansion can ask whether that state was old. A rolling
  window is one uniform proof of this, not the definition of forgetting.
- **Anti-duplication audit:** FIFO scheduling and infinite-graph completeness
  were considered as alternate axes but already have complete small examples
  in notes 03/74/164 and 09/54, so no duplicate cards were added.
- Updated note 54 rather than creating a new thematic note. Added no code,
  test, benchmark, Docker action, or GPU implementation.
## 2026-08-29: question card — do all frontier sizes determine BFS work?

- **Question:** If two rooted graphs have the same complete sequence of BFS
  frontier sizes and the same distance map, must BFS scan comparable work?
- **Smallest family:** Compare a `k`-leaf star with the same graph after adding
  every edge among its leaves.
- **Same semantics:** Both have `F_0={s}`, `|F_1|=k`, and empty `F_2`; every
  nonroot distance is one.
- **Different work:** With undirected adjacency stored both ways, the star has
  `2k` occurrences while the filled star has `k(k+1)`. Its extra `k(k-1)`
  occurrences stay entirely inside `F_1` and accept no state.
- **New intuition:** The frontier profile measures newly certified distance
  mass, not edge density inside or back into the visited ball. Equal waves can
  conceal an unbounded work ratio.
- **Practical consequence:** A work model needs layer-edge/occurrence structure,
  not only `|F_d|`, even before hardware effects are considered.
- **Recording failure:** The first patch attempt was rejected because the
  multi-file patch boundary was malformed; it made no file change.
- Added the hand-worked pair to central note 54. Added no code, test, benchmark,
  Docker action, or new thematic note.
## 2026-08-29: question card — do frontier sizes plus total edges fix level work?

- **Question:** If rooted graphs have equal `V`, equal `E`, and identical BFS
  frontier sizes, must their per-level expansion volumes match?
- **Five-vertex pair:** Both use `s--a,s--b,a--x,b--y`. One adds `a--b` inside
  `F_1`; the other adds `x--y` inside `F_2`.
- **Preserved data:** Both have five undirected edges and layers of sizes
  `(1,2,2)`.
- **Different timing:** Their adjacency-occurrence vectors by expansion depth
  are `(2,6,2)` and `(2,4,4)`, although both scan ten occurrences in total.
- **New intuition:** `E` fixes total explicit adjacency work, while the
  layer-edge placement fixes when that work appears. Frontier sizes determine
  neither coordinate.
- **Hardware boundary:** Equal total work can still produce different peak
  buffers, per-level latency, synchronization exposure, and owner imbalance;
  this is a semantic workload distinction before any GPU claim.
- Added the hand trace to central note 54. Added no code, test, benchmark,
  Docker action, or new thematic note.
## 2026-08-29: question card — does equal occurrence work imply equal duplicates?

- **Question:** If two BFS levels have the same frontier, generated-occurrence
  count, and accepted next frontier, is their discarded work semantically the
  same?
- **Common prefix:** `s->a,s->b`, so `F_1={a,b}`.
- **Old-return case:** `a->x,a->s,b->y,b->s` generates four occurrences, two of
  which hit the already visited root; distinct candidates are `{s,x,y}`.
- **New-merge case:** `a->x,a->y,b->x,b->y` also generates four occurrences,
  but all point outward and collapse by cross-parent convergence to `{x,y}`.
- **Same result:** Both accept exactly `F_2={x,y}`.
- **New intuition:** Equal raw work, duplicate count, and semantic progress do
  not identify the rejection mechanism. Visited returns and new-state
  convergence are different information flows.
- **GPU/multi-GPU boundary:** Old hits require authoritative membership;
  convergence requires equal new candidates to meet. They can induce different
  atomics, sorting, routing, and imbalance despite matching scalar totals.
- Added the hand trace to central note 54. This is the third and final card on
  the frontier/work axis before switching topics. Added no code, test,
  benchmark, Docker action, or new thematic note.
## 2026-08-29: source question — what BFS does CayleyPy's library class run?

- **Question:** How does the retained CayleyPy `BfsAlgorithm` map the abstract
  BFS recurrence into its actual tensors, and where is exactness conditional?
- **Observed recurrence:** `get_neighbors(current)` generates occurrences,
  `get_unique_states` combines candidates, `_remove_seen_states` subtracts
  retained old layers, and the survivors become the next complete layer.
- **Batch meaning:** Batches are individually uniqued and later batches are
  filtered against earlier accepted batches; batching moves the combination
  boundary but does not introduce beam pruning.
- **Rolling visited:** With inverse-closed generators, the implementation keeps
  only the previous and current layer hashes, exactly matching the undirected
  three-layer theorem when identity is exact.
- **Identity boundary:** One-int64 encoded states use identity hashing. Wider
  states are reduced to one 64-bit hash; `get_unique_states` keeps the first
  equal-hash state and performs no full-state collision resolution. Seen tests
  are hash-only as well.
- **Completion boundary:** `bfs_completed=True` is set only after an empty exact
  next layer. Diameter, size, and callback limits return a prefix marked
  incomplete.
- **Output distinction:** The traversal may process a large layer without
  retaining its decoded states in `BfsResult`; layer-size computation and full
  materialized graph output are different contracts.
- **Conclusion:** This is structurally ordinary level BFS, unlike the production
  beam outer loop, but wider-state mathematical exactness is conditional on
  collision-free/injective hashing over the reached domain.
- Source audit only; no code change, execution, Docker action, benchmark, or
  proposed fix.
## 2026-08-29: source question — why does CayleyPy bfs_numpy remember a generator?

- **Question:** Does the last-generator partition in `bfs_numpy` replace
  visited, or only remove a known redundant transition?
- **Observed state:** Every exact frontier state is assigned to one generator
  bucket after cross-bucket uniquing. With tied shortest parents, this records
  one selected incoming label rather than all parents.
- **Skipped move:** While applying generator `i1`, the implementation omits
  states whose selected incoming label is `inverse(i1)`. That move would return
  exactly along the selected parent edge into the previous layer.
- **Safety:** Omitting one guaranteed parent-return cannot remove an outward
  shortest child in an inverse-closed unit graph.
- **Why visited remains:** Other predecessors, same-layer edges, and new-state
  convergence are not encoded by the selected label. The code still subtracts
  every partition of the previous/current layers and uniques the building next
  layer.
- **Output boundary:** Bucket assignment is an internal arbitrary tie outcome;
  the function returns layer sizes, not a canonical parent tree or word.
- **Retained tests:** Several permutation/Coset growth sequences and one bounded
  `lrx(16)` prefix are checked. This is fixture-level regression evidence for
  sizes, not a universal parent/history theorem.
- **New intuition:** Parent-skip removes one certified occurrence; visited
  represents the rest of the old ball information still needed by the graph.
- Source audit only; no code change, execution, Docker action, benchmark, or
  proposed optimization.
## 2026-08-29: source question — how does CayleyPy restore a path without parents?

- **Question:** What information replaces parent pointers in CayleyPy's
  `restore_path`, and what guarantee follows from it?
- **Observed method:** Starting from the concrete target, generate inverse
  neighbors and choose the first whose hash occurs in the preceding retained
  BFS layer; repeat to depth zero and reverse the labels.
- **Shortestness premise:** With complete exact layer membership, every chosen
  edge is real and decreases exact depth by one, yielding a length-`D` shortest
  path without a stored discovery parent.
- **Memory/work trade:** Parent trees store one choice per state during BFS.
  Layer backtracking stores all layer membership and regenerates up to one
  inverse neighborhood per path step and query.
- **Rolling conflict:** Previous/current layers suffice for scalar duplicate
  rejection, but not for later layer-by-layer reconstruction. Therefore
  `return_all_hashes=True` intentionally retains information the traversal hot
  path could otherwise reclaim.
- **Tie semantics:** The first matching inverse generator supplies an arbitrary
  shortest predecessor under generator order, not a canonical word.
- **Collision boundary:** Membership is hash-only. `restore_path` does not
  assert that the final reconstructed start equals the central state; a
  semantic collision can therefore corrupt membership or reconstruction.
- **Test evidence:** Retained path tests call `validate_path` on produced fixture
  paths, which checks positive replay for those cases but is not generic
  collision resolution.
- Added the source mapping to central note 54. This is the third and final
  CayleyPy source card before switching axes. No code change, execution, Docker
  action, benchmark, or proposed implementation.
## 2026-08-29: synthesis correction — REF-046 is no longer not-run evidence

- **Trigger:** A proposed push/pull card was rejected because notes 14 and 190
  already cover the exact predicate, early exit, implicit transfer boundary,
  and distributed protocol.
- **Contradiction found:** Central note 54 still grouped REF-046 with REF-045 as
  `not run`, while the experiment artifact, experiment log, coverage audit, and
  current journal record REF-046 as completed.
- **Authoritative status:** REF-046 passed five Docker/Rust integration tests;
  six local and eighteen three-edge single-stop schedules separate blind drop
  from helpable/logged publication.
- **Preserved history:** Earlier named-pipe/Docker access failures remain valid
  historical evidence and the `-not-run.md` precursor remains retained.
- **Evidence boundary:** Completion promotes REF-046 only to a bounded
  sequential state-model result, not runtime memory ordering, GPU execution,
  multi-GPU recovery, or performance evidence.
- Corrected the source map in central note 54. Added no code, execution, Docker
  action, benchmark, new BFS theorem, or thematic note.
## 2026-08-29: synthesis audit — late notes 182-190

- **Question:** Did the central mental model actually absorb the conclusions of
  the late thematic notes, or merely retain them as disconnected cards?
- **Observed gap:** Central note 54 had no source-map entry for notes 182-190
  and almost no explicit bridge to their topics.
- **Integrated distinctions:** traversal order versus several live-boundary
  widths; finite-branching versus fair positive enumeration; unlabeled graph
  equivalence versus labeled simultaneous action conjugacy; deletion
  invalidation versus full-graph repair; one-edge insertion versus batch
  min-plus closure; and 1D/2D expand-fold versus systolic bottom-up ownership.
- **Unifying insight:** These variations all split quantities that ordinary
  finite level BFS makes easy to conflate. The stable core is still a closed
  distance wave; order, retained information, graph version, action labels,
  and physical ownership determine how that proof is represented and when it
  may be published.
- **Historical connection:** Moore/Lee's search-then-trace wavefront view makes
  the same distinction without defining BFS by a queue.
- Updated central note 54 and its source map. No code, execution, Docker action,
  benchmark, or new experiment was proposed or performed.
## 2026-08-29: source question — what did Moore 1959 actually call BFS?

- **Question:** Can the primary Moore paper now support detailed historical
  claims, and did it contain one BFS algorithm or a broader family?
- **Search result:** No inspectable primary scan was obtained. Google Books and
  WorldCat expose metadata; SciSpace labels a copy open access but returned an
  access error. These do not license quotations or pseudocode attribution.
- **Stronger secondary evidence:** Schrijver reconstructs four Moore procedures
  A--D. Its Algorithm A description is exact layer labeling from source `0`;
  Algorithm D handles general edge lengths. Lawler independently describes the
  unit-edge method and a two-bit tree-recording claim.
- **Correction:** “Moore's algorithm” must not mean the whole 1959 paper. The
  modern BFS correspondence is specifically with the reported unit-edge
  Algorithm A; the paper apparently had a broader shortest-path scope.
- **Evidence boundary:** Lee 1961 remains the inspected primary source for the
  wave/search/trace intuition. Moore's exact text, marking scheme, figures, and
  full procedures remain unverified until the original is obtained.
- Updated note 185. No code, experiment, Docker action, or benchmark was used.
## 2026-08-29: semantic question — which shortest-path trees are BFS trees?

- **Question:** Does pointwise parent validity imply that the whole tree can be
  generated by some ordinary FIFO first-discovery BFS?
- **Answer:** No. With depth-one vertices `u,v` both adjacent to depth-two
  vertices `x,y`, choosing `parent(x)=u` and `parent(y)=v` demands both `u<v`
  and `v<u` in the same dequeue order.
- **Source result:** Manber 1990 treats recognition of a BFS-output spanning
  tree as a separate global problem and gives a linear-time algorithm for
  undirected graphs. Only the publisher abstract was inspected, so its
  recognition construction was not reconstructed here.
- **GPU implication:** A deterministic post-layer parent reduction can produce
  an exact replayable shortest-path tree that is not the discovery tree of any
  serial FIFO history. This is valid unless serial first-in realizability was
  explicitly part of the output contract.
- **Validation split:** distance parity, parent-depth validity, replay,
  deterministic reduction parity, and first-in-tree realizability are distinct
  gates.
- Added note 191 and integrated the distinction into central note 54. No code,
  experiment, Docker action, or benchmark was used.
## 2026-08-29: Cayley question — does symmetry remove tree non-realizability?

- **Question:** Can the crossed-parent obstruction occur in a genuine,
  vertex-transitive Cayley graph with a natural generator set?
- **Exact example:** In `Cay(S_3,{(12),(13),(23)})`, `F_1` is the three
  transpositions and `F_2` is the two 3-cycles. Every `F_1` state is adjacent
  to both `F_2` states, so the layer incidence is `K_(3,2)`.
- **Consequence:** A proposed shortest tree assigning the two 3-cycles to two
  different transposition parents is locally valid but cannot be first-in FIFO:
  the first expanded transposition discovers both.
- **Insight:** Cayley transitivity does not decouple parent choices. Relations
  simultaneously create geodesic multiplicity and cross-state arrival-order
  constraints.
- Added the exact `S_3` proof to note 191 and central note 54. No code,
  enumeration, Docker action, or benchmark was used.
## 2026-08-29: Cayley question — what controls common-child coupling?

- **Question:** Which algebraic quantity predicts when two Cayley frontier
  parents compete for the same next-layer states?
- **Identity:** For right successors `uS,vS`,
  `|uS intersection vS|=|S intersection (u^-1 v)S|`. Equivalently, common
  children correspond to generator pairs satisfying `u^-1 v=s t^-1`.
- **BFS filter:** Only `F_(d+1) intersection uS intersection vS` represents
  shared shortest children; raw autocorrelation may include inward or
  same-layer states.
- **Tree consequence:** Two shared outward children suffice to construct
  contradictory parent precedence demands and hence a shortest-valid tree that
  is not first-in FIFO realizable.
- **Counting correction:** A child with `k` parents contributes `C(k,2)` pair
  intersections but only `k-1` excess occurrences. Pair overlap, duplicate
  rejection, and contention are connected but not interchangeable metrics.
- **GPU boundary:** Translation-invariant semantic overlap does not imply
  warp/device/owner locality; ordering and partition determine where the
  occurrences actually meet.
- Added note 192 and integrated it into central note 54. No code, enumeration,
  Docker action, or benchmark was used. This closes the three-card
  first-in-tree axis before switching topics.
## 2026-08-29: CayleyPy source question — what edge stream is generated?

- **Question:** What occurrence order and label multiplicity does retained
  `CayleyGraph.get_neighbors` implement, and what changes under batching?
- **Layout:** Unbatched neighbors are generator-major. Batched BFS changes the
  global order to batch, then generator, then state within batch.
- **Set/order split:** Exact or collision-free hashing preserves the scalar
  frontier set, but batch order selects first retained hash representatives and
  does not preserve arbitrary first-winner metadata.
- **Alignment boundary:** In the batched non-identity branch, states remain
  stacked by accepted batch while hashes are globally sorted. Hooks must not
  treat rows as aligned unless batching is disabled; the API documents this
  requirement generally.
- **Generator multiplicity:** Duplicate permutation entries are allowed and
  generate separate occurrences, then scalar state dedup collapses them.
- **Inverse-label boundary:** Permutation inverse lookup uses a dictionary from
  transformation to the last matching index, so replay can be transformation-
  valid without preserving duplicate label identity.
- **Model boundary:** Repeated central-state values make a non-free orbit;
  matrix mode does not require invertibility and can describe a semigroup
  action rather than a group.
- Added note 193 and central-note integration. Source audit only; no code change,
  execution, Docker action, benchmark, or proposed optimization.
## 2026-08-29: CayleyPy source question — is BfsDistributed ordinary BFS?

- **Question:** Where do novelty and completion become authoritative in the
  library's multi-device and torchrun BFS paths?
- **Ownership:** Candidates route by `hash mod workers`; producer-local unique
  is followed by owner-side unique, owner-local seen subtraction, and rejection
  against earlier accepted batches of the same next layer.
- **Closure:** Torchrun takes the global maximum local batch count, makes every
  rank execute that many owner exchanges, then globally sums the closed local
  next frontiers. Accepted next states are not expanded early.
- **Visited:** General directed search accumulates owner-local seen chunks;
  inverse-closed search retains previous/current hashes, applying the exact
  rolling-window theorem after authoritative routing.
- **Identity boundary:** Routing co-locates equal hashes but does not resolve
  unequal full states with the same hash.
- **Output boundary:** Distributed results contain no edge list. Under torchrun,
  public `return_all_edges` and `disable_batching` arguments are silently
  removed. Stop hooks gather the full layer and run on every rank.
- **Failure boundary:** Ordered collectives are assumed successful; no retry or
  recovery protocol is present in the inspected path.
- Added note 194 and central-note integration. Source audit only; no code change,
  execution, Docker action, benchmark, or proposed optimization.
## 2026-08-29: CayleyPy source question — what does BfsResult export?

- **Question:** Is an explicit `BfsResult` graph a faithful generated-edge and
  diameter artifact for complete and truncated runs?
- **Diameter boundary:** `diameter()` is always `len(layer_sizes)-1`. On an
  incomplete run it is only last returned depth; on a completed run it is a
  source/source-set eccentricity unless extra symmetry proves graph diameter.
- **Directed prefix boundary:** If BFS is incomplete, `BfsAlgorithm` appends a
  reversed copy of the last edge block to make adjacency symmetric, without
  checking inverse closure. A directed NetworkX export can therefore contain
  arcs absent from the declared generator action.
- **Identity boundary:** Hash-to-index uniqueness checks only retained records;
  they cannot detect a semantic state already lost through hash deduplication.
- **Label boundary:** NetworkX uses a simple Graph/DiGraph and `get_edge_name`
  returns the first generator that replays an endpoint pair. Parallel generator
  labels and the original occurrence are not preserved.
- **Matrix boundary:** HDF5 load reconstructs through the permutation
  constructor and records no matrix generator type/modulus, so matrix-mode
  round trip is not demonstrated.
- Added note 195 and central-note integration. Source audit only; no code change,
  execution, Docker action, benchmark, or proposed optimization. This closes
  the three-card CayleyPy source axis before switching topics.
## 2026-08-29: coverage audit — post-note-195 correction

- **Inventory:** 196 numbered-note files cover 195 unique numeric identifiers;
  number 185 has two distinct notes. The corpus contains 2,048 unique `SEM-*`
  IDs and 43 unique `REF-*` IDs under the audit's existing counting convention.
- **New conceptual cells:** Notes 191--192 distinguish shortest-path trees from
  serial first-in BFS-tree realizability and connect Cayley shared-child
  geometry to generator autocorrelation. Notes 193--195 define the retained
  CayleyPy ordinary BFS and export boundaries.
- **Evidence correction:** Source inspection does not validate installed/runtime
  parity, torchrun behavior, directed incomplete export, or matrix HDF5 round
  trip.
- **Authorization correction:** Every executable gate is now explicitly
  dormant. Missing evidence remains an open question and does not authorize a
  probe, test, benchmark, demonstration program, or Docker run.
- Updated note 177 only. No new theorem, code, execution, Docker action, or
  benchmark was introduced.
## 2026-08-29: infinite-graph question — can a BFS layer be uncountable?

- **Question:** Does uncountable branching require transfinite BFS distance, or
  does it only defeat explicit enumeration?
- **Answer:** Every reached vertex still has a finite path and natural-number
  distance. Since graph `Post` preserves arbitrary unions, the reachable fixed
  point remains the union of finite-depth balls at stage `omega`.
- **Counterexample:** An uncountable star has diameter one and an uncountable
  `F_1`. No countable event stream can explicitly enumerate that layer.
- **Correction to note 183's scope:** Fair dovetailing can cover countable
  successor indices; it cannot enumerate uncountable choices.
- **Representation boundary:** Exact handling requires symbolic descriptors and
  exact image/union/difference/emptiness/membership operations, not a larger
  explicit worklist.
- **Cayley boundary:** A finite or countable generator alphabet has only
  countably many finite words; a finite alphabet gives finite layers. CayleyPy's
  finite generator lists therefore cannot create this failure mode from one
  central state.
- Added note 196, central-note integration, and one coverage-matrix row. No
  code, execution, Docker action, benchmark, or proposed implementation.
## 2026-08-29: symbolic question — when is image iteration BFS?

- **Question:** Does a symbolic reachable fixed point preserve BFS layers and
  shortest distances?
- **Exact recurrence:** `Image(F_d) minus R_d` is the same next-frontier set
  operation on predicates. Accumulated iteration
  `R_(d+1)=R_d union Image(R_d)` also yields exact balls.
- **Distance boundary:** A final reachable predicate forgets first-entry depth.
  Distances require retained layer differences, a valued distance relation,
  predecessor certificates, or recomputation.
- **Schedule boundary:** Saturation/chaotic schedules may propagate across
  several semantic edges per implementation round; fixed-point correctness does
  not make their round number a graph distance.
- **Output boundary:** Boolean `or` preserves membership and discards parent
  multiplicity, counts, labels, and canonical words. Witness extraction is a
  separate backward symbolic operation through decreasing layers.
- **Cardinality correction:** Classic finite-variable BDDs represent very large
  finite Boolean universes, not arbitrary uncountable domains.
- **Source:** Burch et al. 1992 was used as primary evidence for symbolic
  state/relation representation, BDDs, and fixed-point model checking, not for
  a GPU or shortest-layer claim.
- Added note 197, central-note integration, and one coverage row. No code,
  execution, Docker action, benchmark, or proposed implementation.
## 2026-08-29: goal authority correction — no autonomous experiments

- **Problem:** The protocol still allowed an executable counterexample whenever
  it appeared "truly useful." That wording could turn a missing evidence item
  into autonomous code creation or execution.
- **Correction:** The standing BFS-study goal now authorizes source reading,
  inspection of existing code, hand reasoning, written traces, and recording
  uncertainties. It does not authorize creating or modifying executable probes,
  tests, benchmarks, demos, or simulators, nor running new experiments.
- The restriction covers CPU/Rust, Docker, GPU, and multi-GPU work. A separate
  explicit user request is required for each specifically scoped experiment;
  authorization does not carry forward to later evidence gaps.
- Missing executable evidence remains an **Unknown** or dormant evidence gate.
  No code or experiment was created or run while making this correction.
## 2026-08-29: partial-order reduction question — what kind of BFS survives?

- **Question:** If independent actions commute, may BFS retain one ordering and
  still claim the original shortest-path distance?
- **Answer:** Commutation is only a path-equivalence fact. Pruning additionally
  needs a coverage theorem ensuring that each relevant omitted path has a
  retained representative.
- **Length result:** Xu et al.'s stubborn-set theorem retains a permutation of
  every solution sequence. With unit actions this preserves the length of one
  shortest goal path; edge deletion supplies the opposite inequality.
- **Boundary:** Goal-optimality is weaker than the complete BFS metric. Two
  commuting bit toggles show that one shortest goal can be retained while a
  different depth-one state is omitted.
- **Output loss:** Other shortest interleavings, intermediate frontier states,
  labeled path counts, canonical words, and original duplicate work need not
  survive.
- Added note 198, central synthesis integration, one coverage row, and five
  evidence-map claims. No code, experiment, Docker action, or benchmark was
  created or run.

## 2026-08-31: synthesis audit — discovery is not layer closure

- **Previous cycle:** Read-only searches rejected hypergraph, LexBFS, and graph
  covering questions as duplicates (notes 34, 19/114, and 123). The useful next
  action became a synthesis audit, not another numbered note. One Windows
  wildcard search failed and was replaced by an explicit file-list read.
- **Question:** When does "final" mean exact distance, completed metadata,
  complete layer membership, or fully expanded vertex?
- **Evidence:** Compared notes 03, 37, 54, 57 and the publication obligation
  already recorded in note 178. The mathematics was compatible, but schedule
  wording in notes 03/37 could be read as requiring the whole current layer to
  finish before any child distance becomes final.
- **Correction:** At expansion of `F_d`, a genuinely new child is outside exact
  `B_d` and has a length-`d+1` witness. Its distance is final immediately;
  remaining producers may add equal-distance metadata and other frontier
  vertices, not a shorter distance.
- **Concrete trace:** Reused `s -> {a,b} -> x` to distinguish the four events in
  the central synthesis. Gray/black describe expansion status, not tentative
  versus exact distance in ordinary FIFO BFS.
- **Schedule clarification:** No explicit barrier is required in sequential
  FIFO. Abandoning the exclusion of shorter unresolved proposals, rather than
  merely removing a barrier call, is what invalidates irreversible discovery.
- Updated existing notes 03, 37, and 54 only, plus this log. No new thematic
  note, executable code, calculation, experiment, or Docker action.

## 2026-08-31: correction — all finite traces versus goal traces

- **Question:** Does retaining one representative per trace necessarily permit
  loss of intermediate BFS states?
- **Failure found:** Note 198 and SEM-2053 used an unrestricted "per trace"
  quantifier for a loss example that only demonstrated goal-path coverage.
- **Correction and proof:** If every finite source path has a retained valid,
  equal-length representative with the same endpoint, apply this to a shortest
  path to each vertex. The retained distance is at most the original; edge
  deletion gives the reverse inequality. Every source distance and therefore
  every distinct-state frontier is preserved.
- **Small check:** The one-letter paths `a` and `b` are finite traces too.
  Deleting root edge `b` cannot satisfy all-finite-trace coverage. Keeping only
  the goal path `ab` illustrates a weaker guarantee, not a counterexample to
  the stronger theorem.
- **Intuition:** State frontier width can stay identical while generated
  occurrences and shortest-word multiplicity shrink. State count and work
  count are different quantities.
- Corrected note 198, SEM-2053, and the central synthesis. This is direct hand
  reasoning, not a new source theorem or runtime result. No code or experiment.

## 2026-08-31: POR source scope and attribution check

- **Progress classification:** The preceding cycle corrected a substantive
  quantifier error. This cycle checked the remaining source attribution rather
  than extending the topic.
- **Bibliographic failure:** The authors in note 198 were wrong. Live arXiv
  metadata identifies You Xu, Yixin Chen, Qiang Lu, and Ruoyun Huang, not the
  previously listed Xu, Fern, and Yoon. Corrected the citation and retained the
  error history explicitly.
- **Scope:** Definition 10 requires a permuted solution sequence, not a
  separately stated identical final state. The same-endpoint premise in our
  all-target metric proof is explicit and must not be inferred merely from
  the phrase "permutation of actions."
- ArXiv metadata loaded; experimental HTML and the alternate HTML endpoint
  failed. The indexed full-paper copy exposed Definition 10 for inspection.
- Added the scope qualification to note 198. The POR sequence is now closed;
  do not extend it without a distinct BFS question. No code or experiments.

## 2026-08-31: memory intuition — layer count is not retained volume

- **Previous cycle:** Progress: source attribution and theorem scope corrected.
- **Question:** Does replacing permanent visited by three exact layer roles
  necessarily remove most membership states?
- **Read check:** Note 181 already proves when forgetting is safe. The missing
  explanatory link was from layer lifetime to volume, not a new forgetting
  theorem. The first attempted filename was wrong; located the existing file
  with `rg --files` and read it rather than creating another note.
- **Hand calculation:** At the same pre-reclamation boundary, an endpoint-rooted
  path retains `3/(d+2)` of the ball. A b-ary tree retains a fraction tending to
  `1-b^(-3)`; for binary growth this is 87.5 percent. Through depth seven the
  exact counts are 224 retained states out of 255.
- **Interpretation:** Forgetting saves old volume, not depth indices. A rapidly
  growing frontier can contain nearly all accumulated states already.
- **Limits:** Membership-state counts only; no byte/peak-memory measurement,
  runtime speed claim, or best-algorithm lower bound. A known tree has stronger
  parent-only certificates, and Cayley sphere growth need not stay geometric.
- Added the example to note 181 and the central synthesis. No new thematic
  file, code, executable calculation, experiment, or Docker action.

## 2026-08-31: retained Megaminx counts — membership volume versus work

- **Previous cycle:** Progress: explained why a small number of retained layers
  can contain most of a metric ball.
- **Question:** Does the same distinction appear in saved puzzle evidence, and
  what does its candidate counter actually count?
- Read REF-026/027/028 raw text outputs and the existing REF-028 Rust loop.
  No source was modified and no program was run.
- **Membership inference:** Saved layer sizes through F4 are
  `1,24,408,6208,90144`. The pre-reclamation three-role snapshot keeps 96760
  of the ball's 96785 distinct states, discarding only 25. This is a hand
  calculation, not measured allocation or a rolling implementation result.
- **Counter clarification:** `candidate_records` increments only outside the
  old ball. The F3 loop generates 148992 occurrences, rejects 9744 into B3,
  retains 139248 outward records, and yields 90144 F4 states. Its separate
  enumeration counts 274224 shortest words. These are different objects.
- Added scoped explanations to notes 181 and 64. Historical results were read,
  not rerun; no new code, calculation process, Docker action, or experiment.

## 2026-08-31: compatibility gate for the combined Megaminx counts

- **Previous cycle:** Progress: connected retained puzzle counts to the memory
  example and located the outward-candidate counter in the existing source.
- **Question:** Is combining REF-026/027/028 counts justified by a shared input
  and state/action convention, rather than merely similar puzzle names?
- All three saved reports record input digest
  `1780a8368d504fd75f448d25e5bede9adb498b35db6a3251e920bbc8524adfca`.
  Current probe sources share the REF-025 parser, `config.central`, generator
  extraction, gather action, and full-vector equality. REF-026 documents 120
  unique positions and 24 signed face moves.
- **Result:** Historical compatibility is supported. Fresh checksum validation,
  historical executable identity, and independent physical-puzzle validation
  are not established by these reads. Shared parsing is not an independent
  oracle.
- Attached this evidence boundary to notes 181 and 64. No code was changed or
  run, no checksum calculation or experiment was launched. This closes the
  three-cycle memory-volume/counter/compatibility sequence; next work should
  return to a different BFS question rather than extend the provenance audit.

## 2026-08-31: first meeting — corrected counterexample and safe early stop

- **Previous cycle:** Progress: established the scope for joining historical
  Megaminx counters. Switched from memory to bidirectional stopping.
- **Question:** Is any partial layer enough to invalidate first-contact stopping?
- **Error found:** Note 56's old length-two route `s->b->t` already intersected
  both depth-one balls at `b`. With the stated persistent discovery checks the
  supposed missed short route would already be known. It did not isolate the
  claimed minimum-advancement defect.
- **Replacement hand trace:** Edges are exactly the two routes
  `s->a->x->t` and `s->b->y->c->t`. Initial depth-one balls are disjoint.
  Forward expansion of `b`, followed by reverse expansion of `c`, first meets
  at `y` with length four. Pending `a` and reverse `x` reveal length three.
  Each side's FIFO order and discovery-time intersection check remain valid.
- **Safe case clarified:** Starting with disjoint exact balls and keeping the
  opposite ball fixed, the first hit during the active next layer is already
  shortest. Finishing that active layer is unnecessary for one distance/path,
  though required for its complete frontier and some richer outputs.
- Updated notes 08, 56, and the central synthesis; retained correction history.
  No new source theorem, executable code, or experiment. The counterexample and
  safe case were checked by the explicit finite path/queue trace only.

## 2026-08-31: propagate the bidirectional stopping correction

- **Previous cycle:** Progress: replaced a flawed counterexample and separated
  partial active-layer stopping from arbitrary two-sided partial interleaving.
- **Question:** Do dependent summaries still teach the superseded condition?
- Found SEM-314 still citing the invalid length-four-versus-length-two witness.
  Repointed it to the corrected four-versus-three case with error history.
- Updated SEM-08 to state that the active layer need not finish when the
  opposite visited set remains the fixed exact ball from the disjoint start.
- Note 162 had a blanket global-minima requirement for partial stopping.
  Qualified it with the fixed-opposite-ball theorem and clarified that a
  certified optimal answer can coexist with incomplete frontier/work totals.
- Read note 59's REF-010 scope and the open questions: their complete-round
  implementation claim and generic protocol questions remain compatible; no
  edit or stronger runtime claim was needed there.
- Documentation consistency only; no new experiment, code, or execution.

## 2026-08-31: integer stopping threshold — one unit of conservatism

- **Previous cycle:** Progress: propagated the corrected counterexample and
  safe fixed-opposite-ball case into dependent summaries.
- **Question:** Why can a length-three first hit be final while unfinished
  radii remain one on each side and the displayed `a+b>=mu` test fails?
- **Hand proof:** Under the same exact ball coverage and globally completed
  intersection checks, any shorter integer path has length
  `D<=mu-1`. If `mu<=a+b+1`, then `D<=a+b` and that path already intersects
  the guaranteed balls, contradicting the maintained incumbent.
- **Result:** `a+b+1>=mu` suffices under these unit-cost conventions. The old
  rule remains valid but conservative. Pending layer work need not be falsely
  marked complete to justify the shorter test.
- Added scope-qualified proof to notes 56, 08, and the central synthesis. No
  stopping code was changed or run. Weighted costs, pending connector checks,
  and richer outputs do not inherit the refinement without their own proof.
- This completes the three-cycle stopping-proof/correction sequence. Return to
  another foundational question rather than extending this into an optimizer.

## 2026-08-31: multi-source versus independent BFS — what visited identifies

- **Previous cycle:** Progress: proved the scoped integer stopping refinement.
  Switched to source identity rather than extending that proof sequence.
- **Question:** Why does one visited set fail to represent a batch of independent
  source-distance queries even when they share the same graph?
- Note 13 already distinguishes the joint minimum from independent rows and
  supplies the path `s--x--y--t`. Reused it rather than adding a new topic.
- **Hand trace:** `x` is reached from `s` at depth one and from `t` at depth
  two. The second arrival is a new `(source,vertex)` pair, not a duplicate for
  the full distance matrix. It is correctly discarded by the joint-minimum
  problem, which asks for a different answer.
- **Identity model:** Independent searches are BFS over disjoint source-tagged
  graph copies. A source bitset can encode pair membership, but all-nearest
  labels omit farther sources and final reachability masks omit arrival depths.
- Added the small table and pair-state explanation to note 13. No batch
  implementation, runtime performance claim, code change, or experiment.

## 2026-08-31: Cayley source reuse — retain the source in the coordinate map

- **Previous cycle:** Progress: distinguished source-pair novelty from a joint
  nearest-source wave. Its generic representation statement needed an explicit
  symmetry qualification for the intended Cayley setting.
- **Question:** Can one BFS answer independent source rows without taking their
  minimum? Yes, if a proved graph automorphism transports each source to the
  table root.
- **Hand proof:** For right Cayley moves, left translation by `s^-1` preserves
  each directed labeled edge, yielding `dist(s,v)=dist(e,s^-1 v)`. The source
  remains in the transformed query, not in a merged nearest-source distance.
- **Small example:** On directed `Z_6` with `+1`, source 3 to vertex 1 has
  distance four from relative coordinate 4; joint sources `{0,3}` give one.
- Qualified note 13's generic claim: source-specific information is required,
  but separate traversals or explicitly stored rows are not always necessary.
  Stated depth-limited table, Schreier automorphism, output-size and parent-tie
  boundaries. No new executable code, table, experiment, or runtime claim.

## 2026-08-31: source-table synthesis and direction check

- **Previous cycle:** Progress: qualified independent-query storage with the
  full Cayley translation theorem.
- **Question:** Does reuse of an identity-rooted table also justify reading it
  directly as distance to identity? Not for a directed alphabet.
- **Hand check:** `T_to(x)=T_from(x^-1)` in the full right Cayley graph. Directed
  `Z_6` with only `+1` gives `T_from(1)=1` and `T_to(1)=5`.
- A valid word to `x^-1` can be replayed unchanged from `x` to identity;
  reversing and inverting a word to `x` may introduce unavailable move labels.
- Added this table-orientation explanation to note 13, and integrated the
  source-pair versus symmetry-transport distinction into the central synthesis,
  which still contained the earlier generic separate-source phrasing.
- No code or experiments. This closes the three-cycle source-identity,
  translation, and orientation sequence; return to another BFS axis next.

## 2026-08-31: certificate audit — infinity is not a predecessor depth

- **Previous cycle:** Progress: integrated source-table reuse and direction.
- **Question:** What does the local BFS certificate prove beyond valid parent
  replay? Note 41 already answers this with path witnesses and edge feasibility;
  no duplicate explanatory section was needed.
- **Ambiguity found in the adjacent count certificate:** The displayed
  recurrence used `L(u)+1=L(v)` without restricting labels to finite values,
  while its induction claimed every eligible edge strictly decreases depth.
- **Hand counterexample:** An isolated source plus unreachable `p->q->p` has
  infinite labels on both cycle vertices. Under `infinity+1=infinity`, the
  unguarded equations allow `sigma(p)=sigma(q)=7`. They are self-consistent but
  not counts of paths from the source.
- Corrected note 41 to require zero counts for unreachable vertices and to
  sum only finite-depth predecessor contributions for finite non-source
  vertices. This repairs the stated certificate, not any inspected runtime.
- No new thematic note, code, test, experiment, or Docker action.

## 2026-08-31: finite-distance scope of the shortest-path DAG

- **Previous cycle:** Progress: repaired the count certificate's infinity case.
- **Question:** Is the same finite-label premise explicit in the predecessor
  DAG definitions that justify counting and sampling?
- Notes 11, 57, and the central synthesis displayed only `d(u)+1=d(v)`.
  Their reachable-frontier interpretation was sound, but the full-graph
  filtering formula omitted the domain restriction needed for infinity labels.
- Added `d(u)<infinity` to those definitions, clarified unreachable predecessor
  sets/counts, and aligned SEM-323 and SEM-1351. No change to the theorem for
  already reachable vertices; this makes its input domain explicit.
- Infinity is not a last BFS layer. The unreachable-cycle counterexample in
  note 41 now explains why strict-depth induction cannot include it.
- Documentation only; no runtime bug claim, source-code edit, or experiment.

## 2026-08-31: missing entries versus certified infinity

- **Previous cycle:** Progress: made finite-distance scope explicit in DAG and
  count definitions.
- **Question:** Can every missing entry in a bounded BFS result use the
  zero-path rule for unreachable vertices? No: bounded absence and certified
  unreachability carry different information.
- Note 09 already separates incomplete searches from exhaustion. Added the
  corresponding scope boundary to note 41's local/count certificates.
- **Hand example:** `s->a->t` and `s->a` with isolated `t` have the same exact
  radius-one ball. Without scanning boundary `a`, that ball cannot distinguish
  target distance two/count one from infinity/count zero.
- Recorded finite-certified, outside-radius, and unreachable statuses.
  Cancellation without a certified completed radius supports no radius bound
  merely from a missing entry. Storage sentinels do not establish semantics.
- No new code or experiment. This closes the three-cycle count/DAG/infinity
  scope audit; next work should switch away from this axis.

## 2026-08-31: zero-cost edges remove the parent-depth termination proof

- **Previous cycle:** Progress: separated bounded absence from infinity and
  closed the count-certificate audit.
- **Question:** What fails when ordinary discovery/parent semantics are carried
  into 0-1 BFS? Note 12 already had the first-discovery counterexample, so it
  was not duplicated.
- **Hand example added:** `s->a` and `s->b` cost one; `a->b` and `b->a` cost
  zero. Correct labels on both vertices are one, but choosing each as the
  other's parent satisfies local tightness and creates a cycle.
- **Interpretation:** Unit BFS gets parent well-foundedness from strict depth
  decrease. Correct weighted distances plus tight parent edges alone do not
  give that proof when zero-cost edges exist. This is a certificate boundary,
  not a claimed defect in a particular 0-1 BFS implementation.
- Added the example to note 12 and made its tight-edge definition explicitly
  finite-distance. A broad read-only duplicate search stalled and was cancelled;
  no exhaustive corpus absence claim is made. No code or experiment.

## 2026-08-31: weighted certificate — which BFS premises must change?

- **Previous cycle:** Progress: demonstrated tight parent cycles with zero-cost
  edges. Continued with the certificate implication rather than implementation.
- **Question:** Can the unit certificate be reused by only substituting weights
  for one? No: unique zero labels and strict parent-depth descent can both fail.
- **Hand witnesses:** `s->a` of cost zero legitimately gives two zero-distance
  vertices. An isolated source and unreachable zero-cycle with false labels one
  pass local weighted tightness/feasibility while lacking rooted witnesses.
- Recorded the finite-graph nonnegative weighted certificate: source zero,
  explicit root-terminating tight parent chains for finite labels, and complete
  weighted edge feasibility/finite-successor closure. Path summation proves
  both distance inequalities without a strict weighted-depth descent premise.
- Added the qualification to note 41 and central synthesis. Exact arithmetic
  and distance-only scope stated; zero-cycle counts remain a different problem.
- No new source claim, code, validator, experiment, or Docker action.

## 2026-08-31: equal positive costs are a normalization

- **Previous cycle:** Progress: specified what zero-cost weighted certificates
  need in place of strict unit-depth descent.
- **Question:** Is the literal numeric cost one essential to ordinary BFS?
  No: a common positive cost preserves ordering and ties of all hop lengths.
- **Hand proof:** Every k-edge path costs `c*k`; for `c>0`, minimizing either
  expression is equivalent. BFS distances need only be rescaled on output.
- Qualified the "positive unit cost only" wording in note 12 and added the
  same normalization boundary to note 05. Zero is different: every reachable
  walk then costs zero, so hop-shortest paths are only a subset of the
  cost-optimal family. Unequal positive costs lack a common scaling proof.
- No implementation or experiment. This closes the three-cycle weighted
  boundary sequence; next work should return to a different BFS question.

## 2026-08-31: goal-alignment audit and roadmap authorization repair

- **Previous cycle:** Progress: qualified equal positive costs as normalization.
- Read the research protocol, roadmap, coverage audit, experiment index,
  REF-017's execution/oracle protocol, REF-010's simulation protocol, and the
  retained CayleyPy action-contract introduction.
- **Drift risk found:** Roadmap wording still permitted minimal code/probes
  unless substantial implementation was requested. Historical phase imperatives
  could likewise be mistaken for standing authority. Aligned both the roadmap
  and protocol purpose with the later explicit authorization gate.
- Refreshed note 177's evidence scope without relabeling reports as fresh runs:
  bounded S9 GPU evidence exists; distributed owner routing is simulated;
  application-scale parity and real multi-GPU timings remain unknown.
- Named one next read-only question about REF-017 kernel-sum versus end-to-end
  timing boundaries, with a mandatory duplicate check and no new benchmark.
- No code, runtime measurement, checksum, or experiment. Goal remains active;
  this was a scoped alignment audit, not a full completion audit.

## 2026-08-31: REF-017 timing boundaries, read-only clarification

- Previous goal turn made progress by recording the corrected goal authority.
- Question: what does the saved kernel/traversal timing gap teach about BFS?
- Existing synthesis already covers the main distinction; no new thematic note.
- Inspected `rust/src/cayley.rs:677-695` and
  `gpu/cayley_bfs.cu:240-264`: reset precedes the Rust timer; per-level scalar
  resets precede GPU start events; synchronization and scalar copies return
  the next frontier size; Rust verifies each count inside its traversal timer.
- Correction: the report's claim that synchronization dominates was stronger
  than this timing decomposition supports. Non-kernel elapsed time dominates
  the reported totals, but no individual overhead is isolated. Updated report
  and existing synthesis, following the throughput skill's metric-boundary rule.
- Intuition: closing a BFS layer and learning the next frontier size are part
  of this traversal even though they are not all inside its expansion timer.
  Algorithmic layer dependency is not a proof that host synchronization is
  intrinsically required by every implementation.
- No executable code changes or runs. Historical timings remain historical.

## 2026-08-31: exhaustion evidence versus the stopping driver

- Previous turn made progress: corrected unsupported attribution of the
  REF-017 timing gap specifically to synchronization.
- Question: does this example discover when to stop, or verify exhaustion at
  an oracle-specified depth?
- Read both traversal functions in `rust/src/cayley.rs`: their loop bound is
  `expected.len()`, and unexpected early zero fails the next-count assertion.
  The correctness traversal also checks final emptiness after the loop.
- Hand trace `s -> a`: expanding F0 gives F1; expanding F1 gives empty F2.
  Two predetermined expansions and stop-on-empty can agree on this trace,
  without being the same stopping implementation.
- Updated the report and existing synthesis: complete S9 exhaustion remains
  validated by the retained experiment; generic unknown-depth/device-side
  stopping is not demonstrated by it. No new note, code, or run.
- This closes the two-question source check. Return to foundational BFS
  understanding next; do not grow a driver-design or optimization backlog.

## 2026-08-31: why an earlier arrival can replace a later one

- Previous turn made progress by separating exhaustion validation from the
  actual oracle-bounded stopping driver.
- Question: why is retaining one arrival sound for minimum hop distance?
- Existing notes 20, 128 and synthesis already contain history-state and
  unsafe-merging counterexamples. No additional counterexample note needed.
- Added the short replacement argument to the synthesis: prefixes of lengths
  p <= q reaching the same semantic vertex admit the same fixed suffix r;
  replacing the longer prefix gives p+r <= q+r. This connects state sufficiency
  and nondecreasing discovery directly to the reason visited is safe.
- Scope: ordinary walks, fixed graph and state-based goal, scalar hop distance.
  Equal-length arrivals can still contribute distinct path counts; history
  constraints may require a larger vertex. No code or experiments.

## 2026-08-31: reachable-component work excludes dense initialization

- Previous turn made progress by connecting visited to prefix replacement.
- Question: which costs does linear BFS complexity count for a given graph
  representation? The implicit generation-cost question is already covered
  in note 29 and the synthesis; no duplicate treatment was added.
- Found an overstatement in note 29: unreachable vertices need not be touched
  after construction. That is true of traversal adjacency access, not of an
  arbitrary query's initialization or full-distance output.
- Hand example: N isolated vertices, one source. Traversal work is constant;
  clearing N visited entries is linear. Corrected the note to distinguish
  Theta(|R|+A_R) traversal from Theta(|V|+A_R) with dense initialization.
- No implementation or run. The first documentation patch failed its context
  check; inspected the unchanged text and reapplied with the exact context.

## 2026-08-31: connecting frontier width to FIFO capacity

- Previous turn made progress by separating initialization from traversal work.
- Question: is the largest metric layer the largest FIFO queue?
- Note 73 already answers this with exact occupancy m-k+D_k and a hand tree
  with layers [1,100,100]. Hub-first gives 199 queued entries; hub-last gives
  100. No new theorem, experiment or thematic note is needed.
- Added a short cross-reference in note 29 where O(W) live queue memory could
  be mistaken for exactly W slots. Under mark-on-enqueue FIFO, nonempty full
  traversal has queue peak at most 2W-1; this is still O(W), not a prescription
  for buffer allocation or a bound on all BFS memory.
- The learning is a distinction between an asymptotic scale and an exact
  capacity, not a queue-order optimization proposal. No code or runs.

## 2026-08-31: root depth and diameter, connected to Cayley symmetry

- Previous turn connected asymptotic queue memory to exact occupancy.
- Question: when does a single exhaustive BFS determine graph diameter?
- Notes 21 and 195 already contain the theorem and API caveat. Reused note
  21's three-point Schreier witness rather than creating another note.
- Integrated the smallest contrast into the synthesis's relative-element
  lookup explanation: middle-root BFS on 1--2--3 reaches depth one, diameter
  is two; finite strongly connected Cayley graphs instead have equal root
  eccentricities by left translation. A transitive puzzle-state action does
  not by itself prove fixed-generator graph transitivity.
- Documentation only. Removed an accidental duplicated line introduced while
  inserting the explanation; checked the resulting paragraph. No executions.

## 2026-08-31: exact distance policy versus admissible beam ranking

- Previous turn connected root eccentricity to the Cayley symmetry premise.
- Question: does even a perfect heuristic fail under width-one selection?
- Hand proof: with exact h=distance to target, a state of h=r>0 has a
  successor of h=r-1 and no successor below r-1. Complete successor generation,
  minimum-h selection and no obstructing filter yield one geodesic.
- Contrast on the existing two-branch graph: h(a)=1, all other h=0 is admissible
  and consistent, yet width one takes b and returns length four instead of two.
- Added this distinction to note 24 and softened synthesis wording from
  deletion being incompatible to requiring its own proof. Exact policy descent
  preserves one optimal path, not the full BFS frontier or path multiplicity.
- These are hand deductions, not claims about the production scorer, not a
  new algorithm implementation, and not authority for an experiment.

## 2026-08-31: requested radius versus certified radius

- Previous turn proved the distinction between exact target distance and an
  admissible beam score. The present reverse-table question was mostly covered
  by note 49, so no duplicate note was created.
- Found an overly broad incomplete-table conclusion: interruption does not
  erase an already certified smaller complete ball. The miss bound uses that
  ball's radius, never an unfinished requested radius.
- Hand chain b->a->t: a completed reverse radius-one ball gives a miss bound
  two for b, whereas requested radius two would incorrectly imply three.
- Added this example and corrected the conclusion. Also clarified that an
  exact abstract PDB is not necessarily exact concrete distance, connecting
  the previous exact-descent result without contradicting beam caveats.
- Documentation/hand reasoning only; no code, table construction or GPU run.

## 2026-08-31: scramble length versus the distance being learned

- Previous turn distinguished requested table radius from a certified radius.
- Question: does a k-step scramble supply a BFS-distance label for its endpoint?
- Read notes 95 and 39. Reused the existing non-backtracking C4 word aaaa:
  length four, endpoint identity, BFS distance zero. Added its direct
  interpretation as a trajectory label rather than a minimum-distance label.
- Hand direction check: in Z6 with only +1, a one-step forward scramble from
  0 reaches 1, but distance back to 0 is five. The forward witness upper bound
  applies to the forward query; reversing it needs the appropriate move set.
- Documented the conditional training implication without claiming any current
  CayleyPy dataset uses such labels. No dataset/model audit or executable work.

## 2026-08-31: cross-axis review, closed steps and evidence boundaries

- Previous turn made progress by separating scramble witnesses from BFS labels.
- Read the current protocol, roadmap, audit, core recurrence/variant map,
  distributed closure model, REF-010 simulated protocol and REF-017 timing
  boundary. Located the actual REF-010 filename after an incorrect path failed.
- Updated note 177's stale next-step entry: timing/source stopping checks are
  closed. Summarized how recent corrections connect the study axes rather
  than creating another thematic note or repeating the examples.
- Core mathematical and conceptual hardware understanding is represented;
  historical small-GPU evidence and simulated ranks remain explicitly weaker
  than application-scale and real multi-GPU measurements.
- No new foundational gap was established in this selective review. Future
  steps need an actual unresolved question, not inventory growth. This is not
  a whole-corpus correctness claim or authorization for dormant runtime work.
- Documentation only; goal remains active, no completion/blocking claim.

## 2026-08-31: expert question and one source-backed trajectory-label example

- Previous turn refreshed the cross-axis map and closed a stale next step.
- Concrete unresolved question: does a retained project training path actually
  use scramble depth, and is that path tied to the production checkpoint?
- Asked multigpu_beam once, requesting read-only source lines and no runs.
  Expert could not access its checkout; target/checkpoint provenance remained
  unknown. Did not adopt its unsupported action/Q-scoring characterization.
- Independent local read access to D:\100XH100 was available. Read project
  instructions and one staged pilgrim trainer, finding generation-depth Y
  passed directly to the loss after random walks or random-tree sampling.
- Added precise source-path/line references to note 95. The no-immediate-inverse
  sampler does not convert those depths into minimum-distance certificates.
- No checkpoint/run-manifest association established. This answers the bounded
  code-capability question, not the requested production-checkpoint provenance.
  No code changes, training, test or GPU run; only local study notes updated.

## 2026-08-31: word-tree growth is not graph-BFS generation growth

- Previous turn established one retained trajectory-label training path while
  leaving production-checkpoint provenance unknown. Returned to BFS geometry
  instead of extending the training-source audit.
- Question: do two available moves imply exponential BFS work in depth?
- Found ambiguous wording in note 29 grouping raw words with generated BFS
  occurrences. Corrected it: fixed q gives q*|F_d| expansion occurrences,
  not q^d, when each accepted state is expanded once.
- Hand integer-line example: +/-1 gives 2^d words, two states per positive
  sphere, and four attempted transitions per noninitial expansion. Building
  B_d costs 4d-2 applications; expanding its boundary too costs 4d+2.
- Recorded the distinction between constructing a bounded ball and expanding
  that ball, without claiming exhaustion of an infinite graph. No code/runs.

## 2026-08-31: target discovery, dequeue, and predecessor-layer completion

- Previous turn corrected word-tree versus graph-generation growth.
- Question: which layer needs completion for target distance versus counts?
- Note 57 already separates the output contracts. Added one hand queue trace
  to disambiguate the phrase finish the layer: s->{a,b}->t->z gives queue
  [b,t] on first discovery of t (count one), then [t] after b (count two).
- Completing F_(D-1) expansion finalizes standard target counts; expanding
  target layer F_D is unnecessary. First dequeue of t implies the former in
  sequential FIFO, not in an arbitrary asynchronous schedule.
- Preserved the equal-depth accumulation and source base-case premises.
  No standalone note, implementation, test, or run.

## 2026-08-31: reverse access need not be an allowed inverse move

- Previous turn clarified predecessor-layer completion for target counts.
- Question: does bidirectional BFS require invertible/forward-allowed inverse
  moves, or an exact predecessor enumeration interface?
- Note 08 gives the correct reverse-graph contract; roadmap shorthand was
  narrower. Replaced it with the predecessor-enumeration requirement.
- Hand map f(x)=floor(x/2) on {0,1,2,3} is noninjective but admits reverse
  layers {0},{1},{2,3}. This separates a predecessor set from a single inverse.
- Added the Cayley orientation consequence: backward x*s^-1 enumerates an
  original s-edge even when s^-1 is not forward-allowed. Replay retains s.
- No code or graph execution. This is an access/metric clarification, not a
  bidirectional implementation task.

## 2026-08-31: smaller frontier versus cheaper next expansion

- Previous turn separated predecessor enumeration from invertibility.
- Question: when does frontier cardinality predict expansion work?
- Note 08 already gives the policy caveat and REF-009 comparison. Added a
  hand witness: forward {a,b} has two total outgoing entries; backward {t}
  has 100 predecessors. Smaller vertex count need not mean fewer inspections.
- Qualified the example as complete-layer work, not the cost of immediate-hit
  cancellation. Connected to regular Cayley occurrence counting q*|F|, where
  cardinality does predict next-step attempts but not total work or wall time.
- No new policy, benchmark or implementation. This closes the reverse-access
  and side-work pair; do not extend it into selection/tuning work.

## 2026-08-31: completion of the understanding-first study goal

- Previous turn clarified complete-layer work versus frontier cardinality.
- Re-read the corrected goal authority and checked the requested learning
  outcomes against the core proof, graph-interface note, CayleyPy action and
  owner contracts, variant map, GPU control/communication model, historical
  measurement limitations and recorded corrections.
- Corrected one final terminology mismatch in note 03: eccentricity is the
  last nonempty depth, not an unqualified count of expansion iterations.
- Note 177 now records requirement-by-requirement completion of the learning
  outcome. This is not certification of all corpus claims or runtime systems.
  Unknown application performance, real multi-GPU scaling, runtime parity and
  checkpoint linkage remain explicitly recorded and are not declared solved.
- The requested outcome is the connected understanding and durable record,
  not an optimized implementation or an endless catalogue of refinements.
  No new executable work was performed in the completion review.
