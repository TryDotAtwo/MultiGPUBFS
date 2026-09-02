# Open questions

Questions here are hypotheses to investigate, not architecture decisions.

Correction status, 2026-08-31: questions resting on a falsified premise are
resolved or reframed in place. A remaining question is not permission to run a
probe. In particular, a later-shorter K2 hit is impossible under the complete
exact reverse-ball and length-ordered enumeration contract; inspect actual
premise failures instead of searching for that impossible counterexample.

## Next-step understanding gate

- What single thing about BFS am I trying to understand better?
- What do I currently expect to happen on the smallest useful graph, and why?
- Can a hand trace or counterexample answer the question without code?
- What observation would force me to correct the current mental model?
- Does an existing note already answer it well enough?
- Am I about to inspect an adjacent system because BFS requires it, or merely
  because it offers another auditable topic?
- Am I treating bookkeeping (notes, claims, sources, coverage) as progress?
- Can the result be recorded as one short question card rather than another
  thematic note?
- If code is genuinely needed, is it a deliberately small Rust probe executed
  in Docker rather than an implementation project?
- Have GPU performance and multi-GPU engineering been deferred until the
  semantic question is understood and the user separately requests code?

## Discovery-publication continuity questions

- Which semantic event changes a state from absent to claimed?
- At that event, where does recoverable publication responsibility reside?
- May a losing claimant discard its record, or must it prove/help publication?
- Can the novelty winner stop permanently before the frontier payload is
  visible?
- What orders payload visibility before `PUBLISHED` across host, device,
  transport, and persistent storage?
- Is duplicate physical publication idempotent for the requested output, not
  only for the reached set?
- Does capacity exhaustion leave a live obligation or produce explicit failure?
- Can checkpoint/recovery distinguish `CLAIMED` from `PUBLISHED`?
- Is publication credit included in the same consistent cut as visited and
  queues?
- Which actor can complete an orphaned descriptor after device/rank loss?

## Cut, information, and protocol-traffic questions

- Which adjacency, frontier, visited, rank, and action facts does each owner
  initially know?
- Which cross-owner edge occurrences actually introduce information absent at
  the destination?
- Is redundant destination-side generation permitted, and where is its work
  charged?
- Are `C_d`, `U_d`, active owner pairs, encoded bits, actual bytes, and rounds
  retained separately?
- What conditional-information family underlies any claimed lower bound?
- Could replication make traversal messages vanish while increasing memory,
  preprocessing, or replica synchronization?
- Are producer-side rejected candidates proved one-sided safe?
- Does a claimed compression saving include encoding time and headers?
- In an implicit/Cayley graph, is it cheaper to send a child, a parent plus
  label, a shared rank, or a frontier fact, and under which exact identity
  contract?
- Does predictable algebraic ownership merely choose the route, or actually
  reduce information absent at the receiver?
- Are output redistribution and validation traffic separated from traversal?

## Distributed exact reconciliation questions

- What exact object is compared: state set, distance map, parent map, labeled
  DAG, count map, multiset, or sequence?
- Is the state representation injective/collision-resolving for validation?
- Do both executions normalize into one verifier partition independent of their
  runtime owner maps?
- Which evidence proves every source shard and in-flight verifier record
  participated?
- Can validation capacity or transport failure ever be reported as equality?
- Does a local comparator inspect full keys or only hashes/prefixes?
- Is a one-bit global equality flag backed by exact comparison of every local
  block?
- Are Merkle/fingerprint/IBLT results labelled with their collision/failure
  assumptions?
- Does every mismatch retain an exact replayable semantic witness?
- Can a whole omitted shard evade otherwise exact local comparisons?
- Do CPU and GPU validators share the same move table, encoder, or legality bug?
- Which bounded independent oracle calibrates common-mode failure risk?

## Safe visited-forgetting questions

- Which future occurrences can still reach a reclaimed state?
- Is the graph undirected, directed, or a directed Cayley/Schreier action?
- Which structural theorem bounds backward BFS-layer span before the traversal?
- Are previous, current, and building-next layers represented by distinct exact
  epochs before bit rotation?
- Can a delayed device/message record arrive after its membership layer was
  recycled?
- Which output needs old distances, parents, labels, counts, or replay data?
- Is forgotten membership replaced by complete boundary/used-operator metadata?
- Can a directed DAG still jump to an arbitrarily old BFS layer in this graph?
- For every Cayley generator, what is the shortest positive-alphabet word for
  its inverse?
- Does quotienting change the backward-span proof for the requested concrete
  identity?
- Does delayed duplicate detection retain the full old-state certificate
  externally, or actually prove it unnecessary?
- Which consistent cut authorizes global reclamation across GPUs and recovery
  artifacts?

## BFS order and live-boundary questions

- Is “frontier” referring to a metric layer, queue/Open set, left boundary,
  right boundary, edge cut, or owner boundary?
- Which total vertex order is being evaluated, and is it BFS-valid for the
  declared root?
- How large are left and right live boundaries separately at every cut?
- Does a small pathwidth order violate nondecreasing-distance processing?
- How much larger is BFS-constrained separation than unrestricted pathwidth?
- Is a wide Open set materialized, streamed, compressed, or regenerated?
- Does retained used-operator metadata scale with boundary vertices, crossing
  edges, or labels?
- Which parent/count/canonical metadata remains live after graph-boundary
  separation is complete?
- Are temporal and ownership boundaries measured as a joint matrix rather than
  collapsed into remote fraction?
- In Cayley/Schreier search, how do relations and stabilizer aliases change
  state, label, and occurrence boundaries separately?
- Is a combinatorial width being misreported as a direct GPU byte lower bound?
- Which small graphs can independently validate layer width, queue peak,
  left/right separation, edge cut, and optimal/BFS-constrained orders?

## Infinite-branching dovetail questions

- Does every successor enumerator terminate, or merely emit every positive
  occurrence eventually?
- What fairness measure covers parent, depth, and successor index jointly?
- Is the requested result any witness, an eventually exact bound, or a finitely
  certified shortest distance?
- Can an early longer witness precede a late direct target edge?
- Are first claims correctable, and do decreases reactivate every consequence?
- Which finite event proves all paths shorter than the incumbent absent?
- Could two graph presentations share every finite transcript so far but differ
  by a late shallower edge?
- Does a finite semantic state universe still have an infinite redundant
  successor-occurrence stream?
- Which quotient, rank bound, adjacency decision, or lower bound supplies
  finite negative evidence?
- For an infinite Cayley alphabet, is generator enumeration order separated
  from abstract word length?
- Are finite GPU chunks reported as prefixes rather than completed levels?
- Can termination telemetry distinguish apparent label stability from certified
  finality?

## Deferred state-identity direction

- Investigate an injective compact encoding of the concrete puzzle state as an
  alternative to a 128-bit probabilistic fingerprint.
- First compute the information lower bound
  `ceil(log2(number_of_valid_states))`, including permutation, orientation,
  parity, and other legality constraints.
- Compare encoded width, GPU rank/unrank cost, radix-sort cost, visited memory,
  and multi-GPU traffic against the 128-bit-hash design.
- This is explicitly deferred: it is not part of the current BFS pipeline and
  should not block the hash-based implementation.

## Product-graph questions

- Which puzzle state decompositions are true Cartesian products, and which
  moves couple several coordinates and therefore invalidate additive distance?
- When factor growth series are known, how accurately does their convolution
  explain the full-state frontier before quotient relations couple coordinates?
- How much shortest-word multiplicity comes purely from interleavings of
  independent coordinate paths versus relations internal to each factor?
- What evidence would distinguish an exact product from a heuristic coordinate
  abstraction without turning that distinction into an implementation plan?

## Frontier-profile realizability questions

- Which additional sphere sequences are forbidden by fixed regular degree,
  vertex transitivity, or a specified Cayley generator alphabet?
- What proved structural information is sufficient to turn a measured frontier
  prefix into a safe capacity bound rather than an extrapolation?
- How different can edge work and shortest-path multiplicity be among graphs
  sharing the same complete rooted sphere sequence and maximum degree?
- Which non-unimodal profiles occur in concrete finite Cayley graphs rather
  than unrestricted layered constructions?
- How different are the semantic frontier peak, sequential FIFO peak, candidate
  peak, and bulk current-plus-next buffer peak on the same exact traversal?
- Which parent-order prefix union curves explain queue occupancy without being
  misreported as a recommendation to reorder production search?
- At which exact event in each runtime does a candidate become claimed,
  visible, durable, settled, and safe from duplicate re-expansion?
- Are queue capacities justified against unique states, candidate occurrences,
  routed copies, or retry-amplified records?
- Do runtime counters separate stale records safely discarded at pop from stale
  records accidentally re-expanded as new graph work?

## Cayley dead-end questions

- Do the current Megaminx or Cube generator metrics contain interior dead ends,
  and what exact bounded evidence would establish their first depth?
- How should dead-end, escape-depth, retreat-depth, and strong-depth counters be
  named so artifacts cannot silently mix off-by-one conventions?
- How much variation in outward yield exists across one exact Cayley frontier,
  despite uniform labeled generator degree?
- How tight are the mean/max forward-degree bounds on dead-end fraction for
  actual puzzle layers, and what additional histogram information matters?
- Can a bounded-radius lookup distinguish `NOT_FOUND_WITHIN_RADIUS` from a
  genuine local pocket without claiming anything about global reachability?
- Which generator-set changes remove dead ends while also changing the target
  metric, making the apparent improvement semantically incomparable?
- Do current telemetry fields distinguish graph dead ends, BFS-tree leaves,
  terminal-layer states, target-stop suppression, and pruning-induced zeros?

## Contract-map audit

- Can every existing CayleyPy BFS/beam/lookup pipeline fill the exact-BFS
  passport in note 37 without an ambiguous graph, metric, or output field?
- Which current artifacts prove exact set membership rather than only counts,
  hashes, or one replayable target path?
- Where do physical supersteps, owner epochs, and checkpoints align with logical
  completed-level boundaries?
- Which pipelines intentionally solve a quotient, product, directed, bounded,
  or pruned problem but still carry an overly broad `BFS` label?
- Can performance tables group only runs that pass semantic-parity and
  correctness gates before comparing throughput?

## Correctness and identity

- Which existing state keys are proved injective ranks, collision-resolving
  hashes, or probabilistic fingerprints?
- Do any validators currently infer frontier-set equality from counts and
  aggregate fingerprints without an exact bounded oracle?
- Is the requested result minimum-to-set distance, one nearest label, every
  tied nearest label, or a complete per-source distance family?
- For each perfect/minimal-perfect hash proposal, what fixed key set is covered
  and how are queries outside that set recognized?
- Can every Bloom-positive BFS candidate reach an exact backing-set decision
  before rejection?
- Which full-table/probe-limit paths report overflow rather than returning a
  false `seen` result?
- Do canonical serialization, hash seed, graph version, and owner epoch remain
  consistent for all in-flight distributed records?
- Which forced-collision fixtures cover membership, ownership, resizing, and
  concurrent insertion separately?

- Which intended BFS outputs are most naturally specified as a least fixed
  point, and which additionally require distance strata, parents, labels, or
  path multiplicities that the reached-set fixed point forgets?
- Can each distributed design state a precise global-cut condition under which
  an observed empty delta proves successor closure?
- Which delivery assumptions are actually required: exactly once, at least
  once plus idempotence, or recoverable replay after failure?
- On infinite or infinitely branching implicit graphs, is the requested result
  a finite-radius ball, a target path, a symbolic fixed point, or an impossible
  exhaustive materialization?
- When a GraphBLAS formulation changes semiring or mask semantics, what object
  replaces Boolean reachability and which BFS guarantees still survive?

- Which state spaces have a practical bijective rank/unrank encoding?
- When is full-state storage unavoidable for exact visited membership?
- What deterministic parent tie-break costs are acceptable?
- How should synthetic hash-collision testing be exposed in every backend?
- What state summary gives the strongest inexpensive stopping lower bound for
  asymmetric bidirectional schedules with partial layers and in-flight work?
- Which concrete distributed snapshot, epoch, acknowledgement, or credit
  protocol can expose note 56's global minima without a full layer barrier, and
  what evidence proves that executing and in-flight work is included?
- Can a useful per-edge or per-move lower bound safely retire the remainder of
  a partially expanded depth-`d` vertex before every transition is evaluated,
  or does the intended BFS domain provide no stronger information than `d`?
- How should equality-boundary completion be specified for each of one path,
  deterministic path, all labeled shortest paths, and exact path counts?
- For all-shortest-path output, exactly which equality-boundary layers and
  crossing edges must be completed after the shortest distance is known?
- When is multi-source ownership part of the semantic output, and which
  deterministic tie-break should define it?
- Under which fairness and message-order assumptions is an asynchronous
  relaxation execution observationally equivalent to level-synchronous BFS?
- Which algorithms bearing the name BFS compute distance layers, and which
  compute only a useful vertex ordering?
- For each intended domain, is the reachable component known finite, only
  locally finite, or potentially infinitely branching?
- What finite certificate distinguishes `EXHAUSTED` from a merely bounded or
  interrupted no-target result?
- Which parallel scheduling guarantees are sufficient to prove liveness for
  every finite-depth work item, independently of distance safety?
- Does each application require distances, one replayable path, a deterministic
  path, path counts, the predecessor DAG, or explicit all-path enumeration?
- Which current artifacts explicitly version path identity: vertex sequences,
  labeled moves, generator occurrences, sources, or concrete quotient lifts?
- Can each run report validity independently for distance, arbitrary witness,
  canonical witness, DAG, count, enumeration, and sampling rather than one
  aggregate success flag?
- Which metadata currently treated as diagnostic is consumed downstream as a
  semantic key, especially source labels, move labels, parent order, and
  generator multiplicity?
- For canonical output, what exact producer-completion event proves that no
  better equal-distance proposal remains in a device, buffer, route, or retry?
- For all-path outputs, is truncation represented as `PARTIAL` with a resumable
  cursor/certificate rather than as successful complete enumeration?
- Which CayleyPy consumers interpret `solution_length`, `best_length`,
  `solved_count`, `all_solutions`, or `puzzle_solved=0` more strongly than the
  one replay-valid bounded-beam witness contract established in note 58?
- Can each retained CayleyPy result bind source/dirty-tree, binary/container,
  generator, puzzle/target, K1/K2, and model manifests without relying on an
  external directory convention?
- Which historical CayleyPy artifacts contain enough immutable provenance to
  revalidate the exact packaged word today, and which are candidate-only CSVs?
- Which future exact multi-GPU BFS artifact can expose every note 59 transfer
  obligation: epoch, per-peer record identity, capacity, incumbent reduction,
  global minima, parent reconstruction, and global closure?
- Should exhaustive finite artifacts retain every case/path or a Merkle/digest
  commitment permitting selected historical witnesses to be independently
  reconstructed and checked?
- Which artifact comparisons require byte identity, normalized text equality,
  parsed semantic equality, or tolerance-aware numeric equality?
- What defines distinct paths when parallel edges or duplicate generator labels
  connect the same state pair?
- What arithmetic contract should shortest-path counts use: exact big integer,
  checked fixed width, saturation, or an explicitly requested modulus?
- Can a deterministic global parent rule remain stable across rank counts
  without retaining or communicating every same-depth claimant?
- Which intended domains truly have unit move costs, and which silently optimize
  a weighted cost while still calling the result BFS?
- If zero-cost moves exist, are outputs defined over vertices, simple paths, or
  walks, and can shortest-walk multiplicity be infinite?
- What exact event makes a weighted tentative label final in each proposed
  schedule, especially with messages in flight?
- For multi-source queries, is the required output only distance to the source
  set, one nearest label, a canonical label, or every tied nearest source?
- Must graph Voronoi cells have coherent same-label parent paths, and which tie
  rule guarantees the required connectivity?
- Can canonical equal-distance source-label changes be finalized per complete
  layer without later distributed recoloring?
- When goals form a symmetry orbit, is distance to the orbit actually the
  requested problem or only a lower bound for a fixed-orientation target?

## Implicit graphs

- What is the girth of each exact Cayley/Schreier graph under its actual
  directed, labeled, inverse, loop, and parallel-edge conventions?
- Which earliest duplicate events are trivial inverse returns, distinct-word
  convergence, same-level edges, or hits in strictly earlier balls?
- What shortest reduced identity or stabilizer words can be replayed from those
  collision witnesses?
- How far do the measured spheres equal their regular-tree upper counts before
  the first relation closure, and is the observed threshold proved or sampled?
- Which written presentation relators trace simple shortest cycles, and which
  merely decompose into or imply other shorter relations?
- Can a relation witness explain duplicate multiplicity without falsely
  predicting its warp-local or owner-local physical concentration?

- For each concrete domain, what is the smallest payload that, together with
  declared immutable context, still determines every exact successor? Which
  separate equality, replay, and presentation records discharge the other
  output obligations? The generic role separation is settled in note 06; the
  per-domain inventories remain open.
- What independent oracle can validate successor completeness when the full
  state graph is too large to enumerate?
- For a proposed symmetry quotient, are transitions well defined on classes,
  are distances preserved for the requested targets, and how is a quotient path
  lifted with all orientation metadata intact?
- When duplicate generator actions preserve vertex distances but not labeled
  path multiplicity, which output contract does the application require?
- Can a dense rank be updated directly under each generator, or does carrying
  the full state avoid repeated rank/unrank work?
- What fraction of runtime is generator application versus visited processing?
- How quickly do duplicates arise from inverse moves and short group relations?
- At each depth, what fraction of rejected transitions are same-level candidate
  duplicates versus hits in earlier visited levels?
- How much duplicate traffic remains cross-rank after exact local pre-dedup?
- Can identity and exact duplicate generators always be removed safely from the
  supplied graph contract, or must multiplicity/labels sometimes be preserved?
- Does canonicalization save enough states to repay its per-candidate cost?
- For which Cayley graphs is bidirectional BFS materially smaller than
  single-source BFS after accounting for two visited sets?
- How sensitive is bidirectional work to target choice within the same distance
  layer and to strict-alternating versus smaller-frontier policies?
- On irregular graphs, does an edge-work selector still win after paying for
  frontier reduction and global side agreement?
- What are the candidate-stop and batch-stop work distributions over every
  target in the same distance layer, rather than one order-biased sample?

## GPU

- Does any intended traversal physically batch several original-edge depths,
  and if so where are the logical microlevel boundaries retained?
- Can a longer internal-depth arrival ever perform the authoritative visited
  claim before a shorter arrival from the same superstep?
- Does a k-hop primitive enumerate exact-k walks, all lengths through k, or
  only a selected macro-generator language?
- Which intermediate states must change owner before the same physical
  superstep can continue their expansion?
- What global condition proves that no in-flight internal-depth message can
  produce a shorter target or state label?
- How is each macro-parent expanded into a replay-valid sequence of original
  directed moves and Cayley action conventions?
- Are reported depths original move counts, coarse annulus indices, or distances
  in a power graph?

- For each intended graph, how do actual push edge inspections compare with pull
  predecessor checks, including failed scans and mode-conversion overhead?
- Does the state space provide an affordable enumerable unvisited universe, or
  merely inverse moves and a visited rank?
- Which requested parent/label output removes pull's first-hit early-exit
  advantage?
- In distributed pull, what is the minimal exact representation of global
  frontier membership needed by each owner partition?
- Which part of the per-level work vector predicts wall time for each graph
  presentation, and when does that predictor change across the traversal?
- How should generated implicit transitions be normalized against loaded
  explicit edges without hiding state-generation cost?
- Which races are benign for the application's actual output contract: distance,
  one path, deterministic path, or all shortest paths?
- How should frontier, candidate, scratch, and visited capacity failures be
  surfaced so that no truncated run can resemble a valid smaller traversal?
- At what state width and duplicate ratio does sort/unique beat a GPU hash set?
- Can one radix ordering be reused for both deduplication and multi-GPU owner
  routing, amortizing the 1.4 ms large-batch sort observed in REF-015?
- For non-rankable 64/128-bit states, where is the crossover among global sort,
  warp-local aggregation, and an exact fixed-capacity hash table?
- Should states use AoS, SoA, or a packed rank representation?
- Can generator application and key computation be fused without excessive
  register pressure?
- Which load-balancing method is appropriate when generator count is fixed?
- How much does warp-level equal-key aggregation improve concentrated Cayley
  duplicates without penalizing uniformly distributed keys?
- At what acceptance fraction does block-scan compaction beat one global output
  atomic per accepted state on sm_86?
- Where do QTM and HTM backward, same-level, and forward duplicate occurrences
  co-locate physically under parent-major and generator-major GPU layouts?
- Does HTM's larger same-level stream get rejected cheaply by a resident exact
  visited structure, or does it amplify multi-GPU routing before authority?
- How accurately can a cheap sample estimate equal-key multiplicity *within a
  warp*, and is an online baseline/warp selector cheaper than its saved atomics?
- Which frontier and generator layouts naturally place identical successor
  states in the same warp, and can layout be changed without harming coalescing?
- Can an exact GPU frontier retain discovery/locality order without a global
  sort, despite concurrent output reservations?
- Does rank-sorted frontier reconstruction repay its scan/sort cost through
  bitmap-word locality or multi-stage reuse on later levels?
- How stable are REF-016 locality results across S9/S10 and generator sets with
  longer relations, identities, redundant generators, or same-level edges?
- Does block compaction become profitable at a different block size, state
  width, output record width, or when output reservation includes parent data?
- Should duplicate concentration be summarized by maximum multiplicity, bitmap-
  word occupancy, entropy, or a small histogram usable by an online selector?
- What cancellation granularity is practical for persistent kernels versus
  launch-per-level BFS, and how much already-issued work remains after a hit?
- For level-synchronous exact BFS, which removes host round trips most cleanly:
  persistent kernels, CUDA Graph conditionals, fixed-capacity graph launches,
  or device-side dynamic work queues?
- How should a device-driven traversal resize work for the next frontier without
  losing exact overflow reporting or launching the full capacity every level?

## Multi-GPU

- Is the intended recovery unit a clean completed level or a partial consistent
  cut with channel/message-log state?
- Can any durable visited claim exist without a matching durable `PENDING` or
  `EXPANDED` lifecycle and replayable parent/output metadata?
- Which outputs are idempotent sets/minima, and which require stable
  contribution IDs under retry?
- Does a restart preserve the owner epoch or perform a proved migration to a
  new device count and hash partition?
- Which messages are at-least-once, at-most-once, or coupled transactionally to
  owner-side visited/frontier commits?
- Can every claimed terminated checkpoint prove no pending work, messages in
  flight, failed owner, or hidden overflow?
- Which bounded crash-injection schedule covers every write/acknowledgement
  boundary without building a production recovery system?

- Is the intended benefit strong scaling, weak scaling, capacity scaling, or a
  stated combination, and which fixed costs limit each regime?
- Which objects share ownership—adjacency, full state, visited identity,
  frontier, parent metadata—and which require different placement?
- What timeline evidence demonstrates useful communication/compute overlap
  rather than merely asynchronous API issuance?
- Should ownership be based on a stable state rank, a hash, or a partitioning
  learned from measured skew?
- Can a deterministic owner function preserve generator locality without
  producing pathological per-level skew?
- How should balance, remote bytes, and cross-rank duplicate convergence be
  combined into a meaningful partition-quality objective?
- Which owner strategies remain Pareto-competitive across multiple graph
  families without per-graph salt tuning?
- How much local deduplication is worthwhile before communication?
- When does compressed state exchange cost less than recomputation?
- Which collectives and topology assumptions remain valid across nodes?
- At what frontier work does a scalar side-selection all-reduce repay its
  latency compared with a predetermined alternating schedule?
- How should ranks propagate a target hit while preserving the global shortest
  path proof, and how much communication/work continues before convergence?
- What is the smallest wire record that supports exact intersection and later
  path reconstruction: state only, state plus parent owner/key, or deferred
  second-pass reconstruction?
- Does a two-phase exchange of compact keys followed by metadata only for
  accepted states beat sending full parent records eagerly?
- Can last-level rejection ratios predict the eager/two-phase byte crossover
  robustly enough to choose the protocol without oscillation?
- Does bitmap control traffic remain preferable to returning compact accepted
  indices when peer buffers are small or highly fragmented?
- Can side-specific visited tables share one ownership/index structure without
  causing harmful contention or doubling probe traffic?

## External memory and memory hierarchy

- Which parts of a realistic Cayley state record dominate external traffic:
  canonical state, hash, parent/move, or recovery metadata?
- Can exact equality be resolved within stable hash partitions without repeated
  full-state comparisons across partitions?
- Which output contracts remain idempotent under replay of a partially
  committed level, especially path counts and all-parent DAGs?
- When multiple GPUs spill to one host/storage tier, what event proves that all
  candidate runs for the level have been durably incorporated?
- Which external-memory claims rely on undirected explicit adjacency and do not
  transfer to directed or successor-only implicit graphs?

## Cayley actions and Schreier state spaces

- For each intended CayleyPy puzzle, is the concrete state action free, and if
  not, what is the base-state stabilizer?
- Are move arrays composed on the left or right, and does stored parent-label
  replay use the same temporal/product order?
- Does arbitrary-start normalization target one group element or a full
  `a^-1*H*b` subset, and how would that subset be recognized exactly?
- Which puzzle symmetries are intrinsic stabilizers of a concrete state, and
  which are optional quotients that change the requested target semantics?
- Does a claimed parity invariant descend to state cosets, or does the
  stabilizer contain representatives of both parities?
- For a directed allowed-move alphabet, does the positive monoid reach the same
  states as the group generated after formal inverses are admitted?

## Symmetry quotients and lifting

- Which CayleyPy canonicalizers arise from genuine automorphisms of the exact
  allowed-move graph, including direction and move labels?
- For each quotient, is the desired answer distance to a concrete target or to
  its full symmetry orbit?
- What minimal symmetry frame must accompany a canonical parent record so the
  original move sequence can be replayed?
- Do equivalent representatives expose identical unlabeled neighbor classes,
  identical labeled transitions, or only transitions after label conjugation?
- Can forward and backward quotient paths meet in one orbit but fail to align
  at a common concrete state, and what certificate resolves that alignment?
- Are any proposed quotients graph coverings with unique lifts, or merely orbit
  quotients with several possible lifts?

## Asynchronous and barrier-free schedules

- What fairness model is realistic across GPU work queues, host staging, and
  inter-rank transport: bounded delay or eventual delivery only?
- Which event reactivates a vertex after a smaller owner-side label arrives,
  and can capacity pressure lose that event?
- Can proposals be safely coalesced by `(state,min_label)` before routing while
  preserving the winning parent/version contract?
- What exact global predicate counts device-produced but not yet published work
  during termination detection?
- Can target stopping maintain a trustworthy minimum active/in-flight label
  without recreating level or bucket barriers?
- Should parents be versioned with relaxations or reconstructed after distance
  convergence for a simpler proof?
- How much repeated edge expansion and post-first-hit work can asynchronous
  scheduling introduce on Cayley graphs with many short relations?

## Ordering, parents, and canonical words

- Which output is actually required for CayleyPy: arbitrary shortest word,
  deterministic word, or the shortlex-minimal generator word?
- Is generator ordering part of a stable public puzzle contract, or only an
  implementation detail?
- Can a compact frontier path rank represent shortlex prefix order without
  storing complete words, and how would it remain stable across rank counts?
- How often does first-winner parent selection differ from minimum-parent and
  shortlex-parent selection on the intended Cayley families?
- Do duplicate generator transformations with different names remain distinct
  when defining canonical words?
- Is canonical-parent postprocessing acceptable when it produces a valid
  shortest tree that no ordinary first-winner FIFO order could produce?
- Which reproducibility level is desired: same distances, same parents, same
  frontier byte order, or same replayed move sequence?

## Product states and history-dependent legality

- Which existing CayleyPy move filters are proved geodesic-preserving pruning,
  and which actually define a constrained word language?
- Does any puzzle legality depend on last move, phase, orientation frame, or
  other memory absent from the current visited key?
- For constrained search, is the requested output a shortest accepted walk or
  a simple concrete-state path?
- Can a small DFA encode intended move restrictions, and how many product
  states are actually reachable per base state and depth?
- Do any automata contain epsilon/zero-consumption transitions that move the
  problem from ordinary BFS to closure or 0-1 semantics?
- In bidirectional constrained search, how will forward and reverse automaton
  states certify that the concatenated word is accepted?
- Should product ownership colocate all memory states of one base state or hash
  the full pair, and how does each choice affect semantic duplicate accounting?

## BFS certificates and global graph parameters

- Which intended puzzle graphs are genuinely vertex-transitive under the exact
  allowed generator set, rather than merely transitive as abstract state
  actions?
- For directed Cayley alphabets, is diameter defined only after strong
  connectivity, or are unreachable pairs assigned infinity?
- When reporting a maximum reached depth, is the artifact proving source
  eccentricity, orbit eccentricity, a lower bound, or exact graph diameter?
- Can each same-level conflict retain enough parent data to emit a concrete odd
  cycle rather than only a non-bipartite flag?
- Which workloads need only a non-bipartite flag, one arbitrary witness,
  deterministic witness, or the globally shortest odd cycle?
- For each CayleyPy move graph, is the reported girth about the simple graph,
  labeled multigraph, or reduced generator words with inverse cancellation?
- Does the exact represented graph have enough vertex transitivity to justify
  replacing all-root odd-girth BFS by one identity-rooted traversal?
- In multi-GPU traversal, how are equal-depth conflict lengths reduced only
  after all edge occurrences and in-flight candidates for the level complete?
- Which generic diameter bounds are useful before repeated BFS becomes
  necessary, without mistaking a heuristic lower bound for the exact value?
- In distributed exhaustive BFS, what evidence proves global final-layer
  completion and maximum-depth reduction across every owner?

## Dynamic and temporal graph semantics

- Are any intended CayleyPy graphs mutable during a search, or can every run
  bind to one immutable generator/legality version?
- If generators change, is the desired result a fresh snapshot metric, an
  incrementally repaired metric, or a chronological sequence of allowed moves?
- Which parent/version metadata is sufficient to replay paths after adjacency
  or legality updates?
- Can a batch contain insertions and deletions whose intermediate versions must
  remain invisible to queries?
- For temporal puzzles, is the objective earliest arrival, minimum duration,
  minimum moves, or a weighted combination including waiting?
- How should graph-version installation interact with BFS level epochs and
  distributed candidate messages already in flight?
- Are source insertion/deletion treated as new search epochs, and which old
  distance, parent, tie, count, and completion fields are invalidated?
- Do validators compare exact entering/leaving frontier memberships rather than
  only per-layer counts after a source-set change?
- Which fresh exact Docker BFS oracle will validate a maintained post-update
  result at tractable scale?

## Iterative deepening and memory-bounded search

- Is the intended Cayley workload one shortest target word or exhaustive unique-
  state layers/component geometry? The two favor different memory semantics.
- How much larger is bounded word-tree work than unique Cayley-state work at
  each target depth for the intended generator relations?
- Which transposition entry would be semantically sufficient: minimum reached
  depth, maximum fully searched remaining budget, path-language state, and graph
  version?
- Must pruning preserve any shortest word, the shortlex word, or all shortest
  labeled words?
- Can current-path inverse/cycle pruning capture most trivial repetition without
  a global visited table, and which relations remain duplicated?
- How would a distributed IDDFS certify that every smaller depth limit completed
  globally before accepting a found target as shortest?
- Which metrics make a fair comparison: word prefixes expanded, move
  applications, unique group elements, repeated transpositions, and peak memory?

## Exact BFS, beam, and hybrid-search naming

- In each existing CayleyPy pipeline, is width counted in candidate records,
  unique states, paths, or owners' local states?
- At which exact stages do visited filtering, state deduplication, scoring, and
  top-k selection occur?
- Is the declared beam global, or is it a union of partition-local beams whose
  semantics change with GPU count and ownership?
- Can every capacity/overflow path report an exact dropped count and prevent an
  `exact_bfs=true` label automatically?
- What is the precise domain, radius, generator convention, and graph version
  of each BFS lookup table used by a beam pipeline?
- Does a lookup hit certify only a replay-valid suffix, or is there an
  independent lower-bound argument for global optimality?
- Which outputs are wanted from heuristic ordering without pruning: any shortest
  parent, deterministic shortlex parent, a full ball, or early target latency?
- What tie tuple makes global beam selection reproducible across partition and
  device-count changes?

## Metrics

- Which Cayley states have a proved injective dense rank suitable for exact
  bitmap membership, and which have only hashes or sparse keys?
- For each level, how do `log2 binomial(N,k)`, raw list bytes, bitmap bytes, and
  actual allocated/compressed bytes compare?
- Which phases require enumeration, membership, ordered parents, occurrence
  multiplicity, or only an exact set?
- How much conversion work is hidden when switching queue/bitmap or
  compressed/uncompressed frontier views?
- Do owner-local densities, rank clustering, and communication payloads differ
  materially from the global frontier density?
- Can every representation conversion verify exact membership equality rather
  than only cardinality/popcount?

- Which CayleyPy workloads have a proved spherical growth series or recurrence,
  rather than a fit to observed frontier prefixes?
- Are growth formulas bound to the exact generator multiset, action, legality,
  quotient, and simple-versus-labeled graph convention?
- Can a known coefficient oracle validate every frontier while a separate set
  fingerprint/replay check prevents compensating membership errors?
- How far can a workload remain locally tree-like or lattice-like before its
  first relation changes the apparent recurrence?
- When using a growth formula for capacity arithmetic, which candidate, hash,
  parent, routing, allocator, and scratch overheads remain unmodeled?

- Do any intended workloads genuinely contain hyperedges or synchronized
  prerequisites, or only batches of ordinary binary Cayley transitions?
- If hypergraphs occur, is distance measured in hyperedges, incidence edges,
  weighted activations, or an AND-tail schedule?
- Which outputs retain hyperedge identity and multiplicity rather than only
  scalar vertex distance?
- Can vertex-to-hyperedge and hyperedge-to-vertex phase counters expose skew
  hidden by one combined frontier size?
- Would any multi-GPU partition own vertices and hyperedges separately, and
  what constitutes a complete logical two-phase level?

- Which reported quantities are arithmetic walk mass, Boolean exact-length
  support, cumulative reached states, or first-discovery frontier size?
- For each Cayley depth, how is total `q^d` word mass distributed across new
  states, earlier states, and repeated shortest parents?
- Are random-walk or spectral measurements being used only as geometric bounds,
  or accidentally interpreted as exact BFS completion evidence?
- Do any GraphBLAS experiments declare semiring, mask timing, accumulator, and
  overflow semantics strongly enough to identify the computed object?
- Do they also declare `A[u,v]` orientation, `vxm` versus `mxv`, and transpose
  descriptors, with at least one asymmetric directed validation fixture?
- Do reused outputs request replace semantics, are accumulators null or
  intentional, and can sparse Boolean vectors contain explicit false entries?
- Does termination test stored tuple count or valued Boolean support, and what
  invariant makes those quantities equal?
- When a lazy walk is used for spectral analysis, is its changed step alphabet
  kept separate from the original puzzle move metric?

- For each CayleyPy graph and depth, what are the histograms of backward,
  same-layer, and forward generator occurrences per state?
- Do any intended graphs have a proved intersection array, or does apparent
  uniformity disappear when profiles are checked state by state?
- How much candidate-parent multiplicity variance is hidden by the aggregate
  frontier sequence, and does it correlate with owner or hash partitions?
- Which counts refer to simple endpoint edges versus distinct generator labels
  that reach the same endpoint?
- Can small exact profiles distinguish intrinsic graph irregularity from skew
  introduced by state encoding and multi-GPU ownership?

- For independently specified Cube and Megaminx actions, which shortest proved
  relations predict candidate convergence, alternate predecessors, or
  same-level edges at the earliest exact layers?
- Which observed short relation signatures survive translation and overlap in
  larger spheres, and which aggregate counters hide materially different local
  geometries?
- Can an independent move-action oracle replay the predicted equal words before
  any quantitative claim is transferred from the three REF-022 toy groups?
- For Cube and for Megaminx encodings other than REF-025's unique 120-position
  identity representation, is the move-group action free, and if not, what is
  the shortest independently replayed stabilizer word?
- Which apparent short relations are group identities, which are intrinsic
  stabilizer words of a partial state, and which arise only from an optional
  symmetry quotient?
- Are loops and coincident labeled destinations retained in occurrence counts
  even when the distance oracle uses a loopless simple-state graph?
- At what explored scale should native 64-bit and production 128-bit hash-only
  equality be reported as an explicit collision assumption, and what evidence
  would distinguish assumed absence from verified full-state equality?
- Beyond the F4 generator/conjugated-generator commutators, which next
  geodesic equality needs a genuinely different relation family?
- Can a bounded relation signature propagate static and conjugated
  independence through exact frontiers without pretending to be a complete
  word canonicalizer?
- Which later cross-class equalities are explained by the order-five power
  relations, overlaps of the F4 commutators, or genuinely new short relations?
- Which action and metric properties determine whether power, braid, or
  conjugated-centralizer relations are the first to exceed static commutation?
- Beyond Cube QTM F4, what is the first geodesic equality not explained by
  static commutation and `g^2=g^-2`?
- Can the clean DeepCubeA geometric move implementation be compared by an
  explicit coordinate conjugacy against the CayleyPy cycles without importing
  either implementation into the other?
- Does the CayleyPy repeated-color Cube action remain free globally, or what is
  its shortest nontrivial stabilizer word beyond the REF-029 length-eight
  exclusion?

- For every reported `V` and `E`, do they denote the ambient graph, reachable
  component, stored adjacency entries, unique state edges, or generated labeled
  occurrences?
- Which workloads charge state generation and exact equality as constant-time
  primitives, and which need byte/operation-sensitive costs?
- What is the requested output-size lower bound: scalar distance, one path,
  full distance map, parent tree, predecessor DAG, or all paths?
- What are peak frontier, candidate, visited, output, scratch, and communication
  bytes separately?
- Which reported rates use actual event counters and which use a normalized
  graph-volume numerator such as Graph500 TEPS?
- Can work, dependency depth, communication, and capacity be reported without
  collapsing them into one throughput number?

- Which graph-family properties explain the measured sphere sequence: known
  growth function, isoperimetry, bottlenecks, relations, or finite saturation?
- Can short generator relations predict when word-tree growth first diverges
  materially from unique-state frontier growth?
- Which statistic best separates edge boundary, unique outside-vertex boundary,
  and convergence multiplicity for hardware interpretation?
- Generated transitions/s.
- Unique states accepted/s.
- Duplicate and previously-visited ratios.
- Bytes generated, sorted, stored, and exchanged per accepted state.
- Time breakdown by generation, visited, compaction, communication, and barrier.
- Peak memory and capacity headroom at every depth.

## CayleyPy production beam audit follow-ups

- Is the Zobrist `Hash128` mapping proved injective for any supported puzzle
  state domain, or is collision risk explicitly accepted as probabilistic?
- For which actual puzzle/depth/configuration does
  `global_beam_width_effective` first discard an otherwise eligible unique
  child state?
- Do single- and multi-GPU runs retain identical survivor identities for every
  equal-score tie, not merely identical counts and replay-valid solutions?
- Which wrappers and result consumers distinguish “BFS-built goal
  neighborhood” from the outer beam algorithm in their terminology?
- Does any downstream claim require globally shortest solutions, or is a valid
  replayed solution with a reported length the complete output contract?
- In neighborhood/suffix mode, what lower bound—if any—would justify stopping
  at the first retained prefix depth when suffix lengths differ?

## Non-backtracking and word-normalization questions

- Which intended generator sets are symmetric, and where is the exact inverse
  involution recorded, especially for involutory moves?
- Is inverse-move suppression ever applied by an external wrapper or generator
  table even though it is absent from the inspected GPU candidate indexing?
- What is the shortest nontrivial reduced relation, or action stabilizer word,
  for each puzzle and generator convention?
- Through what radius is the reduced-word-to-state map actually injective, and
  how does that compare with measured frontier growth?
- Are any reported “pruned history” byte counts being mistaken downstream for
  eliminated search candidates?
- Would a requested output concern ordinary state paths or the genuinely
  history-sensitive non-backtracking product graph?

## Reverse-BFS goal-neighborhood questions

- Where, if anywhere, are loaded generator rows validated as in-range
  permutations before inverse predecessor construction?
- Is collision-free K1 identity a probabilistic engineering assumption, or is
  injectivity proved for any concrete puzzle state domain?
- Does each K1 artifact record target/generator/Zobrist versions strongly enough
  to prevent lookup against a mismatched table?
- Can a current configuration produce more genuine hits in one depth than
  `SOLVED_RESULT_CAPACITY`, and what output claim follows on overflow?
- Does the actual K1 artifact satisfy the complete exact reverse-ball and
  shortest-suffix premises, and does K2 scan every length including zero in
  nondecreasing order? Under these premises first hit already minimizes the
  combined residual (corrected notes 40 and 42).
- If an actual trace has a later-shorter `K2 + K1` residual, which premise was
  violated: exact identity, ball completeness, stored suffix distance, the
  empty word, or enumeration order? Such a trace cannot refute the exact-ball
  theorem while simultaneously satisfying all its premises.
- In any nonbijective implicit workload, how would complete predecessor
  enumeration be established rather than assumed from one inverse function?

## Distance-certificate questions

- Which existing BFS artifacts contain labels as well as parents, so the
  decreasing-witness and edge-feasibility halves can be checked independently?
- Which validators currently apply an undirected absolute-level rule to a
  directed, positive-generator, or otherwise asymmetric graph?
- For bounded searches, is the permitted open boundary recorded explicitly, or
  can a missing successor below the radius look like intended truncation?
- What independent move interpreter can check implicit successor completeness
  without sharing the production generator's mistakes?
- Can every distributed validator obtain exact remote endpoint labels under a
  fixed owner/version epoch before reducing violation counts?
- Do failed validations retain the first concrete edge/parent witness, or only
  an aggregate Boolean/count?
- Which richer outputs need additional validation beyond exact distances:
  canonical parents, all predecessors, labels, or path counts?

## Bounded lookup and negative-result questions

- Which current table APIs distinguish an exact miss from incomplete or
  version-mismatched `UNKNOWN`?
- Do any result consumers translate “not in K1” into unreachable rather than
  merely farther than the configured radius?
- What artifact proves every requested K1 layer and every K2 word length
  completed without dropped work?
- Are K1/K2 negative bounds ever persisted, or is the lookup used only for
  positive solution detection?
- Can a forced `Hash128` collision fixture demonstrate both direct false hits
  and descendant false misses during table construction?
- Does positive-result overflow affect only best-suffix selection, or can it
  also make downstream code report a misleading negative/complete status?
- Which distributed flags must be reduced together with `found` to make a
  global bounded miss meaningful?
- Where is the searched outer-frontier identity recorded so a local residual
  exclusion cannot be mistaken for an original-source distance lower bound?

## CayleyPy K1/K2 test-evidence questions

- Is there an external or manual test that invokes the real host K1 builder and
  compares every state/distance/suffix with an independent small oracle?
- Has a nonempty K1 suffix ever been reconstructed and replayed in a retained
  targeted artifact?
- Is there a fixture using non-involutory generators that distinguishes inverse
  direction and multi-move composition order?
- Can existing build reports be tied to immutable container images, commits,
  commands, GPU models, and raw logs rather than dated summary files alone?
- Which test, if any, carries one actual candidate continuously through
  Stream1/2/3/4, global selection, materialization, history, suffix append, and
  full-state replay?
- Are first- versus second-bucket K1 lookup, matching-fingerprint/wrong-hash,
  table-capacity failure, and solved-result overflow covered outside the
  inspected unit test?
- What current Docker run would establish parity of base-generator and composed
  K2 backends across all production-generated suffixes?

## Explicit-paper to implicit-Cayley transfer questions

- Which intended implicit state spaces have a proved dense rank that makes
  bitmap visited genuinely comparable with indexed CSR BFS?
- Can any workload enumerate all unvisited states cheaply enough for pull, or
  do inverse generators only answer predecessor queries for known states?
- What fraction of an implicit transition's bytes and time belongs to state
  transformation, legality, canonicalization, key creation, collision
  resolution, and frontier materialization?
- Which throughput numerator should be primary for each claim: generated moves,
  valid moves, unique candidates, accepted states, or completed exact levels?
- Can a paper comparison match graph direction, multiedge/self-loop convention,
  parent output, stopping scope, and exactness before comparing rates?
- How do communication records change when a compact local vertex ID is
  replaced by hash plus full state and parent/move metadata?
- Which observed Cayley frontier regimes resemble low-diameter scale-free
  explicit graphs, and which are structurally outside that evidence?
- What timeline evidence separates transition compute, exact identity,
  communication, and per-level barriers on one versus many GPUs?

## Exact implicit GPU representation questions

- Which intended Cayley or puzzle domains have a proved rank covering every
  valid state component, including orientation, parity, quotient, and history?
- Is `unrank` needed only for validation/output, or would a proposed pull phase
  require enumerating and reconstructing the full candidate universe?
- When a source says "one-bit" or "two-bit BFS", what does each bit mean at
  every level boundary, and which separate structure stores the frontier?
- Which graph or move-schedule theorem prevents a recycled old-layer bit from
  suppressing a genuinely new state?
- Does operator or move-alternation preserve at least one shortest word for
  every semantic state, or only reduce the raw word tree heuristically?
- For a GPU state hash table, are full keys retained, how are collisions
  forced in tests, and what status is returned on full-table/probe exhaustion?
- Can rank/update cost be measured separately from move application, visited
  access, frontier materialization, and output metadata?
- Which abstractions are small enough for a dense bitvector while the full
  puzzle state universe is not, and exactly which distance heuristic do they
  certify?

## Expansion and memory-pressure questions

- For each intended Cayley generator set, is any vertex/edge expansion bound
  proved globally, or are only identity-rooted frontier traces available?
- When an expansion claim is spectral, which Laplacian/random-walk convention,
  regularity assumptions, and degree losses connect it to vertex boundaries?
- What is the largest observed or proved `|S_d|/|V|`, and which exact frontier
  record/output fields turn that ratio into a memory lower bound?
- How far can boundary edge occurrences exceed distinct outside endpoints at
  each level, and where do those duplicates converge physically?
- Does a locality-preserving multi-GPU partition align with graph subsets for
  which expansion forces large cuts, or is ownership essentially hashed?
- Which owner-crossing metric counts candidate occurrences versus distinct
  state identities before and after authoritative deduplication?
- For bidirectional searches, what fractions of the graph are contained in the
  two completed balls when the stopping lower bound first closes?
- Can an abstraction have excellent expansion and low diameter while its full
  state payload makes the forced linear frontier infeasible in device memory?

## Work-span and parallelism questions

- What is the per-level profile of transition, identity, compaction, output,
  routing, and collective work rather than only their run totals?
- Which frontier levels underfill one GPU, and which become limited by memory
  capacity or exact-identity contention rather than parallelism supply?
- What primitive access model supports each span claim: local successor oracle,
  explicit adjacency, global bitmap/matrix, symbolic relation, or preprocessing?
- How much of each level's measured component time lies on the actual critical
  path rather than merely overlapping elsewhere in the timeline?
- At what GPU count does `W/P` fall below the measured per-level dependency and
  communication span for a fixed state space?
- Does k-hop fusion reduce only physical launches/rounds, or also change work,
  candidate materialization, and in-flight logical-depth storage?
- Which output boundary ends the critical path: first candidate hit, globally
  final distance, completed target layer, completed ball, or exhaustion?
- When latency scaling saturates, how much additional exact capacity is still
  gained from aggregate device memory?

## Frontier-separator and exhaustion questions

- Which retained artifacts prove exact frontier-set coverage rather than only
  counts, hashes, or replay of selected positive paths?
- At each checkpoint, is the stored frontier the last completed sphere or the
  fully committed next sphere, and which candidate buffers belong to the cut?
- Can a crash leave a visited claim on the reached side while its only expansion
  record is absent from the durable frontier side?
- Which beam/pruning claims, if any, have an independent theorem that the
  retained subset intersects every relevant source-to-goal path?
- Are local empty queues reduced only after all send, receive, spill, and
  owner-claim phases can no longer reactivate the next frontier?
- For directed searches, are separators and reverse separators formed under
  the correct arc orientations and graph versions?
- How far are BFS metric spheres from minimum source-to-remaining-region cuts in
  the intended Cayley/puzzle graphs?
- Can a compact exact set certificate accompany a large external or distributed
  frontier without relying only on probabilistic fingerprints?

## Pattern-database and abstraction questions

- Which concrete puzzle fields would each proposed abstraction forget, and can
  every concrete move be projected with no greater abstract cost?
- Does the abstract target represent the fixed requested configuration or a
  broader orbit that intentionally weakens the lower bound?
- Are abstract reverse edges generated under the correct directed convention,
  especially when allowed forward generators are not inverse-closed?
- Which tables are full PDBs and which are completed radius-capped tables with
  `R+1` as their strongest safe miss value?
- Can collisions, table capacity, packing, or missing shards ever turn an
  unknown/missing abstract state into an overestimated heuristic entry?
- When several PDBs are combined, is `max` used, or is every concrete move cost
  explicitly partitioned to justify a sum?
- Which abstract edge costs become zero after cost partitioning, requiring
  0-1 BFS rather than an ordinary unit BFS builder?
- Is an abstract path used only as explanatory lower-bound evidence, or is code
  incorrectly attempting to replay it as a concrete suffix?
- Could a dense replicated PDB fit per GPU, and if not, what exact semantics do
  remote/sharded cache misses have?
- Are learned beam scores and exact admissible PDB bounds logged as different
  fields so ranking evidence is not mistaken for a pruning proof?

## Bound-certified heuristic-search questions

- Which fields are empirical ranking scores versus proved admissible lower
  bounds, and are they named differently in artifacts and telemetry?
- Is every incumbent used for pruning a concrete replay-valid solution under
  the same graph, target, move-cost, and version contract?
- Does the requested output permit pruning `g+h=U`, or are all optimal paths,
  parents, counts, labels, or secondary ties required?
- When multiple prefixes reach one visible state, is higher `g` genuinely
  dominated, or does omitted history change future legality or output identity?
- If a heuristic is admissible but inconsistent, which mechanism reopens a
  closed state after a better `g` and updates its parent/version safely?
- What exact event finalizes a goal: generation, owner claim, min-key removal,
  completed bucket, or global lower-bound reduction?
- In multi-GPU search, does the minimum-open-bound reduction include device
  queues, host staging, network messages, spill files, and pending owner claims?
- How many records were ordered, bound-pruned, top-k/capacity-dropped, reopened,
  or processed after the first incumbent?
- Can a concrete K1 suffix improve the global incumbent while a PDB/K1 miss
  provides a separately logged lower bound for the same retained state?
- Where does a purported BFS run begin scheduling across hop layers by `g+h`,
  making A*/best-first terminology and proof obligations more accurate?

## Owner-hashing and load-balance questions

- For each frontier, how close are owner counts to an independently uniform
  multinomial baseline, and which rank/generator correlations explain deviations?
- How many owners are necessarily idle when the frontier is smaller than the
  world size, and which other move-level work remains available locally?
- Are capacity margins chosen from per-rank peak/tail evidence or only from
  global totals divided by `P`?
- Which mapping best balances the vector of frontier, receive, accepted, visited,
  scratch-byte, remote-byte, and wall-time maxima under predeclared constraints?
- How does increasing `P` move equal-child convergence from source-local dedup
  to destination-owner dedup for each puzzle/generator family?
- Which owner mappings preserve useful parent-child locality, and which merely
  make remote fractions approach the independent `1-1/P` baseline?
- Do topology-weighted paths reveal a different bottleneck than uniform remote
  record counts, especially across nodes or shared links?
- What immutable fields define an ownership epoch, and how are visited/frontier
  authorities migrated or restarted when any field changes?
- Could a deterministic two-choice mapping remain state-stable without a global
  live-load directory, and what balance/locality trade would it actually prove?
- Are tuned salts or rank ranges evaluated on held-out graph families rather
  than selected separately on every measured frontier trace?

## Visited-replica and advisory-filter questions

- Which replicas contain only authority-confirmed exact state IDs, and which
  are probabilistic fingerprints or Bloom filters?
- Can any replica bit be set speculatively before owner acceptance, destroying
  the sound-subset invariant?
- For each cache result, is the permitted action explicit: early drop, route,
  exact fallback, prefetch, or priority only?
- Does early duplicate filtering preserve required all-parent, path-count,
  label, deterministic-parent, and product-state metadata?
- How many extra candidate records arise from stale exact-replica negatives,
  and how do those bytes compare with replica update traffic and memory?
- Are Bloom positives confirmed authoritatively before any candidate is removed
  from an exact frontier?
- In bidirectional search, can a replica hint trigger only candidate meeting
  validation, never global optimal termination by itself?
- Which messages can create authoritative work and therefore participate in
  termination detection, versus advisory replica updates that may lag safely?
- Are cache namespaces tied to graph, move, identity, world-size/owner, source,
  target/direction, and visited-generation epochs?
- After restart or repartition, what evidence proves that no old positive bit
  survives under a new search identity?

## Uniform shortest-path sampling questions

- Is the intended sample space vertex sequences, edge-labeled paths, generator
  words, source-labeled paths, quotient paths, or concrete lifted replays?
- Does every shortest predecessor edge contribute exactly once despite local
  dedup, cross-rank convergence, retry, and checkpoint replay?
- What maximum count bit width occurs in intended Cayley/puzzle targets, and is
  exact big-integer overflow handled explicitly?
- If approximate/logarithmic counts are used, what distributional error rather
  than exact-uniform claim is reported?
- How is an unbiased random integer drawn below an arbitrary `sigma(v)` without
  modulo or floating-point bias, and how is the seed recorded?
- For multiple sources, should sampling be uniform over all shortest paths or
  first uniform over nearest sources and then conditional on a source?
- Do duplicate move labels count as different generator words even when state
  sequences coincide?
- What concrete-lift multiplicities are needed before sampling through a
  symmetry quotient or PDB abstraction?
- Which fixed bidirectional distance cut makes every shortest path cross exactly
  once, and are connector weights prefix-count times suffix-count?
- Can a sampled path retrieve sharded counts and labels without changing the
  probability distribution or accepting stale/incomplete layer totals?

## Implicit successor-completeness questions

- What source independently defines every allowed generator label, endpoint,
  legality rule, multiplicity, and action direction?
- Which CPU/GPU validators share permutation rows, canonicalization, hashes,
  composition helpers, or packed-state code with production?
- Can every bounded validation batch record exact coverage of
  `(parent_index,generator_index)` rather than only `N*q`?
- For partial moves, are exceptions/timeouts recorded as `UNKNOWN` rather than
  silently classified as illegal?
- Where are loaded generator rows checked for range, bijectivity, inverse
  composition, and agreement with independently declared physical moves?
- Which small puzzle instances can exhaustively compare labeled successor
  multisets for every valid state and exercise packing-width boundaries?
- Does the validation suite fail when one generator is omitted, duplicated,
  reversed, or mapped to a wrong but still bijective permutation?
- Can fused kernels expose validation-stage counters before goal, visited, and
  compaction filtering hide unevaluated work coordinates?
- Which per-peer or per-record identifiers strengthen multi-GPU conservation
  beyond equal aggregate sent/received counts?
- What specification-to-code or code-generation argument could extend finite
  evidence to intended large domains without claiming more than it proves?

## BFS landmark and distance-coordinate questions

- Which repeated query workloads need exact pair distance, a certified lower
  bound, a certified upper path, or merely a rejection beyond a threshold?
- How much do candidate landmark coordinate vectors actually separate pairs,
  rather than only spreading individual vertices away from the landmarks?
- In a directed implicit graph, can both forward and transpose successor
  contracts be independently validated?
- Which CayleyPy state spaces are genuine group-element Cayley graphs, and
  which are Schreier/quotient actions requiring distance to a stabilizer coset?
- For a bounded identity table, how should a query expose `>k`, unreachable,
  and representation/lookup failure as distinct outcomes?
- Can a replayable landmark detour be produced from retained parent evidence
  without confusing that upper-bound walk with a shortest path?

## Resolving-set and metric-dimension questions

- Which CayleyPy tasks actually need state identification from distances, as
  opposed to bounds for already represented states?
- How many collisions remain after each proposed landmark coordinate, and are
  they caused by automorphisms or by unrelated metric coincidences?
- Can a candidate landmark tuple be certified resolving on a finite orbit
  without confusing exhaustive vector uniqueness with a theorem for other
  puzzle sizes or generator sets?
- For directed state graphs, should resolution use distances to landmarks,
  from landmarks, or both, and how are unreachable coordinates represented?
- When an identity table is bounded, what partial-identification statement is
  justified for vectors containing unknown coordinates?
- How does quotienting by puzzle symmetry change metric dimension and the
  meaning of the vertices being identified?

## Distance-embedding and strong-resolution questions

- For a proposed Cayley landmark set, which pairs are merely distinguished and
  which attain equality in the maximum coordinate difference?
- What contraction distribution remains after a landmark tuple passes an
  exhaustive injectivity test on one finite orbit?
- Can subset-distance coordinates provide useful separation that singleton
  coordinates miss, and what independent multi-source fields are then needed?
- Which exact output is desired: identity of a state, a lower bound on its pair
  distance, or isometric recovery of every tested distance?
- How should a directed Cayley or Schreier workload represent asymmetric
  distance without importing an undeclared symmetrization?
- At what point does storing many BFS coordinate fields cost as much as the
  distance information the representation was meant to avoid?

## BFS-tree stretch and parent-geometry questions

- Which consumers need only one replayable root path, and which incorrectly
  treat the parent tree as a pair-distance oracle?
- How much do different valid parent tie policies change edge stretch, LCA
  depths, and replay locality on the same Cayley frontier?
- Are lateral and non-parent predecessor edges retained anywhere when later
  pair queries need them?
- Does a distributed checkpoint promise arbitrary valid parents, deterministic
  parents, canonical normal forms, or stable tree geometry across replay?
- Which short Cayley relations connect elements whose selected normal forms
  diverge near the identity and therefore induce large tree stretch?
- If an approximate pair-distance structure is desired, what explicit spanner
  guarantee replaces the much weaker shortest-path-tree guarantee?

## Fundamental-cycle and relation-witness questions

- Which non-tree transitions first appear at each Cayley BFS depth, and are
  they same-layer odd witnesses or alternative-predecessor even witnesses?
- How much do parent tie rules change fundamental-cycle lengths and the words
  attached to identical non-tree transitions?
- Which labeled parallel edges or loops disappear if an application exports
  only a simple undirected state graph?
- Are recorded cycles intended as binary cycle-space evidence, ordered replay
  walks, identity words, stabilizer words, or candidate defining relators?
- Can every distributed cycle witness retrieve both parent chains and the
  closing labeled transition from one consistent checkpoint epoch?
- Which short relation families generate many longer fundamental words only by
  conjugation and binary combination, and which genuinely add new structure?

## Fundamental-cut and bridge questions

- Which BFS-tree edges have large fundamental cuts, and how sensitive are they
  to parent ties despite fixed distance labels?
- Are any apparent bridges artifacts of a bounded ball, missing generator
  labels, delayed messages, or a quotient/simple-graph projection?
- Which metrics refer to parent-subtree boundaries, completed frontier cuts, or
  cross-owner communication, and are they named separately in logs?
- Can cycle-cut parity checks catch missing or duplicated transition records in
  a finite audited graph without sharing the same edge-identity bug?
- Do selected Cayley normal-form prefix subtrees resemble useful algebraic
  regions, or is that resemblance destroyed by short relations?
- What evidence would prove a bridge statement for an infinite implicit graph
  without claiming that finite exploration is exhaustive?

## Strong-component and condensation questions

- Which directed CayleyPy alphabets are inverse-complete in traversal, merely
  group-generating with implicit inverses, or genuinely positive-only?
- Can forward and transpose successor oracles be validated independently over
  the same exact state identity and graph version?
- Which outputs need one root SCC, all SCC labels, condensation reachability,
  or original directed distances inside components?
- Are condensation depths ever being reported as original move depths or as a
  topological ordering without the needed qualification?
- For an infinite positive Cayley alphabet, what subgroup
  `<S>_+ intersect <S>_+^-1` and what component quotient actually arise?
- How are finalized SCC IDs and cross-component arcs made consistent across
  owners before constructing a distributed condensation DAG?

## Directed-period and cyclic-class questions

- What periods arise from CayleyPy's directed positive alphabets before inverse
  symmetrization, and which generator relations determine their GCD?
- Can exact depth-slack GCDs be cross-checked against independently replayed
  short directed cycles or adjacency-power support?
- Which missing generator labels would falsely increase an observed period,
  and which stale distances would introduce an invalid smaller divisor?
- Do any state quotients or stabilizers destroy the group-level word-length
  homomorphism while leaving a valid cyclic class on the Schreier graph?
- Is a reported parity phenomenon really bipartiteness, directed period two,
  generator-sign parity, or a bounded-depth coincidence?
- How should cross-owner arcs be assigned so a global slack-GCD reduction has a
  complete, non-overclaiming coverage certificate?

## Eventual-walk and primitive-exponent questions

- For small CayleyPy state graphs, what gaps remain between shortest word
  length and eventual completeness in the permitted residue?
- Which workloads actually request exact-length support or word counts rather
  than first-discovery state frontiers?
- Are any permanent visited structures being reused across those two contracts
  without preserving later revisits?
- How different are diameter, pairwise conductors, and primitive exponent on
  representative directed generator alphabets?
- Does inverse padding exist as legal transitions, or only as an algebraic
  operation outside the positive search alphabet?
- What bounded-step or period-detection evidence should replace quiescence for
  distributed exact-length propagation?

## Walk-versus-simple-path questions

- Which CayleyPy outputs mean generator words/walks, state-simple paths,
  nonbacktracking paths, or geodesics?
- Do any exact-length or constrained searches merge records by endpoint even
  though their used-state histories enable different suffixes?
- What is the true semantic history: all used vertices, used edges, last move,
  an automaton state, or some proved sufficient quotient of these?
- Are relation-padded word spectra being mistaken for simple state-path length
  spectra?
- Which small-`k` investigations justify a parameterized method, without
  turning this study into an unsolicited solver implementation?
- How would a distributed ownership and dedup contract represent `(v,U)` or a
  smaller proved-equivalent history state?

## Trail and edge-history questions

- Does each workload forbid repeated directed arcs, undirected semantic edges,
  generator labels, or labeled transition occurrences?
- Which inverse Cayley transitions share one undirected edge ID across devices?
- Are parallel labels collapsed for state BFS but required to remain distinct
  for trail enumeration or Euler coverage?
- Can any proposed line-graph conversion prove orientation-compatible lifting,
  especially for undirected edges?
- Which tasks are genuinely generic exact-length trails and which have Eulerian
  degree structure that changes the problem?
- How could per-history edge exclusions be represented without mistaking a
  global traversal bitmap for semantic visited state?

## Dominator and unavoidable-gateway questions

- Which CayleyPy tasks need a full all-path dominator, and which need only an
  unavoidable vertex in the shortest-path DAG?
- Can independently generated incoming transitions certify that a proposed
  gateway has no longer bypass, rather than merely replaying the same generator
  implementation?
- Which small directed Cayley and Schreier graphs have nontrivial dominator
  trees under asymmetric positive generator alphabets?
- How much can one missing cross-owner arc inflate an apparent dominated region
  while leaving BFS depths unchanged?
- What compact certificate could validate a distributed dominator tree against
  one immutable graph epoch without adopting a production implementation?
- When path counts are used for shortest gateways, how will edge identity,
  parallel labels, reverse transitions, and overflow be made explicit?

## Menger and route-redundancy questions

- What are the local vertex- and arc-connectivity distributions of small exact
  CayleyPy state graphs under symmetric versus positive-only generators?
- How loose are minimum and median intermediate BFS-sphere widths as upper
  bounds on actual source-target separator size?
- Which apparent multiple-parent regions still reconverge through a later
  dominator?
- Do labeled parallel generators create only arc redundancy, or genuinely
  vertex-disjoint state routes?
- Which workload failures correspond to lost states, lost semantic edges, or
  lost physical owners, and therefore require different connectivity notions?
- Can small independently enumerated graphs expose false disjointness caused by
  duplicate representations of one semantic state?

## Postdominator and inevitability questions

- Which CayleyPy targets are intended as absorbing success states, and which
  retain outgoing moves after first arrival?
- Do any reports call a reverse-reachable state "guaranteed solvable" when they
  mean only that one completion exists?
- Which directed puzzle abstractions contain reachable cycles that avoid the
  goal forever under adversarial move choice?
- Is nontermination ignored, modeled as a virtual exit, or constrained by a
  fairness assumption in each workload?
- How often do longer completion paths destroy gateways visible in the reverse
  shortest-path DAG?
- Could a mixed graph epoch preserve reverse distances while manufacturing a
  false postdominator by omitting one longer bypass?
- What evidence distinguishes distributed algorithm quiescence from a semantic
  claim that every path in the state graph reaches the goal?

## Reachability-equivalent representation questions

- Which CayleyPy generators are algebraically redundant but materially shorten
  the declared word metric?
- Are any benchmark variants comparing the same state set under different
  generator alphabets while labeling the change as implementation-only?
- How do reduced, original, and macro-augmented transition sets change frontier
  peaks and synchronization depth on the same small exact state universe?
- Which macro moves carry an exact old-distance weight and replay witness, and
  which silently become new unit moves?
- Can SCC condensation artifacts distinguish their reachability order from the
  particular arc representation used for BFS distance?
- What validation would detect equal final visited sets but incorrect layer
  migration after a generator-set change?

## Word-metric comparison questions

- What are the exact mutual substitution constants between CayleyPy's commonly
  used move alphabets?
- Are those constants uniform across puzzle-size families or growing with the
  instance?
- Which observed changes are coarse growth changes and which are only exact
  sphere/growth-series changes under a new alphabet?
- How much do Schreier stabilizers tighten the group-level substitution bounds
  on actual puzzle-state distances?
- For positive-only alphabets, which inverse or alternate moves have bounded
  positive expansions and which change directed reachability?
- Do macro-parent records retain enough expansion data for replay in the base
  move alphabet without claiming the expansion is geodesic?
- Can geometric radius rescaling be reported alongside, but not substituted
  for, measured candidate, memory, and communication traces?

## Amenability and BFS-boundary questions

- Do CayleyPy's actual BFS balls behave like Følner sets, or do only other
  specially shaped subsets have small boundary?
- What are the vertex-boundary, edge-boundary, and generated-occurrence ratios
  across exact small puzzle layers?
- Which finite puzzle families have a uniform pre-saturation expansion bound?
- How differently do the group Cayley graph and puzzle Schreier orbit expose
  boundary-to-volume ratios under the same moves?
- Does a small distinct-state frontier still generate a large crossing-edge
  bag through labeled duplicates?
- How strongly does the owner partition decorrelate semantic graph boundary
  from actual inter-device traffic?
- At what radii does finite saturation invalidate an infinite-growth analogy?

## Random-walk hitting and coverage questions

- Which CayleyPy uses need uniform samples or heuristic hits rather than exact
  shortest distances and component closure?
- How do target hitting-time distributions compare with BFS depths on small
  exact puzzle graphs?
- After apparent mixing, how many states remain unvisited and where do they lie
  in the BFS-layer decomposition?
- Do labeled parallel moves bias a state random walk relative to a
  unique-neighbor walk?
- What independent total-state or closure evidence would turn a union of walker
  visits into an exact coverage claim?
- How much overlap remains among multiple GPU walkers, and does higher step
  throughput increase unique-state discovery proportionally?
- Are probabilistic confidence bounds ever being reported with deterministic
  BFS language such as complete, unreachable, or exact radius?

## Flooding and message-time questions

- Which multi-GPU paths enforce one logical round per Cayley edge, and which
  allow remote discoveries to arrive after later local work?
- Are first arrivals used only for reachability, or incorrectly frozen as hop
  distances under variable delivery order?
- What loss, retry, acknowledgement, and duplicate-message assumptions exist at
  each communication layer?
- Does a timeout represent performance monitoring or a claimed semantic
  exhaustion certificate?
- Which experiments sample generators/peers and therefore measure rumor/random
  exploration rather than exact all-neighbor BFS?
- If macro transitions have heterogeneous costs, is message time interpreted as
  weighted distance, temporal earliest arrival, or merely runtime latency?
- What global evidence proves no unprocessed or in-flight discovery remains
  when every owner appears locally quiet?

## Graph-end and BFS-separator questions

- Which infinite Cayley groups, if any, are the intended ambient limits of the
  finite CayleyPy puzzle families?
- How many infinite outside components do exact balls expose before finite
  quotient wraparound reconnects or saturates them?
- Can two generator alphabets reveal the same end structure at radically
  different radii and frontier widths?
- Which puzzle Schreier actions change the ambient group's number of ends?
- Are any finite-prefix observations being described as asymptotic end
  structure without a group-theoretic proof?
- How should directed positive-alphabet notions of forward/backward ends be
  declared before using an undirected analogy?
- Does any owner partition align with persistent outside components, or is that
  merely accidental at sampled radii?

## Percolated-BFS questions

- Which random model is meaningful for CayleyPy: missing semantic moves,
  unavailable states, transient communication loss, or deliberately sampled
  search edges?
- What semantic identity pairs inverse orientations into one undirected bond?
- How strongly do short Cayley relations reduce unique frontier growth below a
  tree branching approximation?
- Which finite puzzle sequence, if any, supports a meaningful giant-component
  scaling claim?
- How often does a supercritical parameter still produce an early-extinct root
  cluster in the tested sizes?
- Which frontier and peak-memory quantiles are needed beyond the mean for GPU
  capacity evidence?
- Can paired seeds isolate implementation timing changes from realization
  variability without treating overflow as a sampled closed edge?

## Geodesic-language and automaton questions

- For which CayleyPy generator families is a regular geodesic language known,
  and is the claim uniform or only per finite instance?
- Does any available normal form cover puzzle orbit states uniquely, or only
  elements of the ambient group before quotienting by a stabilizer?
- Which candidate languages are geodesic in the exact move metric rather than
  merely freely reduced or canonical under a rewriting order?
- Is the language prefix-closed, and if not, what state is required to enumerate
  length layers without traversing rejected prefixes?
- How does automaton state count scale with puzzle size and generator choice?
- Can accepted-word counts be compared against independently exact BFS sphere
  counts to expose missing states, duplicates, or nongeodesic representatives?
- Under reversal, do inverse convention and action orientation preserve the
  intended normal-form semantics?
- Would automaton filtering reduce evaluated state candidates in practice, or
  merely move work into control-state transitions and irregular batching?

## BFS phase-building questions

- Which other algorithms use BFS levels as a temporary admissibility proof
  rather than as their final output?
- For an implementation that stops on first sink discovery, which tied
  shortest level edges are omitted from the subsequent phase?
- What exact residual snapshot and synchronization event separates two BFS
  phases on multiple GPUs?
- Can phase-local dead-end information be reused safely after augmentation, and
  what monotonicity proof would justify it?
- How much of each phase is BFS scanning versus blocking/disjoint augmentation,
  mutation conflict handling, and global synchronization?
- Does a wide level graph contain many independent improvements or mostly
  paths converging on a small capacity or vertex bottleneck?
- What evidence proves global blocking when admissible paths and residual
  capacities are partitioned across owners?
- How should incomplete blocking be reported when it preserves correctness but
  loses the strict per-phase distance-growth argument?

## BFS-profile and refinement questions

- Which retained CayleyPy artifacts contain exact frontier identities versus
  only per-depth counts or digests?
- Can richer backward/lateral/forward multiplicity histograms distinguish
  failures that ordinary layer sizes currently hide?
- Which puzzle actions are vertex-transitive under the exact labeled move
  convention, making every rooted profile necessarily identical?
- Are any profile matches currently used as if they proved state-set equality
  or replay correctness?
- What independently exact small instances could calibrate collision rates of
  proposed radial fingerprints?
- Does root-individualized color refinement expose useful state classes in a
  Schreier graph, or merely rediscover already known stabilizer orbits?
- How many roots would a sampled-profile claim need, and what population-level
  statement could it legitimately support?
- When distributed layer counts agree, what exact set evidence still checks
  for compensating missing and spurious states?

## BFS and local-message-passing questions

- Which CayleyPy models receive explicit goal/source markers, move labels,
  coordinates, or precomputed distance features that break graph symmetry?
- Are any embeddings compared as if equality implied exact puzzle-state
  identity?
- Which learned targets are exact distance, bounded lookup, ranking, or
  probability, and how are unknown values represented?
- Does training supervise first-arrival distance or a walk-aggregated signal
  with different semantics?
- At what radii do real puzzle sphere sizes create the sharpest fixed-width
  information bottlenecks, and where do relations suppress tree growth?
- Which long-range tasks genuinely require information from most of a ball
  rather than one recoverable path or sufficient statistic?
- How does an implicit, on-demand Cayley graph alter the materialization
  assumptions of ordinary full-graph message passing?
- What measurements separate feature exchange, state generation, exact
  deduplication, and learned inference across multiple GPUs?

## Succinct and implicit-complexity questions

- What is the compact input parameter for each CayleyPy puzzle family: pieces,
  coordinates, permutation width, generator tables, or serialized bytes?
- How do valid-state count, orbit size, and reachable component size scale
  against that parameter?
- Which successor APIs enumerate named legal moves directly, and which perform
  broader legality or destination searches?
- Which requested outputs genuinely require explicit BFS layers, and which ask
  only reachability, membership, bounded distance, or one path?
- Are there group/action algorithms that answer those narrower queries without
  enumerating the orbit, and what shortest-path information do they lose?
- At what state parameter does doubling aggregate GPU memory buy only one
  additional feasible bit, piece, or coordinate?
- Which symbolic frontier families remain compact empirically, and which
  relations cause representation blow-up?
- Are runtime claims normalized by compact input size or by an already
  exponentially expanded number of state/move records?

## BFS and topological-wave questions

- Which existing pipelines call a dependency-ready wave a BFS level without
  declaring the underlying `max` recurrence?
- Are derived puzzle DAGs graded, or can shortcut and long dependency paths
  reach the same state at different lengths?
- Which DAGs are shortest-path DAGs built from certified distances, and which
  merely store selected parents or evaluation dependencies?
- Does any multi-GPU readiness protocol prove that every remote predecessor
  completion was counted exactly once before release?
- How much nominal wave parallelism is lost to weighted tasks, memory pressure,
  or the last remote predecessor?
- Are cycle certificates global over the complete dependency graph or only over
  a materialized reachable subgraph?
- Which reported level count is minimum search depth, maximum dependency depth,
  or a physical synchronization-round count?
- Can a derived graded DAG expose exact BFS concurrency without being mistaken
  for the original Cayley graph itself?

## Distance-transform and wavefront questions

- Which CayleyPy uses of "distance" mean move count, weighted move cost,
  embedding-space norm, or physical/geometric displacement?
- Are any grid or tensor probes using four-, eight-, or larger-neighborhood
  stencils without naming the induced metric?
- Do diagonal or macro moves carry unit cost because they are one action, or a
  geometric cost based on displacement?
- Which obstacle/corner rules determine whether diagonally touching regions are
  connected?
- Are nearest-source labels and equidistant ties retained, or only scalar
  distances?
- When would a dense dilation sweep outperform a sparse active frontier on the
  same exact ball, and which work count explains the crossover?
- Can word-metric sphere collisions be interpreted as front interference in a
  way that predicts duplicate multiplicity without hiding state identity?
- Which claims invoke fast-marching intuition while actually using a unit-edge
  graph recurrence, or vice versa?

## BFS and union-find questions

- Which CayleyPy tasks need only orbit/component membership, and which require
  minimum move count or a replayable move sequence?
- Are any representative-parent chains treated as if they were legal graph
  paths after path compression?
- For incremental move/edge additions, how many BFS labels change while the
  component partition remains identical?
- Which deletion scenarios require detecting whether a removed move occurrence
  was a bridge in the semantic simple graph?
- Are directed positive-alphabet edges ever unioned symmetrically, thereby
  replacing forward reachability with weak connectivity?
- What completeness evidence proves every relevant implicit edge reached the
  DSU before a negative connectivity answer?
- How are cross-GPU representative races, duplicate union messages, and final
  canonical component labels validated?
- Which benchmark reports component-label throughput under a BFS name despite
  omitting distances and frontiers?

## BFS-tree and MST questions

- Are any unit-weight puzzle spanning trees called "minimum" as if that
  certified geodesic root paths?
- Which retained trees preserve root distances, minimum total edge cost,
  bottleneck paths, or merely connectivity?
- Do weighted generator costs represent action cost, physical time, or a tree
  construction objective?
- Which tie rules determine reproducible BFS parents or MST edges, and which
  downstream artifacts depend on that choice?
- Can a selected Cayley tree have severe root-distance stretch despite optimal
  total unit weight?
- Which validation uses distance/predecessor conditions and which uses MST
  cut/cycle exchange conditions?
- After an edge or weight update, is the required repair for shortest paths,
  minimum total tree weight, or both?
- Are shared sort/scan/union primitive timings being transferred between MST
  and BFS without end-to-end work accounting?

## BFS-distance and effective-resistance questions

- Which CayleyPy state pairs have equal word length but measurably different
  effective resistance because their relation/cycle neighborhoods differ?
- Can exact small Cayley graphs calibrate `R_eff(e,g)` against sphere depth,
  shortest-path multiplicity, and cut size?
- Which generator changes shorten distance but increase or decrease route
  redundancy independently?
- Are random-walk hitting or commute observations being explained by BFS depth
  without accounting for total edges and effective resistance?
- Do any retained layer profiles contain enough cross-layer edge data to bound
  resistance, even though they cannot determine it exactly?
- Which directed positive-alphabet workloads would require a nonstandard
  directed-resistance definition rather than symmetrization?
- How should approximate Laplacian-solver tolerance be validated without
  confusing it with exact BFS state identity?
- Which shared sparse primitives have transferable timings, and which
  end-to-end synchronization and convergence costs remain workload-specific?

## BFS-geodesic and hyperbolicity questions

- Under which precise convention should finite Cayley puzzle hyperbolicity be
  reported: vertices only, unit-edge realization, thin triangles, or four
  points?
- What exact or proved upper/lower bounds are practical for the current small
  Cayley fixtures without mistaking sampled witnesses for exact values?
- How do `delta`, diameter, sphere growth, duplicate convergence, and shortest
  path multiplicity vary independently when the generator set changes?
- Do current Megaminx short commutation and conjugation relations create local
  fat triangles, or do they mainly alter multiplicity inside thin corridors?
- Which graph families provide a meaningful uniform scaling question instead
  of the vacuous statement that each finite puzzle has some finite `delta`?
- Can a stated hyperbolicity theorem justify a particular bidirectional
  corridor bound without assuming small frontiers or unique meeting states?
- Which Schreier actions preserve useful hyperbolicity bounds from an ambient
  group, and what additional hypotheses are required?
- How should directed positive-alphabet reachability be studied without
  silently replacing it by a symmetric word metric?
- If quadruples are sampled on GPU, what sampling design and independent
  distance validation make the resulting lower-bound evidence reproducible?
- Which throughput belongs to repeated distance computation, and which claims
  still require an actual frontier/visited multi-GPU traversal?

## BFS-ball convexity, gates, and Helly questions

- Which current puzzle Cayley or Schreier graphs have convex, weakly convex, or
  gated small-radius balls, and at what first radius does each property fail?
- Can short relation families predict the first nonconvex BFS ball without
  enumerating the whole graph?
- Are any generator choices for the same group Helly while others are not?
- Which finite Cayley fixtures admit a direct four-ball non-Helly witness like
  `C4`?
- When a family of puzzle distance constraints is pairwise feasible, what
  independent evidence would justify inferring a common state?
- How large can the common intersection be even when the Helly property
  guarantees existence, and what additional rule would select one state?
- Which action quotients preserve gated subsets or ball-Helly behavior, and
  which destroy them?
- Can the Helly radius identity be turned into a certified procedure only with
  additional graph-class structure, rather than an unjustified two-sweep rule?
- How should forward, reverse, and round-trip balls be distinguished for
  positive directed generator alphabets?
- In a distributed distance-constraint probe, where is globally consistent
  state identity established before claiming a common witness?

## BFS intervals, medians, and partial-cube questions

- Which small puzzle state graphs have triples with empty, unique, or multiple
  interval intersections?
- At what smallest Cayley radius can current Megaminx relations witness failure
  of median structure?
- Can any useful puzzle subgraph be proved isometric to a partial cube even when
  the complete Cayley graph is not median?
- Which state coordinates correspond to genuine convex cuts, rather than merely
  compact encodings or hash features?
- How does changing generators alter triple medians in the same group beyond
  the `Z2^2` calibration example?
- Which quotient actions preserve an isometric cube embedding, and which merge
  vertices across its cut coordinates?
- When does a weighted graph-median set contain several states despite unique
  medians for every triple?
- Can interval-intersection evidence explain shortest-path multiplicity without
  materializing the entire predecessor DAG?
- What certificate would validate a claimed partial-cube embedding independently
  of the code that constructed it?
- Which GPU measurements concern coordinatewise majority, and which still
  require complete BFS frontier and visited-state work?

## BFS layers and weak-modularity questions

- What is the first TC or QC violation in the current small puzzle Cayley
  fixtures, and can it be expressed as a short relation witness?
- Does Cayley translation genuinely reduce a proposed audit to one root under
  the exact implemented left/right action convention?
- Which successor omissions or hash collisions could fabricate a TC/QC
  violation?
- Can a compact certificate prove absence of every qualifying lower common
  neighbor without trusting the same generator implementation under test?
- Which generator additions move a fixed group between median, modular, weakly
  modular, and unrestricted graph classes?
- Which Schreier stabilizers create multiple medians or destroy TC/QC after an
  ambient Cayley graph satisfied them?
- Are any bounded-radius local-to-global claims applicable only after proving
  simple connectedness of an associated triangle-square complex?
- How many candidate TC and QC diagrams occur per BFS layer, independently of
  frontier size and raw successor count?
- In multi-GPU checking, how is a global absence result distinguished from a
  missing owner response or incomplete partition?
- Which measurements belong to graph-class auditing rather than the ordinary
  BFS hot path?

## BFS frontier and treewidth questions

- What treewidth or lower-bound evidence exists for the actual finite puzzle
  Cayley graphs, rather than for an encoding or move-interaction graph?
- Can small-radius implicit balls have low treewidth while still containing too
  many states for exact visited storage?
- Which BFS roots align with any known low-layered-width decomposition, and
  which do not?
- Does a proposed decomposition cover every implicit edge and satisfy the
  running-intersection condition under exact state identity?
- How much vertex replication appears if frontier states are assigned to
  overlapping decomposition bags?
- Which generator additions increase treewidth without proportionally changing
  early BFS frontier size, or vice versa?
- Do any Schreier quotients lower treewidth while introducing stabilizer-driven
  duplicate pressure?
- Can separator evidence yield a useful capacity bound without materializing a
  full tree decomposition?
- Which planar or minor-free local-treewidth theorems, if any, apply to a real
  puzzle state graph rather than an unrelated physical drawing?
- Are bag-DP throughput and frontier-BFS throughput being measured and named as
  separate workloads on one and many GPUs?

## BFS, LexBFS, and chordal questions

- Which ordinary BFS tie orders on small chordal fixtures reverse to PEOs by
  accident, and which fail like the diamond example?
- Can a failed PEO check be reconstructed into a compact induced-cycle witness
  under an implicit successor oracle?
- How can absence of a later-neighbor edge be independently certified when the
  graph is generated rather than stored explicitly?
- Are any proposed puzzle graphs being called chordal after adding auxiliary
  fill edges that changed their word metric?
- Which fill edges shorten BFS distances or alter shortest-path multiplicity in
  those auxiliary completions?
- Does an implemented "LexBFS" preserve dynamic label histories, or only sort
  adjacency once inside FIFO layers?
- For finite Cayley puzzles, does any noncomplete chordality claim contradict
  the vertex-transitive simplicial-vertex argument because the actual graph is
  directed, has quotient identity, or is not the claimed Cayley graph?
- Which infinite Cayley examples separate chordality from completeness and from
  finite puzzle behavior?
- What communication is required to verify later-neighbor cliques under a
  distributed vertex partition?
- Which measurements belong to LexBFS/PEO recognition rather than ordinary BFS
  frontier expansion?

## BFS and distance-hereditary questions

- What is the smallest induced non-distance-hereditary obstruction in current
  puzzle Cayley and Schreier fixtures?
- Can a short relation witness be converted into an induced hole, house, gem,
  or domino certificate?
- Which retained puzzle subsets are genuinely isometric even though the full
  graph is not distance-hereditary?
- Does an alleged pruning sequence check twins after every earlier removal, or
  only compare neighborhoods in the original graph?
- Which true/false twin states remain distinct named outputs or carry different
  parent/path multiplicities?
- What quotient outputs admit a complete lifting proof, and which require every
  original state to remain in visited?
- How do generator changes move a fixed finite group into or out of the
  distance-hereditary class?
- Which Schreier stabilizers create twin states that do not exist in the ambient
  Cayley graph?
- In a distributed obstruction audit, how is global absence distinguished from
  an incomplete owner response?
- Are structural compression and BFS frontier throughput reported as separate
  measurements?

## BFS spanner, emulator, and hopset questions

- Which current generator alphabets contain moves with short exact words over a
  smaller retained alphabet?
- Are those replacement words group identities or only state-specific Schreier
  coincidences?
- What worst-case generator replacement length follows algebraically, and how
  loose is it versus observed pair stretch?
- Which retained generator subsets disconnect the state graph despite looking
  adequate inside a shallow BFS ball?
- How do fewer generator edges trade against extra BFS depth, frontier migration,
  and synchronization rounds?
- Which virtual emulator or hopset edges retain replayable original move
  witnesses?
- Can an approximate path upper bound be paired with a completed exact layer or
  another independent lower bound to certify optimality?
- Do long-range shortcut edges reduce rounds while increasing owner-routing
  volume or skew in multi-GPU execution?
- Are weighted shortcut relaxations being incorrectly timed or described as
  ordinary unweighted BFS?
- Are preprocessing, approximate query, unpacking, and original exact traversal
  reported as separate workloads?

## BFS doubling-dimension and metric-net questions

- What lower bounds on doubling constant follow immediately from current puzzle
  generator degrees?
- At which radii do identity-ball packing or cover numbers first exceed those
  local degree bounds?
- Can short-relation convergence lower cover numbers without proportionally
  shrinking exact frontier size?
- Which generator changes alter the numerical doubling profile most strongly?
- Are measured covers complete over the exact ball, or only over sampled states?
- What family of increasing puzzle instances would make a uniform doubling
  claim nonvacuous?
- Can a hierarchical net support a useful approximate query while retaining
  replay witnesses for original moves?
- Which output contracts, if any, permit net-center aggregation without losing
  exact state identity?
- In multi-GPU cover validation, how is a genuinely uncovered point separated
  from a missing owner contribution?
- Are cover/packing throughput and original frontier/visited throughput reported
  as different workloads?

## BFS replacement-path and fault-tolerance questions

- Which future use cases need one failed edge on one chosen path, all single
  graph-edge failures, vertex failures, or whole move-label failures?
- Is a reversible physical move failure modeled as both directed orientations,
  and is that convention stable across artifacts?
- Which current Cayley generator removals preserve connectivity, and which split
  the action into proper subgroup orbits?
- Are globally valid generator replacement words available, or only state-local
  Schreier detours?
- When a selected parent edge fails, can the complete predecessor DAG certify
  an equal-length replacement before any new traversal?
- Which replacement paths necessarily leave the original shortest-path DAG?
- What independent lower-bound evidence certifies a returned detour as shortest
  in the surviving graph?
- Can a surviving-cut certificate prove disconnection without rerunning full
  reachability, and what complete adjacency evidence would it require?
- Which exact FT-BFS guarantees are worth their worst-case redundancy, and which
  consumers explicitly permit stretch?
- In a batch of failure scenarios, how are graph epochs and failure identities
  kept separate through deduplication and distributed ownership?
- Is failed-owner recovery ever being misreported as deletion of owned states?
- Are preprocessing, retained memory, per-scenario traversal, worker recovery,
  and exact validation measured as separate workloads?
- Does the implementation first classify preserved old labels by reachability
  in the surviving complete old shortest-path DAG?
- Is the chosen-parent subtree reported only as an overapproximation of scalar
  invalidation after a tree-edge deletion?
- Are old-support invalidation and longer-distance repair represented by
  separate states and termination conditions?
- Which vertices keep scalar distance while losing parents, DAG arcs, path
  count, or the canonical word?
- For a batch failure, is combined `D-F` reachability checked instead of
  intersecting independent single-failure summaries?
- Is every parallel invalidation result compared with a fresh exact BFS on the
  named surviving graph version?

## Incremental BFS single-edge sensitivity questions

- Is the update one directed edge, one undirected edge, a batch, or a global
  generator family?
- Does the inserted head strictly improve, tie its old depth, or remain worse?
- Are strict scalar decreases separated from equal-length new predecessor and
  path contributions?
- Is the exact change cone checked against
  `d_G(s,a)+1+d_G(b,v)` rather than graph reachability from `b` alone?
- Does validation include suffixes shortest from `b` that were absent from the
  old source-rooted shortest DAG?
- For an undirected insertion, are both orientations evaluated without
  double-counting one semantic edge path?
- Can batch paths chain several inserted edges, and where is that closure
  represented?
- Is a sampled local Cayley transition being misreported as evidence for
  insertion of the globally translated generator family?
- Are strict decreases, equal-label changes, routed proposals, duplicates, and
  accepted updates reported separately across GPUs?

## Incremental BFS batch-closure questions

- How many inserted-edge occurrences can an updated shortest path use?
- Is an independent one-edge minimum being mistaken for closure over chained
  insertions?
- Are closure rounds labeled as inserted-edge-use rounds rather than original
  BFS depths?
- Which old terminal-distance rows are available, recomputed, or only assumed?
- Does endpoint compression preserve only scalar distance, or is a canonical
  segmentation proved before counting paths?
- Can one concrete old path be represented through optional intermediate
  terminals more than once?
- Are equal proposals closed separately for DAG, count, and canonical outputs?
- Is the update an atomic batch or a query-visible sequence of graph versions?
- For undirected insertions, do two arc records retain one semantic edge
  identity?
- For generator insertion, is the number of new-edge occurrences confused
  with the number of new labels?
- What global condition proves no remote proposal using additional batch edges
  can still lower or enrich an output?

## Distributed BFS 1D/2D expand-fold questions

- What adjacency orientation and frontier-vector orientation define processor
  rows and columns?
- Does every active source reach every block holding required outgoing edges?
- Where is destination visited state authoritative after fold?
- Which expand/fold objects can still be in flight when a local buffer is empty?
- How many blocks store each active source adjacency, and how much frontier
  identity is replicated?
- Are fold records measured before and after current-level and persistent local
  aggregation?
- Is static edge cut being confused with root- and time-dependent fold traffic?
- Which output contract governs equal candidates during fold?
- Where do adjacency, frontier, visited, parent, and candidate buffers peak
  simultaneously?
- How are logical row/column communicators mapped onto NVLink, NVSwitch, PCIe,
  NUMA, NIC, and inter-node links?
- For an implicit graph, what real second axis replaces stored matrix columns?
- Does generator sharding replicate wide full states or recompute hashes/ranks
  on every shard?
- Are same-parent generator aliases discarded only under a compatible output
  merge algebra?
- Is explicit Graph500 evidence being transferred to Cayley successor work
  without accounting for generation and exact state identity?

## Distributed bottom-up systolic questions

- Is the frontier bitmap an exact closed snapshot for every predecessor shard?
- Which incoming-adjacency shards can contain a parent for each candidate?
- Does every uncompleted candidate visit all of them before a negative result?
- Can a completion bit become visible before its parent and next-frontier
  record are durable?
- What epoch prevents stale frontier or completed bits from changing depth?
- How much work continues inside a shard or device after another worker finds a
  parent?
- How do `p_c` substeps trade adjacency work against synchronization latency?
- Are shard and neighbor order allowed to choose an arbitrary parent, or is a
  canonical output required?
- Which output contracts require scanning later shards despite a first hit?
- Are frontier, completed, and parent-update bytes reported separately?
- Does a GPU mapping preserve candidate responsibility without serializing the
  entire device grid?
- For implicit pull, where does the exact enumerable unvisited universe come
  from?
- Do generator-shard aliases carry labeled multiplicity that first-hit would
  discard?
- What consistent cut proves every rotation, scan, completion transfer, and
  publication has finished?

## BFS degree--diameter and Moore-capacity questions

- What are the true simple degrees of the current puzzle graphs after identity,
  duplicate-label, and inverse conventions are normalized?
- How loose is the Moore capacity at each exact BFS depth, not only at the final
  known diameter?
- At which layer does most of the Moore defect first accumulate for each move
  alphabet?
- How much of each layer deficit is explained by low degree, same-layer edges,
  older-layer edges, or convergent next-layer candidates?
- Which Cayley relation families explain the largest early deviations from
  collision-free capacity?
- Are any current diameter statements actually one-root eccentricities or
  lower bounds from incomplete BFS?
- For directed positive alphabets, are directed diameter and strong
  connectivity established before applying a directed Moore sum?
- How does changing generators move a puzzle simultaneously in degree,
  diameter, order-normalized defect, and actual frontier width?
- Which bounds sharpen the unrestricted Moore ceiling for the relevant Cayley,
  Schreier, or permutation-group class?
- Can a near-capacity layer coexist with severe owner imbalance under the
  current partition hash?
- Are combinatorial capacity and allocated GPU capacity explicitly separated in
  benchmark reports?
- Does any memory-sizing argument use a Moore ceiling where the known finite
  state count or measured exact layer is already much sharper?

## BFS distance-sum, closeness, and Wiener questions

- Which current exact BFS artifacts retain complete layer histograms suitable
  for integer farness reconstruction?
- Are reported average distances source averages, unordered-pair averages, or
  samples over a declared distribution?
- Which closeness normalization and unreachable-node convention is intended in
  every downstream consumer?
- For directed move graphs, is the desired statistic outgoing, incoming, or a
  combination of both?
- Which puzzle state graphs are genuinely vertex-transitive under automorphisms
  preserving the measured generator graph?
- For Schreier actions, can root-independent distance profiles be proved, or do
  stabilizers create distinct root types?
- How does changing the move alphabet change farness and harmonic profiles in
  addition to diameter and peak frontier?
- At what radius does a truncated distance sum become numerically close to the
  exhaustive value, and can that approximation have a certified error bound?
- If total state count is known, what lower and upper farness bounds follow from
  a completed radius prefix and diameter bounds?
- What accumulator width is required for the largest intended state count and
  diameter before any floating conversion?
- Can multi-GPU layer-count reductions be independently reconciled with exact
  visited cardinality at every depth?
- Are traversal time, scalar reduction time, all-source batching, and sampled
  estimation reported as different workloads?

## BFS shortest-path betweenness questions

- Which downstream consumers want raw, normalized, ordered, unordered, or
  endpoint-inclusive betweenness?
- Do parallel labeled moves count as distinct shortest paths or one simple
  state-edge path?
- Which stored BFS artifacts contain every predecessor and exact `sigma`, rather
  than one parent and an operational hit count?
- What integer or rational arithmetic contract prevents silent path-count and
  dependency corruption?
- Can aggregate identities detect lost or duplicated cross-owner predecessor
  contributions at every source epoch?
- Which puzzle Cayley graphs have multiple generator-edge orbits with different
  edge betweenness?
- Is any relevant Schreier action actually vertex-transitive by graph
  automorphisms preserving the chosen moves?
- How does changing generators alter the common Cayley vertex betweenness even
  though symmetry keeps every vertex equal?
- Can one identity-rooted dependency computation recover label-orbit edge
  scores by a proved translation argument?
- Which quotient maps preserve weighted path fractions, rather than merely
  reachability or distance?
- If sources are sampled, what estimator, confidence statement, and target
  distribution are actually claimed?
- Are forward BFS, path counts, reverse dependencies, communication, and
  validation timed as different workloads?

## BFS ordering, Cuthill--McKee, and matrix-layout questions

- Which explicit graph or symmetric matrix representation, if any, actually
  needs a bandwidth/profile ordering in this project?
- Is the target metric bandwidth, profile, envelope, wavefront, fill, locality,
  or end-to-end solve/traversal time?
- What root, degree convention, generator order, and tie-break define a
  reproducible CM result?
- Does any claimed RCM bandwidth change accidentally compare different roots or
  tie policies, since pure reversal preserves bandwidth?
- How loose is the two-layer frontier-width bound relative to measured edge
  spans on current graphs?
- Can exact lower bounds show how far a CM/RCM result is from optimal, rather
  than merely from the original numbering?
- Which pseudo-peripheral candidates are truly peripheral under exhaustive
  eccentricity checks on tractable graphs?
- In regular Cayley graphs, what non-degree within-layer signal could matter,
  and would it preserve generator symmetry?
- Does a Schreier graph's irregular degree profile make CM materially different
  from a plain deterministic BFS order?
- What complete enumeration and remapping cost would a global implicit-state CM
  order require?
- If IDs determine multi-GPU owners, can layout effects be separated from the
  changed partition induced by renumbering?
- Are preprocessing, remapping, storage metrics, traversal correctness, and
  end-to-end hardware measurements reported separately?

## BFS graph-covering and universal-tree questions

- Which current quotient or canonicalization maps are genuine locally bijective
  graph covers rather than homomorphisms or orbit quotients?
- What are the first radii of vertex-fiber collision and induced-boundary edge
  closure for each puzzle move graph?
- Can those radii be certified algebraically from relations or stabilizers
  rather than only observed by BFS?
- Which short reduced words collide as group relations and which only as
  action-specific stabilizer words?
- Do any generator images become identity or collide after quotienting, thereby
  destroying local bijectivity?
- How are loops, inverse darts, and parallel labeled moves represented in a
  proposed cover?
- At what depths does one universal-cover sphere begin mixing multiple base BFS
  distances?
- For finite covers, how are fiber representatives distributed across lifted
  depths rather than merely counted globally by sheet number?
- Can bounded anonymous local views distinguish any intended finite puzzle from
  another graph sharing the same universal-cover prefix?
- Does any experimental candidate count actually measure lifted histories while
  being labeled as unique base states?
- How would cross-owner fiber collisions be reconciled under authoritative
  base-state visited semantics?
- Are cover/history and base-state memory, throughput, and validation reported
  as separate workloads?

## BFS deletion, contraction, and graph-minor questions

- Which proposed coarse graphs use only contraction, and which also delete
  edges or vertices so that bound direction changes?
- What connected branch sets and original cross edges witness every coarse
  vertex and edge?
- What are the exact or certified upper bounds on every contracted cluster's
  original diameter?
- How loose is the cluster-diameter lifting bound on representative puzzle
  paths?
- Which loops or parallel move labels appear after quotienting and are they
  retained, merged, or discarded?
- Does any quotient-group path end at the requested concrete state, or only in
  its normal-subgroup coset?
- What kernel/stabilizer correction is required to lift a coarse solution to the
  fixed target?
- Can coarse lower bounds be paired with independent replayable upper bounds to
  certify exact distance on tractable cases?
- How do contraction and deletion separately change frontier widths, duplicate
  profiles, and shortest-path multiplicities?
- If branch sets cross GPU owners, which owner becomes authoritative for the
  supervertex and how is internal state retained?
- Does coarsening reduce communication or merely move it into construction and
  path lifting?
- Are coarse and original state counts, memory, rounds, lifting, validation,
  and throughput reported separately?

## BFS subdivision, topological-minor, and integer-weight questions

- Which future weighted examples have strictly positive integer lengths and
  therefore admit an exact unit-edge subdivision model?
- Are queries restricted to original branch vertices or allowed to end at
  artificial transit vertices?
- Is a subdivision uniform, so depth can be divided by one global `k`, or
  nonuniform, so the metric is genuinely weighted?
- Are replacement paths internally disjoint and tied to stable original edge
  identities?
- How are parallel edges, directions, generator labels, and inverse moves
  represented during expansion and suppression?
- Does a claimed path-count equality count original edge paths or artificial
  unit-edge histories?
- How much do transit vertices change frontier peaks, total work, and depth on
  representative graph families?
- Which diameter, radius, center, distance-sum, or betweenness claims quantify
  only branch vertices and which include the expanded vertex set?
- For a topological-minor occurrence, what are the three separate distances in
  the abstract pattern, selected subdivision, and full host?
- Can a labeled expanded path be collapsed and replayed as complete original
  moves, rejecting partial macro-edges?
- If transit states are implicit, what product/history identity prevents their
  accidental deduplication with genuine Cayley or Schreier states?
- How are long degree-two chains partitioned across GPUs, and how many logical
  weighted steps versus physical unit rounds and owner crossings are reported?
- When comparing with a bounded-integer weighted algorithm, are expansion cost,
  traversal work, memory, and output semantics measured separately?

## BFS sweep, eccentricity-bound, and diameter-certificate questions

- Which current diameter-like outputs are exact eccentricities, lower bounds,
  tree-diameter upper bounds, or matched exact certificates?
- What start and farthest-vertex tie policies define every recorded sweep?
- Do any repeated sweep chains plateau below known exact diameter on current
  puzzle or explicit graph families?
- How much do additional pivots raise `L` and lower
  `max_x min_p(d(p,x)+ecc(p))` on tractable graphs?
- Are every pivot distance field and eccentricity complete for the same graph
  epoch?
- Which roots minimize the number and size of outer fringe layers that require
  complete eccentricity sweeps?
- Has any claimed fringe upper bound accidentally sampled rather than exhausted
  an outer layer?
- How do four-sweep midpoint choices vary across valid BFS parent trees and
  farthest ties?
- Which graph-family theorems, if any, justify stronger sweep conclusions for
  the actual inputs?
- Can a purported Cayley state graph be proved vertex-transitive under its
  exact generator and identity conventions before using one-sweep diameter?
- Which Schreier quotients lose the automorphisms needed for equal
  eccentricities?
- In directed graphs, what forward/reverse or strongly connected certificate
  replaces the invalid undirected sweep argument?
- For GPU experiments, are concurrent independent roots distinguished from one
  BFS distributed across multiple devices?
- Does reported exact-diameter time include heuristic root selection, every
  completed fringe sweep, reductions, and certificate validation?

## BFS complement-graph and nonedge-frontier questions

- Which prospective workloads genuinely ask for complement distance rather
  than using complement language informally?
- Are graphs simple and loop-free, or must directed arcs, labels, parallel
  edges, and weights receive separate complement conventions?
- Which original graphs or Cayley generator sets yield disconnected
  complements and what are their co-components?
- Do measured complement frontiers match
  `U_i \ intersection_(v in F_i) N_G(v)` exactly?
- Has any implementation confused union and intersection when batching a
  complement frontier?
- What evidence makes the unvisited set authoritative after concurrent
  complement discoveries?
- Are logical complement edges ever materialized, or is adjacency decided from
  the original representation?
- Which counters measure original adjacency, nonedge predicates, candidate
  filtering, logical edges, and physical storage separately?
- For a Cayley complement, is `Gamma\({e} union S)` represented exactly and
  inverse closed under the declared graph convention?
- Does any reported complement word accidentally get presented as a legal word
  over the original puzzle generators?
- Can a Schreier complement adjacency be realized by a meaningful action
  alphabet, or is it only an unlabeled simple graph?
- Under multi-GPU sharding, which owner can certify that an original edge is
  globally absent?
- How do candidate ownership and frontier ownership change communication for a
  dense logical complement of a sparse stored graph?
- Are construction, traversal, replay/parent validation, and graph-epoch checks
  reported separately?

## BFS bisimulation, simulation, and state-merging questions

- Which proposed state merges are intrinsic equality, automorphism orbits,
  strong bisimulation, simulation abstraction, or mere fingerprints?
- What observation map must remain constant inside every class?
- Is goal membership saturated by whole classes, or does a requested concrete
  target force further refinement?
- Do all representatives match every allowed labeled transition in both
  directions?
- Does any symmetry permute move labels, requiring a frame rather than
  same-label bisimulation?
- Which abstract paths have been concretely lifted from every relevant start
  representative?
- Are degree, transition multiplicity, path counts, probabilities, or costs
  claimed even though the chosen equivalence preserves only existence?
- Would equitable/count refinement split any class accepted by plain
  bisimulation?
- Has a universal or overly coarse bisimulation arisen because goal/error/type
  observations were omitted?
- For weighted moves, what relation matches transition costs exactly?
- Does any stutter quotient report abstract hops as original BFS distance?
- In directed bidirectional search, are both outgoing and incoming partitions
  stable?
- Which Cayley or Schreier equivalences are congruent under every declared
  generator action?
- Can quotient parents be lifted into a replay-valid fixed-target path with all
  necessary representative/frame metadata?
- Under multi-GPU refinement, can a remote transition split a locally accepted
  class?
- Are partition construction, proof/validation, quotient traversal, lifting,
  original replay, and communication reported separately?

## BFS Myhill--Nerode, DFA-minimization, and residual-language questions

- Which finite-memory constraints currently contain DFA states with identical
  residual suffix languages?
- Are automata deterministic, complete, trimmed, and epsilon-free under the
  declared word-length metric?
- Is a missing partial-DFA transition represented consistently as rejection or
  an implicit dead sink?
- Has any merge used equal depth, acceptance bit, or nearest-accept distance
  instead of full residual-language equality?
- Does the requested output need one shortest accepted word, all shortest
  words, all accepted words, or original prefix provenance?
- Is the alphabet order fixed when shortlex output is claimed?
- For NFAs, what reachable subset states arise before deterministic
  minimization?
- Which minimized DFA states are actually reachable in each labeled base-graph
  product?
- Can a product-specific automaton reduction be safely reused across another
  graph or generator epoch?
- Does automaton minimization preserve the constraint language while the
  language itself still fails Cayley coverage, uniqueness, or geodesicity?
- What reverse automaton relation makes forward/backward memory states
  compatible at a bidirectional meeting?
- Are residual classes assigned stable identical IDs on every GPU?
- Which metrics count DFA states, residual classes, prefixes, accepted words,
  base walks, product states, and base states?
- Are trim/minimization, product construction, traversal, witness replay, and
  end-to-end costs reported separately?

## BFS NFA subset-state, antichain, and dominance questions

- Which constraints are genuinely nondeterministic and what full subset is
  active after each word?
- Are epsilon closures complete before symbol depth advances?
- Has any frontier accidentally unioned configurations belonging to different
  prefixes?
- How many of the theoretical `2^n` subsets are reachable in each actual
  automaton and graph product?
- What is the distribution of active-state counts inside reachable subsets?
- Are subset equality keys exact, or only popcount/hash/overlap fingerprints?
- For every inclusion prune, are base vertex, target, automaton, resource phase,
  graph epoch, and all other future-relevant fields identical?
- Was the dominating superset reached at no greater depth?
- Is the requested output only one shortest existentially accepted path, or
  does it require shortlex order, all words, all paths, or run provenance?
- Which different subsets become equal only after residual-language
  minimization?
- Are accepted words and accepting NFA runs counted separately?
- What reverse-subset compatibility condition closes a bidirectional accepted
  word at the meeting state?
- In Cayley/Schreier products, do distinct accepted words still collide after
  group or orbit evaluation?
- Under multi-GPU sharding, which reduction proves that every member's
  successors entered the authoritative next subset?
- Are determinization, epsilon closure, dominance, minimization, traversal,
  witness replay, and communication costs reported separately?

## BFS AND/OR, reachability-game, attractor, and rank questions

- Which intended problems contain genuine adversarial or universal choices
  rather than only solver-selected moves?
- What ownership and target labels belong to every game state?
- Are nonterminal arenas total, and if not, who wins at each kind of dead end?
- How much larger is ordinary reverse reachability than the exact winning
  attractor on representative arenas?
- Do computed layers satisfy the forced-within-`k` rank interpretation?
- Has any universal vertex been finalized after observing only one winning
  successor?
- Can a self-supporting universal cycle enter through an incorrect greatest or
  arbitrary fixed point?
- Is every existential strategy edge strictly rank-decreasing?
- Does the complement of the attractor satisfy the expected trap conditions?
- Which missing or spurious transitions can cause each polarity of error under
  current ownership?
- Does a reported witness encode one favorable path or a positional strategy
  valid against every adversary response?
- Are path counts, play-tree sizes, and strategy counts kept distinct?
- For alternating automata, what exact run-tree and terminal acceptance
  convention replaces existential NFA semantics?
- In Cayley/Schreier games, does translation preserve ownership, control phase,
  moves, and target semantics?
- Under multi-GPU sharding, what proves complete universal outdegree and
  all-successor status at one consistent epoch?
- Are arena construction, ordinary reachability, attractor rounds, ranks,
  strategy validation, communication, and end-to-end costs reported separately?

## BFS support-graph, probabilistic-reachability, and MDP questions

- Is the intended transition model a plain graph, a fixed Markov chain, an MDP,
  or an adversarial game?
- Does an edge mean possible, positive probability, controllable choice, or an
  environment response?
- Is the requested claim positive-probability, almost-sure, sure/adversarial,
  bounded-horizon, or maximum/minimum-probability reachability?
- Does support BFS distance equal the first horizon with a nonzero hit value?
- Which reachable BSCCs or MDP end components avoid the target?
- Are non-hitting paths assigned infinite hitting time, excluded by
  conditioning, or charged another declared terminal cost?
- Which arguments rely on the state space being finite?
- Can a probabilistic self-loop be escaped almost surely even though its
  adversarial interpretation loses?
- Does any shortest support path have negligible success probability or lead
  to a trap on its other outcomes?
- When parallel generator labels share an endpoint, is their probability mass
  summed rather than discarded by endpoint deduplication?
- Can a numerically tiny but positive edge enter a target-free recurrent class?
- What proves every distributed transition row is complete and sums to one?
- Are support traversal, BSCC/end-component analysis, zero/one classification,
  numerical solve, policy validation, communication, and timing separate?

## BFS de Bruijn, Kautz, overlap, and shift-register questions

- Is the intended object a directed word-shift graph, its underlying undirected
  graph, or a physical network using that topology?
- What alphabet size, word length, loop, parallel-arc, and Kautz adjacency
  convention is authoritative?
- Do measured distances match `n` minus maximum suffix-prefix overlap?
- Which source borders and periods explain each observed frontier profile?
- Are appended histories, candidate words, unique words, and newly discovered
  words counted separately?
- Does any claim of uniform root layers incorrectly follow from regular degree?
- Is a de Bruijn sequence/Hamiltonian enumeration being confused with a BFS
  order or shortest-path tree?
- Has orientation been silently removed, changing the metric and diameter?
- Are fixed-symbol shift maps mistakenly described as bijective Cayley
  generators?
- Which special cases, if any, admit an independently proved Cayley or Schreier
  representation?
- Does small logical diameter translate into fewer global BFS barriers under
  the actual stopping contract?
- Are logical state edges, owner routing edges, and physical GPU links kept as
  three separate graphs?
- Do multi-GPU reports separate generated words, duplicate collisions, owner
  traffic, physical hops, synchronization, and end-to-end time?

## BFS lamplighter, wreath-product, and configuration-position questions

- What base graph, lamp group, finite-support rule, and generating set define
  the intended lamplighter state graph?
- Are toggle, movement, switch-walk, and walk-switch charged as separate or
  combined generators?
- Does the visited key include both the complete lamp configuration and cursor?
- For two states, is the required lamp set their symmetric difference under the
  exact group-coordinate convention?
- Does a claimed word length equal lamp-change cost plus a certified shortest
  base walk visiting all changed lamps?
- Is a line/cycle visiting formula being transferred to a harder base graph
  without proof?
- Is the wreath-product graph being incorrectly treated as a Cartesian product
  of lamp states and position?
- How is each BFS frontier distributed by lamp popcount, route cost, and cursor?
- Which frontier states are radial dead ends, and which merely lie on the finite
  graph's diameter layer?
- Are dead-end depth and distance from the identity kept distinct?
- Are generator histories, visiting routes, shortest words, DAG parents, and
  unique group states counted separately?
- Does a bidirectional meeting compare the complete state rather than cursor
  position alone?
- Does a finite lamp window report boundary escape as unknown instead of
  unreachable?
- For finite cyclic probes, do all states match the independent route-cost
  oracle?
- Are logical Cayley transitions, owner traffic, and physical links reported as
  separate graphs and costs?

## BFS Tower-of-Hanoi, Schreier, recursion, and frontier questions

- Which peg count, disk identity, placement rule, and allowed peg-move graph
  define the puzzle?
- Is a forbidden peg-pair action absent, retained as a fixed-point loop, or
  treated as an error?
- Does the visited key encode the peg of every distinguishable disk?
- Are states group elements, an orbit of configurations, or words in the move
  generators?
- Does the implementation generate three labeled peg-pair attempts or only
  two/three distinct legal neighbors?
- Are loop occurrences, simple edges, and candidate attempts counted
  separately?
- Does fixing the largest disk expose exactly three `H_(n-1)` copies and the
  expected bridge edges?
- Do corner distances and all-pairs diameter equal `2^n-1` under the exact
  classical contract?
- Do corner-root layers satisfy `|F_k|=2^popcount(k)` and sum to `3^n`?
- Is the last corner layer incorrectly assumed to contain only the two other
  perfect-stack states?
- Are recursive calls, one canonical solution, shortest paths, histories, and
  distinct BFS states kept separate?
- Is corner symmetry being transferred to arbitrary roots despite the simple
  graph's non-vertex-transitivity?
- Does bidirectional stopping use proved frontier bounds rather than a guessed
  recursive midpoint?
- Which claims fail after adding pegs or restricting move directions?
- Are recursive logical bridges, owner-routing messages, and physical GPU links
  distinguished in multi-GPU measurements?

## BFS pancake Cayley, prefix-reversal, and collision questions

- Is the state an unsigned permutation, signed/burnt permutation, or repeated-
  symbol string?
- Which prefix lengths are generators, and does each reversal cost one?
- Is left/right permutation composition fixed consistently for translation,
  parents, and replay?
- Does visited use complete permutation identity or a proved bijective rank?
- Are hashes/fingerprints, breakpoint summaries, and parity treated only as
  advisory unless collision/equivalence safety is proved?
- Do the first layers match `n-1`, `(n-1)(n-2)`, and
  `(n-1)(n-2)^2-1`?
- Is the depth-three deficit traced to the six-cycle relation rather than a
  missing successor?
- Where does effective new-state branching depart materially from the
  nonbacktracking history tree?
- At what depth does each measured frontier peak and begin contracting?
- Is vertex transitivity being overextended to edge transitivity, distance
  regularity, path counts, or physical balance?
- Does a diameter claim contain both a hard-state lower bound and a global
  upper-bound/exhaustion certificate?
- Are unsigned results incorrectly transferred to the `2^n n!` burnt state
  space?
- Is graph bipartiteness incorrectly inferred from generator involutions despite
  mixed generator parity?
- For bidirectional search, are endpoint translation and move reconstruction
  verified under the same action convention?
- Are histories, candidates, unique permutations, new states, rank work,
  owner traffic, synchronization, and end-to-end time reported separately?

## BFS star-transposition, cycle-metric, and generator-contrast questions

- Is the generator set star transpositions, prefix reversals, adjacent swaps,
  or another permutation metric?
- Is the distinguished center symbol/position and composition convention fixed?
- Does every claimed star distance match `s+c-2delta` from complete cycle
  decomposition?
- Are cycles containing the center charged `l-1` and disjoint cycles `l+1`?
- Does the measured diameter equal `floor(3(n-1)/2)`?
- Does depth parity match permutation sign for every reached state?
- Are involution and common parity homomorphism distinguished when asserting
  bipartiteness?
- Do early layers match the cycle-type counts for depths one, two, and three?
- Is cycle type being used incorrectly as complete visited identity?
- Which vertices at one depth have different cycle types and shortest-word
  multiplicities?
- When comparing star and pancake BFS, are state count, degree, root, identity,
  and representation genuinely held fixed?
- Is any finite observed diameter/peak ordering being overgeneralized to all
  `n`?
- Is the star graph the computed state graph, the owner-routing topology, or the
  physical interconnect?
- Are swap/reversal work, distance-oracle work, candidates, duplicates, visited,
  routing, synchronization, and end-to-end time reported separately?

## BFS all-transposition, cycle-count, and Stirling-frontier questions

- Are all `C(n,2)` transpositions present and unit-cost, or only a generating
  subset?
- Does every BFS label equal `n-c(pi)` with fixed points included as cycles?
- Does each generated edge merge or split cycles and change depth by exactly
  one?
- Do measured layers equal unsigned Stirling numbers `[n over n-d]`?
- Does the layer polynomial equal `product_(i=1)^(n-1)(1+iq)` and sum to `n!`?
- Does the farthest layer contain exactly the `(n-1)!` full cycles?
- Is cycle type being confused with authoritative permutation identity?
- At one depth, how do inward/outward degrees vary across cycle types?
- Do 3-cycles and pairs of disjoint transpositions expose non-distance-
  regularity at depth two?
- Are minimal factorization words, shortest-DAG parents, and unique permutation
  states counted separately?
- When comparing generator sets on `S_n`, are state identity and `n` held fixed
  while degree and logical edge cost are reported explicitly?
- Is reduced diameter being mistaken for reduced candidate, byte, or
  communication work?
- Are computed Cayley edges, owner-routing edges, and physical links kept
  separate?
- Are cycle-oracle, rank, candidates, duplicates, visited, frontier,
  synchronization, routing, and end-to-end costs reported separately?

## BFS coverage, domination, and k-center questions

- Is the supplied center set fixed, or is the task to select centers?
- Is coverage radius, center cardinality, or both constrained?
- Does the declared universe include every vertex needed for a coverage proof?
- Is an unreached component represented as infinite radius rather than ignored?
- Does a claimed dominating set cover every closed radius-one neighborhood?
- For distance domination, is the radius and open/closed ball convention fixed?
- Is a nearest-center owner label being confused with scalar distance to the
  center set?
- Are arbitrary, canonical, and set-valued Voronoi ties distinguished?
- Is maximum degree being used incorrectly as a proxy for minimum
  eccentricity?
- Is farthest-first described as a metric factor-two approximation rather than
  an exact selector?
- Are start-vertex and farthest-tie rules included in a greedy result's
  workload identity?
- Does an upper bound from a proposed center set have a separate lower-bound
  witness before optimality is claimed?
- For directed coverage, is the required distance `d(c,v)` or `d(v,c)`?
- In a Cayley graph, is singleton transitivity being overextended to the
  relative placement of several centers?
- For iterative selection, are traversal count, per-run work, total work, and
  selection quality reported separately?

## BFS shortest-hop secondary-cost and Pareto questions

- Is the objective minimum hops, minimum secondary cost, lexicographic order,
  a resource constraint, or all nondominated trade-offs?
- Does a one-parent BFS promise only an arbitrary minimum-hop path or a stronger
  secondary tie contract?
- Are all edges satisfying `d(v)=d(u)+1` retained before secondary reduction?
- Is secondary metadata finalized for a complete layer before descendants use
  it?
- If equal-depth improvements arrive late, are they repropagated?
- Are hop-first `(h,c)` and cost-first `(c,h)` being confused?
- Can one vertex require several incomparable Pareto labels?
- Is dominance pruning proved safe for every continuation-relevant resource?
- Does a constraint require product-state identity rather than metadata on the
  original vertex?
- Are negative secondary costs safe only because the shortest-path subgraph is
  depth-acyclic?
- Are arbitrary, deterministic, all-parent, and count outputs distinguished?
- In a Cayley graph, is secondary cost invariant under translation and
  additive on declared transitions?
- If cost depends on move history, is that history included in state identity?
- On multiple GPUs, does the authoritative owner reduce every shortest-parent
  candidate rather than accepting message-order first arrival?
- Are BFS traversal, secondary reduction, label multiplicity, communication,
  and end-to-end time reported separately?

## BFS Hamming-graph questions

- Is `H(d,q)` using every one-coordinate symbol replacement, or a smaller
  generator alphabet?
- Does the represented state space contain exactly `q^d` distinct words?
- Does every measured distance equal Hamming disagreement count?
- Do layers match `C(d,i)(q-1)^i` and sum to `q^d`?
- Is the predicted mode near `(q-1)d/q` distinguished from the diameter `d`?
- Do inward, same-layer, and outward degrees match `i`, `i(q-2)`, and
  `(d-i)(q-1)` for every state in layer `i`?
- Is binary bipartiteness being incorrectly transferred to `q>2`?
- Does visited include the current frontier when nonbinary same-layer edges are
  generated?
- Are previous-layer hits, same-layer hits, and next-layer convergence counted
  separately?
- Does every depth-`i` target have `i!` shortest paths under direct coordinate
  replacement?
- Are shortest histories being confused with unique word states?
- Is the dense base-`q` rank proved bijective before bitmap visited is used?
- Is Cartesian/Cayley symmetry being preserved under the declared secondary
  cost or ownership rule?
- When Hamming balls are seeded from a code, is covering radius distinguished
  from minimum code distance?
- Are logical regularity and closed duplicate counts being overgeneralized to
  hardware locality or owner traffic?
- Are state count, candidates, same-layer traffic, remote traffic,
  synchronization, and end-to-end time reported separately?

## BFS Johnson-graph and fixed-weight questions

- Are states exact `k`-subsets, ordered tuples, multisets, or quotient orbits?
- Does every move exchange one selected and one unselected element?
- Is `J(n,k) ~= J(n,n-k)` used only as an isomorphism, not as state identity?
- Does distance equal half symmetric difference under the declared move set?
- Do layers match `C(k,i)C(n-k,i)` and exhaust `C(n,k)` states?
- Do inward, same-layer, and outward degrees match `i^2`, `i(n-2i)`, and
  `(k-i)(n-k-i)`?
- Is a fixed-weight invariant being incorrectly interpreted as bipartiteness?
- Are same-layer triangle edges included in current-frontier visited checks?
- Are previous-layer, same-layer, and next-layer convergence occurrences
  counted separately?
- Does each depth-`i` target have `(i!)^2` shortest exchange sequences?
- Is the Johnson graph being confused with the fixed-weight induced subgraph of
  the ordinary hypercube?
- Are two-bit exchange edges being silently charged as one or two logical
  steps?
- Is the state graph a Cayley graph, a Schreier graph, or an unlabeled simple
  neighbor graph?
- If all transpositions are generated, are stabilizer self-loops counted and
  reported separately from `k(n-k)` distinct neighbors?
- Is intersection size being used incorrectly as a complete visited key?
- Is combinadic ranking proved bijective and its arithmetic cost measured
  separately?
- Are logical regularity, owner balance, routing locality, and physical
  throughput kept as separate claims?

## BFS Grassmann-graph and subspace-identity questions

- Are vertices semantic subspaces or particular matrices/bases?
- Is the field `F_q`, including its arithmetic representation, fixed?
- Are two states equal by row space under a proved canonical convention?
- Does state count match the Gaussian binomial `[n choose k]_q`?
- Does graph distance equal `k-dim(U intersection W)`?
- Is graph distance distinguished from the factor-two subspace coding metric?
- Do layers match `q^(i^2)[k choose i]_q[n-k choose i]_q`?
- Do inward and outward degrees match `[i]_q^2` and
  `q^(2i+1)[k-i]_q[n-k-i]_q`?
- Are same-layer neighbors included in current-frontier visited checks?
- Does every depth-`i` target have `([i]_q!)^2` graph shortest paths?
- Are basis multiplicity and shortest-path multiplicity counted separately?
- Can different hyperplane/extension or scalar choices emit the same semantic
  neighboring subspace?
- Is RREF or another canonical encoding deterministic across devices and
  processes?
- Is a claimed Grassmann rank bijective over the complete declared universe?
- Is intersection dimension being used incorrectly as a complete visited key?
- Are group elements, stabilizer occurrences, candidate bases, distinct
  neighbors, and accepted states reported separately?
- Does owner routing use canonical subspace identity rather than raw basis
  bytes?
- Are field operations, canonicalization, visited, communication,
  synchronization, and end-to-end time separate measurements?

## BFS sparse-random-graph questions

- Is the workload `G(n,p)`, `G(n,m)`, a configuration model, a random regular
  graph, or percolation on a fixed graph?
- Is one graph frozen for the traversal, or is adjacency resampled on query?
- Are `n`, `p/c`, PRNG, seed, pair canonicalization, and graph version recorded?
- Is the root fixed, uniformly random, or selected from the largest component?
- Is a branching process used only as a scoped local approximation?
- Is mean degree being confused with realized or excess degree?
- Are extinction probability, giant fraction, and expected random-root
  component fraction distinguished?
- Are asymptotic high-probability claims being applied incorrectly to one
  finite sample?
- Near `c=1`, are ranges/quantiles and extinction frequency retained rather
  than only a mean frontier?
- At what exposed fraction do frontier growth and the tree approximation begin
  to diverge materially?
- Are previous-layer, same-layer, repeated-next-parent, and new-state
  occurrences counted separately?
- Is outward-occurrence multiplicity being mistaken for the complete duplicate
  ratio?
- Does implicit generation make one stable symmetric decision per unordered
  pair?
- Can different ranks or retry orders generate inconsistent edge samples?
- Is graph-generation time excluded from or included in end-to-end timing
  explicitly?
- Are per-seed traces, load skew, routing, synchronization, and throughput
  reported rather than one selected favorable sample?

## BFS random-regular and pairing-model questions

- Does “regular” mean fixed degree, distance regularity, or a regular group
  action?
- Is the sample a simple uniform regular graph, pairing multigraph, conditioned
  pairing, switch-chain output, or biased constructive graph?
- Are `n`, `r`, PRNG, seed, shuffle, rejection, and retry contracts recorded?
- Are loops/multiedges rejected, conditioned away, or silently removed?
- Is graph-generation/rejection work separated from BFS work?
- Does the early envelope use root branching `r` and later branching `r-1`?
- At which first depth does the actual frontier fall below
  `r(r-1)^(d-1)`?
- Are inward, same-layer, and outward ranges measured within each layer?
- Is equal adjacency-list length being mistaken for equal radial or hardware
  work?
- Are outward multiplicity ratios paired with absolute next-layer sizes,
  especially in tiny tail layers?
- Is local weak tree convergence being overextended to growing radius or the
  whole finite graph?
- Are `r=0,1,2` excluded before applying `r>=3` connectivity claims?
- Is one-root eccentricity being reported incorrectly as graph diameter?
- Is ensemble label exchangeability being confused with automorphisms of one
  realized graph?
- Do all owners share one accepted pairing and identical undirected adjacency?
- Are generation, CSR construction, edge scans, collision classes, routing,
  synchronization, and end-to-end time separate measurements?

## BFS stochastic-block and multitype-frontier questions

- Is one conditional SBM edge sample frozen for the complete traversal?
- Are block proportions, coefficient matrix, PRNG, seed, and root type recorded?
- Is the branching matrix orientation declared rather than silently transposed?
- Is a local branching approximation being mistaken for an exact finite BFS
  recurrence?
- Are Perron total growth and signed type-contrast evolution distinguished?
- Is irreducibility checked before claiming one unique global giant?
- Are scalar frontier sizes hiding persistence or alternation of vertex types?
- Are root conditioning and largest-component conditioning distinguished?
- Are block labels given, inferred, or unavailable to the traversal?
- Does the owner partition reduce routing at the cost of reachable-owner load
  balance or memory capacity?
- Is a low or zero edge cut being mistaken for high multi-GPU utilization?
- Are type-to-type and owner-to-owner candidate matrices retained per depth?
- Are previous-ball, same-layer, repeated-next-parent, and new-state outcomes
  separated by source and destination type?
- Are community inference quality, BFS correctness, and partition performance
  reported as separate claims?
- Are graph generation, traversal, routing, synchronization, and end-to-end
  measurements separated?

## BFS heterogeneous-degree and configuration-model questions

- Is the graph a direct pairing multigraph, a simple-conditioned configuration
  graph, or a collapsed support graph?
- Are self-loops and parallel occurrences retained as candidate work even when
  they do not change reachability?
- Are the degree sequence, label shuffle, pairing PRNG, seed, and root rule
  recorded?
- Is the root degree law `D` distinguished from edge-endpoint law `D*`?
- Is average degree being substituted incorrectly for
  `E[D(D-1)]/E[D]`?
- Are finite-second-moment and maximum-degree assumptions checked before using
  branching or giant asymptotics?
- Are mean degree, excess mean, maximum degree, and degree variance reported
  separately?
- Does a higher branching mean coexist with a smaller giant because of leaves
  or isolated vertices?
- Is fixed-root extinction retained rather than hidden by conditioning on the
  largest component?
- How does frontier degree composition change after early hubs are depleted?
- Can a narrow hub-rich frontier scan more edge occurrences than a wider
  leaf-rich frontier?
- Are loops, parallel hits, previous-layer hits, same-layer occurrences,
  repeated next parents, and unique new states distinguished?
- Does vertex-balanced ownership leave incident-edge work or routed fanout
  highly skewed?
- Are heavy-tail cutoffs and their dependence on `n` explicit?
- Are graph generation, pairing validation, traversal, routing,
  synchronization, and end-to-end time separate measurements?

## BFS directed-random and bow-tie questions

- Are arcs sampled independently by ordered pair and then frozen?
- Is forward reach distinguished from reverse, weak, and strong reachability?
- Is transpose adjacency complete, or are incoming neighbors being guessed from
  outgoing storage?
- Are GIN and GOUT named relative to flow into and out of the giant SCC?
- Is `SCC(s)` computed as the intersection of two completed exact traversals?
- Is one root SCC being mistaken for a complete SCC partition?
- Is root conditioning on GIN, GOUT, or GSCC explicit?
- Are conditional giant traversal size and unconditioned mean reach separated?
- At `c<=1`, are finite largest-SCC reach sets incorrectly called giant IN/OUT?
- Are forward and reverse marginal symmetry being confused with equal realized
  frontier profiles?
- Are per-level outgoing and incoming degree masses recorded separately?
- Can two orientations have similar total reach but different peak memory and
  synchronization counts?
- Are owner-to-owner routing matrices allowed to be asymmetric?
- Are transpose construction/storage, traversal, SCC classification,
  intersection, routing, synchronization, and end-to-end time separate?

## BFS random-geometric and spatial-wave questions

- Is the point process, ambient metric, dimension, radius, and boundary
  convention fixed?
- Is the graph built on a square, torus, sphere, obstacle domain, or another
  space?
- Are all radius decisions frozen before BFS?
- Is `ceil(d_E/r)` used only as a lower bound rather than an exact hop oracle?
- Are unreachable pairs distinguished from finite geometric detours?
- Are finite critical-window samples being mistaken for a sharp asymptotic
  threshold estimate?
- Is maximum finite BFS depth in a disconnected graph mislabeled as diameter?
- Are center, corner, uniform, and largest-component-conditioned roots kept
  separate?
- Are stretch ratios weighted by samples, roots, or eligible vertex pairs, and
  are zero-eligible cases retained?
- How do frontier geometric extent, vertex count, and scanned degree mass differ
  by depth?
- Does a boundary truncate the wave or merely lower local degree?
- Does a torus partition include wraparound owner interfaces?
- Does a low spatial edge cut leave early or late frontier work on one owner?
- Are total owner balance and per-level owner balance reported separately?
- Are point generation, neighbor construction, BFS, routing, synchronization,
  and end-to-end time separate measurements?

## BFS small-world and shortcut-wave questions

- Is the model rewiring local edges, adding shortcuts, or fixing their exact
  count?
- Are loop/duplicate rejection and shortcut endpoint sampling recorded?
- Does the shortcut carry unit cost or the old path cost it represents?
- Is the augmented graph metric being confused with faster execution of the old
  BFS problem?
- Is edge-addition distance monotonicity checked against a baseline oracle?
- At what depth does the first useful shortcut endpoint enter the frontier?
- How many separated local wave intervals coexist before they collide?
- Are frontier widening and eccentricity reduction reported together?
- Is mean degree hiding a large change in distance distribution and BFS rounds?
- How many destinations benefit from each sparse shortcut set?
- Is shortcut edge fraction being mistaken for fraction of affected paths?
- Where is the root relative to contiguous owner boundaries?
- At what depth does every owner first receive useful frontier work?
- Does low total edge cut coexist with long periods of idle owners?
- Are base/shortcut scans, changed metric, visited collisions, routing,
  synchronization, and end-to-end time separate measurements?

## BFS preferential-attachment and age-core questions

- Is the exact growth process specified rather than merely called scale free?
- Are seed, attachment offset, edge multiplicity, loop/parallel semantics, and
  sequential endpoint choices recorded?
- Is birth-time correlation being confused with a theorem that every old vertex
  is a hub?
- Are root degree, root age, and uniform-root conditioning kept separate?
- At what depth does the wave first hit a declared old or high-degree core?
- Does candidate multiplicity spike before, with, or after unique frontier
  width?
- Are frontier degree and birth-time quantiles retained per complete level?
- Is a configuration-model comparison conditioned on the same realized degree
  sequence and graph semantics?
- Has degree-preserving randomization actually mixed enough to erase age
  correlations?
- Are typical distance, root eccentricity, and diameter stated separately?
- Is a `log n / log log n` statement transferred only to the precise model and
  observable covered by its theorem?
- Does birth-contiguous ownership concentrate old hubs or only vertex counts?
- Are total owner balance and per-level scanned-edge/candidate balance both
  reported?
- Does hash ownership relieve hub concentration at the cost of remote traffic?
- Are graph generation, randomization, BFS, routing, synchronization, and
  end-to-end time separate measurements?

## BFS growing-tree and rerooting questions

- Is the growth seed being confused with the current BFS query root?
- Which parent edges reverse when the query root changes?
- Is birth depth being reported as BFS distance from a nonseed root?
- Does the implementation exploit a proved tree contract or merely assume that
  early layers contain no cycles?
- If visited is omitted, is the incoming parent excluded exactly?
- Are same-layer and repeated-next-parent counts asserted zero as a tree check?
- Does measured `|F_(d+1)|` equal frontier excess-degree sum exactly?
- Is a wide hub layer producing unique children or duplicate candidates?
- Is collision-heavy language restricted to cyclic attachment models?
- Does birth-contiguous ownership concentrate incident-edge work despite equal
  vertex counts?
- Are cross-owner tree edges distinguished from duplicate merging, which should
  be absent?
- Are growth orientation, BFS orientation, layout, and owner authority treated
  as separate structures?

## BFS unicyclic and first-duplicate questions

- Is the graph certified unicyclic, or has only one cycle been observed so far?
- Where does the source's unique path enter the cycle?
- Is cycle parity inferred from the correct BFS signature?
- For odd parity, is the closing edge counted as two same-layer scan
  occurrences rather than a repeated next-state proposal?
- For even parity, are both shortest predecessors retained when the output
  requires a shortest-path DAG?
- Are downstream vertices with one immediate predecessor but two complete
  shortest paths counted correctly?
- Does a one-parent result state how the antipode's winner is chosen?
- Are attached-tree degrees separated from the unique cycle duplicate event?
- Is parent-only exclusion being transferred unsafely from trees?
- When cycle arcs cross owners, which owner is authoritative for the antipode?
- Are two routed proposals distinguished from two accepted states?
- Is the finite simple undirected Cayley specialization correctly reduced to a
  cycle rather than an irregular decorated cycle?
- Are directed, Schreier, loop, and parallel-edge conventions kept outside the
  simple unicyclic theorem?

## BFS cactus and composable-cycle questions

- Is the graph actually a cactus, or do two cycles share a path?
- Is the unique source-target block sequence distinguished from local arc
  choices inside cycle blocks?
- Which traversed even cycles are entered and exited antipodally?
- Is total shortest-path count computed as a product of local choices rather
  than from only the target's immediate predecessor count?
- Can a target with one predecessor inherit multiple upstream shortest paths?
- Is each odd cycle's same-layer edge distinguished from each even cycle's
  repeated next-state proposal?
- Does measured cycle rank equal the number of cactus cycle blocks?
- Are frontier profiles being inferred incorrectly from pairwise block-route
  formulas?
- When block partitions minimize cuts, do they also balance frontier vertices,
  incident edges, and candidate traffic per level?
- Are legitimate antipodal contributions distinguished from retried messages?
- Which richer output contract requires preservation of losing parents?
- Does the graph contain a theta subgraph that invalidates independent cycle
  choices?
- Are cactus cycles being used only as an independent-relator control rather
  than as a model of a general Cayley graph?

## BFS theta and overlapping-cycle questions

- Are the three path lengths and simple-graph conventions declared?
- How many paths attain the branch-to-branch minimum?
- Is a three-way candidate meeting being forced into a pairwise duplicate
  model?
- For each longer path, what is the parity of `d+L_i`?
- Does another path's length change layers on the path currently being
  inspected?
- Are cycle rank two and three simple cycles being conflated?
- Is a selected cycle basis hiding the physical multiplicity of one meeting?
- Do path-count contributions retain upstream multiplicity carried by a single
  immediate predecessor?
- Can two graphs with equal cycle rank place work in same-layer edges versus
  repeated-next-state proposals differently?
- Does the authoritative owner distinguish three graph predecessors from
  retransmission of one message?
- Are one frontier insertion, parent list size, and `sigma` accumulation
  reported separately?
- Are theta word families in a Cayley graph checked by exact endpoint equality
  rather than binary cycle-space analogy alone?
- Do chords or extra cross-links invalidate the two-endpoint distance formula?

## BFS layer-edge and duplicate-conservation questions

- Are `A_d`, `B_d`, and `|F_d|` measured from the same completed BFS layers?
- Does every edge satisfy the consecutive-layer inequality?
- Does `sum A_d + sum B_d` reproduce the graph's edge count?
- Does `B_d` equal the summed predecessor count of `F_(d+1)`?
- Does the radial decomposition reproduce `m-n+1` exactly?
- At which depths is cyclomatic charge `q_d` concentrated?
- How does rerooting move `q_d` without changing total cycle rank?
- Are reverse tree-edge scans separated from structural non-tree occurrences?
- Are same-layer scans separated from repeated outward proposals?
- Is the full-traversal rejected-occurrence identity applicable, or did the run
  stop early or avoid scans structurally?
- In bipartite inputs, are all measured `A_d` exactly zero?
- Are excess predecessors treated as output when DAG or path counts are
  requested?
- Do physical duplicate records co-locate enough for a proposed GPU mechanism
  to see the semantic excess?
- Are retries, parallel labels, and distinct predecessor edges distinguished?
- Has simple-support cycle rank been transferred incorrectly to directed or
  implicit occurrence semantics?

## BFS successor-occurrence and label-multiplicity questions

- Does the successor interface return a set, multiset, ordered list, or labeled
  occurrence stream?
- Is endpoint state identity distinct from label and record identity?
- Are support arcs materialized or only inferred from occurrences?
- Do all occurrences from `F_d` partition into visited-ball and next-layer
  endpoints as required by exact distances?
- What are `T_d`, `X_d`, `Y_d`, and distinct support-predecessor count `P_d`?
- How much excess comes from same-parent labels versus distinct parents?
- Are duplicate generator elements and identity moves rejected or retained?
- Is the action free, or can stabilizers create state-dependent aliases and
  loops?
- Are Cayley cancellation arguments being transferred incorrectly to a
  Schreier action?
- Does path identity distinguish parallel labels?
- Are message retries assigned stable identity separately from graph
  occurrences?
- Is an equal endpoint record discarded, combined into metadata, or retained as
  a distinct labeled predecessor according to the output contract?
- Can instrumentation reconstruct word histories already collapsed in previous
  layers?
- Do same-parent aliases co-locate in the chosen generator/frontier layout?
- Does the authoritative owner receive all distinct parent contributions while
  suppressing retries idempotently?

## BFS Schreier stabilizer-coset questions

- What is the current state's stabilizer under the declared action convention?
- Which right cosets `Ks` of that stabilizer intersect the generator collection
  under the declared right action?
- What are the alias-class sizes `|S intersect Ks|`?
- Which generator labels lie in the stabilizer and therefore loop?
- Is simple support degree counting or excluding the loop endpoint explicitly?
- Are stabilizers merely equal in order, or does `S` intersect their conjugates
  uniformly?
- Is `S` invariant under the conjugations relating orbit states?
- Does transitivity of the action get mistaken for label-preserving
  vertex-transitivity of the fixed Schreier graph?
- Can two frontiers with equal state count and generator count have different
  support-arc output before visited lookup?
- Are loop mass and new-endpoint alias mass separated into `Y_d` and
  `X_d-P_d`?
- Did symmetry canonicalization introduce aliases absent in the free covering
  action?
- Does removing quotient loops preserve required labeled paths and replay?
- Is owner assignment correlated with stabilizer or orbit type?
- Are support-degree and alias histograms retained per complete BFS level?

## Reverse BFS and Schreier directionality questions

- Does backward traversal explicitly use `S^-1` or another proved predecessor
  oracle?
- Is the original support graph actually symmetric, or merely generated by
  invertible group elements?
- What are the forward right-coset `Ks` and reverse left-coset `sK`
  intersection histograms at each frontier state under the right action?
- Is the current stabilizer normal?
- Is `S` inverse-closed as a set and as a labeled occurrence collection?
- Do forward and reverse loop counts agree while nonloop aliases differ?
- Does the side-selection policy compare state frontier size, occurrence work,
  support arcs, or routed bytes?
- Can the smaller state frontier generate more unique or remote candidates?
- Are backward distance labels interpreted as distance to the target?
- Are meeting and stopping bounds independent of the physical side policy?
- Is every reverse label inverted correctly when reconstructing an original
  forward path?
- Are reverse aliases collapsed only when allowed by the requested path output?
- Does one owner partition yield different direction-specific routing matrices?
- After quotienting, were reverse generation and path lifting revalidated?

## BFS stabilizer-aware work-waterfall questions

- Are loop occurrences counted before they are collapsed into one support
  endpoint?
- Is nonloop same-parent alias excess computed after removing all loop labels?
- Does `G_d=L_d+R_d+P_d` hold for the declared generator occurrence semantics?
- Do all nonloop support arcs partition into visited and next-layer endpoints?
- Does `C_d-|F_(d+1)|` match cross-parent support convergence?
- Does the full waterfall reproduce `|S||F_d|` exactly?
- Are loop, alias, visited, and cross-parent terms recorded separately rather
  than as one duplicate ratio?
- Which terms are discardable for the requested output and which become
  metadata reductions?
- Does a free Cayley action actually give zero loop and same-parent alias terms
  under the loaded generator collection?
- Are generator transformation costs equal across states and labels, or only
  occurrence counts equal?
- At which physical scope can each semantic class meet?
- Is visited filtering attempted before routing with authoritative or merely
  advisory information?
- Are owner matrices retained before and after each combination boundary?
- Are retries excluded from graph occurrence counts?
- Is any composition ratio being presented incorrectly as expected speedup?

## Directed BFS arc-surplus and back-depth questions

- Is the count over distinct support arcs or labeled occurrences?
- Are all vertices and outgoing arcs in the source-reachable set included?
- Do arcs from `F_d` partition exactly into `B_d` and `F_(d+1)` endpoints?
- Does one predecessor arc per nonroot state form the claimed arborescence?
- Does the surplus identity reproduce `m-(n-1)` exactly?
- How much surplus is visited-ball mass versus next-state predecessor excess?
- What is the full `Q_(d,k)` back-depth spectrum?
- Is same-layer or backward-depth placement being mistaken for a directed cycle
  witness?
- Has return reachability or SCC membership actually been checked?
- Is `m-n+1` being mislabeled as directed cycle count?
- Can a large surplus occur in an acyclic input such as the ordered complete
  DAG?
- Do condensation arcs still jump toward shallower BFS layers?
- Which nonaccepting arcs remain required by the requested DAG, SCC, cycle, or
  graph output?
- Can visited-ball arcs be rejected authoritatively before routing?
- Do forward and reverse traversals retain separate surplus and lag profiles?

## BFS prefix-conservation and early-stop questions

- Which exact frontier depths have been expanded completely?
- Does the declared cut construct `B_d`, `B_(d+1)`, or only a partial next
  candidate set?
- Are logical successor obligations distinguished from resident records?
- Does the prefix occurrence identity hold through the last completed depth?
- For radius `R`, were all parents through `F_(R-1)` expanded?
- Is a partial parent subset being mistaken for complete `F_d`?
- If a target was found mid-layer, which output contracts are finalized?
- Were all equal-depth parents processed before claiming canonical/DAG/count
  closure?
- Does a negative result quantify over every relevant parent and successor?
- Are `NOT_WITHIN_RADIUS`, `UNREACHABLE`, and `UNKNOWN` separated?
- How do parent order, batch size, check location, and cancellation granularity
  affect reported work?
- Did an intermediate buffer overflow despite a fitting final frontier?
- Are claimed states durably published before work is considered complete?
- Have every rank, owner, message, retry, kernel, and spill reached the same
  consistent completion cut?
- Are completed semantic totals and partial latency totals reported separately?
- Does partial bidirectional stopping use a global unfinished-depth bound rather
  than first intersection?

## BFS conservation and verification-ladder questions

- Which checks are exact identities, and which are finite fingerprints?
- Can a balanced lost/extra-state mutation pass the current counters?
- Are totals retained per level, owner, and semantic category before reduction?
- Do retries carry stable logical occurrence identities?
- Is frontier equality collision-resolving and based on canonical states?
- Which successor and identity dependencies are shared by the implementation
  and its reference?
- What is the largest exhaustively validated domain?
- Are worker-count parity results being reported as evidence rather than proof?
- Which forced omission, insertion, alias, and retry mutations are detected?
- Are performance-scale claims separated from correctness-validation scope?

## BFS schedule-contract questions

- Which contract is claimed: closed layers, global minimum settlement, or fair
  label correction?
- Which physical event supplies the corresponding semantic closure evidence?
- Can a state be improved after first generation, first claim, or first pop?
- If labels improve, does every improvement reactivate dependent propagation?
- Are parent and move records versioned with the winning distance?
- What global lower bound makes a target final?
- Does a distributed minimum include delayed and device-produced work?
- Does quiescence include messages, kernels, retries, spills, and publications?
- Which outputs are schedule-confluent besides scalar distances?
- Are work metrics compared only between executions with compatible contracts?

## BFS work-coordinate questions

- What is the semantic unit: frontier state, generator occurrence, support arc,
  unique candidate, or accepted state?
- Which logical objects are materialized as physical records?
- How many probes, claims, transactions, and bytes correspond to each retained
  boundary?
- At which spatial and temporal scope do duplicates actually meet?
- Are terminal levels reported without dividing by zero accepted states?
- Are total and maximum-owner work and bytes both retained?
- Is semantic dependency depth separated from physical synchronization calls?
- Are remote occurrences, routed records, payload bytes, and protocol bytes
  measured separately?
- Which record fields account for changes in bytes per routed record?
- Do one- and multi-GPU runs use the same semantic counters and validation
  contract?
- Is retry, replay, or reactivation amplification separated from graph work?
- Which scalar summary is answering which named question?

## BFS scaling-regime questions

- Is the claim about one-query latency, fixed-work strong scaling, weak scaling,
  capacity, or independent-query throughput?
- Is the semantic workload identical at every GPU count used for speedup?
- Does a feasible matching one-GPU baseline exist?
- Which resident, scratch, replicated, routed, and allocator bytes limit usable
  capacity?
- What exact quantity is held constant per GPU in a weak-scaling study?
- How do diameter, reachable fraction, and frontier profile change with the
  growing instances?
- Are independent query batches kept distinct from multisource BFS?
- Are latency distributions reported alongside aggregate query throughput?
- Which levels gain or lose time as GPU count changes?
- Does an apparent superlinear result coincide with fit, spill, cache, or code
  path changes?
- Are topology and real transport paths recorded for multi-GPU claims?
- Are isolated primitive and simulated-routing results labeled below
  end-to-end scaling evidence?

## Cayley and Schreier ownership questions

- Which action side and matching coset side define ownership?
- Which generator labels lie in the chosen subgroup `H`?
- Are labeled occurrence locality and distinct support-arc locality separated?
- How is `F_d` distributed across cosets at every depth?
- Is `H` normal, and is quotient structure being used only when valid?
- How are many algebraic blocks mapped onto the available GPUs?
- For Schreier states, what are the `H`-orbit/double-coset sizes?
- Which stabilizers cause variable orbit capacity or outside-label locality?
- How do hash and algebraic ownership compare in per-owner frontier work rather
  than only total state balance?
- Do routing matrices expose hot destinations and convergence before scalar
  reduction?
- Which exact owner remains authoritative when replicas reject old states?
- Does lower routing coexist with worse frontier skew or critical time?

## Cayley quotient and owner-activation questions

- Is the subgroup normal under the declared Cayley action?
- Does quotient distance mean distance to a fiber or to one representative?
- Is first owner activation proved exact or only lower-bounded?
- How does quotient shell depth compare with observed per-coset frontier
  occupancy over all later levels?
- Are block transitions representative-independent for every generator label?
- If not, can two adjacent abstract arcs require incompatible middle states?
- Is path lifting proved by congruence, bisimulation, covering structure, or an
  explicit representative witness?
- Do Schreier `H`-orbits remain stable under every outside generator?
- Are abstract paths replayed concretely before being used as connectors?
- Is a quotient lower bound being mistaken for a load or byte prediction?
- How are several quotient blocks mapped onto physical GPUs?
- Are exact-fiber, lower-bound, and heuristic quotient claims clearly labeled?

## BFS fiber and re-entry questions

- What is the full distance multiset inside each fiber, beyond its minimum?
- At which nonconsecutive original depths is each owner block active?
- Does `S intersect H` generate the subgroup fiber?
- Are fibers geodesically convex or isometric under the full generator metric?
- Can a shortest path leave and re-enter its target owner block?
- Which lifted quotient entry representative minimizes a fixed-target path?
- Is an additive quotient-plus-local formula actually proved?
- Are owners retained and reactivated after empty frontier intervals?
- Which concrete parent fibers contribute to later states in one block?
- Is one quotient parent being mistaken for a concrete predecessor DAG?
- Does a bidirectional block meeting contain a common exact state or a proved
  connector?
- Are repeated active intervals and retained resident bytes visible in traces?

## Cayley quotient routing-matrix questions

- What quotient image and multiplicity does every generator label have?
- Which distinct generator elements collide only at the owner-image level?
- Does the observed coset-to-coset occurrence matrix match
  `f_d(C)mu(C^-1D)`?
- Are quotient routing aliases kept separate from concrete state duplicates?
- How does the coset-to-GPU map change diagonal and off-diagonal occurrence
  totals?
- Which logical occurrences are removed or combined before physical routing?
- Are records, messages, retries, payload bytes, and protocol bytes separated?
- Which concrete endpoint identities occupy each quotient destination bin?
- Does reverse traversal use inverse quotient images and its own frontier
  histogram?
- Is Cayley convolution structure being assumed for a Schreier action without
  transition congruence?
- Are accepted states being inferred incorrectly from raw destination counts?
- Which validation rung supports each predicted or measured matrix?

## Cayley convolution and frontier-nonlinearity questions

- Does the raw destination histogram match quotient convolution exactly?
- How many raw occurrences collapse to distinct concrete endpoints per block?
- How many distinct endpoints are already in the authoritative visited ball?
- Can two concrete frontiers with the same block histogram produce different
  next frontiers on the target workload?
- At which levels does collision-free equality `f_next=y` stop holding?
- Are occurrence and remaining-capacity ceilings being mistaken for forecasts?
- Which within-block identity/intersection information is retained by a model?
- Does local precombination change owner-received multiplicity without changing
  logical raw routing?
- Are quotient-bin collisions distinguished from state and label aliases?
- Do validation artifacts compare canonical frontier sets, not only block
  histograms?
- For Schreier blocks, are variable orbit sizes used in capacity ceilings?
- Which collision/visited assumptions behind accepted-progress predictions have
  been tested independently?

## BFS idempotence and merge-algebra questions

- Is the output merged by set union, minimum, set-of-parents union, addition,
  multiset accumulation, or sequence concatenation?
- Which operations are associative, commutative, and idempotent?
- Does every logical successor arrive at least once despite retries and loss?
- Can a visited claim become durable without durable pending frontier work?
- Which metadata side effects must commit with membership?
- Are arbitrary and canonical parent contracts distinguished?
- Does canonical parent finalization wait for every equal-depth contender?
- What stable identity prevents retry from double-counting a path contribution?
- Are same-layer reorder tests separated from cross-layer disorder tests?
- Can a deeper first claim suppress a delayed shorter proposal?
- Does final-set parity conceal wrong parents, multiplicities, or ordering?
- Which typed reduction and closure event belongs to each reported output?

## BFS proof-obligation independence questions

- Are graph soundness and graph completeness proved in both directions?
- Is exact identity validated independently of the successor oracle?
- Does reachable-set correctness use the requested metric and logical step?
- Which theorem makes tentative or first-claim distances final?
- Does oracle completeness quantify over calls only, or traversal coverage over
  every required parent and route?
- Can reached membership exist without pending-or-expanded publication state?
- Which richer outputs remain unproved despite exact scalar distances?
- Is termination safety separated from termination-detection liveness?
- Can a correct execution lack adequate evidence, or matching evidence share a
  common-mode bug?
- Which assumptions does each output theorem actually consume?
- Which one-GPU quantifiers expand when workers and channels are introduced?
- Which meaning of “complete” is attached to each status and artifact?

## BFS obligation-conservation and termination-cut questions

- What stable identity names each logical successor obligation?
- At which semantic event is an obligation retired?
- Can responsibility disappear between sender decrement and receiver increment?
- Are channel/device/kernel locations part of the conserved cut?
- Do retries preserve one logical ID while recording physical copies separately?
- Can duplicate or lost acknowledgements create false zero or permanent nonzero?
- Is accepted-state publication complete before causal credit returns?
- Which dynamically created child obligations enter the outstanding total?
- Are termination safety and liveness failure counters distinguished?
- What is the minimum unfinished depth/key, not only the total count?
- Which output-specific metadata can still change after scalar target finality?
- Does checkpoint/repartition retain or recreate every obligation in the right
  epoch?
- Are per-rank totals sampled from one consistent cut?
- Which device-side objects can still emit authoritative work after host idle?

## BFS shortlex-rank and distributed-determinism questions

- Is the requested deterministic object a state order, parent ID, vertex path,
  or move word?
- Does the parent key encode canonical path order rather than semantic state ID?
- Are all equal-depth contender keys visible before per-child minimum finalizes?
- Is the next frontier ranked globally by selected canonical-word keys?
- Can owner-local ranks disagree with the global shortlex order?
- Are retry duplicates harmless while lost smaller contenders remain detectable?
- Do canonical parents and ranks remain identical across GPU/owner counts?
- Which generator alphabet, label order, action side, and path identity define
  the canonical epoch?
- Does multi-source ordering prioritize source ID or path word first?
- What closure proves no smaller target word remains at the same distance?
- Does a quotient canonical word lift to the requested concrete target?
- Are exact frontier sets compared separately from canonical parent/word output?

## Bidirectional shortlex and connector-closure questions

- Does reverse BFS rank forward-oriented suffixes by prepend order?
- Are inverse traversal operations separated from stored forward labels and
  their lexical order?
- Can ordinary reverse discovery order choose `ba` before forward-shortlex
  `ab`?
- Are both vertex meetings and crossing-edge connectors considered?
- Are canonical prefix and suffix reductions closed for every optimal
  connector?
- How are full solution words compared across different forward/reverse split
  depths?
- Is a meeting state/owner ID being used as a path-order surrogate?
- Which theorem excludes smaller equal-length words after distance is final?
- Can an equal-length lexical contender remain in flight on another owner?
- Does canonical output remain invariant across GPU counts and side schedules?
- Are quotient suffixes lifted to the fixed concrete target and frame?
- Is one canonical connector being mistaken for all-shortest path/count closure?

## CayleyPy and DeepCubeA action-transfer questions

- What single 54-position bijection conjugates all six face actions at once?
- Does each CayleyPy positive face label map to DeepCubeA sign `1` or `-1`?
- Is any apparent mismatch caused by position flattening, action side, or a
  genuinely different face convention?
- Do the complete signed-label map and position map replay mixed-face words,
  rather than only one-face powers?
- Are exact 54-ID search equality and six-class neural observation kept
  separate in every oracle and visited check?
- Which outputs need only unlabeled graph isomorphism, and which require exact
  labeled action conjugacy?
- Are source commits and both generated permutation-table hashes emitted by the
  Docker comparison?
- Does the cross-runtime state map rename both positions and unique sticker
  IDs, so that solved state maps to solved state?
- After fixing one anchor, is the simultaneous position map unique, or does an
  action centralizer leave multiple valid reports?
- Which mixed-word fingerprints are used only for candidate rejection, and
  where is the final all-generator equation checked?
- Does the signed-label map preserve the requested shortlex alphabet order, or
  only shortest-path validity?
