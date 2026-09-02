# Research roadmap

## Latest recorded direction — 2026-08-31

Correct the retained research findings first. Plugin design and implementation
are deferred by the user. The phases below describe the study's historical
structure, not a renewed autonomous goal. Track corrections through the
[audit](reviews/2026-08-31-bfs-research-audit.md) and
[correction record](reviews/2026-08-31-bfs-audit-corrections.md);
unavailable runtime evidence remains unavailable until separately requested.

## Goal

Develop an evidence-backed, intuitive and mathematical understanding of exact
BFS: what object it computes, why its guarantees hold, how its variants differ,
and how graph/representation structure changes its behavior.

GPU and multi-GPU performance are study subjects, not an instruction to design
a production optimizer. Existing code and retained results may be inspected;
creating or modifying executable code and running any new probe requires a
separate explicit user request, even for a minimal oracle or counterexample.
Negative results and conceptual corrections are first-class deliverables.

### Anti-drift rules

- Start with one plain-language BFS question. If the proposed activity cannot
  say how answering it will change the mental model, do not do it.
- Choose the next activity from an explicit knowledge gap, not from an urge to
  add another note, claim, source audit, implementation, or optimization.
- Before adding a synthesis note, check whether that synthesis already exists.
  Prefer a tiny hand-worked example or counterexample over a new thematic note.
- Note count, claim count, source count, topic coverage, and a filled evidence
  matrix are bookkeeping, never success metrics and never reasons for the next
  step.
- Do not follow self-generated chains of narrow lemmas. After at most three
  related question cards, switch back to synthesis or another foundational BFS
  axis.
- Prefer a simple trace that improves the ordinary BFS mental model over a new
  result in adjacent algebra, graph theory, distributed systems, or hardware.
- Treat excessive formal depth without a clearer BFS explanation as scope
  drift, even when every individual statement is correct.
- A library, benchmark, validator, paper implementation, or distributed
  protocol is inspected only when a previously stated BFS question requires
  it. Do not browse such systems merely because they are adjacent to BFS.
- Conceptual understanding, bounded evidence, and real target-runtime evidence
  are separate statuses. Never promote one to another without the corresponding
  proof or measurement.
- GPU and multi-GPU performance is deferred until the underlying semantic
  phenomenon is understood. A probe requires explicit authorization for that
  named experiment; understanding a hypothesis does not authorize execution,
  architecture design, tuning, or a reusable optimized engine.
- Code is exceptional, not the default study medium. First try a hand trace,
  minimal graph, counterexample, or proof sketch.
- Rust is used for research and host/oracle code. C++ is permitted only inside
  explicitly requested CUDA/GPU translation units.
- Every build, test, calculation, benchmark, and executable probe runs in
  Docker. If Docker is unavailable, record `not run` and continue with
  non-executable study; do not repair infrastructure as part of this goal.
- Real multi-GPU work starts only from a narrow evidence question and must
  preserve an independent semantic oracle. A conceptual gap alone does not
  authorize a production implementation.

## Phase 1: correctness model

**Status of the phase lists below:** These are the historical study structure
and descriptions of potential evidence, not executable instructions or an
active backlog. Imperatives such as write, test, compare, generate, or measure
do not override the authorization gate in `research-protocol.md`. Existing
reports record past activity only. Read-only source/artifact study and hand
reasoning remain permitted; each new executable task needs a separate explicit
user request. Current evidence gaps are tracked in note 177.

- Define level-synchronous BFS precisely.
- Write a small deterministic CPU reference.
- Validate distances, frontier disjointness, and parent paths.
- Cover self-loops, duplicate edges, disconnected components, and multiple
  shortest parents.

Deliverable: reference implementation and exhaustive tests on small graphs.

## Phase 2: implicit state spaces

- Replace adjacency lists with a `neighbors(state)`/generator interface.
- Compare full-state visited keys with hash-indexed exact comparison.
- Test deliberately colliding hashes.
- Measure branching factor, duplicate ratio, frontier growth, and bytes per
  discovered state.

Deliverable: CPU reference for small permutation/Cayley graphs.

## Phase 3: CPU representation experiments

- Queue/hash-set baseline.
- Sort/unique frontier processing.
- Dense-ID bitmap baseline where a bijective rank is available.
- Bidirectional BFS where original-graph predecessors can be enumerated;
  inverse generator actions are one way to provide that interface, not a
  requirement that inverse moves be allowed in the forward graph.

Deliverable: reproducible representation comparison, not a universal winner.

## Phase 4: single-GPU explicit BFS concepts

- Start with integer/CSR graphs to isolate GPU frontier mechanics.
- Study expand, flag, prefix-sum compact, and visited marking as conceptual
  decompositions; use small probes only where they answer a concrete question.
- Compare work-efficient and idempotent duplicate handling.
- Measure load balance and memory bandwidth.

Deliverable: evidence-backed understanding of which primitive answers which BFS
obligation, not a production implementation.

## Phase 5: single-GPU implicit BFS concepts

- Generate neighbors by applying state transformations.
- Evaluate conceptually how state layout and generator placement affect work.
- Compare exact visited designs through sources and bounded measurement probes.
- Keep full-state equality in the correctness path unless a bijective encoding
  has been proved.

Deliverable: a clear model of implicit-graph bottlenecks and correctness risks.

## Phase 6: multi-GPU concepts

- Define deterministic ownership.
- Study local pre-deduplication and variable-size exchange.
- Make the owner authoritative for visited membership.
- Formalize collective level termination and target detection.
- Measure communication volume, skew, and synchronization time.

Deliverable: ownership, communication, termination and correctness models,
supported by small simulations rather than an unsolicited production system.

## Phase 7: performance understanding

- Classify workloads by explicit/implicit representation, state width,
  branching factor, diameter, duplicate ratio, and frontier shape.
- Study frontier, visited, expansion, and communication strategies using papers,
  traces and bounded probes.
- Interpret kernel timelines, memory traffic, collectives, and synchronization
  when existing evidence makes them relevant.
- Describe trade-offs among throughput, memory capacity, latency and
  determinism without attempting to build a universal selector.
- Keep exact semantics explicit whenever discussing hardware behavior.

Deliverable: evidence-backed intuitions, applicability boundaries and open
questions rather than an automatically optimized implementation.

## Later, separate investigations

- Direction-optimizing push/pull.
- Bidirectional and multi-source search.
- State canonicalization and symmetry quotients.
- Compression and perfect/rank encodings.
- Host/NVMe external-memory BFS.
- Multi-node topology-aware routing and checkpoint/restart.

These are separate study topics, not an implicit implementation backlog.
