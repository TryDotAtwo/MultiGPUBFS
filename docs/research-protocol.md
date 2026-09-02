# Research protocol

## Purpose

### Corrected goal and authority

Изучить и прочувствовать BFS: его сущность, инварианты, гарантии, вариации,
frontier/visited и поведение на явных, неявных и Cayley-графах. Разбирать
небольшие примеры вручную, читать источники и существующий код, записывать
наблюдения, ошибки, выводы и открытые вопросы. GPU и multi-GPU рассматривать
как контекст понимания алгоритма, а не задачу разработки или оптимизации.
Не писать и не изменять исполняемый код, не запускать новые эксперименты,
зонды, тесты или бенчмарки без отдельного явного запроса пользователя.
Это относится и к минимальным учебным примерам. Если код отдельно разрешён:
Rust для host/research, C++ только для GPU-кода, сборки и запуски в Docker.
Критерий прогресса — понимание алгоритма, а не скорость реализации,
количество заметок или непрерывная активность.

Historical scope correction: an earlier goal text permitted measurement probes
and restricted only substantial GPU code. The user's later restriction above
superseded it. That old text is not a current permission or tool-state claim;
the audit's 2026-08-31 `get_goal` check returned no active goal. Do not manufacture
completion merely to replace an objective or infer authority from stale text.

Latest recorded direction, 2026-08-31: correct the findings in the retained
research before considering a plugin. Plugin planning and implementation are
deferred. This is a bounded correction task, not a restart of open-ended study.
Existing implementations and historical results are explanatory evidence, not
authority to create or run new ones. Any separately requested executable work
remains scoped to that request. Preserve how conclusions were reached,
including failed ideas and ambiguous evidence.

## Understanding-first gate

The purpose of a study step is to change or sharpen the mental model of BFS.
Producing a note, increasing coverage, inspecting an implementation, or adding
an experiment is not progress by itself.

Before starting a step, write one plain-language question and answer these
checks:

1. What about BFS should become more intuitive or less ambiguous?
2. What is the smallest example, counterexample, source passage, or trace that
   can answer it?
3. What observation would actually change the current understanding?

Do not proceed when the honest purpose is merely to:

- add another topic to the corpus;
- increase note, claim, source, test, or coverage counts;
- audit a library, benchmark, validator, or distributed protocol without a
  prior BFS question that requires it;
- turn a conceptual observation into an architecture, optimization backlog, or
  production implementation;
- measure performance before the semantic phenomenon being measured is clear.

The default study unit is a short question card recorded in the research log:
question, concrete example, prediction, observed reasoning, correction or new
intuition, and remaining uncertainty. Create a standalone thematic note only
when the result cannot be stated clearly in that card or materially revises an
existing synthesis.

## Depth and drift limit

The goal is to build a coherent, gradually deepening intuition for BFS, not to
enumerate every theorem that can be derived from it. A study cycle addresses
exactly one question and should normally end after one smallest useful example
and one plain-language conclusion.

Stop the cycle without adding material when any of these is true:

- the question is a minor refinement of a claim already recorded;
- answering it requires several new layers of terminology before it clarifies
  BFS itself;
- its relevance is mainly to path-count algebra, group theory, distributed
  protocols, or hardware rather than to the current BFS mental model;
- the result would increase formal coverage but would not change how BFS is
  explained, traced, or recognized in practice;
- the next step is chosen only because it follows from the previous lemma.

After at most three closely related question cards, return to the roadmap and
choose a different foundational axis or synthesize what was learned. Do not
continue a self-generated chain of lemmas. Breadth, synthesis, and revisiting
simple examples take priority over novelty and corpus size.

Progress is evaluated by whether the current mental model can be explained
more simply and connected back to ordinary BFS. Note count, claim count, token
use, mathematical novelty, and uninterrupted activity are explicitly not
progress metrics.

Code and execution are outside the default authority of this study goal. Use
sources, inspection of existing code, hand reasoning, and small written traces.
Missing, ambiguous, or desirable executable evidence must remain an **Unknown**
or a dormant evidence gate; its usefulness is never authorization to obtain it.

A separate explicit user request is required before any of the following:

- creating or modifying a probe, test, benchmark, demo, simulator, or other
  executable code, including small CPU/Rust counterexamples;
- building or running newly created experimental code;
- starting a Docker container for an experiment, calculation, measurement, or
  demonstration;
- performing a new CPU, GPU, or multi-GPU experiment or optimization attempt.

When such an experiment is explicitly authorized, research/host code is Rust
and all builds and runs happen in Docker. C++ is reserved for explicitly
requested GPU code. The authorization is scoped to the named experiment and
does not carry forward to later evidence gaps.

## Evidence labels

Every substantial note should use one of these labels when the status is not
obvious:

- **Definition**: mathematical or interface meaning adopted by the project.
- **Fact**: supported by a primary source, inspected code, or reproducible test.
- **Observation**: directly seen in a named experiment or trace.
- **Hypothesis**: plausible explanation or expected result that needs testing.
- **Inference**: conclusion derived from listed facts and assumptions.
- **Decision**: current engineering choice, including scope and reasons.
- **Failure**: attempted method that did not meet correctness or performance
  criteria.
- **Unknown**: unresolved question or missing evidence.

Source-backed facts should link the primary paper, specification,
documentation, code revision, or local artifact. Measurements should name the
exact command/configuration and retained raw output.

## Experiment record

Each experiment should record:

```text
id and date
question
hypothesis
algorithm and exact semantics
code revision
hardware and topology
software/toolchain versions
dataset or implicit graph definition
parameters and memory capacities
commands
correctness oracle and result
raw artifact paths
metrics
unexpected observations
failure mode, if any
interpretation
next experiment
```

An out-of-memory run, timeout, incorrect result, noisy measurement, or negative
speedup is still a useful result. It must not be silently omitted from a sweep.

Retain each attempted run under a distinct immutable artifact identity, including
rejected runs. Record which corrected run supersedes it instead of overwriting
raw evidence. If an old artifact was already overwritten, mark its absence
explicitly; do not invent samples or treat a new run as recovery of old bytes.

## Correctness gate

Performance results are eligible for comparison only after the implementation
passes the relevant correctness oracle:

- exact distances or frontier sets against a trusted CPU reference at tractable
  scale;
- valid parent edges and replayable paths;
- synthetic collision tests for hash-indexed visited structures;
- no silent frontier, visited, exchange, or output overflow;
- multi-GPU parity across different rank counts and ownership layouts;
- explicit declaration of any quotienting, canonicalization, or bounded-depth
  semantics.

Approximate search, probabilistic loss, and beam pruning belong to separately
named algorithms and must not be reported as exact BFS.

## Performance gate

A claim of improvement requires:

- the same BFS semantics and graph instance;
- the same correctness gate;
- warm-up policy and repeated measurements;
- distribution statistics rather than a single favorable sample;
- component timing sufficient to explain the change;
- peak memory and capacity headroom;
- hardware utilization and communication evidence where relevant.

Useful metrics include:

- generated transitions/s;
- traversed edges/s for explicit graphs;
- unique accepted states/s;
- time per level and end-to-end time;
- duplicate and already-visited ratios;
- bytes read, written, sorted, stored, and communicated;
- peak host RAM and VRAM;
- load imbalance across blocks, GPUs, and nodes;
- kernel, collective, and synchronization time;
- energy or power when hardware exposes reliable counters.

## Single-GPU efficiency questions

- Is traversal limited by neighbor generation, memory bandwidth, visited lookup,
  sorting, atomics, or launch/synchronization overhead?
- Does the representation permit dense bitmap visited, or require exact
  full-state comparison?
- Which frontier sizes justify changing kernels or traversal direction?
- Does fusion reduce traffic without causing register pressure or lost
  concurrency?
- How do candidate structures trade throughput, memory capacity, and exactness
  across different workload shapes?

## Multi-GPU efficiency questions

- What entity owns a state, visited record, parent, and frontier slot?
- How much work is eliminated before network transfer?
- Are payloads better compressed, ranked, or regenerated remotely?
- Is communication all-to-all, neighborhood, hierarchical, or partitioned into
  passes?
- What are the synchronization and termination costs per level?
- How do topology, ownership skew, and frontier evolution change scaling?
- Is measured speedup strong scaling, weak scaling, or merely additional
  capacity?

## Research hygiene

- Keep general BFS, explicit CSR BFS, and implicit state-space BFS separate.
- Distinguish algorithmic work reduction from hardware throughput improvement.
- Do not infer current hardware performance from historical paper numbers.
- Preserve rejected hypotheses and the evidence that rejected them.
- Revisit old conclusions when state width, graph family, GPU generation, or
  scale changes.

## Correction and completion discipline

- Correct an erroneous canonical passage and its dependent synthesis, claim
  rows, questions, and conclusions together. Adding a later correct note alone
  leaves the earlier false statement available for reuse.
- Keep a dated correction trail. A retracted assertion may remain as history
  only when the retraction and corrected result are explicit at that location.
- Check both a theorem and a warning against their actual premises. In
  particular, a counterexample outside the claimed graph/output/schedule
  contract does not refute the conditional theorem.
- Record proof, hand witness, source inspection, finite execution, and actual
  target-runtime evidence separately. Matching fingerprints are not exact set
  equality; aggregate timing is not a causal bottleneck decomposition.
- A status index states the latest recorded outcome and links earlier failed
  attempts. Chronological prose can be stored out of order; neither file
  position nor an old `not run` paragraph proves the current status.
- Completion names the bounded task and its verified deliverables. It does not
  certify every statement in the archive, fill unavailable evidence, or create
  permission for a follow-on experiment or plugin.
