# Universal CayleyPy BFS Implementation Plan

> Execute directly with superpowers:executing-plans. User forbids subagents and local development/builds; implementation and tests run on authorized remote 2 RTX3060. The user approved design and instructed execution with "Proceed".

**Goal:** One convenient graph-definition entry point for all CayleyPy graph semantics, bounded measured automatic planning and explicit rank/node placement, retaining exact BFS and compact SHARD_AB.
**Architecture:** Versioned lossless graph contracts separate successors, equality, traversal history and optional output retention. CUDA generic successors feed the existing shard pipeline; planner inventories topology/memory and admits preallocated candidate profiles before calibration. Exact directed graphs use all-visited membership.
**Tech stack:** Rust1.75 core/runtime, CUDA/NCCL, Python startup adapter, CayleyPy pinned for validation.
**Spec:** universal-bfs-design.md from private HF audit revision6885c771dccc58e84739d37669c4903ac038d0e9.

## Global constraints
No local domain edits/builds, no subagents, no B200/B300 rental, no states downloaded to PC, total authorized budget$10 (reserve$1 for earlier audit), one GitHub branch, all semantic fallbacks explicit and verified.

## Review focus
Directed generators must never select two-layer history. Matrix multiplication wraps int64 before modulus as CayleyPy does. Repeated labels and rectangular/singular starts preserve actions. Rank mapping must not conflate global/local indices or assume peer access. Startup tuning must not enter the hot loop or claim a global optimum.

### 1. Lossless graph contract and adapter
Files: crates/mgbfs-core/src/graph_definition.rs, tests/graph_definition.rs; multigpubfs/graph_definition.py, tests/test_graph_definition.py.
Interface: GraphDefinitionV2 validates permutation and matrix data with per-generator modulus, shape, exact i64 start and canonical identity. Python from_cayleypy(definition) serializes without constructing a different graph. CPU successor/oracle is independent full-state reference only.
- [ ] Failing fixtures for LX/direction, repeated labels, rectangular/singular matrices, mixed moduli and overflow.
- [ ] Implement validation/canonical digest and adapter.
- [ ] Remote CPU tests and actual pinned CayleyPy definition roundtrip.

### 2. Generic CUDA successors and exact equality
Files: cuda/generic_graph.cu/.h, Rust CUDA bridge and native dispatch.
Interface: coalesced preallocated SoA int64 state action; supplied permutation/matrix tables; rows/global-local shape explicit; child keys coupled to exact state/origin metadata.
- [ ] GPU primitive oracle matching every child, including overflow, width and per-generator arithmetic.
- [ ] Integrate generic successors with hash-first selected regeneration, no CPU child generation and no placeholder LRX substitution.
- [ ] Resolve hash collisions with full state comparison.

### 3. Directed visited membership and retention
- [ ] Persistent visited per shard for directed/non-inverse-closed graphs; explicit budget and exact append/membership.
- [ ] Prove inverse-closed two-layer selection by graph semantics, never library name.
- [ ] Complete small layer/state oracles, cancellation/resource/fault retention and full/compact writer tests.

### 4. Unified launch and auto plan
Files: Python package public API/CLI, planner modules, native query integration, installation metadata and examples.
- [ ] No HF/deadline/source-path requirement for local use.
- [ ] Device/host/cgroup inventory, actual warmed native allocation candidates, bounded performance calibration/cache identity.
- [ ] Batch/shard/bank/hash-sort candidates selected using matched prefixes and existing counters at safe boundaries; distinguish capacity admission from performance evidence.
- [ ] Clean install and 1/2GPU test against CPU oracle.

### 5. Node/rank topology
Files: Rust topology/bootstrap/reference_launch; distributed launcher/merge.
- [ ] Vector rank validation and global/local node placement,128rank simulated contracts.
- [ ] Actual multihost rendezvous/output identity/control faults; HOST fallback and LSA capability gate.
- [ ] Never call simulated topology a128GPU data-plane or throughput validation.

### 6. Acceptance and delivery
- [ ] Catalog family coverage + honest parameter/resource boundaries.
- [ ] Heavy same-prefix baseline timing for changed paths.
- [ ] Publish single branch and immutable sources/proofs; checksums/readback.
- [ ] Token removal, exact rental deletion/API ABSENT; completion audit of every requirement.
