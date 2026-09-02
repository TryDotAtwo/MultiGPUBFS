# MultiGPUBFS

Research workspace for understanding exact breadth-first search, from a small
CPU reference to conceptual and measurement-based study of GPU and multi-GPU
traversal over explicit and implicit graphs.

The repository contains study notes, deterministic CPU correctness oracles,
and reproducible reference experiments.

The current implementation target is specified in
[Multi-GPU Cayley BFS architecture](ARCHITECTURE_NEED.md).

Native matrix implementation is in `crates/` and `cuda/`. It is **incomplete**:
CPU contracts and individually tested GPU generation/hash/routing primitives
exist; there is no production multi-GPU BFS executable yet. See the
[implementation status and local test commands](docs/native-matrix-implementation.md).
The Python package and `gpu/` / `rust/` trees below remain research prototypes,
not fallback implementations of the new runtime.

The notes deliberately separate three domains:

1. classical BFS theory and correctness;
2. high-performance BFS over explicit integer/CSR graphs;
3. BFS over implicit state spaces such as Cayley graphs.

These domains share frontier-processing primitives, but their memory models and
bottlenecks are not interchangeable.

## Study index

- [Research audit, 2026-08-31](docs/reviews/2026-08-31-bfs-research-audit.md)
- [Audit corrections and remaining evidence limits](docs/reviews/2026-08-31-bfs-audit-corrections.md)
- [Research roadmap](docs/roadmap.md)
- [Research protocol](docs/research-protocol.md)
- [Evidence map: facts, observations, rejected claims, and unknowns](docs/evidence-map.md)
- [A mental model of BFS: from metric balls to hardware](docs/notes/54-a-mental-model-of-bfs-from-metric-balls-to-hardware.md)
- [Validating implicit successor completeness](docs/notes/55-validating-implicit-successor-completeness.md)
- [Partial-layer bidirectional stopping and global bounds](docs/notes/56-partial-layer-bidirectional-stopping-and-global-bounds.md)
- [BFS output contracts and finalization boundaries](docs/notes/57-bfs-output-contracts-and-finalization-boundaries.md)
- [CayleyPy output-contract audit](docs/notes/58-cayleypy-output-contract-audit.md)
- [REF-010 exact distributed BFS output audit](docs/notes/59-ref010-exact-distributed-bfs-output-audit.md)
- [Short relations and BFS duplicate signatures](docs/notes/60-short-relations-and-bfs-duplicate-signatures.md)
- [Relations, stabilizers, and BFS state identity](docs/notes/61-relations-stabilizers-and-bfs-state-identity.md)
- [Current CayleyPy Megaminx vertex and equality contract](docs/notes/62-current-cayleypy-megaminx-vertex-and-equality-contract.md)
- [Real Megaminx short relations in BFS](docs/notes/63-real-megaminx-short-relations-in-bfs.md)
- [Word history versus frontier-record multiplicity](docs/notes/64-word-history-versus-frontier-record-multiplicity.md)
- [Static and conjugated independence in Cayley BFS](docs/notes/65-static-and-conjugated-independence.md)
- [Trace quotient versus equality in a Cayley group](docs/notes/66-trace-quotient-versus-group-equality.md)
- [Generator-order parity and early BFS signatures](docs/notes/67-generator-order-parity-and-bfs-signatures.md)
- [Generator-set changes alter visited work](docs/notes/68-generator-set-changes-visited-work.md)
- [Cartesian-product BFS: additive distance and frontier convolution](docs/notes/69-cartesian-product-bfs.md)
- [Direct-product Cayley graphs and BFS shuffle signatures](docs/notes/70-direct-product-cayley-bfs.md)
- [How arbitrary can a BFS frontier profile be?](docs/notes/71-arbitrary-bfs-frontier-profiles.md)
- [Dead ends, pockets, and radial progress in Cayley BFS](docs/notes/72-dead-ends-and-cayley-bfs.md)
- [FIFO queue occupancy versus frontier width](docs/notes/73-fifo-queue-occupancy-versus-frontier-width.md)
- [Discovery, settlement, and duplicate-tolerant BFS queues](docs/notes/74-discovery-settlement-and-duplicate-queues.md)
- [Matrix orientation, vxm/mxv, and directed BFS](docs/notes/75-matrix-orientation-and-directed-bfs.md)
- [Multi-source balls superpose; distance frontiers do not](docs/notes/76-multisource-balls-superpose-frontiers-do-not.md)
- [Source-set updates and BFS layer migration](docs/notes/77-source-set-updates-and-layer-migration.md)
- [BFS landmarks, triangle bounds, and Cayley homogeneity](docs/notes/78-bfs-landmarks-triangle-bounds-and-cayley-homogeneity.md)
- [Resolving sets, metric dimension, and BFS coordinates](docs/notes/79-resolving-sets-metric-dimension-and-bfs-coordinates.md)
- [BFS distance embeddings and strong resolution](docs/notes/80-bfs-distance-embeddings-and-strong-resolution.md)
- [BFS-tree root exactness and pairwise stretch](docs/notes/81-bfs-tree-root-exactness-and-pairwise-stretch.md)
- [BFS trees, fundamental cycles, and Cayley relators](docs/notes/82-bfs-trees-fundamental-cycles-and-cayley-relators.md)
- [BFS trees, fundamental cuts, and bridges](docs/notes/83-bfs-trees-fundamental-cuts-and-bridges.md)
- [BFS, strong components, and condensation distance](docs/notes/84-bfs-strong-components-and-condensation-distance.md)
- [BFS depth slack and directed period](docs/notes/85-bfs-depth-slack-and-directed-period.md)
- [Eventual walk lengths, primitivity, and BFS first arrival](docs/notes/86-eventual-walk-lengths-primitivity-and-bfs-first-arrival.md)
- [BFS walks, simple paths, and history state](docs/notes/87-bfs-walks-simple-paths-and-history-state.md)
- [BFS trails, edge history, and line digraphs](docs/notes/88-bfs-trails-edge-history-and-line-digraphs.md)
- [BFS trees, shortest gateways, and dominators](docs/notes/89-bfs-trees-shortest-gateways-and-dominators.md)
- [BFS separators, dominators, and Menger paths](docs/notes/90-bfs-separators-dominators-and-menger-paths.md)
- [Reverse BFS, postdominators, and inevitable targets](docs/notes/91-reverse-bfs-postdominators-and-inevitable-targets.md)
- [Reachability-preserving graphs, BFS metric, and generators](docs/notes/92-reachability-preserving-graphs-bfs-metric-and-generators.md)
- [Cayley word metrics, generator changes, and BFS growth](docs/notes/93-cayley-word-metrics-generator-changes-and-bfs-growth.md)
- [BFS boundaries, Følner sets, and Cayley amenability](docs/notes/94-bfs-boundaries-folner-sets-and-cayley-amenability.md)
- [BFS versus random-walk hitting and cover time](docs/notes/95-bfs-versus-random-walk-hitting-and-cover-time.md)
- [BFS, flooding, rumor spreading, and message time](docs/notes/96-bfs-flooding-rumor-spreading-and-message-time.md)
- [BFS balls, separators, and ends of Cayley graphs](docs/notes/97-bfs-balls-separators-and-ends-of-cayley-graphs.md)
- [BFS on percolated Cayley graphs and random frontiers](docs/notes/98-bfs-on-percolated-cayley-graphs-and-random-frontiers.md)
- [BFS, geodesic languages, and automatic Cayley graphs](docs/notes/99-bfs-geodesic-languages-and-automatic-cayley-graphs.md)
- [BFS level graphs, blocking flows, and matching phases](docs/notes/100-bfs-level-graphs-blocking-flows-and-matching-phases.md)
- [BFS distance profiles, color refinement, and isomorphism limits](docs/notes/101-bfs-distance-profiles-color-refinement-and-isomorphism-limits.md)
- [BFS, local message passing, and neural receptive fields](docs/notes/102-bfs-local-message-passing-and-neural-receptive-fields.md)
- [BFS, succinct graphs, and description size](docs/notes/103-bfs-succinct-graphs-and-description-size.md)
- [BFS versus topological waves and critical-path levels](docs/notes/104-bfs-versus-topological-waves-and-critical-path-levels.md)
- [BFS distance transforms and discrete wavefronts](docs/notes/105-bfs-distance-transforms-and-discrete-wavefronts.md)
- [BFS versus union-find and connectivity state](docs/notes/106-bfs-versus-union-find-and-connectivity-state.md)
- [BFS trees versus minimum spanning trees](docs/notes/107-bfs-trees-versus-minimum-spanning-trees.md)
- [BFS distance versus effective resistance](docs/notes/108-bfs-distance-versus-effective-resistance.md)
- [BFS, geodesic triangles, and graph hyperbolicity](docs/notes/109-bfs-geodesic-triangles-and-graph-hyperbolicity.md)
- [BFS balls, convexity, gates, and the Helly property](docs/notes/110-bfs-balls-convexity-gates-and-helly-property.md)
- [BFS intervals, triple medians, and partial cubes](docs/notes/111-bfs-intervals-triple-medians-and-partial-cubes.md)
- [BFS layers, modular graphs, and weak modularity](docs/notes/112-bfs-layers-modular-and-weakly-modular-graphs.md)
- [BFS frontiers, treewidth, and layered treewidth](docs/notes/113-bfs-frontiers-treewidth-and-layered-treewidth.md)
- [BFS, LexBFS, chordal graphs, and elimination orders](docs/notes/114-bfs-lexbfs-chordal-graphs-and-elimination-orders.md)
- [BFS, isometric subgraphs, and distance-hereditary graphs](docs/notes/115-bfs-isometric-subgraphs-and-distance-hereditary-graphs.md)
- [BFS, spanners, emulators, hopsets, and generator substitution](docs/notes/116-bfs-spanners-emulators-hopsets-and-generator-substitution.md)
- [BFS balls, doubling dimension, and metric nets](docs/notes/117-bfs-balls-doubling-dimension-and-metric-nets.md)
- [BFS replacement paths and fault-tolerant distance](docs/notes/118-bfs-replacement-paths-and-fault-tolerant-distance.md)
- [BFS degree-diameter Moore capacity and defect](docs/notes/119-bfs-degree-diameter-moore-capacity-and-defect.md)
- [Moore, Lee, and the wavefront view of BFS](docs/notes/185-moore-lee-and-the-wavefront-view-of-bfs.md)
- [BFS distance sums, closeness, and Wiener index](docs/notes/120-bfs-distance-sums-closeness-and-wiener-index.md)
- [BFS shortest-path DAG and betweenness centrality](docs/notes/121-bfs-shortest-path-dag-and-betweenness-centrality.md)
- [BFS orderings, Cuthill-McKee, bandwidth, and profile](docs/notes/122-bfs-orderings-cuthill-mckee-bandwidth-and-profile.md)
- [BFS graph coverings, universal trees, and fiber collisions](docs/notes/123-bfs-graph-coverings-universal-trees-and-fiber-collisions.md)
- [BFS edge deletion, contraction, and minor metrics](docs/notes/124-bfs-edge-deletion-contraction-and-minor-metrics.md)
- [BFS, edge subdivision, topological minors, and integer weights](docs/notes/125-bfs-edge-subdivision-topological-minors-and-integer-weights.md)
- [BFS sweeps, eccentricity bounds, and diameter certificates](docs/notes/126-bfs-sweeps-eccentricity-bounds-and-diameter-certificates.md)
- [BFS on complement graphs, frontier algebra, and Cayley complements](docs/notes/127-bfs-on-complement-graphs-frontier-algebra-and-cayley-complements.md)
- [BFS, bisimulation, simulation, and safe state merging](docs/notes/128-bfs-bisimulation-simulation-and-safe-state-merging.md)
- [BFS, Myhill--Nerode equivalence, DFA minimization, and residual languages](docs/notes/129-bfs-myhill-nerode-dfa-minimization-and-residual-languages.md)
- [BFS, NFA subset states, antichains, and dominance](docs/notes/130-bfs-nfa-subset-states-antichains-and-dominance.md)
- [BFS, AND/OR reachability games, attractors, and ranks](docs/notes/131-bfs-and-or-reachability-games-attractors-and-ranks.md)
- [BFS support graphs, probabilistic reachability, and MDPs](docs/notes/132-bfs-support-graphs-probabilistic-reachability-and-mdps.md)
- [BFS on de Bruijn and Kautz overlap digraphs](docs/notes/133-bfs-de-bruijn-kautz-overlap-graphs-and-frontiers.md)
- [BFS on lamplighter Cayley graphs: state, metric, and dead ends](docs/notes/134-bfs-lamplighter-cayley-state-metric-and-dead-ends.md)
- [BFS on Tower-of-Hanoi graphs: Schreier state, recursion, and frontiers](docs/notes/135-bfs-tower-of-hanoi-schreier-recursion-and-frontiers.md)
- [BFS on pancake Cayley graphs: prefix reversals and frontier collisions](docs/notes/136-bfs-pancake-cayley-prefix-reversals-and-frontier-collisions.md)
- [BFS on star-transposition Cayley graphs: cycle metric and generator contrast](docs/notes/137-bfs-star-transposition-cayley-cycle-metric-and-generator-contrast.md)
- [BFS with all transpositions: cycle-count distance and Stirling frontiers](docs/notes/138-bfs-all-transpositions-cycle-count-stirling-frontiers.md)
- [BFS dominating sets, covering radius, and k-center certificates](docs/notes/139-bfs-dominating-sets-covering-radius-and-k-center-certificates.md)
- [BFS shortest-hop paths, secondary cost, and Pareto boundaries](docs/notes/140-bfs-shortest-hop-secondary-cost-and-pareto-boundaries.md)
- [BFS on Hamming graphs: coordinate distance and binomial frontiers](docs/notes/141-bfs-on-hamming-graphs-coordinate-distance-and-binomial-frontiers.md)
- [BFS on Johnson graphs: fixed-weight exchange frontiers](docs/notes/142-bfs-on-johnson-graphs-fixed-weight-exchange-frontiers.md)
- [BFS on Grassmann graphs: subspace identity and q-binomial frontiers](docs/notes/143-bfs-on-grassmann-graphs-subspace-identity-and-q-binomial-frontiers.md)
- [BFS on Erdos-Renyi random graphs: branching, collisions, and giant components](docs/notes/144-bfs-on-erdos-renyi-random-graphs-branching-collisions-and-giant-components.md)
- [BFS on random regular graphs: tree bounds, pairing, and radial variance](docs/notes/145-bfs-on-random-regular-graphs-tree-bounds-pairing-and-radial-variance.md)
- [BFS on stochastic block models: multitype frontiers and owner cuts](docs/notes/146-bfs-on-stochastic-block-models-multitype-frontiers-and-owner-cuts.md)
- [BFS on configuration models: size-biased frontiers and hubs](docs/notes/147-bfs-on-configuration-models-size-biased-frontiers-and-hubs.md)
- [BFS on directed random graphs: IN, OUT, and strong cores](docs/notes/148-bfs-on-directed-random-graphs-in-out-and-strong-cores.md)
- [BFS on random geometric graphs: spatial waves and boundaries](docs/notes/149-bfs-on-random-geometric-graphs-spatial-waves-and-boundaries.md)
- [BFS on small-world graphs: shortcuts and wave branching](docs/notes/150-bfs-on-small-world-graphs-shortcuts-and-wave-branching.md)
- [BFS on preferential-attachment graphs: age, hubs, and core entry](docs/notes/151-bfs-on-preferential-attachment-graphs-age-hubs-and-core-entry.md)
- [BFS on growing trees: birth orientation, rerooting, and exact frontiers](docs/notes/152-bfs-on-growing-trees-birth-orientation-rerooting-and-exact-frontiers.md)
- [BFS on unicyclic graphs: cycle parity and the first duplicate](docs/notes/153-bfs-on-unicyclic-graphs-cycle-parity-and-first-duplicate.md)
- [BFS on cactus graphs: block trees and multiplying geodesics](docs/notes/154-bfs-on-cactus-graphs-block-trees-and-multiplying-geodesics.md)
- [BFS on theta graphs: overlapping cycles and multiway meetings](docs/notes/155-bfs-on-theta-graphs-overlapping-cycles-and-multiway-meetings.md)
- [BFS layer-edge accounting: cycle rank and duplicate conservation](docs/notes/156-bfs-layer-edge-accounting-cycle-rank-and-duplicate-conservation.md)
- [BFS successor occurrences: support arcs and Cayley label multiplicity](docs/notes/157-bfs-successor-occurrences-support-arcs-and-cayley-label-multiplicity.md)
- [BFS on Schreier graphs: stabilizer cosets and variable support degree](docs/notes/158-bfs-on-schreier-graphs-stabilizer-cosets-and-variable-support-degree.md)
- [Reverse BFS on Schreier graphs: inverse generators and asymmetric aliases](docs/notes/159-reverse-bfs-on-schreier-graphs-inverse-generators-and-asymmetric-aliases.md)
- [Stabilizer-aware BFS frontier work waterfall](docs/notes/160-stabilizer-aware-bfs-frontier-work-waterfall.md)
- [Directed BFS arc surplus: back depth and arborescence accounting](docs/notes/161-directed-bfs-arc-surplus-back-depth-and-arborescence-accounting.md)
- [BFS prefix conservation: partial layers and early-stop boundaries](docs/notes/162-bfs-prefix-conservation-partial-layers-and-early-stop-boundaries.md)
- [BFS foundations](docs/notes/01-bfs-foundations.md)
- [BFS invariants and schedules](docs/notes/03-bfs-invariants-and-schedules.md)
- [Frontier and visited semantics](docs/notes/04-frontier-visited-semantics.md)
- [BFS variants and their guarantee boundaries](docs/notes/05-bfs-variants-and-boundaries.md)
- [Explicit, implicit, and Cayley graphs under one model](docs/notes/06-explicit-implicit-cayley-model.md)
- [Conceptual single-GPU and multi-GPU BFS cost model](docs/notes/07-gpu-multigpu-conceptual-model.md)
- [Bidirectional BFS meeting and stopping proof](docs/notes/08-bidirectional-bfs-stopping-proof.md)
- [Completeness and termination on finite and infinite graphs](docs/notes/09-completeness-termination-infinite-graphs.md)
- [Frontier growth and metric-ball geometry](docs/notes/10-frontier-growth-geometry.md)
- [BFS tree, shortest-path DAG, and path counts](docs/notes/11-shortest-path-tree-dag-counts.md)
- [Ordinary BFS, 0-1 BFS, and relaxation-based SSSP](docs/notes/12-bfs-zero-one-dijkstra-boundary.md)
- [Multi-source BFS, graph Voronoi labels, and ties](docs/notes/13-multisource-bfs-voronoi-ties.md)
- [Push, pull, and direction-optimizing BFS](docs/notes/14-push-pull-direction-optimization.md)
- [External-memory BFS and delayed duplicate detection](docs/notes/15-external-memory-bfs.md)
- [Cayley versus Schreier BFS and action conventions](docs/notes/16-cayley-schreier-action-conventions.md)
- [Symmetry quotients, distance semantics, and path lifting](docs/notes/17-symmetry-quotients-and-path-lifting.md)
- [Asynchronous BFS relaxation, reactivation, and termination](docs/notes/18-asynchronous-bfs-relaxation-and-termination.md)
- [BFS ordering, shortlex paths, deterministic parents, and LexBFS](docs/notes/19-bfs-order-shortlex-and-lexbfs.md)
- [Product-state BFS, path constraints, and safe history pruning](docs/notes/20-product-state-bfs-and-history-constraints.md)
- [BFS certificates: components, bipartiteness, eccentricity, and diameter](docs/notes/21-bfs-certificates-components-bipartite-diameter.md)
- [Static snapshots, dynamic BFS maintenance, and temporal graphs](docs/notes/22-static-dynamic-and-temporal-bfs.md)
- [BFS versus iterative deepening: word trees, graph visited, and memory](docs/notes/23-iterative-deepening-tree-vs-graph-search.md)
- [Exact BFS versus beam search, top-k, and local BFS lookup](docs/notes/24-exact-bfs-versus-beam-search.md)
- [BFS as a least fixed point: balls, deltas, and quiescence](docs/notes/25-bfs-as-least-fixed-point.md)
- [k-hop batching, graph powers, and logical BFS depths](docs/notes/26-k-hop-batching-and-graph-powers.md)
- [Girth, generator relations, and tree-like BFS growth](docs/notes/27-girth-relations-and-tree-like-bfs.md)
- [Exact state identity: ranks, hashes, fingerprints, and Bloom filters](docs/notes/28-exact-state-identity-hashes-and-fingerprints.md)
- [What BFS complexity means across explicit, implicit, and distributed graphs](docs/notes/29-what-bfs-complexity-means.md)
- [Exact checkpoint/restart, replay algebra, and fault semantics](docs/notes/30-checkpoint-replay-and-fault-semantics.md)
- [BFS bipartite witnesses and shortest odd cycles](docs/notes/31-bfs-bipartite-witnesses-and-odd-girth.md)
- [Distance regularity and BFS intersection profiles](docs/notes/32-distance-regularity-and-bfs-intersection-profiles.md)
- [Adjacency powers, walk mass, and BFS layers](docs/notes/33-adjacency-powers-walk-mass-and-bfs-layers.md)
- [Hypergraph BFS and incidence semantics](docs/notes/34-hypergraph-bfs-and-incidence-semantics.md)
- [Cayley growth series and frontier extrapolation](docs/notes/35-cayley-growth-series-and-frontier-extrapolation.md)
- [Frontier set representations and information bounds](docs/notes/36-frontier-set-representations-and-information-bounds.md)
- [Exact BFS contract map and validation ladder](docs/notes/37-exact-bfs-contract-map.md)
- [CayleyPy production beam: read-only BFS contract audit](docs/notes/38-cayleypy-production-beam-contract-audit.md)
- [Non-backtracking words versus state BFS](docs/notes/39-nonbacktracking-words-versus-state-bfs.md)
- [Reverse BFS goal neighborhoods and suffix certificates](docs/notes/40-reverse-bfs-goal-neighborhoods-and-suffixes.md)
- [Local certificates for BFS distance labels](docs/notes/41-local-certificates-for-bfs-distance-labels.md)
- [Bounded BFS negative results and three-valued lookup](docs/notes/42-bounded-bfs-negative-results-and-three-valued-lookup.md)
- [CayleyPy K1/K2 test evidence audit](docs/notes/43-cayleypy-k1-k2-test-evidence-audit.md)
- [What explicit GPU BFS papers transfer to implicit Cayley search](docs/notes/44-what-explicit-gpu-bfs-papers-transfer-to-implicit-cayley-search.md)
- [Exact implicit GPU BFS: ranks, bitmaps, and hash tables](docs/notes/45-exact-implicit-gpu-bfs-ranks-bitmaps-and-hash-tables.md)
- [Expansion, diameter, and BFS memory pressure](docs/notes/46-expansion-diameter-and-bfs-memory-pressure.md)
- [Work, span, and frontier parallelism in BFS](docs/notes/47-work-span-and-frontier-parallelism.md)
- [BFS frontiers as separators and exhaustion certificates](docs/notes/48-frontiers-as-separators-and-exhaustion-certificates.md)
- [Pattern databases: abstraction and BFS distance heuristics](docs/notes/49-pattern-databases-abstraction-and-bfs-heuristics.md)
- [BFS, A*, and bound-certified heuristic pruning](docs/notes/50-bfs-a-star-and-bound-certified-pruning.md)
- [Owner hashing, load balance, and routing in distributed BFS](docs/notes/51-owner-hashing-load-balance-and-routing.md)
- [Authoritative visited, stale replicas, and advisory filters](docs/notes/52-authoritative-visited-replicas-and-advisory-filters.md)
- [Uniform sampling from the shortest-path DAG](docs/notes/53-uniform-sampling-from-the-shortest-path-dag.md)
- [BFS conservation checks, fingerprints, and the verification ladder](docs/notes/163-bfs-conservation-checks-fingerprints-and-verification-ladder.md)
- [BFS schedule contracts: layer-setting and label-correcting execution](docs/notes/164-bfs-schedule-contracts-layer-setting-and-correcting.md)
- [BFS work coordinates and hardware amplification](docs/notes/165-bfs-work-coordinates-and-hardware-amplification.md)
- [BFS scaling regimes: latency, throughput, and capacity](docs/notes/166-bfs-scaling-regimes-latency-throughput-and-capacity.md)
- [Cayley and Schreier ownership: cosets, orbits, and routing](docs/notes/167-cayley-schreier-ownership-cosets-orbits-and-routing.md)
- [Cayley quotient BFS, owner activation, and path lifting](docs/notes/168-cayley-quotient-bfs-owner-activation-and-path-lifting.md)
- [BFS fibers, re-entry, and the failure of quotient-first search](docs/notes/169-bfs-fibers-reentry-and-the-failure-of-quotient-first-search.md)
- [Cayley quotient generator images and routing matrices](docs/notes/170-cayley-quotient-generator-images-and-routing-matrices.md)
- [Cayley raw convolution and nonlinear frontier evolution](docs/notes/171-cayley-raw-convolution-and-nonlinear-frontier-evolution.md)
- [BFS level union, idempotence, and output merge algebra](docs/notes/172-bfs-level-union-idempotence-and-output-merge-algebra.md)
- [BFS proof obligations: independence and counterexample matrix](docs/notes/173-bfs-proof-obligations-independence-and-counterexample-matrix.md)
- [BFS logical obligations, credit conservation, and termination cuts](docs/notes/174-bfs-logical-obligations-credit-conservation-and-termination-cuts.md)
- [BFS shortlex-rank recurrence and distributed determinism](docs/notes/175-bfs-shortlex-rank-recurrence-and-distributed-determinism.md)
- [Bidirectional BFS shortlex suffix ranks and connector closure](docs/notes/176-bidirectional-bfs-shortlex-suffix-ranks-and-connector-closure.md)
- [BFS study coverage audit and next evidence gates](docs/notes/177-bfs-study-coverage-audit-and-next-evidence-gates.md)
- [BFS discovery, publication, and helpable commit](docs/notes/178-bfs-discovery-publication-and-helpable-commit.md)
- [BFS cuts, information, and protocol communication](docs/notes/179-bfs-cut-information-and-protocol-communication.md)
- [Distributed exact BFS set reconciliation](docs/notes/180-distributed-exact-bfs-set-reconciliation.md)
- [BFS safe forgetting, rolling windows, and boundary certificates](docs/notes/181-bfs-safe-forgetting-rolling-windows-and-boundary-certificates.md)
- [BFS orders, live boundaries, and pathwidth](docs/notes/182-bfs-orders-live-boundaries-and-pathwidth.md)
- [BFS dovetailing, infinite branching, and distance finality](docs/notes/183-bfs-dovetailing-infinite-branching-and-distance-finality.md)
- [CayleyPy and DeepCubeA Cube actions: a conjugacy audit](docs/notes/184-cayleypy-deepcubea-cube-action-conjugacy-audit.md)
- [Simultaneous conjugacy of labeled permutation actions](docs/notes/185-simultaneous-conjugacy-of-labeled-permutation-actions.md)
- [Decremental BFS: shortest-DAG invalidation versus distance repair](docs/notes/186-decremental-bfs-shortest-dag-invalidation-and-repair.md)
- [Incremental BFS: one-edge distance and output change cones](docs/notes/187-incremental-bfs-single-edge-distance-and-output-cones.md)
- [Batch incremental BFS: endpoint metric closure and its limits](docs/notes/188-batch-incremental-bfs-endpoint-metric-closure.md)
- [Distributed BFS: 1D/2D expand-fold semantics and implicit transfer](docs/notes/189-distributed-bfs-1d-2d-expand-fold-and-implicit-transfer.md)
- [Distributed bottom-up BFS: systolic early exit and exact snapshots](docs/notes/190-distributed-bottom-up-bfs-systolic-early-exit.md)
- [Research log](docs/research-log.md)
- [Experiment log](docs/experiment-log.md)
- [Open questions](docs/open-questions.md)

## Current status

Latest recorded direction, 2026-08-31: correct the audited research first;
plugin design and implementation are deferred. The audit found errors despite
earlier completion labels. Its coverage was 63 fully read and 136 sampled
notes, not an independent verification of every statement in all 199 notes.
See the correction record above for what was repaired and what remains unknown.

Historical complete, implicit labeled, target-stopping, and bidirectional CPU
references exercise selected core correctness contracts. Recorded transition-work
experiments study duplicates, partitioning, termination, and side selection.
Single-GPU bitmap variants and a complete CUB sort/unique pipeline were
measured on synthetic batches; fingerprint agreement and full-set fixture
comparisons have different evidence scopes. Retained S8 Cayley levels quantify how frontier
order and expansion layout change duplicate locality, and a fused exact S9
traversal evolved complete GPU-resident frontiers for its checked finite graph.
Application-scale and real multi-GPU scalability remain unmeasured. Notes 163--183 supply
conceptual contracts for verification, schedules, scaling regimes, algebraic
ownership, termination, deterministic output, and discovery-publication
continuity, plus a separation of graph cuts, information obligations, and
physical protocol traffic, an exact distributed reconciliation contract, and
safe visited-reclamation conditions, BFS-constrained live-boundary distinctions,
and infinite-branching finality boundaries; these are not runtime measurements.
Evidence gaps remain explicitly recorded; they do not automatically start a
new study cycle, benchmark, source-code change, or plugin task.

## Containerized GPU path

GPU code is built and run only in Docker. Rust owns host orchestration and
validation; C++ is restricted to CUDA translation units behind a C ABI.

```powershell
docker build -f docker/Dockerfile.gpu -t multigpubfs-gpu:dev .
docker run --rm --gpus all multigpubfs-gpu:dev
```

The image contains Rust correctness/oracle commands, CUDA bitmap and sort/unique
backends, reproducible sweeps, and artifact validators. See
[the GPU benchmark contract](docs/notes/02-gpu-benchmark-contract.md) and the
[experiment index](docs/experiment-log.md); isolated primitive timings are not
yet end-to-end BFS throughput.
