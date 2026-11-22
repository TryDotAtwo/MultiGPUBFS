# Experimental device-driven BFS on four GPUs

Scope: standalone test programs, not an installed production backend. They do
not exercise the existing segmented24/warp pending-claim implementation or
establish Explore, production archives, autotuning or device fallback.

On Vast machine 54815 the eight V100-SXM2-32GB devices expose native peer
atomics for only some pairs. CUDA's peer-atomic query confirmed that devices
0,1,2,3 form a fully connected group; the eight-device set does not. The tests
therefore used exactly devices 0,1,2,3 and four independent processes. NCCL
2.30.7 reported a single LSA team for that four-rank group.

The bitmap test uses four producer lanes, incoming SPSC rings and a separate
coordinator in one cooperative block per GPU. For every test it compares the
complete per-rank (depth, packed state) multiset with an independently
constructed CPU FIFO BFS, including duplicate counts and incomplete prefixes.
The suite includes S5, S6, repeated symbols, a directed rotation, empty ranks,
queue sizes 1 and 7, normal and reversed device order, and a duplicate on a
full owner. The independent experimental serial full-history hash program adds
forced equal-hash collisions, two nonempty shards per rank, and the same
comparison of complete layer multisets for its own fixtures.

An additional bitmap test completed S8, all 40,320 states and 29 exact
layers, with queue size 7. Each rank's entire layer/state multiset was compared
to the CPU oracle. The program materializes this small output for verification;
this is not a production-scale performance measurement. 282,240 generated
edges were reported across the four ranks.

Eight-GPU native peer atomics, eight-GPU execution, production storage
semantics, archived output and normal user-facing BFS are not established by
these results. Full logs and test binaries are retained with the experiment
output. No conclusion about B200 or large-graph speed follows from V100.
