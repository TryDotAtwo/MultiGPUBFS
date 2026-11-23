# Experimental independent-bucket resident hash owner

Status: source checkpoint only. This revision has not yet been built or checked on a physical GPU; it is not a production default or a performance claim.

The opt-in `MGBFS_RESIDENT_HASH_INDEPENDENT_BUCKETS=1` requires bucket-owned extents and more than one virtual bucket. One CUDA block owns one bucket while accepting its bounded tile, so independently owned tables do not wait on a whole-device barrier for each tile. Stable exact-key ownership, full equality, extent leases and global drain before packet acknowledgement remain required. The source includes tests for collisions, small extents and limits, plus a persistent multi-GPU layer-multiset fixture. No result is implied by their presence.

This checkpoint does not implement A/B overlap, dynamic device-side segment growth, or guaranteed replay when the current capacity is exhausted. The existing backend and defaults remain unchanged. Complete acceptance requires building this source, running the fixtures on physical GPUs, and comparing identical completed work and memory on the same hardware.

Source checkpoint: `2f4d78f54a8acb522cde71c3a5ee3a84b1560ca5` (tree `8d16f3d93005552376b4cee36d60ce947458c226`).
