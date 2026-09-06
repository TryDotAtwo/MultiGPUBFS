# Remove a redundant route-count host round trip

Current `cuda/route.cu` preserves cardinality when pre-dedup is OFF: radix sort,
device copies and count publication do not select or drop records. Therefore
the distributed native reference can enqueue exchange packing on that same
stream with `candidate_count`, without synchronizing and downloading the same
count first. This applies to DENSE and HASH_FIRST; it does not change payloads,
owner mapping, scratch allocation or the preselected profile.

Pre-dedup ON still reads the compacted count, now checking it cannot exceed the
input count. Device-count-driven packing remains future work, not implemented
by this change. Other reference synchronization points also remain.

RED: new `route_count` tests initially failed on the missing module. GREEN:
two tests pass, covering no read callback at all for OFF (including empty
input), exactly one callback for ON, count bounds and read errors. The callback
is the actual synchronization/readback branch used by `distributed_native.rs`.

Hardware correctness and timing for this change are pending. Kaggle v39 is
pinned to earlier source `47cbb1a` and does not include it. No speedup is claimed;
the removed operation is established by the source diff, not a GPU timeline.
