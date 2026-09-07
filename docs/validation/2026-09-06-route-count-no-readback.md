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

## Hardware regression verified 2026-09-07

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` v40 completed on source
`ebcbb9998d52689c53d50bbd1dd7cc23243cca22`, package
`c3a1fc85783ab0d653c9403fca6978457787df8f`. Two distinct Tesla T4 devices
(15360 MiB each) were recorded:

- `GPU-2b723a57-97d4-f7a4-bd25-a9b0d45f4a83`
- `GPU-aa210c1e-dee7-1106-2567-341e40088465`

Plain execution and Compute Sanitizer memcheck, racecheck, initcheck and
synccheck passed. The downloaded raw artifacts were reconciled against the
pinned source: 40 tool logs, 36 measured rank records, 36 warmup rank records
and 36 archive verifiers. Reference profile checks cover DENSE/HASH_FIRST,
CUB/BMMA, pre-dedup ON/OFF and one/two ranks; the small graph layer counts are
`[1, 3, 5, 6, 5, 3, 1]`.

This confirms the regression gate, not a speedup or completion of the
asynchronous production dispatcher. Timing remains unmeasured for this change;
the removed operation is established by the source diff, not a GPU timeline.
