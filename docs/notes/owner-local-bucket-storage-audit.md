# Owner-local bucket storage audit

Current distributed native runtime uses 256 global hash-prefix buckets and
ownership from the top hash bit. Consequently only 128 buckets can receive
states on each of the two ranks, including when the rank map is swapped.

The accepted-hash allocation nevertheless reserves all 256 buckets per rank:
`buckets * bucket_capacity * 16`. The default bucket capacity is
`ceil(layer_capacity / 128) + 4096`.

At 178M layer capacity, this reserves 5,712,777,216 bytes/rank. Half,
2,856,388,608 bytes/rank, belongs to the other logical owner and is unreachable
under correct routing. This is a code-derived allocation observation, not a
measured post-fix saving. No storage layout was changed by this audit.

An implementation must consistently translate global prefix IDs to local IDs
in directory construction, jobs, accepted counts, shard/lane validation and
FinalizeDepth compaction. Merely halving allocation while retaining global
indices is invalid. Avoid pointers formed before the allocation as an offset
shortcut. Validate both rank maps and equality of full small-graph layers,
then run sanitizer gates before a new S13 capacity claim.
