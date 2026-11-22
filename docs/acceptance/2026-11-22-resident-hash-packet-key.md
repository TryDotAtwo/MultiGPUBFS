# Reusing the routing hash in resident packets: no measured gain

This experiment did **not** enter the implementation. On this workload it did
not improve speed beyond measurement variation. The
[reproducible patch](artifacts/2026-11-22-resident-hash-packet-key.patch.b64)
applies to `5914dcd9430d8814a66d38635623e68622898182` and has SHA-256
`a6fa47701c7e16945d90506e08d1c4a50feb615d34c1eaead58398a0d034e6c8`.
Decode it with `base64 -d 2026-11-22-resident-hash-packet-key.patch.b64 > packet-key.patch`.
The tested executable SHA-256 was
`45ed245bbe2f9f3ef1dd77c1c9f8ec5d00198a952cd8959577ed6339fd06b364` and the
loaded CUDA library SHA-256 was
`217288101ba03af6ac2dc1076c6a74e5617e7de1907c2e5fef0fb4b8eeb75cf5`.

The patch optionally places the producer's deterministic 64-bit routing hash
beside the lossless packed state in the existing 16-byte packet. A separate
packet format distinguishes it, and the receiver still compares full states
on collisions. The source is identified by the ring, rather than a payload
field. Hash seed, algorithm, dictionary, option and format are agreed before
the run. The optional verification flag recomputes the hash on receipt and
fails the job on mismatch; it was disabled in the timing comparison.

With verification on, exact layer-multiset tests passed on four GPUs for S4,
repeated-symbol and signed/wide-symbol graphs and on two GPUs with every hash
forced to zero. With verification off, the same four-GPU layer tests and
CayleyPy persistent duplicate-at-full, resource-prefix and healthy-next cases
passed. Resident bitmap tests also passed on two GPUs. These tests do not
cover all possible graphs, packet corruption or eight-GPU topology.

## S9 comparison on the same executable

Four V100-SXM2-32GB GPUs, 362,880 states, 37 layers, two shards, 8192
rows/segment, 46 extents/rank, indexed extents, parallel shard owners, eight
lanes and eight packet records. Ordering was off, on, on, off. Instrumentation
and recompute-on-receipt were disabled. Each run matched all Mahonian layer
sizes and the final state.

| Hash in packet | BFS seconds | End-to-end wall seconds |
|---|---:|---:|
| off | 15.203369 | 22.393102 |
| on | 15.287399 | 22.529734 |
| on | 15.348828 | 22.352242 |
| off | 15.229212 | 22.439934 |

Two paired observations show no useful gain on this graph and hardware. This
does not rule out benefits on a different topology or much larger workload.
[Machine-readable receipts](artifacts/2026-11-22-resident-hash-packet-key.json)
include per-rank geometry; detailed logs and tested binaries are preserved in
the evidence archive. The published implementation remains the preceding
[indexed-extent checkpoint](2026-11-22-resident-hash-indexed-extents.md).
