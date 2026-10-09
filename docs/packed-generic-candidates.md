# Exact packed candidate transport

Permutation byte states of width 1 through 16 use their exact values inside the existing 32-byte route record. Generation computes each element once, packs it and computes the same routing hash. A separate regeneration kernel and payload lane are omitted. Owner shards still compare exact full values, including forced hash collisions, and materialize only accepted unique states. Queue metadata remains 32 bytes; this is not hash-only identity.

Wider permutations and integer matrix actions retain the separate SoA payload path. Directed all-visited history is unchanged. Packed routing is an internal automatic representation, with no extra public mode. Source A/B leases, inbox retirement, resource snapshots and retry votes retain the same lifetime contracts. Physical multihost and Blackwell performance remain unverified.

Validation: actual 2xRTX3060, CUDA13.2. Full CPU layer oracle for byte/int64 codecs, forced hash zero, repeated byte values, 8/16/17-element boundaries, matrix fallback, queue skew retries and resource current/previous snapshots. Paired nine-element graph: 362880 states, three alternating repetitions, fixed batch32768, capacity362880, 16 shards; median BFS 0.219753989 s before versus 0.183127461 s packed (1.2000x). This is a small matched graph; the 39916800-state comparison is still awaiting CPU validation.
