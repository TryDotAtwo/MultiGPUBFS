# Owner retry and fixed-capacity scratch acceptance, 2026-10-10

Base source: a5fe587b4dc7dc9a9608ca991a9fac17187eb032. Hardware: two RTX3060 12GB, CUDA13.2, SM86. Current modifications are not yet released.

The previous unconditional second carry graph launch is replaced by one launch. A device-side conditional loop permits at most one extra allocator attempt after actual shared-credit pressure. Successful accept publishes the retry credit with its carry. Persistent pressure preserves the input owners and surfaces the existing collective resource failure. No CPU readback was added to decide whether to retry.

Verified evidence on this rental:
- Six native pressure fixtures: no failure, one transient failure, two persistent failures on each GPU. Attempts are respectively one, two, two. Persistent failure preserves the allocation bitmap and input owner tokens; cleanup returns the bitmap to zero.
- Compute Sanitizer memcheck on those six fixtures: zero errors.
- Twenty-four disjoint merge/duplicate transaction fixtures, both GPU devices, byte/int64 states, widths2/25/129, normal/full-collision hash modes.
- Native metadata query: descriptor16, tiers260, carry128, snapshot1032, reservation28 bytes. Carry allocation remains256 bytes after alignment.
- Ten two-rank full-layer CPU-oracle BFS fixtures, including directed/all-visited, inverse-closed/rolling, packed/full/parent transport, matrices, signed states and resource-stop retention.
- Nineteen runtime library CPU tests and fourteen generic distributed-memory integration tests. Logical8/128-rank memory geometries are not physical GPU acceptance.

The removal of the redundant null-storage CUB scan query passed independent native-gateway acceptance and repeated ten-fixture BFS gates. Cold native shape admission already queries the identical fixed scan geometry. The linked CUB implementation checks insufficient storage in util_temporary_storage.cuh before execution. A new native-gateway regression requires cudaErrorInvalidValue for a one-byte scratch budget and verifies no state/history publication. Its memcheck completed with zero errors and all repeated BFS gates matched the CPU oracle.

Delivery still required: accepted final source commit, matched-prefix timing for this changed path, updated immutable native package and clean-installed acceptance, private proof checksum/readback, source-only GitHub delivery and exact rental teardown. Physical multi-host8/128 GPU and Blackwell remain unverified; neither this checkpoint nor CPU geometry tests prove their throughput.
