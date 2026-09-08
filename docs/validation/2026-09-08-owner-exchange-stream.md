# Owner / exchange stream separation: hardware verification pending

Source `b3c191a5c32654a779e325d4b3bbe8e2f8226362`.

The reference runtime now creates a nonblocking exchange stream and a
completion event before work begins. The already-completed pack/count
publication makes immutable sender ranges readable. Count exchange still
requires a host observation; the hash and state payload operations then run
on the exchange stream. The event is recorded after both payload operations.

Local owner jobs remain on the compute stream and do not depend on remote
arrival. Before the remote owner consumes its range, that stream waits for
the exchange event. This dependency is inserted for empty remote input and
after local owner failure as well. The subsequent failure collective and
batch rollover therefore cannot overtake the earlier P2P operations or reuse
the sender/receiver buffers early. HASH_FIRST keeps the same origin lifetime.
No payload buffer is added. Runtime stream/event internal memory remains
outside the explicit allocation ledger and inside the existing reserve.

NCCL calls remain serial on the host, in identical rank order, and each
send/receive group uses only one stream. NVIDIA's [CUDA stream semantics]
(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)
describe asynchronous stream execution and the unwanted cross-stream
synchronization imposed when multiple streams are mixed inside one group.
This change does not mix streams within a group.

`process_owner_pair` is used by the runtime, not only a fixture. Three CPU
tests failed with the old wait-before-local schedule, then passed after
changing the order. They verify local-before-ready, remote-after-ready,
first-error retention, and suppression of unsafe remote reads after failed
readiness. Owner-result and abort tests also pass (seven targeted checks in
total), as does Rust CUDA-feature type checking of the library and both
native-scatter/distributed-archive fixtures.

The prepared full-runtime sanitizer package pins this source. It has **not**
been launched yet: the two permitted Kaggle notebooks are running the S11
one/two-T4 panels at the earlier verified source `66d82d0`. Do not interrupt
those runs or treat their measurements as evidence for this later change.
Launch this gate after one slot is available. Correctness and actual overlap
need target-hardware verification; speedup needs unprofiled timings and a
timeline. This is not yet the full admitted BFS dispatcher.
