# Payload-bank consumer fanout contract

`PayloadLease` tracks a fixed number of host-side consumer descriptors for one
physical payload bank. Reserve precedes byte-admission ACK; each downstream
job obtains a ticket-bound token before launch. Completed tokens are not reused
within the ticket. Explicit sealing means the splitter has enumerated every
consumer; zero outstanding jobs before sealing does not authorize bank reuse.
Duplicate, foreign/stale completion, late registration, and capacity violations
poison the lease. Source-local slot tokens may decrease when the source changes;
global epochs must increase on physical bank reuse.

Storage is one fixed `Vec<bool>` of `jobs` elements allocated at setup, plus
constant metadata. Successful hot-path operations do not allocate. This is CPU
control metadata, not per-state GPU storage or a device-side allocator.

RED: `cargo test --locked -p mgbfs-runtime --test payload_lease` failed on the
missing `payload_lease` module. GREEN: four contract tests pass; the complete
local `cargo test --locked -p mgbfs-runtime` suite passes. CUDA-gated tests do
not execute in that local run.

The two-bank native scatter fixture now reserves leases before admission and
uses two actual D2H readers per payload, checking that the first reader cannot
drain the second. Empty transfers seal an empty fanout. This changed CUDA
fixture has not yet been validated on hardware.

The lease does not itself observe events: the caller must prove transfer and
individual consumer completion. Existing `NativeEvent` guards remain responsible
for transfer completion in the fixture. Production buffer ownership, event
binding and exhaustive BFS dispatcher integration remain unimplemented.
