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

## Hardware result

Kaggle v37 completed at `d226dc919fc90760a4634c4fc912150019d3b91e`
(package `45727da`). Devices were distinct Tesla T4s, 15360 MiB each:
`GPU-113bf917-c34a-b996-e6a1-4aca29c0bb1f` and
`GPU-30992c1d-f9b6-758e-c5cd-a5986f78595d`.
The changed fixture passed plain, memcheck, racecheck, initcheck and synccheck;
all error summaries and race warnings/hazards are zero.

Full raw audit on 2026-09-06:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v37/distributed-sanitizer test_results/distributed-sanitizer-v37-summary/distributed-sanitizer/summary.json d226dc919fc90760a4634c4fc912150019d3b91e
```

Result `RAW_GATE_RECONCILED`: 40 tool logs, 36 measured ranks, 36 warmup
ranks and 36 archive verifiers. The existing runtime, macro and archive
regressions pass; all 24 reference profile selections preserve S4 layers
`[1,3,5,6,5,3,1]`. Only logs and metadata were downloaded locally.

This validates sealed fanout with synchronous test readers. It does not validate
the later cross-stream dependency change (`ea844ff` / import fix `93cdf30`),
the production dispatcher, or any throughput gain.
