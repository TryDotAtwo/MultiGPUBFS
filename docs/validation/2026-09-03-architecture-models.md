# Architecture v2: first executable contract gates

Scope: CPU models and owner-lane allocation arithmetic. No CUDA hot-path changes,
new GPU benchmark, production scheduler, or multi-rank claim.

## Changes

- `bounded_owner_ledger(I,J,K,library_queries)` enumerates individually aligned
  SoA allocations from architecture v2. No layer-capacity argument or hidden
  layer-sized temporary. Checked arithmetic rejects invalid/overflowing plans.
  Library byte counts are inputs; this does NOT yet perform CUDA/CUB queries
  or constitute a complete rank memory planner.
- `StateRing` now models origin leases independently from archive completion
  and enumeration. Reclamation requires all three conditions. Registration is
  allowed only while current, and release underflow/stale refs fail.
- `BatchReceipts` models one source batch across owners. Responses may finish
  before terminal receipt. Closure requires receipt totals and every response
  send completion. Owners receiving no emitted records require no receipt;
  owners receiving records but accepting zero still require one.
- Duplicate responses/receipts, wrong owner/emitted counts, excess accepted or
  served counts poison the batch. CPU sets are verification bookkeeping, not
  a proposed GPU container. This model assumes request identity validation and
  batch routing happened at the transport boundary.

## RED -> GREEN

Observed initial failures:

- owner memory fixture: OWNER_PLAN_NOT_IMPLEMENTED;
- parent reclamation fixture: reclaimed 8 records instead of 0 with live origins;
- receipt fixtures: missing closure and missing duplicate-message rejection.

After implementation, seven new tests pass. The integration fixture enumerates
all 720 orders of two receipts, three response completions, and archive D2H.
For every order, an eight-record parent cannot retire before the last required
event and retires at completion. Additional cases test zero survivors, lease
underflow/late registration, errors, alignment and arithmetic overflow.

Full default workspace: **32 tests passed**, zero failures. CUDA-feature tests
are excluded, deliberately. Existing matrix oracle m2..6, ring wrap/capacity,
archive fault injection and previous sequencer fixtures also passed.

Environment: existing `multigpubfs-rust-toolchain:dev`, CPU-only container,
offline dependency cache `/src/build/cargo-home`. Windows Cargo failed to fetch
dependencies due to Schannel SEC_E_NO_CREDENTIALS; it was not a test failure.

Command inside the existing image with workspace mounted at `/src`:

```sh
cargo fmt --all
cargo test --locked --offline
```

## Remaining architecture gates

- Complete query-backed rank allocation plan including all profile pools,
  archive/control, runtime overhead and reserve.
- New schema2 frozen config/wire/archive byte vectors and codecs.
- Bounded sharded owner execution model with touched-range/byte accounting.
- Integrate the typed transport model below with owner/StateRing/receipt jobs;
  TCP/NCCL realization and transport failure propagation remain pending.
- Integrated whole-depth/multi-depth model for both profiles, including credit
  admission, failure propagation and drain. The 720-order test is one boundary
  fixture, not exhaustive model checking of that whole system.
- GPU/real two-rank implementation and hardware gates remain pending.

The legacy StateRing model still uses monotonic extent IDs rather than the v2
absolute-record StateRef wire encoding. Origin-lease behavior is usable evidence,
but does not validate the new ABI. No architecture-ready promotion is made.

## Typed transport gate (subsequent change)

`transport::Transport` is a separate v2 CPU oracle; old candidate-only
`exchange::Sequencer` remains unchanged for its existing fixtures.
Candidate/request/response/receipt/finalize share one monotonically increasing
sequence, including across depth rotation. Multiple tickets can be in flight;
each rank completes them in its own comm-stream issue order. Empty ranks must
acknowledge too. Send completion and receive consumption are distinct.

Credits bound source offers, receive banks and live ticket metadata. They are
partitioned by message kind. Candidate metadata cannot occupy response-reserved
capacity, including when candidate payloads are empty. A RED test exposed that
the initial shared metadata limit did not provide this guarantee; partitioning
fixed it. The architectural contract now states that requirement explicitly.
An additional RED case split ten records across two destinations under an
eight-record offer limit. Validation now checks the checked SUM across peers,
not only each destination count, before consuming any slot or sequence number.

Response/request/receipt drain has priority over new candidates; eligible sources
round-robin within a kind. Closed sources may still serve materialization.
Finalization requires closed candidate sources, no pending/in-flight/received
tickets and no registered external work. Finalization stops new admissions;
depth rotation requires every rank's final acknowledgment and preserves seq.

Seven new tests were observed failing before implementation, then passing.
`cargo fmt --all && cargo test --locked --offline` in the same existing CPU
toolchain now passes **39 tests**, zero failures; no GPU tests claimed.

Limits: the model issues one source offer per ticket, no optimized multi-source
aggregation. Metadata uses CPU collections, not a proposed GPU layout. Error
calls return without partial mutation so fixtures can inspect invariants; the
production caller must treat protocol errors as group-fatal. It does not model
NCCL abort, TCP disconnects, wire codecs, bytes/plane packing, or whole-BFS jobs.
External jobs must register work BEFORE consuming their parent receive ticket;
the integrated dispatcher model must prove that ordering rather than assuming it.

## Integrated small-graph BFS oracle (subsequent change)

`simulation::run` now connects per-prefix-bucket OwnerModel, StateRing,
BatchReceipts and typed Transport across complete BFS layers to exhaustion.
It uses real matrix successors and seeded hashes, not canned layer counts.
Ownership uses high hash bits and an explicit logical-to-physical rank map.
Full canonical state sets are compared against the independent visited-set
layer traversal (`MatrixGroup::exact_layers`); both traversals share the matrix
successor contract, so this is not an independent matrix-arithmetic test.

216 completed configurations: U3 mod2, U3 mod3, U4 mod2; DENSE/HASH_FIRST;
pre-dedup OFF/ON; rank maps [0], [0,1], [1,0]; six schedule seeds, each paired
with a corresponding hash seed 1..6 and immediate/delayed archive completion.
Every full layer matches, generated=states*degree, committed=states-1,
HASH_FIRST request/response counts=committed; DENSE sends no such requests.

Candidate batches are one parent, grouped into bucket jobs. Owner preview is
a clone of the CPU semantic oracle, allowing reservation before commit; it is
NOT a proposed double-merge implementation. Target extents are reserved in
request order; regenerated responses are checked against canonical state and
committed hash. Receipts may precede requests/responses, but responses require
their request. Pending work and archive leases prevent premature finalization.

A two-state one-rank fixture with one state-ring record completes in DENSE
after packing/reclaiming the parent. HASH_FIRST cannot reuse that record while
origins are live, and DENSE with delayed archive cannot reuse it before D2H;
both fail explicitly. Additional ring/bucket exhaustion and bad rank map cases
return errors, never a completed Simulation. The semantic integration test was
observed RED (SIMULATION_NOT_IMPLEMENTED), then GREEN.

Full default suite now **42 CPU tests passed**, no failures, with
`cargo fmt --all && cargo test --locked --offline` in the same cached toolchain.
No GPU run or performance claim.

Important remaining limits: parent batches and data transfers are serialized
in this integration oracle. Event reordering is within a parent's materialization
wave; independent transport tests exercise multiple in-flight tickets. This is
not a full concurrent dispatcher/deadlock proof. Archive models D2H leases only,
not pinned/disk queue draining; existing disk fault tests remain separate.
CPU Vec/BTree containers hold reference values, not proposed GPU allocations.
No BMMA execution, CUDA scratch query, schema2 codec, general fault-injected
whole-system schedule or production StateRef ABI is validated here.
