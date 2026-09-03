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
- One typed transport sequencer for candidates/requests/responses/receipts.
- Integrated whole-depth/multi-depth model for both profiles, including credit
  admission, failure propagation and drain. The 720-order test is one boundary
  fixture, not exhaustive model checking of that whole system.
- GPU/real two-rank implementation and hardware gates remain pending.

The legacy StateRing model still uses monotonic extent IDs rather than the v2
absolute-record StateRef wire encoding. Origin-lease behavior is usable evidence,
but does not validate the new ABI. No architecture-ready promotion is made.
