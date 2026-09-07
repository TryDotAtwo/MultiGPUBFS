# Dispatcher integration: current code boundary

Inspection baseline: `18e5f4e`. This is a source audit, not an implemented
dispatcher or a performance result.

## Existing pieces to reuse

- `control_pump.rs`: admitted BEGIN / TicketBytes / LAUNCH, ordered transfer
  completion, independent consumer retirement and Finalize / Publish.
- `control_admission.rs`: fixed, plane-partitioned metadata; root capacity
  includes the world-size factor for independently delayed consumers.
- `payload_lease.rs`: aligned physical receive offsets and sealed fanout.
  `reserve` requires a TicketKey, so it cannot reserve a producer's storage
  before the sequencer has assigned that producer an epoch.
- `event_generation.rs`: generation-bound record, device stream wait, poll
  and explicit retirement. Buffer ownership remains the caller's obligation.
- `jobs.rs`: bounded splitting of incoming bucket directories without crossing
  shards. It does not itself retain a payload bank or observe GPU completion.

## Actual integration seams

`examples/distributed_bench.rs` retains bootstrap as `_control_group` and uses
its NCCL ID; the reference stepper does not consume its control connections.
In `distributed_native.rs::advance_inner`, generation, route, packing, count
readback, peer exchange and owner processing still share the batch loop and
mutable singleton workspaces. The peer exchange uses `rank ^ 1`; replacing
only the send/receive call does not turn this into an admitted dispatcher.

The source workspace must exist before READY, whereas `PayloadBanks` begins
at TicketBytes. Source-local batch identity and physical bank lifetime must
therefore be linked to the later ticket without treating epoch as a producer
allocation ID. Receive storage is an independent pool. The fixture avoids
this missing boundary by prearranging source payloads and epoch order.

The first missing integration test should exercise real generated batch
readiness out of order, bind each source bank to its assigned ticket, and keep
one receive consumer alive while another bank retires. It must check that
neither source bytes nor receive bytes are reused early. The test must drive
the dispatcher adapter used by BFS, not another unrelated fixture driver.

HASH_FIRST also retains parent obligations beyond candidate transfer; DENSE
can retire the parent prefix only after its independent copy and archive
lease complete. Existing reference retirement cannot simply be moved behind
NCCL COMPLETE for both profiles. All per-batch `all_max` calls and finalization
must be accounted for before independently progressing ranks are enabled.

No CUDA edits or new hardware run were made by this audit. Existing targeted
CPU checks passed: control_pump 13, event_generation 8, jobs 2, payload_lease 7.
These 30 checks do not cover the missing production integration.

## Source identity implementation update

`source_banks.rs` now reserves aligned source offsets before epoch assignment.
The source token is monotonic and independent of physical slot and transport
epoch. Ready batches may bind in a different order from allocation; a retired
physical slot receives a new token. Errors poison the pool, and stale handles,
wrong ticket identity, unready binding and duplicate live epochs are rejected.

RED: the initial two tests failed to compile because the module was absent.
GREEN: four source-bank tests pass, including malformed bindings, overflow and
unbound retirement. The runtime CPU suite passed with the initial two tests;
the expanded four-test target also passed. This is host bookkeeping only:
event completion, device allocation and the production dispatcher are not
implemented by this type. Pool handles must not be mixed between instances.
No new Kaggle gate is warranted until this is connected to the data plane.

## Admitted command queue regression

A real two-rank TCP test delayed rank zero's command consumer while rank one
offered two tickets on each of four planes. Eight BEGIN plus eight TicketBytes
commands exceeded the legacy `4*slots+1` bound (9 for this setup), causing
`CONTROL_COMMAND_CAPACITY` despite valid receive credits. This is a host
metadata sizing error, not GPU exhaustion.

Admitted mode now preallocates `12*slots+1` command entries: three commands
(BEGIN, TicketBytes, LAUNCH) per live ticket across four planes, plus the
finalization command. Dispatch still refuses growth. Legacy mode is unchanged.
The reproducer failed before the change and passed afterward; all 14
ControlPump tests pass. No new performance or full-dispatcher claim follows.
