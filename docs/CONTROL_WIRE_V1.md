# Native control frame V1

Status: implemented codec, rank-bound connection, bootstrap rendezvous, and real
loopback TCP tests. The reference benchmark now calls rendezvous before NCCL;
setup integration passed the [v30 two-T4 gate](validation/2026-09-06-bootstrap-two-t4-gate.md).
**Not yet wired into the GPU dispatcher**.
This specifies transport framing, not a complete
sequencer or proof of asynchronous NCCL issue order.

`EpochCoordinator` implements the bounded **data-epoch admission** slice. It
mirrors per-rank leases, holds preallocated FIFO offers per plane/rank, prioritizes
RESPONSE/REQUEST/RECEIPT before CANDIDATE, and emits one pinned BEGIN per rank
under a shared sequence. A new ticket needs conservative receive credit on every
rank. Each ticket selects exactly one source by per-plane round-robin; other
ranks receive NO_SLOT. This avoids accumulating W independent source maxima
into one receive bank. The future payload admission must still prove each
source fragment's maximum count/bytes fits every potential receiver.
It reserves credit on every
rank, including empty sources. COMPLETE and CONSUMED are distinct; credit stays
until each rank's consumers retire. Wrong plane/epoch acknowledgements poison
the coordinator. Caller-owned BEGIN output storage and outbound queue capacity
must be reserved before issue. Eight CPU tests cover priority, empty ranks,
cross-plane progress, source round-robin, source closure and invalid acknowledgements.
SOURCE_CLOSED rejects later candidate offers without deleting already queued
candidates or blocking dependent response/request/receipt traffic. Duplicate or
wrong-depth closure is fatal. Once every source is closed, queues are empty and
all consumer leases retired, it closes admission and issues FINALIZE at the
next transport watermark. Every rank must acknowledge FINALIZED after local
finalization/drain. Only then is PUBLISH emitted and the next depth admitted;
transport sequence does not reset. Late READY or duplicate acknowledgement is
fatal. Payload counts/fragment admission, socket pumping, actual CUDA drain
verification and NCCL execution are not implemented by it. Do not treat it as the full
production sequencer or as proof of payload-capacity safety.

Required caller invariant: publish descendant offers before sending CONSUMED
for their input ticket, and register all local jobs/leases before that retirement.
SOURCE_CLOSED follows all initial producer offers. FIFO control order then
preserves the causal watermark. Untracked/detached jobs cannot be proven drained
by this state machine; the GPU dispatcher must enforce this invariant and
verify local drain before FINALIZED. A counter snapshot alone is insufficient.

`rank_epochs::RankEpochs` now supplies bounded rank-local admission bookkeeping:
exact offered slot selection, one sequence across planes, outstanding-source
slot ownership, and conservative per-plane receive credits (including zero
source offers). COMPLETE marks ordered transfer completion without releasing
credit. CONSUMED marks caller-proven consumer retirement and can arrive out of
order across independent epochs. FinalizeDepth advancement requires no pending
offers/live epochs plus external local-drain confirmation; sequence survives
depth rotation. Storage is allocated once; protocol errors poison the state.
Eight CPU tests cover these transitions. After local finalization it forbids
new offers/BEGIN until matching PUBLISH. A composed real loopback TCP test also
checks a later READY arriving before the earlier epoch's completion, exact-slot
BEGIN admission and reuse only after consumption. This component is not yet
connected to the production TCP pump or CUDA; it does not prove global credits, coordinator drain,
actual event readiness or multi-GPU issue order.

A second composed loopback TCP test now drives coordinator + both rank states
through two full control depths, alternating which rank supplies candidates.
It checks that one rank's FINALIZED cannot publish alone, and that the next
depth keeps the transport sequence. This is host control integration evidence,
not CUDA/NCCL data-plane execution or a production progress loop.

Each frame is exactly 64 bytes, little-endian, with no variable-length payload.
The codec uses stack arrays on the successful path. It accepts a caller-owned
`Read`/`Write` stream; the connection owner is responsible for bounded I/O
timeouts, connection identity, and lifecycle. Partial reads/writes are handled;
EOF or an I/O error is fatal to that connection. Never retry a partial frame on
the same stream or scan for a new magic marker.

`FrameReader` holds one fixed 64-byte frame per peer. With a caller-configured
nonblocking socket, `poll` returns immediately on WouldBlock/Interrupted and
retains partial bytes. Each call returns at most one decoded frame. A malformed
frame, EOF, or other I/O error permanently poisons that reader. A slow peer
therefore need not block polling another peer. Blocking `read_from` remains
available for bounded setup/test exchanges; it must not be used by the roulette
dispatcher. `FrameWriter` supplies one fixed pending send frame per peer and
retains its exact byte offset across WouldBlock/Interrupted. Enqueue while busy
returns an explicit capacity error without overwriting the pending frame.
Zero writes and other I/O failures permanently poison it. Local send completion
only means bytes were accepted by the stream, not that the peer or GPU finished.
`ControlOutbox` now provides that bounded FIFO: capacity includes the partially
written head frame, storage is allocated at construction, and each poll retires
at most one frame. `ControlConnection::with_send_capacity` selects its capacity
at setup; the existing constructor/handshake retain capacity one. Partial writes
keep the same FIFO head until complete. Standalone queue-full errors preserve
queued data; through ControlConnection they poison and close the connection as
before. Tests cover short writes, WouldBlock, wraparound, exact frame order,
write-zero poisoning, and the real TCP multi-frame path. Dispatcher sizing and
bootstrap propagation of larger configured queues remain to be connected.

`ControlConnection` owns its TcpStream and configures nonblocking I/O/TCP_NODELAY
at construction. It accepts only a valid rank-0 star edge, checks the sender
against its assigned peer, and enforces command direction. Codec, I/O,
direction, sender mismatch, and pending-send-capacity errors fail the connection
and shut down both socket directions. Bootstrap must establish the run digest
and peer assignment before wrapping the stream; the claimed rank is not
cryptographic authentication. The caller must also enforce overall progress
deadlines while polling (nonblocking I/O alone cannot detect a silent peer).

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | ASCII `MGBCTRL1` |
| 8 | 2 | version = 1 |
| 10 | 2 | action |
| 12 | 4 | sender rank |
| 16 | 8 | depth |
| 24 | 8 | exchange epoch |
| 32 | 8 | local slot, or `u64::MAX` for no slot |
| 40 | 4 | plane |
| 44 | 4 | fatal code, zero except FATAL |
| 48 | 16 | reserved, all zero |

Actions: READY=1, BEGIN=2, COMPLETE=3, SOURCE_CLOSED=4, FATAL=5,
FINALIZE=6, CONSUMED=7, FINALIZED=8, PUBLISH=9. These are additive actions in this pre-release V1
codec; older builds reject unknown actions rather than reinterpreting them. Run/config
identity must match before connecting. Planes: NONE=0, CANDIDATE=1, REQUEST=2, RESPONSE=3,
RECEIPT=4. Unknown values, nonzero reserved bytes, and ranks outside the
configured world are rejected before dispatch.

READY carries a real local slot and data plane, with epoch=0 (the sequencer
has not assigned one). BEGIN is sent by rank 0 with a data plane and the
receiver's pinned slot, or NO_SLOT for a zero offer. Rank 0 issues one BEGIN to
each rank for the same epoch/plane but potentially different slots. The receiver
must use that exact registered READY slot, not resample its newest readiness:
another READY can be in transit when BEGIN arrives. This concretizes the
sequencer's offer snapshot and matches the CPU `exchange::Sequencer` oracle.
The slot identity is receiver-local, so the bootstrap connection must establish
the recipient rank. COMPLETE
acknowledges transfer completion for the specified epoch with a data plane and
no slot; it does not release receive storage. CONSUMED has the same fields and
acknowledges local consumer retirement after transfer completion. SOURCE_CLOSED
has no slot/plane and epoch=0. FATAL has no slot/plane and a nonzero fatal code.
FINALIZE is sent by rank 0 with no slot/plane. FINALIZED echoes its depth/epoch
from each completing rank. PUBLISH is sent by rank 0 with the next depth and
the same FINALIZE epoch; it is a control acknowledgement, not another NCCL
transport ticket. Both carry no slot/plane. All nonfatal messages have a zero
fatal code. Exact send counts remain in the ordered NCCL metadata exchange,
not in this frame.

Required dispatcher checks still to implement: bind sender rank to the
bootstrapped connection/run digest; enforce depth/epoch monotonicity, slot
capacity and legal lifetime, action direction, duplicate acknowledgements,
receive credits, and complete depth drain. The codec does **not** authenticate
a peer or establish that these semantic checks occurred. FINALIZE receipt alone
does not authorize StateRing reuse.

Validation:
`cargo test --locked -p mgbfs-runtime --test control_connection --test control_wire --test exchange --test transport`
passes 5 connection tests, 13 codec/TCP tests and 10 existing CPU sequencing tests. Frozen READY
bytes, short I/O, EOF, invalid fields, and actual loopback READY/BEGIN/COMPLETE
exchange are covered, including nonblocking partial-frame arrival, peer
independence, frame reuse, pending-send capacity/offsets, and terminal error poisoning. This is not a multi-GPU
or Linux hardware gate.
# Setup identity handshake

`ControlConnection::accept_peer` / `connect_peer` exchange fixed 80-byte
hellos before returning a nonblocking connection. Layout: magic `MGBHEL01`
at 0, little-endian u32 version 1 at 8, world at 12, rank at 16,
zero reserved bytes at 20..24, config digest at 24..56, run ID at 56..72,
zero reserved bytes at 72..80. The coordinator replies with rank 0 only
after validating the complete peer hello. One deadline covers both transfer
directions; each partial transfer uses the remaining timeout.

This is run-identity validation, not cryptographic authentication. The caller
must provide the actual shared bootstrap identity and reject duplicate rank
admission. Bootstrap is integrated; GPU dispatcher integration remains pending.
Real loopback TCP tests cover matching identity followed by READY traffic and
rejection of differing world, config digest, or run ID. Linux/GPU execution
of this handshake is covered by the v30 setup gate linked above.

## Bootstrap record

`BootstrapRecord` is a 200-byte setup record: magic `MGBBOOT1` at 0,
u32 version 1 at 8, u32 world at 12, digest at 16..48, run ID at 48..64,
IPv4 octets at 64..68, little-endian u16 port at 68..70, zero reserved
bytes at 70..72, opaque NCCL ID at 72..200. Only loopback addresses and
nonzero ports are accepted for this single-node transport. The caller must
supply its independently established expected run identity when reading.

Publication uses an exclusively created `.rank0.staging` sibling, a complete
write and file sync, then a same-filesystem hard link that cannot replace an
existing destination. Unsupported hard links are fatal, with no rename
fallback. Failed publication retains staging for diagnosis; successful
publication removes its own staging link. Directory crash durability and
recovery are not promised. This is setup metadata, not an archive commit.
Readers bound the read to exactly 200 bytes and reject size/identity mismatch.
`BootstrapRecord::connect` validates configuration before network I/O, connects
to the recorded endpoint and completes the run-identity handshake with the
remaining shared setup timeout. It returns the nonblocking control connection;
NCCL communicator creation and peer-group admission remain the caller's work.

`BootstrapListener` binds loopback before publication and preallocates rank
admission flags. Setup accepts ranks in arrival order, checking duplicate rank
before sending the handshake acknowledgement. Any accept/handshake error
permanently poisons admission. Each `accept_next` call has a shared accept plus
handshake timeout; the caller must impose a total group-startup deadline and
abort already returned peer connections on failure. This setup-only accept
loop is not the GPU hot-path dispatcher. Actual TCP tests cover rank order
2 then 1, duplicate rejection on both sides, and terminal missing-rank timeout.
`accept_all` provides the total group-startup deadline and owns all connections
until every nonzero rank is admitted. Failure drops every connection collected
by that call; success returns a preallocated rank-indexed vector with slot 0
empty. It cannot be mixed with earlier individual admissions. TCP tests verify
successful out-of-order group setup and peer EOF when another rank is absent.

`rendezvous` composes publication, bounded waiting for a new file, validation,
and group admission under one elapsed-time budget. Rank 0 alone calls the
supplied NCCL-ID factory; peers read that ID from the validated record. One
rank explicitly skips file/TCP setup. The returned `BootstrapGroup` owns the
rank-indexed control sockets. The NCCL-ID factory and filesystem system calls
are synchronous setup operations and cannot be forcibly cancelled by this
budget; their elapsed time is checked before the next stage. This helper is
tested with real files and TCP. The reference benchmark now calls it before
NCCL initialization and retains the returned sockets through the run. Its
run identity derives from the required torchrun launch ID and per-pass bootstrap
path, separating warmup and measurement. Its agreement digest includes the
reference archive digest, world, geometry, reserve and archive switches.
This does not implement runtime epoch dispatch or distributed failure polling.
The benchmark integration has not yet passed a GPU execution gate.

The codec and file primitives are tested separately from GPU execution. The
reference example replaces `MGBNCCL1` setup with this record; legacy benchmark
measurements retain their pinned old source. GPU dispatcher integration remains
pending. Primitive tests do not prove the benchmark migration correct on GPUs.
# Host control pump integration status

Current wire schema is **2** (the historical filename is retained). The 64-byte
frame retains magic `MGBCTRL1`, but its u16 version at offset 8 is 2. Offset
48 now holds `source_rank: u32 LE`; bytes 52..64 remain zero. BEGIN requires
source_rank < world and identifies the selected sender on every rank, including
empty receivers. Other actions require source_rank = 0. Version 1 is rejected;
all ranks must run the same build. Earlier V1 descriptions below are historical
where they conflict with this paragraph. Source rank is independent of the
coordinator/sender identity in the existing `rank` field.
Rank admission additionally requires a real slot exactly when source_rank equals
the local rank. A receiver cannot consume its own pending offer for another
source's command, and the selected sender cannot claim an empty source slot.

`ControlPump` now composes the rank admission state, coordinator and nonblocking
TCP connections into one caller-driven polling loop. Construction validates the
star topology and reserves command and send queues before traffic starts. Each
poll handles at most one incoming frame per peer and one issued command group.
Protocol errors poison the pump and close its connections.

The caller receives `BEGIN`, `FINALIZE` and `PUBLISH` commands; it must report
actual transfer completion separately from consumption. Descendant offers must
precede consumption of their input. `finalized(true)` is a caller assertion of
local drain, not a CUDA event check performed by this module.

The two-depth lifecycle is tested both without sockets (one rank) and through
real loopback TCP (two ranks, alternating source, empty receiving peer). These
tests also exercise the Candidate→Request→Response→Receipt chain after source
closure: transfer completion alone must not finalize a layer, and descendants
offered before consumption must still execute. A duplicate local offer must
close the remote connection and permanently poison both pumps.
`poll_before(deadline)` checks a caller-owned deadline before and after polling.
Expiry closes connections and permanently poisons the pump; a later deadline
cannot revive it. The caller must retain a shared deadline across repeated
polls, advancing it only according to the configured progress policy. This is
not an independent watchdog: it cannot interrupt a caller blocked in CUDA/NCCL.
TCP tests cover expiry propagation and normal operation before expiry.
These are CPU control tests only. The GPU dispatcher, NCCL execution, watchdog
and hardware overlap validation are not implemented by this pump.
