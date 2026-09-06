# Native control frame V1

Status: implemented codec and real loopback TCP test; **not yet wired into the
GPU dispatcher or bootstrap**. This specifies transport framing, not a complete
sequencer or proof of asynchronous NCCL issue order.

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
The eventual dispatcher must provision its bounded outbound queue separately.

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
FINALIZE=6. Planes: NONE=0, CANDIDATE=1, REQUEST=2, RESPONSE=3,
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
acknowledges the specified epoch with a data plane and no slot. SOURCE_CLOSED
has no slot/plane and epoch=0. FATAL has no slot/plane and a nonzero fatal code.
FINALIZE is sent by rank 0 with no slot/plane. All nonfatal messages have a zero
fatal code. Exact send counts remain in the ordered NCCL metadata exchange,
not in this frame.

Required dispatcher checks still to implement: bind sender rank to the
bootstrapped connection/run digest; enforce depth/epoch monotonicity, slot
capacity and legal lifetime, action direction, duplicate acknowledgements,
receive credits, and complete depth drain. The codec does **not** authenticate
a peer or establish that these semantic checks occurred. FINALIZE receipt alone
does not authorize StateRing reuse.

Validation:
`cargo test --locked -p mgbfs-runtime --test control_wire --test exchange --test transport`
passes 13 codec/TCP tests and 10 existing CPU sequencing tests. Frozen READY
bytes, short I/O, EOF, invalid fields, and actual loopback READY/BEGIN/COMPLETE
exchange are covered, including nonblocking partial-frame arrival, peer
independence, frame reuse, pending-send capacity/offsets, and terminal error poisoning. This is not a multi-GPU
or Linux hardware gate.
