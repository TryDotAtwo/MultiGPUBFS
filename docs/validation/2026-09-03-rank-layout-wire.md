# Rank layout, absolute origins and schema2 frames

Scope: architecture-v2 dependency contracts, not a production GPU runtime.

## Implemented

`mgbfs_core::rank_plan` combines fixed persistent/state/generation/route/owner
planes with twelve mandatory, named query groups. Queries are bound to the
entire rank shape, policy digest, build digest and device UUID. Each lane gets
its own aligned span; alignment is applied BEFORE multiplying by lane count.
Owner fixed scratch uses I,J,K only. Increasing L does not grow that scratch.
All arithmetic is checked. VRAM budget uses free bytes after runtime warmup,
minus the untouched reserve; pinned budget is separate. Errors report required
and available totals. No allocations or GPU actions occur in this function.

Query responsibility is explicit:

- Generation and Hash contain internal packed matrices, generator/coefficient
  tables, int32 products/partials and library workspaces, per generation lane.
  The external parent, child-state and hash banks are separately fixed planes.
- Route contains additional local sort/select/scan scratch, per route lane;
  two hash and two ordinal banks are separately fixed planes.
- OwnerMerge/OwnerSelect/OwnerScan contain queried library storage per owner lane.
- Materialize contains ALL Qmat planes, per materialization lane.
- Transport contains ALL typed send/receive banks, framing, directories and
  ticket metadata across the rank, including independent progress credits.
- Obligations, FixedDevice, ArchiveDevice and ControlPinned contain the remaining
  explicitly named rank-wide buffers. No silent missing-query zero is allowed.

An empty allocation vector with provenance means an explicit zero query, not an
unavailable implementation. These records are trusted implementation inputs,
NOT cryptographic evidence that a library was actually queried. Tests supply
literal synthetic inputs; they do not certify production scratch sizes. Actual
CUDA/NCCL query adapters, disk-format capacity accounting and post-allocation
cudaMemGetInfo validation remain required. `RankShape` is an allocation contract,
not the completed RunConfigV2 wire schema. Report offsets describe flat pools;
an allocating adapter must honor the largest requested base alignment.

`StateRing` now records each extent's absolute allocation sequence. Parent refs
address individual records, not descriptor IDs. Resolving checks live extent
ranges and readability; wrap gaps and recycled addresses are rejected. A parent
after enumeration is readable only while its origin lease remains live.
Concurrent HASH_FIRST checks these refs at request and response processing.
The CPU model scans extent metadata; this is not a proposed GPU pointer container.

`mgbfs_core::wire` implements schema2 field-wise LE frame headers and OriginRef.
Each plane has explicit logical bytes and aligned reserved bytes. Decode checks
magic/schema/reserved fields, expected kind/session tag/sequence/depth/ranks,
record and byte caps, exact derived payload size. Payload validation rejects
nonzero padding, malformed plane layout and truncation. Concurrent transport
roundtrips headers for every non-final ticket and every peer, including empty
peers. Payload objects remain CPU reference objects in that simulation; this is
not a full serialized data-plane integration. Full UUID TCP envelopes, receipt
payload codecs, archive schema2/BLAKE3 commits and config vectors remain pending.

## Tests

- Rank-layout fixtures cover missing queries, stale shape/build/device, lane
  replication, int32 intermediates, both profiles' transient state banks,
  pinned alignment, exact budget boundaries and arithmetic/name errors.
- StateRef fixtures exercise real wrap with a padding hole and a recycled
  physical address, plus request lease release before reclaim.
- Wire fixtures use literal independent 64-byte header and 16-byte origin
  vectors; test all five kinds, empty and partial frames, corrupt metadata,
  padding and capacity limits.
- Existing 144 concurrent complete-graph schedules now validate parent refs
  and non-final header coverage; complete state layers remain the oracle.

Observed RED: RANK_PLAN_NOT_IMPLEMENTED; STATE_REF_NOT_IMPLEMENTED;
WIRE_NOT_IMPLEMENTED; integrated wire frame count0 versus expected75.
The pinned-layout fixture also caught an avoidable 3840-byte leading gap;
4096-aligned archive storage is now placed before smaller control storage.

No GPU performance or full-runtime correctness claim follows from these tests.

Final local gate: **56 Rust CPU tests + 2 Python hardware/source guard tests
passed**, no failures. Command in cached `multigpubfs-rust-toolchain:dev`:

```sh
cargo fmt --all && cargo test --locked --offline && python3 -m unittest discover -s tests -p test_native_kaggle_gate.py
```

Additional mutation fixture rejected a changed source-batch field at header
offset32 after adding it to expected ticket validation (first observed RED).
Baseline checkout was verified clean at f0f2b8e5ee61173039ab9742f3a7756c9b6365e6.
No CUDA source changed, no GPU job launched in this slice.
