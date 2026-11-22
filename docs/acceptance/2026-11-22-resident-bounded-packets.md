# Experimental resident bitmap: bounded packets

This is an experimental source checkpoint, not a package release. The ordinary
host-driven backend remains the default. Resident packet size defaults to one
record; the opt-in MGBFS_RESIDENT_PACKET_RECORDS accepts 1 through 8, subject
to the loaded native library advertising sufficient payload capacity.

Each producer holds at most eight packed records per destination and flushes
the partial packet before closing the channel. A receiver retains its position
in the packet when a shard is busy and returns credit only after all records
are durably accepted or recognized as duplicates. Failure does not publish a
partial BFS layer. All ranks agree on the mailbox descriptor before the window
is prepared. The reported packet count is not a measurement of physical link
traffic. The one-record wire geometry remains supported.

## Correctness and compatibility

Physical tests used 2 and 4 V100 SXM2 32 GB (SM70), CUDA 12.8.93 and NCCL
2.30.7. They do not establish compatibility or performance on eight GPUs.

- The exported library and transport fixtures passed with packet capacities
  1, 3 and 8, including one- and three-slot rings, exact CPU-oracle layer
  states, duplicate-at-full and resource-stop cases.
- Persistent run_graph passed the full-multiset CPU layer oracle for S4,
  directed C6 and repeated-symbol graphs on both 2 and 4 GPUs, including
  exactly sufficient and one-less rank caps. Receipts showed capacity 8.
- Real CayleyPy persistent calls on 4 GPUs passed S4, C6, repeated symbols,
  LRX3, resource stop and duplicate-at-full; layer sizes and the full final
  layer matched the CPU oracle.
- Killing one rank during a 2-GPU job returned an error, invalidated the
  session and allowed an explicit new healthy generation. The 1→8→1 packet
  switch on 4 GPUs retained the same persistent worker PIDs.
- The previous resident CUDA library rejected a request for eight records
  before resident launch and succeeded with one record using the updated
  executable. The capability query prevents a silent fallback.

Compute Sanitizer 2025.1.0 memcheck with default API reporting completed the
fixture but returned a nonzero result: 40 CUDA error-209 reports during NCCL
kernel queries and two error-800 reports from NCCL cuMemCreate; both child
ranks returned zero. The complete log is preserved. Repeating memcheck and
synccheck with --report-api-errors no produced zero reported errors on the
same 2-GPU fixture. Those scoped runs do not validate suppressed CUDA API
errors or establish absence of inter-GPU races.

## S9 measurements

The graph is adjacent-transposition S9: 362,880 states, 37 Mahonian layers,
2,903,040 generated children. All layer sizes and the last state were checked;
complete sets on every layer were checked only for the smaller cases above.
All comparisons below use the same executable, CUDA library, four physical
V100s, graph, two shards and requested segment_rows=65536. Measured times
are BFS only; setup and export are excluded.

| Route | BFS seconds | Number of sent packets |
|---|---:|---:|
| Resident, 1 record | 25.130, 25.599, 25.314 | 2,903,040 each |
| Resident, up to 8 records | 14.427, 14.481, 14.424 | 363,607–363,622 |
| Existing host-driven bitmap | 0.451, 0.470 | Different host/NCCL transport |

Resident runs alternated 1,8,8,1,1,8 in the same persistent worker processes.
The median resident times are 25.314 s and 14.427 s, a 43% reduction on this
case. The batched resident path is still about 31 times slower than the
ordinary host path here. This is not a large-layer or B200 result.

The resident admission chose 21 segments of 4,567 rows per bank per rank,
2,302,272 B total segment storage and an 11,344 B bitmap per rank. Its mailbox
grew from 2,432 B to 4,224 B per rank. The host run chose two segments per
rank, so the comparison does not isolate transport alone. A 0.5-second NVML
sample peaked at 2,342 MiB on each tested GPU across the resident A/B series
and 1,282 MiB on each tested GPU in the separate host series; these are
sampled device totals, not guaranteed allocation peaks. Peak live rows and
layer-specific timings are not measured by this resident path.

Executable SHA-256:
3f3839546ae4eae14df94cb526db0ddc2b00d1d5ac6339332d6398a5ab0663e3.
CUDA library SHA-256:
5abb36e88a8d4d037264536064d790b5ccd4aac0ccfd1bf6c635f1476485de25.

## Reproduction

The opt-in tests/device_resident_s7_ab_gpu.py helper takes
RESIDENT_AB_N=9, RESIDENT_AB_SEGMENT_ROWS=65536,
RESIDENT_AB_ORDER=old,new,new,old,old,new,
RESIDENT_AB_OLD_PACKET_RECORDS=1 and RESIDENT_AB_NEW_PACKET_RECORDS=8.
Set RESIDENT_OLD_LIB and RESIDENT_NEW_LIB to the same matching library
directory, both SHA variables to its checksum, and RESIDENT_LD_TAIL to the
CUDA/NCCL library paths. RESIDENT_AB_ORDER=host,host measures the existing
path separately. Correctness tests and fault tests remain opt-in and require
a compatible physical GPU group; the 2-GPU compatibility fixture additionally
requires the previous native library.
