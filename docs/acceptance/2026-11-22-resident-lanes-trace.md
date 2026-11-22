# Experimental resident bitmap: channel count and public API trace

This is a source checkpoint, not a release or a default change. The resident
engine remains an explicit experimental opt-in and currently supports pack4
bitmap/atomic, three banks and no autotuning. Device hash and Explore are not
implemented. The evidence below is for four compatible GPUs, not eight or B200.

## Change

The job-agreed resident mailbox can use 1–8 producer/consumer channels per
rank (MGBFS_RESIDENT_LANES); unset still uses four. Each channel retains its
own single-producer/single-consumer ring and consumer warp, with the existing
nonblocking shard lease. The CUDA ABI capability reports the channel limit.
An older library without this capability accepts only its original four-channel
layout; unsupported explicit requests fail before resident launch. The new
library keeps a four-channel kernel specialization.

## Physical correctness

Hardware: four Tesla V100-SXM2-32GB, SM70, CUDA 12.8.93, NCCL 2.30.7. The
eight-device host is not all-pairs native-peer-atomic compatible; it does not
establish an eight-device result.

With packet size eight and channel count eight:

- CTest library/transport fixtures passed on 2 GPUs; the same fixtures passed
  with 1, 3 and 4 channels. The accept fixture passed.
- Public native runner and persistent workers matched independent CPU BFS
  **multisets of every complete layer** for S4, directed C6 and a repeated-symbol
  graph, separately on 2 and 4 GPUs. These include two active shards, tiny
  segments, a duplicate at a full rank cap, and one-below-cap INCOMPLETE.
- CayleyPy persistent on 4 GPUs matched all layer counts and the complete final
  state multiset for those cases and LRX3, including tight-cap and duplicate-full
  cases, with unchanged worker PIDs.
- Killing one of 2 active rank processes failed and poisoned the group; an
  explicitly new group completed a subsequent job.
- The prior packet-capable library rejected channel counts 1 and 8 before GPU
  launch and completed the four-channel/eight-record case.

The native executable SHA-256 was
425829af00d60a817ed8f318e8e359d725584146bc453939234d1b7dbccd0f18;
the loaded CUDA library SHA-256 was
796b516a4e24c0de4843a42cb90c2aa9606a0aed1484c2aabbdcc8d926e514ce.

## Same-artifact S9 comparison

Four GPUs, same executable/library, same persistent worker PIDs, same
Coxeter S9 graph, segment request 65,536, 2 shards, 21 admitted extents with
4,567 rows per bank per rank, one slot/ring, 8 records/packet, compact output.
All 37 layer sizes agree with the independent Mahonian formula; the final
state is the reversed permutation. This is not a full-state comparison of all
S9 layers. Sequence was 4, 8, 8, 4, 4, 8 channels.

| Channels | BFS seconds, in run order | Median |
| --- | --- | --- |
| 4 | 14.2632, 14.2687, 14.2452 | 14.2632 |
| 8 | 13.3797, 13.2397, 13.3869 | 13.3797 |

The median is 6.2% lower with eight channels on this case. The previous
ordinary host backend on this same four-V100 S9 finished in 0.45–0.47 s on a
different CUDA library from the immediately preceding source checkpoint. Thus
this experiment remains far slower and should not be selected for normal work.

NVML was sampled every 0.5 s: the first four-channel run showed a maximum of
2,342 MiB on each GPU; subsequent runs, including later four-channel runs,
showed 2,406 MiB. Persistent contexts and the 8-channel kernel had already been
loaded, so these samples do not isolate a memory cost for either geometry and
may miss shorter peaks. Mailbox allocations in the receipts were 4,224 and
8,320 bytes/rank respectively; the numbers are not full VRAM accounting.

## Public CayleyPy execution trace

Nsight Systems 2025.1.1 recorded the actual public CayleyPy graph.bfs() call
on the same four GPUs, S8 (40,320 states), eight channels and eight
records/packet. The run completed with all layer sizes and full final layer
matching the independent CPU oracle. The four resident kernels ran for
1.326–1.331 s each, with one 320-thread CTA per rank. During each kernel the
CUPTI trace contains no new kernel launch and no GPU memcpy on that rank; it
contains one cudaEventRecord and 261–262 cudaEventQuery calls/rank. The CPU
polls completion/fault handling every 5 ms; it does not determine the next
packet or layer. This does not assert zero host activity or prove behavior on
an unsupported topology. A first attempt with Nsight Systems 2025.3.2 did not
capture CUDA activities (CUPTI_ERROR_INVALID_DEVICE); the successful 2025.1.1
trace and the failed trace are both retained.

Full logs, receipts, binary copies, trace and sampled telemetry are retained
under D:/MultiGPUBFS-evidence/2026-11-21-61786414/device-bfs-8gpu-alt-20261122/.
The comparison is bounded to this workload and hardware; large-layer,
performance/capacity, device-hash, Explore, and eight-compatible-GPU gates
remain open.
