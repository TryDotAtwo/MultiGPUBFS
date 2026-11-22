# Carried exact ID: experimental branch, slower on S9

This branch is a bounded experiment, not a production setting or a default.
The accepted channel-geometry checkpoint remains on codex/packed-bitmap-leaf.
It supports only the experimental resident pack4 bitmap path, not device hash
or Explore.

## Contract

All ranks agree on a versioned packet identity mode before the symmetric
window is prepared. In carry mode the producer constructs the child packed
word, ranks that same word with the exact multiset rank, applies the agreed
owner mapping, and publishes word and mapped ID in the same release of the
bounded ring packet. Only this trusted, job-owned producer can use this path.
The receiver validates the format, rank range and owner before accepting it.
Verify mode additionally re-ranks every received packed word and rejects a
different ID. Legacy mode retains receiver-side ranking. An older library
without the capability rejects carry and verify before resident launch.

The experiment is slower on the measured case and is not promoted to the main
working branch. Its trust contract and results do not establish a performance
benefit on other graphs, GPUs or topologies.

## Correctness

CUDA 12.8.93, NCCL 2.30.7, physical V100-SXM2-32GB SM70:

- With 8 lanes/8 records, exact full-state layer multisets matched independent
  CPU BFS for S4, directed C6 and a repeated-symbol graph on 2 and 4 GPUs in
  carry mode, and on 2 GPUs in verify mode, including tiny segments, rank caps,
  duplicate-at-full and incomplete prefixes.
- For both carry and verify, page ownership rotations 1 and 2 matched full
  multisets at every layer of S4 and a repeated-symbol case on 2 GPUs, with
  persistent worker PIDs unchanged across modes and pages.
- The GPU verifier accepted matching repeated-symbol word/ID pairs and rejected
  deliberately changed IDs. The physical library/transport fixtures passed in
  legacy, carry and verify modes.
- CayleyPy on 4 GPUs matched every layer count and the complete final layer for
  S4, directed C6, repeated symbols and LRX3 in carry mode, including cap and
  duplicate-full cases. A killed rank invalidated its persistent group and an
  explicitly new group completed a job.
- New executable plus old CUDA library rejects the new modes and completes the
  legacy mode. The previous executable plus new library also completed legacy.

The native executable SHA-256 was
1674116e7c048c063a00408bfc70637db01983a8b2ad62631834cf2bf3b2aa17;
the loaded experimental CUDA library SHA-256 was
6d77ea0e86d8f01e6747827d331d7c85b77d75b08460a15caa70ada816f5ba8e.

## Matched S9 A/B

Same binary, same persistent rank processes, four V100, adjacent-transposition
S9, 8 lanes, 8 records, one slot/ring, two shards, segment request 65,536,
21 admitted extents of 4,567 rows/bank/rank, compact output. Run order was
legacy, carry, carry, legacy, legacy, carry. All 37 layer counts matched the
Mahonian oracle and the final state was the reverse permutation. Full state
multisets of every layer were compared only on the smaller cases above.

| Packet identity | BFS seconds, in run order | Median |
| --- | --- | --- |
| Receiver reranks, legacy | 13.7364, 13.8827, 13.8196 | 13.8196 |
| Producer sends exact ID | 17.5242, 17.6000, 17.5427 | 17.5427 |

Carry was 26.9% slower in this comparison, despite fewer blocked send attempts.
The cause was not established by hardware counters; reduced blocked attempts
alone do not show an end-to-end improvement. NVML sampled every 0.5 s observed
a maximum 2,452 MiB on each GPU in each run. Sampling can miss brief peaks and
does not establish the isolated VRAM difference. The 8-lane kernel in the
experimental binary used 96 registers and a 2,880-byte stack per thread
according to cuobjdump; the previous checkpoint used 96 registers and a
2,592-byte stack in a different binary. This observation does not identify the
source of the slowdown.

No eight-GPU resident or B200 result is claimed. Full logs, GPU artifacts and
source checkpoint are retained under
D:/MultiGPUBFS-evidence/2026-11-21-61786414/device-bfs-8gpu-alt-20261122/.
