# Bucket-owned extents and dense packed hash: combined experiment

The bucket-owned extent index and the `MGBFS_RESIDENT_HASH_LOAD_PERCENT` 50–75 setting are both experimental and opt-in. **Do not select this combined geometry as a default**: under pressure its allocation can fragment before useful capacity is reached.

Four Tesla V100-SXM2-32GB GPUs, one resident worker/GPU, NVLink topology, driver 580.178.04. Source tree `bf3f6aece87c51733eaf020e30d21c985e84a8af` at original tested commit `55285abc5bb56a2a838558d85bf353b1c06a84bd`; the code in this branch matches that tested tree, with additional documentation. Executable SHA-256 `07f4da780f6fa91350d92a63861f94f32f85ce386bcfb980d9f4274d6da5427b`, CUDA library SHA-256 `3cb83c0a0a03043218197f4a205e6267df31e46545808b1c216051c5aad78ee7`. The earlier branches document the separate experiments.

32 physical shards, 4 virtual buckets/shard, three banks, exact packed keys, compact positions, table rebuild, GPU timing instrumentation, resident batch scheduler, 262144-edge send and receive buffers. S12 used 262144 configured rows/segment and 8M target-layer rows/rank; S13 used 1048576 and 80M respectively. Each result is one observation.

| Graph | Load | State | Depth | Completed states | BFS s | Worst NVML sample MiB |
|---|---:|---|---:|---:|---:|---:|
| S12 | 50 | COMPLETE | 66 | 479,001,600 | 15.600 | 8,088 |
| S12 | 75 | COMPLETE | 66 | 479,001,600 | 15.856 | 6,296 |
| S13 | 50 | COMPLETE | 78 | 6,227,020,800 | 434.472 | 30,448 |
| S13 | 75 | COMPLETE | 78 | 6,227,020,800 | 452.673 | 26,472 |
| S13, 9 GB segment budget/rank | 50 | INCOMPLETE / RESOURCE | 3 | 441 | 0.015 | 6,740* |
| S13, 9 GB segment budget/rank | 75 | INCOMPLETE / RESOURCE | 4 | 1,715 | 0.016 | 9,074* |

The pressure budget limits segment storage, **not total VRAM**; queues, transport, CUDA/NCCL reserve and other allocations are additional. A large fixed segment assigned exclusively to each active virtual bucket strands capacity, so these runs are not a valid comparison of total BFS time against longer prefixes. *For fast stopped runs, a 0.5 s NVML sample misses startup peaks; the sample is not the allocation peak.*

Every completed layer count in these cases matched the independent Mahonian coefficients, and complete runs returned the exact reverse terminal permutation. They do not prove all intermediate large state sets. Small-graph oracle runs on these four physical GPUs compared complete layer multisets with ordinary and forced-collision hashes. S6 rehash tests also covered forced collisions and sufficient/insufficient limits; an S4 persistent test switched 50→75→50 and checked a resource stop followed by a healthy job.

Per-case machine-readable compressed receipts are in [artifacts](artifacts/). Full logs, source, binaries and topology are in `multicta-hash-combined-retry-evidence.tgz`, SHA-256 `943babae88ab839315d8b2f2c748afd83e12aa1cdb63fb85875105478bbb2a67`. No 8-GPU, B200, full intermediate-state-set or dynamic-growth claim is made.

## Allocation receipts and CTA coverage

Rank 0's segment allocation receipt for the complete S13 load-50 case was 7,095,676,824 bytes of state, 22,455,760,844 bytes of index and 6,792 bytes of counters, for 29,551,444,460 bytes of segment storage. For load 75 it was 8,959,033,344 bytes of state, 16,424,894,464 bytes of index and 8,544 bytes of counters, for 25,383,936,352 bytes of segment storage. The 9 GB cases reported 8,999,995,852 and 8,999,993,936 segment-storage bytes, respectively. These include reserved/allocated capacity, not the number of simultaneously live states.

For the complete S13 cases the rank-0 receipt separately reported 18,875,904 queue bytes, about 14.7 MB resident-batch scratch and a 4,195,264-byte mailbox; these categories may overlap with other accounting and must not be blindly summed. The receipts do not measure peak live rows or the number of simultaneously active CTA/SM for these combined runs. The physical multi-CTA execution has independent earlier evidence, but the occupancy of this specific combination remains unverified.
