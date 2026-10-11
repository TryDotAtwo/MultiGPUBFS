# VRAM admission in bytes

Generic HASH and SORTED admission no longer subtract a percentage of free VRAM.
Cold NCCL admission retains an absolute 2 GiB allowance. Once a healthy,
all-rank-approved communicator is parked, its allocations are already reflected
in live free memory; only 256 MiB additional allowance remains. Explicit
state/history/transport/GEMM/sorted buffers and sorted driver graph storage are
included in device_bytes. Reusable idle application and CUDA pool buffers are
credited once. CUDA pool retention uses total physical memory as its ceiling,
with explicit trimming before external allocations; this ceiling is not an
allocation or an admission reserve.

MGBFS_GENERIC_VRAM_RESERVE_BYTES overrides the additional allowance in bytes.
Inventory records additional_reserve_bytes and nccl_warmed. Global admission
rejects inconsistent reserve accounting across rank inventories.

Specialized SHARD_AB already queries actual NCCL-warmed native allocation:
its existing 256 MiB reserve plus 64 MiB query-to-run margin remain unchanged.
The cold 2 GiB allowance is a conservative absolute policy, not a measurement
of NCCL on every topology. No B200/B300 or 128-GPU acceptance is claimed.

## Physical GPU validation

2x RTX3060 12 GiB, driver 580.126.20. Full n8 exact CPU layers and final states
pass before and after near-capacity n14 HASH/SORTED allocations. n14 has only
two completed BFS layers, checked independently against CPU; this is memory
admission acceptance, not a full n14 calculation or throughput claim. All jobs
reuse the same two resident rank processes. Raw reports, inventories, memory
samples, source and SHA readback are in the corresponding HF evidence folder.
