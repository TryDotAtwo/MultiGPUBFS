# Resident pair transition validation, 2026-10-11

Hardware: eight physical RTX3060 12GiB on one host, driver595.58.03.
Native CUDA kernels unchanged from72befa. Tested small LRX(9,1), bounded
LRX(14,1), then small again in one resident group. All eight PIDs stayed
unchanged. All n9 layers and retained states matched an independent CPU
oracle. n14 matched the pinned previous run through complete depth43;
its deadline stop is not a complete graph calculation.

Session hardware metadata and selector digests are cached. Digest keys use
file identity/size/mtime/ctime; session metadata keys use device visibility
and rank generation. Failed queries and free-memory queries are not cached.
An exact graph profile can reuse its selector result only after fresh native
VRAM admission yields the same full plan/rank geometry. Graph, capacity,
devices, budget, options, loaded dependencies, source and process generation
are part of the key. Geometry or generation change uses the full selector;
CUDA/rank errors propagate without a blind restart. Production admission
remains live. The redundant preliminary auto-capacity query was removed:
the final query still validates the requested batch and current VRAM.

Measured before removing that last redundant query:
- previous warm profile check:0.726–0.767seconds
- new warm profile check:0.030941seconds
- request to BFS after large→small:0.106370seconds
- complete warm n9 launch including BFS and receipts:0.208145seconds
- n9 BFS:0.105340seconds
- first large allocation request→BFS:0.711267seconds

The first large allocation is not millisecond-only reuse. New geometries
still require setup and live admission. Cold tuning remains about18seconds.
This test does not prove a globally optimal large-frontier profile, maximum
hardware utilization, or B200/B300/128GPU readiness. CPU oracle and HF
publication are outside request→BFS and run_graph timing.
