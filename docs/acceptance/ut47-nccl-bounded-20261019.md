# Bounded NCCL submission fix (dev30)

The reproduced eight-rank, 256-shard UT(4,47) startup hang is fixed by limiting
each deterministic peer/lane group to at most 256 NCCL send/receive calls.
Previously one group contained 60,942 calls at width 16. Counts, payload bytes,
buffer offsets and per-peer ordering are unchanged. No autotuning fallback,
shard restriction, CUDA device synchronization or deadline increase is added.
Empty payload rounds submit only the counts group.

## Evidence

- Untouched dev29 times out on the first layer; reverting after the candidate
  passes reproduces the timeout (78.9 seconds including cleanup).
- Patched full shared library passes persistent 1 -> 128 -> 256 -> 1 transitions
  on eight RTX 3060. All four prefixes are `[1, 6, 26, 110, 414]`.
- UT(4,3), 256 shards, completes all 729 states with CPU-oracle layer equality.
- Ordinary UT(4,47) measured autotuning and subsequent BFS pass the diagnostic
  contract. The calculation remains resource-partial; it is not a full UT(4,47).
- 96 CPU geometry contracts preserve every directed peer payload and group
  ordering. Four injected error/cancellation cases close the active group.

The full JSON receipt records precise timings, layer sizes, hashes and limits.
The precise internal NCCL queue/frame was not captured; bounded submission and
reversion establish the causal trigger in our transport, not a universal
explanation for every CUDA999 or the server's earlier InfiniBand fault.

## Distribution

All production SM components and the universal component are rebuilt. The dev29
executable, CUDA runtime and NCCL component are reused; the fix is in the CUDA
shared library. Legacy diagnostic contrast libraries built from different source are excluded
from this release; selected-SM and universal comparisons remain available. Compilation for an SM is not an execution test on that SM.
Use the dev30 wheel attached to this GitHub release after publication. Start a
new Python process. No LD_PRELOAD or manual cache replacement is required.
