# Automatic fixed-r pruning GPU validation

Source: `6c1c0cbc926b99ad3714d44622a8146cd6bf49ca` on the isolated
`codex/bfs-tail-archive` branch. Hardware: two RTX 3060, 12 GiB each.
The native binary was built at `fd05c5ff3c67b3937c22bac24138f722ae27075a`;
the later commit changes only the Python pruning driver and its tests.
Binary SHA-256 and complete runtime configuration are recorded per case.

No range arguments were supplied. The automatic list contains all 527
`(n,r)` pairs for `2 <= n <= 32` and `1 <= r <= n`.
Configured capacity was deliberately small (8 rows per rank, future capacity
16, batch 4, 64 MiB owner pool) to exercise actual native resource stops.
This is a correctness test of pruning, not the hardware's maximum capacity
or a new performance comparison.

Results:

- 100 attempted GPU cases: 69 COMPLETE, 31 INCOMPLETE.
- 283 unattempted pairs pruned only above an observed stop at the same r.
- 144 structural exclusions: four-bit alphabet or native u64 orbit limit.
- No pending pairs. Every pruned case has its exact parent and heuristic flag.
- Other r branches continue; pruned pairs have no native run directory.
- Resume invokes no native runner and preserves the normalized ledger.
- All 69 COMPLETE graphs match independent full-word CPU BFS layer sets
  computed on the GPU host, including the 16-byte packing path.

The GPU run exposed native layer-capacity code
`LIBRARY_RANK_DEPTH_FATAL_16_16`. The driver now recognizes sticky capacity
code 16 specifically; unrelated rank-depth codes, remote cancellation,
HF failures and CUDA device-selection failures do not trigger pruning.
Nine CPU unit tests pass. The pruning-policy fingerprint is v2.

An initial container run failed CUDA device selection with status 804 because
the image selected a forward-compatibility driver library on GeForce.
Prepending the real host-driver directory `/usr/lib/x86_64-linux-gnu` to
runtime `LD_LIBRARY_PATH` fixes the device initialization. Those failed runs
are not used as resource or graph-completion evidence.

States remain on the leased GPU host and are published directly to
`TryDotAtwo/multigpubfs-bfs-results`. Publication uses payloads first, then
manifests and the complete sweep ledger. No states or bulk native logs from
this test are downloaded to the user's computer.

HF verification passed for all 510 packed payload files (19,576 bytes),
all 100 manifests, and the exported source archive (8,929,280 bytes).
All 100 manifests match native completed-layer counts and whole-layer times;
orbit totals and retention checks passed. VRAM uses a 50 ms monitor; short
layers can have explicit null observations, not fabricated zero peaks.

Frozen evidence:

- GPU report at HF commit `b65b80a314c9e35ef1288b67389cfde8674c0321`:
  `evidence/20261002-autoprune-3060/report.json`.
- Metadata audit and source at `be8aae07f1ce94fe5c4c9460c9e7edba1e1dffbd`:
  `evidence/20261002-autoprune-3060/metadata-audit.json` and `source.tar`.
- Source archive SHA-256:
  `3caa7b85240bcdfe9856a173823564c27a6f7b622ebaeec23927d170becfcffc`.
- Sweep ledger: `tail-sweeps/20261002-autoprune-3060-v3/sweep.json`.

The leased instance was destroyed after verified publication; its remote HF
credential was removed first. This test does not re-run the earlier physical
10 GB storage stress or claim that pruned graphs cannot fit different resource
settings or larger hardware.
