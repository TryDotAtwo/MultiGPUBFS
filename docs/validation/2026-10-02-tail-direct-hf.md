# Isolated two-GPU exact BFS tail validation

Native program: `4c82501db1fc38de396d49af6b187ab22fc14568`.
Published branch code and helpers: `3e2bab97093e39cde92df7079913a324722d3ef9`.
Hardware: two RTX 3090 GPUs on the separate Vast lease 53888860.
New states and bulk logs stayed on the GPU host and were uploaded directly to HF.

## Verified baseline finite grid

All 119 pairs `2 <= n <= 15, 1 <= m <= n` were attempted; `m` maps to native `r`.
With 256 host archive ring slots: 93 COMPLETE, 26 INCOMPLETE, zero pending.
The final payloads and manifests are frozen at HF commit
`745f6dfff88013ae13d32806c606fed11d9406f1`.

Remote streamed readback passed for all 2591 files, totaling 2,161,864,688 bytes.
The verifier checked bytes and SHA-256, admission only from completed layers,
COMPLETE retention of at least the final three whole layers (or graph beginning),
10 decimal GB or graph beginning, and INCOMPLETE retention of at most 1 decimal GB.
Every COMPLETE run's sum of layer counts matched `n! / m!`.
This is a count/retention check; separate small full-state CPU and two-rank GPU
oracles check actual reachable words.

[Baseline readback evidence](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261002-directhf/deferred-grid/verification.json)

## Additional runs and metadata

All 26 INCOMPLETE pairs were rerun with 2048 host archive ring slots:
8 additional COMPLETE, 18 INCOMPLETE. Across unique pairs: 101 COMPLETE,
18 INCOMPLETE. Publication/readback of this additional cohort remains a
separate gate until its verifier finishes; it must not be counted as verified
offsite merely from its local manifest.

Required metadata checks passed for all 145 manifests: exact start and actions,
8/16-byte nibble packing, source commit and binary SHA-256, full launch/runtime
configuration and GPU inventory, contiguous completed layer statistics,
whole-layer duration and two per-GPU observed peaks. Monitoring requests samples
every 50 ms; a layer with no observation records null, not an invented peak.

[Metadata audit](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261002-directhf/metadata-audit.json)

## Matched owner and archive comparison

At `(15,4)`, identical two-GPU configuration, three repetitions per owner/mode,
all twelve runs have the same 26-layer completed prefix. Median sums of
whole-layer maximum-rank durations:

| Owner | Archive off | Archive on |
| --- | ---: | ---: |
| CUCO_RANK | 0.461959 s | 0.646725 s |
| CUCO_INDEXED | 0.520959 s | 0.742327 s |

CUCO_RANK is selected for this measured configuration. The experiment does not
prove full `(15,4)` completion or a universal performance/memory advantage.

## Storage, cancellation and reproducibility

The GPU host's physical SSD stress wrote 13 decimal GB, retained four whole
layers totaling 10.4 GB, evicted the oldest layer and verified retained SHA-256.
The INCOMPLETE record-aligned suffix was 1 GB and its readback hash matched.
These are synthetic payloads, not reachable BFS states or GPU-search results.

Early deadline cancellation passed: INCOMPLETE, completed depth -1, no admitted
files. Separate native capacity-stop validation retained completed layers 0..3.
Both owners at `(7,4)` and wide packed cases `(17,16)`, `(32,31)`, `(32,32)`
passed CPU full-state comparison and direct HF streamed checksum verification.
Fourteen CUDA runtime unit tests and the two-rank full-state oracle passed.

The final source bundle and eight helper/evidence files were read back from HF:
all nine SHA-256 values matched, total 9,107,472 bytes, frozen source commit
`4cf485ba57bc3b8cc24c7dab68d193cc38ff6a95`.

[Source readback evidence](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261002-directhf/final-source/verification.json)

The additional cohort's readback and explicit rental deletion are pending
acceptance steps. Native ring-stop reasons remain explicit INCOMPLETE outcomes;
they are not promoted to confirmed VRAM capacity causes.
