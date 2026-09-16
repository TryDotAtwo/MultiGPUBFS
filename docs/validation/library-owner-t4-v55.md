# Eight-shard library owner screen (Kaggle v55)

Evidence retrieved 2026-09-16 from completed `trydotatwo/mgbfs-library-owner-t4`.
This is the earlier two-rank implementation, NOT validation of the new eight-rank code.

- Library source: `f5b52c9f240e89c5b8b30828919ef56c367fdad6`.
- Native comparison: `013ed5c979f4225db273e0015fa9ed72fd230c90`.
- Two distinct Tesla T4 devices, 15360 MiB each.
- Notebook log confirms eight global shards and 256 buckets for both backends.
- S10, DENSE, five independent measurements; archive enabled for both.
- Library pool: 96 MiB per rank.

| Backend | GPUs | Search median, s | MAD, s | Durable median, s | Observed VRAM per rank, MiB |
|---|---:|---:|---:|---:|---|
| cuCollections | 1 | 0.488662411 | 0.004066038 | 4.837173577 | 901 |
| Native CUB | 1 | 1.032989698 | 0.003263622 | 4.797039825 | 835 |
| cuCollections | 2 | 0.465600246 | 0.010600967 | 3.843545656 | 529 / 529 |
| Native CUB | 2 | 0.824494743 | 0.027558312 | 3.995748132 | 457 / 457 |

cuCollections improves search time here but consumes more observed VRAM. Durable
completion is not accelerated by the same factor. This small S10 screen does not
establish the best backend on H200 or LRX13. Sampled VRAM is not a byte-exact peak.

## Correctness evidence inspected

All fifteen `bfs-{gpu0,gpu1,two-gpu}-{plain,memcheck,racecheck,initcheck,synccheck}.log`
files report their test passed. All twelve sanitizer runs report zero errors;
the three racecheck runs additionally report zero warnings/hazards.
The two-GPU fixture uses device threads and NCCL, not separate torchrun processes.
The CLI logs were separately audited: six two-process cases have twelve COMPLETE
rank records and twelve VERIFIED archive checksum/count reports. Cases are each
of cuCollections/cuDF on S4 DENSE, S4 HASH_FIRST and U4m2 DENSE. There is no
U4m2 HASH_FIRST CLI case in this artifact inventory; do not infer it from the
summary's abbreviated description. Archive checksum/count verification alone
does not establish equality of full state sets.

Local raw evidence (ignored by git):

- `test_results/library-owner-v55-summary/library-owner/summary.json` (all samples).
- `test_results/library-owner-v55-summary/mgbfs-library-owner-t4.log`.
- `test_results/library-owner-v55/library-owner/bfs-*.log`.

The new multi-peer source `fce7de137d2ffb121854759efcea330b7c25a803` is under a
separate capacity notebook run; do not transfer these gates to that revision.

Full log download completed successfully. All 30 per-rank archive verification
reports from the measured S10 runs report VERIFIED (checksum/count scope):
two backends times five repeats times (one plus two ranks).
