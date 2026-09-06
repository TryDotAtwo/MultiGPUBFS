# S11 profile panel v2: platform failure

Live evidence checked on 2026-09-06:

- Kaggle kernel `trydotatwo/mgbfs-distributed-bench-s11` reports `KernelWorkerStatus.ERROR`.
- Browser identifies version 2, script run `347581713`, source `d99d2b765e4e9f5f66de8f60a6fe733e85ff4acd`.
- Logs page: `Version 2 failed to run after 3587.1s`.
- Platform message: `high /tmp usage, possibility out of system disk: Cannot connect to the Docker daemon at tcp://172.23.192.108:2375. Is the docker daemon running? 0`.
- CLI output selection for `summary.json` and output-file listing both returned no files. No complete measurement inventory is available locally.

This is not evidence of GPU OOM, a BFS correctness error, or an identified leaking directory. The platform suggests temporary-disk pressure but also reports its Docker daemon unavailable. The exact filesystem consumer and last completed benchmark require additional evidence; do not publish partial timing estimates as a completed panel.

The one-active-T4 panel `trydotatwo/mgbfs-distributed-bench` was still RUNNING at the same check. It has not been restarted. S13 publication is unaffected and must not be repeated.

Before retrying this panel, instrument temporary-disk usage at job boundaries and preserve incremental small result summaries remotely. Existing per-job archive unlinking alone does not establish peak temporary-disk safety.

## Instrumented v3 live checkpoint (not a completed panel)

Version 3, script `347588746`, source `c75f803cc80fd9d639972733dca1cc4ab4b17872`, built CUDA and Rust successfully and entered S11 measurements on two physical T4s. The browser log at 567.8 seconds contains completed DENSE CUB/BMMA and HASH_FIRST scalar CUB jobs, with both rank archive verifiers reporting VERIFIED. Named allocation planes are present in real Linux/GPU rank results. Full panel and all repetitions remain pending.

The new disk telemetry reports `/tmp` filesystem total 8,656,922,775,552 bytes and about 1.10 TB free; output filesystem total 20,957,446,144 bytes. These are filesystem statistics, **not a demonstrated per-notebook quota**. After the first native job, `/tmp` used space increased only 192,512 bytes between job boundaries, not the approximately 2.29 GB of reserved rank archives. This supports release between these observed jobs, but does not prove peak usage, later-job release, or explain v2's platform failure. Do not treat the reported terabyte of free space as permission to allocate it.

One-active-T4 version 10, script `347585212`, was independently observed progressing through repetition 1 (zero-based), not stalled. Its source is `b02bf3f2526c5bce0f5331cb30c5317c2c80b91a`; do not mix its raw measurements with version 9's S10 panel.

## One-active-T4 v10 subsequently failed

CLI now reports ERROR. Its version-specific Logs page reports `Version 10 failed to run after 5260.5s` and `high /tmp usage, possibility out of system disk: Cannot connect to the Docker daemon at tcp://172.23.192.102:2375. Is the docker daemon running? 0`. Selecting `summary.json` through the CLI returned no files. Before failure the live log had completed native repetition 2 (zero-based), including both INT_MMA/BMMA pre-dedup variants; this is not a full five-repeat result or preserved rank-file inventory.

Do not repeat the uninstrumented version or claim its approximate live timings as a complete comparison. Two-active-T4 v3 remained RUNNING at this observation. Two separate platform failures with this message strengthen the need to inspect notebook-specific temporary storage, but still do not identify the consumer or establish a BFS fault.
