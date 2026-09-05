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
