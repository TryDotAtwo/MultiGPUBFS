# Physical 2xT4 S10 A/B

## Result

Kaggle kernel `trydotatwo/mgbfs-distributed-bench`, version 8, completed successfully on two Tesla T4 GPUs. The native and CayleyPy runs produced the same 46 S10 layer counts and the same total of 3,628,800 states.

| Runtime | Selected batch | Search median (5 runs) | MAD | External peak VRAM / rank | Total external peak VRAM | Output contract |
|---|---:|---:|---:|---:|---:|---|
| Native NCCL DENSE | 262,144 | 0.512600 s | 0.003527 s | 3,431 MiB | 6,862 MiB | Mandatory lossless state + Hash128 archive |
| CayleyPy torchrun | 1,048,576 | 1.379884 s | 0.012133 s | 9,223 / 9,207 MiB | 18,430 MiB | Layer counts only; no archive |

For the timed BFS search, native is **2.692x faster** and uses **62.77% less total peak VRAM**. Native durable archive completion has a 3.257536 s median; it is reported separately because the baseline does not produce an archive.

## Repetitions

- Native seconds: `0.494387898, 0.512600460, 0.509072983, 0.522081290, 0.516118640`.
- CayleyPy seconds: `1.427816838, 1.367695773, 1.392017317, 1.371051432, 1.379884390`.
- Native calibration batches: `16,384`, `65,536`, `262,144`.
- CayleyPy calibration batches: `65,536`, `262,144`, `1,048,576`.

## Provenance

- Native source: `4ef9ce1d16c8cef62fc610cd6d36e32e673e623b`.
- CayleyPy baseline: `f0f2b8e5ee61173039ab9742f3a7756c9b6365e6`.
- Raw summary: `artifacts/kaggle/distributed-bench-v8/distributed-bench/summary.json`.
- Environment and every raw run, rank record, log, and `nvidia-smi` sample are retained beside the summary.
- CUDA/NCCL initialization and dummy collective were outside the BFS timer. Every measured repetition ran in a fresh process, alternating native and baseline.

## Interpretation boundary

This is an exact exhaustive S10 comparison on physical 2xT4. Search timing is directly comparable. Durable completion is not symmetric because native writes the required lossless archive while CayleyPy does not archive states.
