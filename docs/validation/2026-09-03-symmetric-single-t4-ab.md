# Physical T4 A/B: native BFS versus CayleyPy

Kaggle kernel `trydotatwo/mgbfs-symmetric-single-gpu-a-b`, version 3,
completed on one Tesla T4.  Every timed measurement ran in a fresh process.
The calibration winner was selected before the five alternating repetitions.

## Correctness gate

S8 was exhausted by both implementations.  All 29 full-state layer SHA-256
digests and all layer cardinalities match exactly; both runs contain 40,320
unique states.  The native run used macro depth 2 for this gate.

## S10 result

Both implementations exhausted the same 3,628,800-state graph and returned the
same 46 layer cardinalities.

| implementation | selected batch | repetitions | search median, s | MAD, s | external peak VRAM, MiB | output contract |
|---|---:|---:|---:|---:|---:|---|
| native DENSE/CUB, K=1 | 262,144 | 5 | 0.465951 | 0.005155 | 3,195 | mandatory asynchronous state+Hash128 archive |
| CayleyPy single-GPU BFS | 262,144 | 5 | 1.474593 | 0.005595 | 11,037 | no archive |

Native search is 3.16x faster and its observed peak device consumption is
71.1% lower.  These figures compare search completion; native durable archive
completion is a separate stronger contract and took about 4.6 seconds.

Native samples: `0.460795, 0.463237, 0.484394, 0.465951, 0.492460` seconds.
CayleyPy samples: `1.468998, 1.468332, 1.484681, 1.477467, 1.474593` seconds.

## Macro-depth calibration

For S10, K=2 lost to K=1 at every tested batch.  At batch 262,144 the search
times were 1.002 s and 0.463 s respectively.  Macro lookahead is therefore a
supported workload parameter, not an assumed universal optimization.

The raw machine-readable evidence is retained in ignored local artifacts at
`artifacts/kaggle/symmetric-single-v3/symmetric-single-gpu/summary.json`.
