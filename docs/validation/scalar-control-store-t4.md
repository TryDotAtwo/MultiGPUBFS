# Stream-ordered scalar control store: T4 gate

The `Buffer::put()` path copies a borrowed host slice and drains its CUDA
stream before returning. Commit `dd679d5411c6335f79b37cf70c087c62c6056111`
adds `mgbfs_device_store_u32` for scalar controls whose consumers are on the
same stream. The value is a kernel launch argument, so there is no borrowed
host-buffer lifetime. The legacy exchange count is consumed on another stream
and deliberately retains the synchronous upload. Generic bulk uploads, host
result reads, and the remaining fatal-vote waits are unchanged.

- Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v26, source `dd679d5`,
  status `COMPLETE`: two peer-accessible T4s; CUCO_RANK DENSE LSA full layers
  and archives matched the CPU oracle. HostSized CUCO_RANK and library control
  fixtures passed. Raw output: `test_results/kaggle_scalar_store_full_v26/`.
- Kaggle `trydotatwo/mgbfs-device-count-owner-pack-gate` v11, source
  `d7c59842c367da319c9a48c0be36d1629706e2bc`, status `COMPLETE`:
  standalone primitive passed plain, memcheck, racecheck, initcheck, and
  synccheck on each T4. `d7c5984` changes only the fixture's fail-closed
  link stubs after the first standalone link attempt failed. Raw output:
  `test_results/kaggle_device_store_v11/`.
- Local `cargo check --locked -p mgbfs-runtime --features cuda,library-owner`
  passed as a Windows type-check with configured library search paths; it did
  not link or run GPU code. `cargo test --locked -p mgbfs-core -p mgbfs-runtime
  --tests` passed the CPU test suite.

Kaggle `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v27 ran source `d7c5984`
on two P2P T4s: five fresh-process S10 runs per transport, alternating order,
with warmup and two verified rank archives per run. Batch 32,768, four shards
per rank, 256 buckets, 96 MiB fixed library pool per rank, DENSE, pre-dedup ON.

| Transport | Search median s (MAD) | Durable median s (MAD) | Sampled MiB/rank |
|---|---:|---:|---:|
| HostSized NCCL | 0.420060 (0.009924) | 3.839984 (0.035375) | 529, 529 |
| NCCL LSA | 0.367077 (0.000966) | 3.752394 (0.033378) | 567, 567 |

LSA search median is 12.6% lower than HostSized in this session. The 50 ms
`nvidia-smi` samples are not exact peaks. Compared with the older v4 paired
screen (HostSized 0.420099 s, LSA 0.366300 s), this change has **no
demonstrated end-to-end speedup**; sessions differ and the LSA shift is within
run variation. Raw v27 data, including all ten samples, is at
`test_results/kaggle_scalar_store_benchmark_v27/`.

The scalar store removes host drains at the named same-stream uploads, but
this is **not** a CPU-free owner-to-retirement pipeline or a full-app sanitizer
pass. Per-batch archive-error and post-owner fatal votes still return to CPU.
