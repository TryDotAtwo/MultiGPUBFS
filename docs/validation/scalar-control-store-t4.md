# Stream-ordered scalar control store: T4 gate

The `Buffer::put()` path copies a borrowed host slice and drains its CUDA
stream before returning. Commit `dd679d5411c6335f79b37cf70c087c62c6056111`
adds `mgbfs_device_store_u32` for scalar controls whose consumers are on the
same stream. The value is a kernel launch argument, so there is no borrowed
host-buffer lifetime. At this first source revision, the legacy exchange
count was consumed on another stream and deliberately retained the synchronous
upload. Generic bulk uploads, host result reads, and fatal-vote waits remain.

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

Commit `3fffcbcb44167cfd17e627d51e47a129e0cbc145` also publishes the
HostSized NCCL peer-count word through `mgbfs_device_store_u32` on the
**consuming exchange stream**, immediately before that stream's size exchange.
The producer pack is already drained for the HostSized path, and the count
value is a kernel argument rather than a borrowed host slice. This removes
the cross-stream `Buffer::put` drain at that peer-round boundary; it does not
remove the host readback of the received count or the following payload-size
decision. The reused collective word is ordered with later owner/fatal work
through the existing exchange-completion event.

Kaggle v34, exact source above, status `COMPLETE`, ran on two P2P-capable T4s.
The complete LSA CUCO_RANK DENSE oracle/archive fixture passed, then the
HostSized `library_two_rank_layers_and_archives_match_oracle` passed across
native/CUB, cuDF and cuCollections indexed owners, DENSE/HASH_FIRST profiles,
two rank maps and symmetric/asymmetric fixtures. The HostSized CUCO_RANK
DENSE oracle/archive fixture passed as well. All three tests reported one
passing test and zero failures. Raw summary/logs:
`test_results/kaggle_host_count_store_v34/lsa-bfs-gate/`.
The local CUDA-feature Rust type-check and the full CPU test suite passed at
this source. V34 is plain full-state correctness, not a sanitizer, overlap,
large-graph or speed result for the changed transport boundary.

In parallel, Kaggle `trydotatwo/mgbfs-library-owner-t4` v60 built the same
source on a separate two-T4 host and reported `PASS` with `full_bfs_gate=true`.
Its plain 1/2-GPU library-owner fixtures and two-process torchrun CLI/archive
cases for S4 and U4m2 completed across cuDF, indexed cuCollections and
CUCO_RANK (the latter DENSE only). The gate explicitly requested no new
sanitizer tools, and these tiny graphs are correctness checks, not a speed
or VRAM comparison. Raw manifest and logs:
`test_results/kaggle_host_count_store_library_v60/library-owner/`.

Kaggle LSA notebook v35, source `6dd41bb1f669a7008e08ab2c2b0769ad64fe71b7`
(same runtime code as `3fffcbc`), completed a fresh paired S10 screen on two
P2P T4s: five unprofiled runs per transport, archive enabled and all 20
per-rank archive verification logs reporting `VERIFIED`.

| Transport | Search median s (MAD) | Durable median s (MAD) | Sampled MiB/rank |
|---|---:|---:|---:|
| HostSized NCCL | 0.435774 (0.007724) | 4.008849 (0.053205) | 529, 529 |
| NCCL LSA | 0.390792 (0.012448) | 4.057025 (0.018930) | 567, 567 |

LSA search median was 10.3% lower within v35, with 38 MiB more sampled VRAM
per rank. This session does **not** isolate the HostSized count-store change:
the earlier v27 run used another host/session, and both transports' medians
shifted upward. No causal speedup from the new scalar store is claimed. Raw
samples, 50 ms VRAM traces and archive logs are under
`test_results/kaggle_poststore_benchmark_v35/lsa-bfs-gate/`.
