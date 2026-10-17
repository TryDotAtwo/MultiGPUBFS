# CUDA999 original-graph reproduction: 2026-10-17 UTC

The original CUDA999 **has not been reproduced or fixed by this investigation**. The current `0.2.0.dev29` package successfully executed the recovered graph and original first-candidate geometry on eight physical SM120 GPUs, including the original hardware class. No failed CUDA candidate was silently accepted and no error was injected.

## Recovered input

The graph is `UT(4,47)`, identity start, with forward elementary generators `E01(+1), E12(+1), E23(+1)` followed by their three inverses. The graph digest exactly matches the supplied failing pilot: `ff71582e6018295a7b1d23605f350623df9059f4e3d595c25ec29a75f9666e7e`. The [definition](acceptance/cuda999-ut47-graph.json) preserves the original generator order.

The captured geometry was used unchanged: eight ranks, capacity 3,080,192/rank, batch 174,762, queue capacity 524,288, 33,554,432 table slots, one shard, three history banks, full transport and CUDA generation. The first four layer sizes were independently checked against a CPU oracle: `1, 6, 26, 110, 414`.

## Physical evidence

| Check | 8×RTX 5060 Ti | 8×RTX PRO 6000 Blackwell Server Edition |
|---|---|---|
| Driver / compute capability | 580.173.02 / 12.0 | 580.159.03 / 12.0 |
| Original-plan four-layer resident and fresh runs | Passed | Passed |
| Actual measured autotuning | Five candidates; zero candidate failures | Five candidates; zero candidate failures |
| Common candidate prefix | Recorded per candidate | All five reached the same 15 completed layers |
| Resident graph/buffer transitions | Five transitions | Thirteen transitions |
| Instrumented native ranks | 8/8 | 8/8 |
| Memory-only memcheck | Zero errors | Zero errors, stream-ordered allocation race checks enabled |
| Driver peer-access capability | No | All 56 directed peer pairs |
| Explicit P2P/CUMEM original-plan probe | Not supported | Passed; transport confirmed in every rank log |
| Thirteen warm transitions with actual P2P | Not supported | Passed |
| Explicit P2P/IPC and cuMem-host variants | Not supported | Both passed |

The default NCCL transport on both rented topologies was `SHM/direct`. P2P was tested separately with `NCCL_P2P_LEVEL=SYS`, and its actual use was verified from the rank logs. These are test settings, **not new production defaults or a recommended workaround for the original server**.

The long 5060 Ti prefix ended after 15 completed layers with `INCOMPLETE / RESOURCE_258`. It was not a full graph calculation. The original harness expected a profile/deadline stop and recorded an assertion failure; both the original result and the resource-limited classification are retained. Successful bounded/resource stops in tuning are not CUDA exceptions and are not claimed as full graph timings.

## Sanitizer boundaries

An initial unfiltered memcheck run completed BFS but returned exit 86 with 160 CUDA API 400 reports in NCCL kernel-attribute discovery (`cudaFuncGetAttributes` / `cudaGetLastError`). That failed result is retained, not relabeled PASS. Separate memory checks disabled CUDA API error reporting, applied no kernel-name filters, verified instrumentation in all eight live native processes, and reported zero memory errors. The PRO 6000 check additionally enabled stream-ordered allocation race tracking. This does not prove absence of all inter-GPU protocol races or identify the original CUDA999 cause.

## Identity and remaining gap

The tested wheel SHA-256 is `ab5d6452dd1588bf4c1a003ad911b4cea03d3e6e61e7a6ea3cc18d2d40bf7a51`; native executable SHA-256 is `10e423aee53f450d5a14f064db0cb1c7784e8080e7b94045adbad378ad304fce`. The current native CLI source is `1067171010ae0a27e283aa0b19365da768affb28`. The CUDA library is compiled for SM120, with NCCL 2.30.7 and CUDA 12.9.

The original archive identifies the graph, plan, failing API and layer. It does **not** identify all loaded original binaries, the original driver or the original NCCL transport. An API reporting an asynchronous error does not by itself identify its initiating kernel or communication operation. The next useful evidence is a failure on the original node with the current pinned package, complete rank logs and `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,P2P,SHM,NET`, `MGBFS_TRACE_NATIVE_CALLS=1`. Repeating the entire graph catalog is unnecessary for this recovered input.

No engine/allocator/transport patch is asserted here. The current tests establish that the actual probes work on the tested hardware; they do not establish that the original server problem is resolved. Existing failed-candidate containment is separate from this root-cause investigation.

## Artifacts and cost

The [structured receipt](acceptance/cuda999-sm120-reproduction-20261017.json) includes every bounded result, configuration, measured candidate, sanitizer boundary and deletion receipt. Complete owned test scripts and rank logs are available as `cuda999-sm120-reproduction-20261017-logs.tar.gz` in the [dev29 release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261017-v29). Container hostnames and private IPv4 tokens are redacted; `SANITIZATION.json` records original and published file checksums. No BFS state payloads or original private-server logs are included.

Both rentals are deleted and their API responses confirmed `ABSENT`. The combined conservative compute/disk bound was $3.57 within the new $10 stage budget; final invoices and traffic are not confirmed. No B200/B300 was rented, no credentials were transferred, and product source/build/test execution was not performed on the local computer.
