# One-shot recovery diagnostic

Run one Python parent inside the GPU allocation, with every requested GPU visible:

```sh
python -m multigpubfs doctor --recover --devices 0,1,2,3,4,5,6,7 --total-timeout 1800
```

This diagnostic is separate from production BFS. The 30 minute budget applies to experiments only; the generated production command uses `max_seconds=None`. It neither resets GPUs nor changes drivers or system permissions. It never retries an interrupted production BFS.

## What it checks

* Verified native component delivery and runtime identity; filesystem, device access, independent driver and management evidence, topology and available kernel errors.
* Context/memory readback, hardware attributes and real CUDA kernels concurrently on every selected GPU.
* Safe-first all-peer NCCL exchange, scalar collectives, measured NCCL memory reserve and a distributed BFS prefix. Alternatives are isolated and their exact settings are retained.
* After each failed experiment: fresh-process CUDA health. If CUDA is unavailable, remaining GPU experiments are blocked, not reported as additional transport failures.
* Complete independent CPU-oracle UT(4,3), repeated persistent requests and real measured autotuning plus a bounded UT(4,47) prefix on the full selected GPU set. A partial graph is never called complete.
* Bounded loading-order/API contrasts and exact BFS mode fixtures for hash, sort, GEMM, indexed/parent transport, radix order, shards, repeated symbols, matrices and wide states.
* When no whole-set candidate passes and CUDA remains usable, pair probes localize failing links. Pair success never counts as whole-set acceptance.

The suite uses finite fixtures, not every graph and parameter combination. Exhausted time, unsupported capabilities, denied kernel logs and blocked tests remain explicit. A missing `dmesg` permission is not evidence that no driver error occurred.

## Output and interpretation

The directory contains `report.json`, `recovery-summary.txt`, `findings.json`, phase/rank/NCCL logs and one SHA-verified `diagnostics.zip`. Large state arrays, full job environments and credentials are excluded; bounded log truncations are listed in the archive manifest.

`recovery.status=READY` means one exact configuration passed exchange, oracle BFS, measured tuning, a real prefix, final health and the public launcher check. `report.status=FAIL` can coexist with READY when an earlier alternative failed: those failures are preserved. READY does not prove every graph fits, peak throughput, or that a node cannot fail later.

Only READY produces `launch-command.txt` and `recovery-profile.json`. The command launches UT(4,47) with automatic tuning on the selected GPUs and no artificial time limit. `--graph /path/graph.json` selects another native graph; that graph's complete traversal is not certified by the diagnostic. Production logs are archived on normal exit or Python exceptions; SIGKILL or node loss cannot guarantee final archive creation, although existing logs remain.

Profiles are bound to the node/boot, GPU/driver evidence, visibility, relevant loader/transport environment and exact package/native identity. Changing those requires a new check, not blindly reusing a transport choice from another server.

If CUDA becomes unavailable in fresh processes, the tool cannot repair the kernel driver without privileges. It states the last observed call and affected ranks, preserves transport errors and recommends a new allocation/node or operator recovery only when warranted. It does not assert that CUDA999 alone proves faulty hardware or insufficient memory.
