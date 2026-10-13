# Broad diagnosis without server administrator access

Install the current pinned GitHub wheel, then run `multigpubfs doctor`. For a Python-only job submission, start a fresh process:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "multigpubfs.diagnostics"])
```

The mode diagnoses the current authorized GPU-job environment. It never requests more GPUs, removes visibility masks, changes cgroups/security policy, installs a driver, resets GPUs or restarts production BFS. The native algorithms are unchanged. All dynamic probes use isolated diagnostic processes and bounded timeouts.

## Evidence collected

| Layer | Actual observation | Important limit |
|---|---|---|
| Process creation | Spawn error/errno, parent resource limits, child PID and last phase | EAGAIN, EPERM, ENOMEM and SIGKILL are not automatically CUDA errors or proof of OOM |
| Device access | NVIDIA device-node metadata; open/close of control/UVM nodes without ioctl | A successful control-node open does not establish every GPU permission; numeric/MIG nodes are stat-only |
| Security | Current-process seccomp/NoNewPrivs/capability flags, LSM context, own cgroup metadata | Active security flags alone are not proof of blocking; cgroup-v2 device BPF policy has no devices.list interface |
| Allocation/visibility | Whitelisted CUDA/container/scheduler GPU settings | Management-device count or missing scheduler variables do not prove allocation |
| Driver selection | Actual loaded libcuda path, CUDA Driver API version, initialization/count and symbolic errors | A library directory named stubs alone is only a warning |
| CUDA runtime | Actual runtime path/version and independent device query | A smaller minor driver API version alone is not incompatibility |
| NVML | Driver management-library path, initialization, reported driver version and management count | Management access/count is not CUDA-use permission; CUDA/MIG enumeration can differ |
| PyTorch | Installed build/runtime, CUDA discovery and context initialization in original client environment | Optional; CPU-only/missing Torch is skipped. A small observed memory headroom skips import rather than risking job exhaustion |
| Library search | Bounded existing libcuda candidates, symlink resolution, ELF class/machine, loader settings | No whole-disk scan or download; alternate probes are failure-only and never persist a fix |
| Native loading | Bundle hashes, CPU key-info, actual dependency inspection and missing ABI/library errors | CPU execution is distinguished from a failed ldd/helper inspection; paths containing spaces are supported |
| Loader contrast | At most four bounded isolated variants; remove a present preload/audit override or try at most two existing non-stub x86_64 driver libraries | All visibility/allocation settings remain unchanged. Success is verified through both direct API and actual native graph-count |
| Contrast reproduction | Original baseline is rechecked after a successful variant | Without a reproduced baseline failure, loader causality remains a hypothesis; no real BFS is restarted |
| Native loader trace | Failure-only graph-count with filtered LD_DEBUG library/version evidence | Collection success is not GPU acceptance; graph-count never runs BFS |
| Filesystem/IPC | Atomic exchange, temporary/shared-memory write/read, free space and relevant mount flags | /dev/shm denial is not blamed for an earlier CUDA discovery failure |
| Local networking | Bounded TCP loopback bind/connect with no external traffic; interface names and configured NCCL flags | This does not validate multi-node connectivity or prove one NCCL transport was selected |
| Fabric initialization | Read-only systemctl is-active state when available | Inactivity is not proof this hardware requires Fabric Manager; system-not-ready remains scoped to the actual Driver API error |
| GPU execution | Context, 4-KiB readback, real native successor kernel, worker protocol, NCCL and exact S3 BFS | A smoke is not full-capacity, throughput, all-family or all-architecture acceptance |

## Read the result

The new directory contains:

- `report.json`: all stages, errors, durations, commands, log tails, last phases, partial snapshots and synthesized analysis.
- `report.txt`: stage outcomes plus the readable analysis.
- `analysis.json` / `analysis.txt`: primary failure, best-supported explanation, observed findings with evidence references, hypotheses, exclusions, unknowns and next user actions.
- Per-stage logs and partial `driver.json`, `environment.json`, `runtime.json` snapshots.
- `manifest.json` and `diagnostics.zip` with SHA256-checked logs/metadata.

An unavailable optional component is SKIPPED, not silently counted as a passed test. A required failure prevents PASS. Timeout snapshots remain explicitly incomplete; library/version observations before a stuck cuInit survive. GPU checks not reached are not blamed for the failure.

For error 35, compare the driver/runtime stage, NVML and Torch evidence, and loader contrasts. Possible distinctions include: no usable driver library, a confirmed stub, an API-reported kernel/library mismatch, an older major Driver API plus runtime incompatibility, explicit visibility masking, an observed device-open denial, or a reproduced loader-environment difference. If the evidence does not establish one cause, the report says so.

The analysis also distinguishes missing native kernel images, incompatible GLIBC/GLIBCXX/CXXABI, inability to execute a helper/native binary, cgroup-v1 device rules not granting a recorded operation, shared-memory restrictions and local TCP denial. It does not infer a particular denied security mechanism solely from a flag or missing file.

## Acting without an administrator

A stable successful loader contrast provides a candidate user-level configuration for the **same** allocated GPUs. It changes only a diagnostic child's environment; the original failure is retained. Discovery success must then be followed by the ordinary kernel/NCCL/BFS checks before treating it as a usable fix.

An explicit visibility filter or GPU-less submission is addressed through the service's authorized GPU-job selector. Do not unset allocation masks to gain access outside the job. Missing build targets or ABI versions require a matching package/build or the correct existing compiler-library environment; a pip package cannot replace the host kernel driver.

Enforced permissions, kernel-driver faults, unreadable device BPF policy and some fabric failures may remain operator-level. The mode records the specific failed operation and what evidence is inaccessible; it does not claim user-space can always identify or repair hidden kernel policy.

The baseline analysis is single-host. The per-stage timeout is configurable with `--stage-timeout`; loader contrasts additionally have a maximum 30-second matrix budget and 6-second individual query limits, keeping a failed experiment bounded. No successful contrast is installed automatically.

## Privacy and reproducibility

Reports stay in the job output; nothing is automatically uploaded. The bundle excludes BFS state payloads, raw worker-job environment files and binaries/symlinked candidate driver libraries. Only a whitelist of diagnostic environment settings is read, not all variables, credentials or other users' processes. Paths, UID/device ownership, cgroup paths, CUDA visibility identifiers and LSM context can still identify the server/job: inspect or redact the bundle before public sharing. Published GPU acceptance contains only sanitized stage summaries and test evidence.

References: [NVML device count semantics](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html), [NVML system queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlSystemQueries.html), [cgroup-v2 device controller](https://docs.kernel.org/admin-guide/cgroup-v2.html), [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
