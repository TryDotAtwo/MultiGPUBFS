# HOST_SIZED_NCCL correctness gate — 2026-09-30

Separate from NCCL_LSA. Prepared runner:
[kaggle/host-sized-nccl-gate/kernel.py](../../kaggle/host-sized-nccl-gate/kernel.py).
Runner SHA256 `7405f051bf7c52c50adb49e9cb4010dec607df4bf8c9663c417e3a210916b973`. Uses existing private boundary notebook; no new notebook.

Frozen archive SHA2564021009152c7b8de41a07f0c65bc22cf5e126cc575d9f59338a5bb0e1e7d503c,
base f5ac5a62506b10520f75b5f94d86ba4e7991a028 plus recorded working edits including
D2H fatal-before-drain patch. Private source input unchanged; verify archive and
955 source hashes. Later documentary ledger edits are outside this fixed package.

HOST_SIZED_NCCL: existing distributed_native.rs non-LSA exchange branch gets
row counts with its size handshake then submits device payload using wrapper
nccl_send_recv. cuda/nccl_transport.cpp uses ncclGroupStart, ncclSend/ncclRecv,
ncclGroupEnd, await_nccl and existing cancellation/abort. CUDA peer mapping is
not a required application ABI. NCCL chooses supported transport; retain its
actual INFO transport/setup logs. Do not force P2P/SHM/socket choices or change
backend on error. This remains a host-sized control path, not a claim that
CPU orchestration/count readback is eliminated. Library-internal staging is
not CPU owner/dedup/algorithm fallback.

NCCL_LSA instead allocates/registers a symmetric NCCL window, requires the full
world LSA team and directly accesses remote memory through ncclGetLsaPointer.
Its v27 P2P0->1/1->0 allowed0 remains UNSUPPORTED_HOST; no LSA sanitizer pass.
Primary NCCL semantics: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
Transport configuration: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

Main gate configuration: two physical T4; cudaDeviceCanAccessPeer is recorded
but does not reject this mode; same pinned NCCL2.29.7; MGBFS_NCCL_LSA=OFF;
cuda-gdb normal parent launch (no attach); existing Compute Sanitizer memcheck;
180s monotonic supervision and90/140s application-inferior stacks where possible.
Fixture is nonignored cuco_rank_two_gpu_dense_layers_and_archives_match_oracle,
so its exact launch intentionally omits --ignored. U3(3), then S4; both rank maps
and prededup states; exact oracle and verified archives. No throughput expansion.

Invariant scope: fixed owner64MiB and fixed capacities, >=1GiB reserve/T4,
no growth/spill/application CPU fallback; same first-failure notification and
original error preservation; completion-gated buffer/receive/parent/pinned reuse.
No runtime change or protecting dependency removal. Observe actual termination;
external process killing bounds diagnostic cleanup, not successful GPU retirement.
This healthy-path test does not inject D2H/NCCL faults: those gates remain OPEN.

Report source/binary/hash/build/config/transport, oracle and sanitizer summary.
PASS requires actual selected fixture passed with exit0 and clean summary;
TIMEOUT/unsupported/runtime failure/missing evidence remain distinct. Parent-MI
mode is scheduling-sensitive diagnostic, not equivalent to untraced acceptance.
MI sanitizer child-following is still to be exercised. All four sanitizers and
independent-process failure/retirement tests remain separate acceptance.
No paid resource, installation, security change or other notebook interruption.

## Actual submission v28 (supersedes prepared source identity above)

Notebook `trydotatwo/mgbfs-nonblocking-boundary-t4` v28 submitted successfully;
authenticated CLI subsequently returned RUNNING. Only one BFS notebook active.
Independent account UI check at 15:41–15:42 UTC showed one Cube555 active,
its extra width-audit cancelled and MGBFS sanitizer draft off; total quota
interpretation was not proven. No other notebook was interrupted by BFS.

Runner now clones public GitHub and checks out exact commit
`a1712ad0b881ff1b7c819536d0f74d9ecdeac7fb`; verifies HEAD and clean checkout,
records tree and SHA256 of every tracked file before build. No dataset input
or uncommitted working-tree snapshot enters this build. Runner SHA256:
`955e1af44df529623676dacfc8731c924e514520fa84b616eca0d899818e49e0`.
Python AST and private metadata validation PASS before submission.

Hardware result PENDING. This HOST_SIZED_NCCL configuration does not execute
the LSA-specific changed error branches at a1712ad; it checks the same build's
ordinary NCCL correctness/sanitizer control. LSA asymmetric-fault acceptance,
CPU-free timeline, four sanitizer gates and paired A/B remain OPEN.
