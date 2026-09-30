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

## v28 terminal result: diagnostic launcher failure

CLI COMPLETE is not a test pass. Downloaded summary/diagnostic bind the actual
source to a1712ad and two physical T4, P2P allowed1 both directions. Native and
Rust builds completed. Debugger rejected the compute-sanitizer launcher as
`not in executable format`; `-exec-run` reported no executable specified.
No application inferior started, no oracle or sanitizer summary: status
DIAGNOSTIC_FAILURE, not a BFS correctness failure or healthy timeout.
Logs: test_results/kaggle_nonblocking_boundary_v28/lsa-bfs-gate/.
Kaggle output CLI printed a Windows charmap error after downloading the
summary, diagnostic, source manifest and full logs; files were read directly.

Prepared correction: preserve original sanitizer argv and use an ELF-verified
`env` process as debugger parent if sanitizer is a script; normal launch only,
no attach or security changes. Three behavioral launcher tests RED then GREEN.
Next source pinned to 0303f37c3b86119e286c9d311a00e624c35ecfc7, including
CancelGroup owner-pair early exit. Runner SHA256
bd769e972e1227790ca08edc3a8e3d8199ada5fb89209d07c85a5144183f9347.
Correction hardware execution remains NOT_RUN until an actual next submission.

Corrected diagnostic actually submitted as v29; CLI RUNNING. Do not resubmit
while live. It pins runtime0303f37 and uses the recorded ELF-parent correction.

The same existing runner now has a prepared `process_faults_only` mode (not
submitted): explicit LSA build ON and bidirectional-P2P preflight; three fault
classes on both ranks, six independent-process cases. Existing process_scope
subreaper supervises torchrun descendants on timeout or terminal launch, checks
cleanup failure, preserves logs/elapsed/exit code and rejects group COMPLETE.
This is a prepared acceptance harness, not physical fault evidence. Changing
the mode requires a new submission only after v29 is terminal. Default remains
the active HOST_SIZED_NCCL diagnostic; no live notebook was modified.

## v29 terminal result and direct gate

v29 CLI COMPLETE, diagnostic TIMEOUT at180.0145s. Source0303f37, two physical
T4, P2P allowed0 in both directions. Debugger ran env/dash/readlink launcher
inferiors but never the BFS application; readlink exited and the active inferior
remained exited. Samples reported program not running. No oracle or sanitizer
summary. This is debugger scheduling failure, not measured BFS deadlock.
Full logs downloaded under test_results/kaggle_nonblocking_boundary_v29.

Next default `full_bfs_direct_gate` runs the existing CUCO_RANK HostSized
full-state/archive fixture directly: plain then all four full-BFS sanitizer
tools, no debugger, kernel filters or suppressed API errors. Existing
eight_gpu_gate.run_command supplies process_scope subreaping and strict
validate_log checks actual one-fixture completion and clean summaries. Each
stage has180s bound; failed plain skips tools; tool failures are individually
retained, not relabelled PASS. This is a two-GPU in-process control gate, not
independent-process LSA acceptance, capture/timeline or performance evidence.
LSA remains OPEN/UNSUPPORTED_HOST; no random P2P retry.
