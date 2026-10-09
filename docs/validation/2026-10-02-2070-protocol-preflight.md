# RTX2070 diagnostic admission: unsupported peer access

Own lease53906867, machine123544, label mgbfs-lrx13-2070-protocol-20261002.
Two real NVIDIA GeForce RTX2070,8192MiB each; driver API version13000.
`nvidia-smi topo -m` reports PIX. Direct libcudart12 query nevertheless returns
cudaDeviceCanAccessPeer status0/allowed0 for both0->1 and1->0.

This host cannot execute the required NCCL_LSA diagnostic. No library build,
BFS, archive, token upload or two-rank acceptance run was started. Only own
lease was deleted immediately after admission failed; unrelated rentals
were untouched. No BFS data existed to recover.

Initial actual price$0.13111111111111112/hour including40GB disk. The external
guard was armed as PID37064 with WAITING records and deadline21:26:09UTC;
early deletion avoids waiting for that deadline. This guard depends on the
Windows host and network, not a provider hard cap. Final billing may lag.

The immutable runtime candidate remains bb97d5a. RTX2070 inventory support
is an explicit diagnostic opt-in and does not admit these GPUs as T4.
Inventory tests pass7/7; full Python discovery passes203,8skipped.

Separate bootstrap investigation: full ordinary parallel CPU cargo test
passed under strace on c28e020.350 traced selected syscalls across528 trace
files; longest selected syscall fsync7.301ms. This passing sample does not
explain or resolve the prior seven BOOTSTRAP_TIMEOUT failures. Trace data
is retained in the existing Linux build volume under
bootstrap-parallel-c28e020.trace.*. No timeout or filesystem semantics changed.

Required physical2xT4 failure/sanitizer/timeline and paired A/B gates remain
open. Failed host admission must not be called a passing BFS test.
