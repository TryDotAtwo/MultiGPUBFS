# Host errors must escape before GPU progress

Production change after `2be1007`, DENSE+LSA+CUCO_RANK only:

- Archive submission error returns to `advance_archived` before the next
  device fatal collective. Its existing outer handler Release-publishes the
  sideband failure token, then aborts NCCL on the dispatcher thread.
- Owner host/API error returns to the same outer handler before the post-owner
  GPU vote. Failed recording of `owner_consumed` also returns without draining.
- Healthy epochs retain both GPU votes, bounded completion credits, receive
  last-reader event, zero-payload exchange order and FinalizeDepth.
- Device capacity failures still poison ring/control and participate in the
  ordered GPU votes. They are not classified as host/API failures.

The removed error-only path previously attempted GPU submissions and waited on
the communication stream before notifying the sideband. A failed rank cannot
be required to issue another collective to report its inability to issue GPU
work. Cancellation deliberately ends ordinary issue order on the fatal path;
the peer dispatcher observes cancellation and aborts its own communicator.

Verification: Linux x86_64 CUDA/library-owner typecheck of library_multi_gpu
passed. Previous CPU suite and four failure_report tests passed at `2be1007`;
they do not execute this changed CUDA dispatch. Independent review and real
two-process asymmetric owner/archive/API injections remain OPEN. Do not claim
this fixes the historical healthy memcheck timeout or proves bounded LSA GPU
reclamation. No memory lease is released early by this change.

Prepared Kaggle source archive `4021009152c7...` predates this change. Rebuild
and freeze a new source package before testing this runtime. HOST_SIZED_NCCL
is an explicit synchronous control, not acceptance of the CPU-free LSA path.
