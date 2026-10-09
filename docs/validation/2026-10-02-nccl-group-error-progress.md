# NCCL group error progress

send_recv/scatter previously returned a send/recv error without processing
GroupEnd progress. The correction always settles successful/in-progress
GroupEnd through existing cancellable await_nccl before returning the first
operation error. No healthy-path readback/new buffer/thread/fallback added.

Actual production wrapper against deterministic API boundary double:
RED pending progress queries were not consumed; GREEN send_recv send/recv
and scatter sender/receiver settle progress and retain original codes3/4.
Existing cancellation, startup, abort and retirement fixture passes too.
This is wrapper protocol evidence, not a reproduced vendor/GPU fault.
Ordinary immediate errors often propagate directly through GroupEnd in the
inspected NCCL2.29.7 source; no frequency/performance claim is made.

Real CUDA/NCCL rebuild passes;20 explicit CUDA/library-owner runtime unit
tests and14 CLI tests pass, including full-state/hash S4 archive oracle and
worker failures. Existing reference_bench.rs:578 unused_mut warning unchanged.
Two-T4 asymmetric errors, registration initcheck and paired A/B remain open.
