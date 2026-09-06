# Flat physical payload-bank reservation

`PayloadBanks` preallocates host leases for fixed disjoint device ranges.
The allocation owner supplies one device allocation of `bytes()` bytes.
For capacity C, alignment A and slots N, the checked layout is
`stride = align_up(max(C,1), A)` and `bytes = N * stride`.
Returned bank handles bind a physical index to the full ticket key; validated
offsets, consumer tokens and retirement refer to that same bank.

A busy pool returns no available bank without changing state: the dispatcher
must not send an admission ACK yet. Byte overflow, duplicate live epochs,
stale handles or consumer errors are terminal. Completed consumer descriptors
are not recycled within one ticket. Pools belong to individual resource
classes/planes; their handles are not interchangeable or wire identifiers.
Transfer/consumer event observation remains the dispatcher's obligation.

RED: three added tests failed on missing `PayloadBanks`. GREEN: all seven
payload tests pass, including bank 1 being released/reused while bank 0 still
has a consumer, checked alignment/overflow, and duplicate/stale tickets. The
full local runtime suite also passed after implementing the pool. Local CUDA
tests remain gated out; this is not GPU evidence.

The changed native scatter fixture uses the pool's actual allocation size and
returned offsets for CUDA/NCCL receives, with transfer/consumer events indexed
by physical bank. Ordered transport COMPLETE is separated from deliberately
reversed consumer retirement. Source-local views still reference their held
send ranges. This hardware change is pending Linux type-check and 2xT4 gates;
the already reconciled v38 run does not cover it.

Production BFS scheduling and owning device-allocation wrappers remain separate
work. The host pool is not a device-side allocator or a throughput claim.

## Hardware result (reconciled 2026-09-06 UTC)

Kaggle v39 completed at `47cbb1a2ee26a9678847433ea6b06c92656bced4`,
package `8afbf3d`. Distinct Tesla T4 devices, each 15360 MiB:
`GPU-81624915-b8ae-101b-b16b-d5853d0f4c14` and
`GPU-39023a04-85b3-b150-3523-f3da475de283`.

The physical-offset and reversed-consumer-retirement fixture passes plain,
memcheck, racecheck, initcheck and synccheck, with zero errors and zero race
warnings/hazards. Full reconciliation:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v39/distributed-sanitizer test_results/distributed-sanitizer-v39-summary/distributed-sanitizer/summary.json 47cbb1a2ee26a9678847433ea6b06c92656bced4
```

Result `RAW_GATE_RECONCILED`: 40 tool logs, 36 measured ranks, 36 warmup
results and 36 archive verifiers. All 24 reference profile selections preserve
S4 layers `[1,3,5,6,5,3,1]`; existing runtime/macro/archive regressions pass.
No state payload datasets were downloaded locally; S13 was not rerun.

This does not validate subsequent `ebcbb99` route-count optimization or the
full asynchronous BFS dispatcher. No performance claim follows from this gate.
