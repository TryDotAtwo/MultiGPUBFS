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
