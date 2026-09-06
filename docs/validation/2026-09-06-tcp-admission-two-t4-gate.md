# TCP byte admission to NCCL: two-T4 gate

Kaggle sanitizer v33 completed at source
`e078a0e0665e69bbbd4252d0b17a4d5a538d63a3` on two distinct Tesla T4 devices.
Raw logs and source SHA were reconciled, including 40 tool logs, 36 measured
rank results, 36 warmups and 36 archive verifiers (`RAW_GATE_RECONCILED`).
Evidence: `test_results/distributed-sanitizer-v33/distributed-sanitizer/`.

The scatter fixture now exchanges native schema-3 TCP byte descriptions,
rank-local capacity acknowledgments and coordinator Launch frames before
calling NCCL. Both source ranks, exact received device bytes, self views,
empty epochs, receive-overflow rejection before NCCL, health polling and
terminal repeated abort pass plain and all four Compute Sanitizer tools.
There are zero reported errors or racecheck hazards/warnings.

The existing 12 runtime fixtures, single-device macro/archive fixtures and
24 one/two-rank profile selections also passed their recorded checks.
Reference profile layer counts reconcile to `[1, 3, 5, 6, 5, 3, 1]`.

This is a serialized integration fixture, not the asynchronous BFS dispatcher.
It uses stream synchronization for completion and globally increasing test
slot tokens. It does **not** validate the later NativeEvent implementation,
handshake version negotiation, or the source-local slot ordering fix at
`c8e9a08`. In particular, lower slot tokens after changing source rank require
the later fixture. No performance/overlap claim follows from this gate.
No published S13 data was recomputed or downloaded locally.
