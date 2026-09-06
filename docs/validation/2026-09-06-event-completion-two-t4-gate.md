# Generation-bound event completion: two-T4 gate

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` completed at source
`c8e9a08cb1f480b817d43668931c49ad0af0b865`, packaged by `8c3995c`.
Evidence is retained under `test_results/distributed-sanitizer-v34/`;
the independently fetched summary is under `distributed-sanitizer-v34-summary/`.

The inventory identifies two distinct Tesla T4 devices, each with 15360 MiB:
`GPU-ec442d64-3b1c-5d23-1740-24968faf0dd0` and
`GPU-fba6e465-b54e-4010-ecd7-2ba4ccc24bf0`.

The scatter fixture passes plain execution and memcheck, racecheck, initcheck,
and synccheck. All sanitizer summaries are clean, including zero race hazards
and warnings. It covers schema-3 TCP byte admission before NCCL, both source
ranks, decreasing source-local slot tokens when changing source, exact received
bytes, source-local views, zero payload epochs, receive-capacity rejection,
NCCL health polling and repeated terminal abort. Completion now uses the
generation-bound CUDA event wrapper instead of host stream synchronization.
The event reuse generation is the global epoch; it is deliberately distinct
from the source-local ticket token.

The full evidence reconciliation command was:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v34/distributed-sanitizer test_results/distributed-sanitizer-v34-summary/distributed-sanitizer/summary.json c8e9a08cb1f480b817d43668931c49ad0af0b865
```

It returned `RAW_GATE_RECONCILED`: 40 tool logs, 36 measured rank results,
36 warmup results and 36 archive verifier results. The existing twelve runtime
fixtures, macro and macro-archive fixtures passed all five execution modes.
All 24 profile selections passed, with S4 global layers `[1,3,5,6,5,3,1]`.
Sixteen local admission/event contract tests also passed in this verification.

## Limits

This remains a serialized exchange fixture, not an overlapping BFS dispatcher.
It does not prove concurrent slot retirement, production ControlPump admission
integration, elimination of production host synchronization, or performance.
Handshake revision 2 is included in this source and the Linux control-contract
suite, but the GPU fixture itself directly constructs its TCP connections.
The existing native BFS is still the synchronous reference implementation.
No S13 rerun or dataset upload was performed for this gate.
