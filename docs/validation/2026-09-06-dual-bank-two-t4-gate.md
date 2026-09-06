# Two live NCCL payload banks: two-T4 gate

Kaggle sanitizer version 36 completed at source
`cc80e7b82450290d9e4a63ff347e4b725d36f79e`, package `284f7be`.
Two distinct Tesla T4 devices each report 15360 MiB:
`GPU-ff327863-13ac-937d-1ffc-264a4b322f0f` and
`GPU-80b5d044-cf36-5682-1e9c-5ae10544094f`.

The admitted ControlPump fixture submits two transfers into distinct payload
banks before polling either completion event or retiring either consumer.
Each bank has independent expected bytes and a generation-bound CUDA event.
Both source ranks, decreasing source-local slot tokens, source-local views,
zero payloads, receive-capacity rejection, health polling, all-rank retirement,
Finalize/Publish and repeated abort remain covered.

Plain execution, memcheck, racecheck, initcheck and synccheck all pass.
Sanitizer error summaries are zero, including race warnings and hazards.
The existing twelve runtime fixtures, macro and macro-archive tests also pass
plain and all four sanitizer modes.

Raw reconciliation on 2026-09-06:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v36/distributed-sanitizer test_results/distributed-sanitizer-v36-summary/distributed-sanitizer/summary.json cc80e7b82450290d9e4a63ff347e4b725d36f79e
```

Result: `RAW_GATE_RECONCILED`, 40 tool logs, 36 measured rank results,
36 warmup rank results and 36 archive verifiers. All 24 reference profile
selections preserve S4 layers `[1,3,5,6,5,3,1]`. Only logs and metadata were
downloaded locally; S13 was neither rerun nor reuploaded.

## Boundary

This proves the tested two-bank lifetimes, not simultaneous kernel execution
or BFS throughput. NCCL calls use one stream; fixture copies still synchronize.
Bank selection is fixture-controlled, not production dispatcher reservation.
The production BFS reference still needs admitted control-plane integration,
bounded bank/event ownership and measured generation/sort/exchange/owner/archive
overlap. No end-to-end performance improvement is claimed by this gate.
