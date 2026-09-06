# Admitted ControlPump to NCCL: two-T4 gate

Kaggle sanitizer version 35 completed at source
`3c4ad4034acc7f22eb48071dc4b868c70fa7e99d`, package `3c1b5f0`.
Two distinct Tesla T4 devices each report 15360 MiB:
`GPU-97b890ea-1dbe-18e1-7363-3a4ab6dce5ac` and
`GPU-6597d459-0522-2132-5214-93b07b47080d`.

The GPU scatter fixture now uses the real admitted ControlPump rather than
manually driving the individual byte-admission state machines. It passes plain
execution and all four sanitizer modes: memcheck, racecheck, initcheck and
synccheck. All error summaries are zero; race hazards and warnings are zero.
The fixture covers both source ranks, decreasing source-local slot tokens,
exact received bytes, source-local views, empty payloads, receive-capacity
rejection, CUDA event completion, NCCL health polling, all-rank CONSUMED drain,
Finalize/Publish, and repeated terminal abort.

Raw reconciliation:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v35/distributed-sanitizer test_results/distributed-sanitizer-v35-summary/distributed-sanitizer/summary.json 3c4ad4034acc7f22eb48071dc4b868c70fa7e99d
```

Result: `RAW_GATE_RECONCILED`, 40 tool logs, 36 measured ranks, 36 warmup
results and 36 archive verifier results. All 24 reference profile selections
preserve S4 layers `[1,3,5,6,5,3,1]`. The twelve existing archived runtime
fixtures, macro and macro-archive tests pass plain and all sanitizer modes.
A connection reset interrupted the first metadata/log download; the resumed
download completed and the full reconciliation passed. No state datasets were
downloaded locally and S13 was not rerun.

## What this does not establish

The exchange fixture is still serialized. It is not the asynchronous BFS
dispatcher, does not demonstrate concurrent CUDA jobs or a speedup, and does
not remove host synchronization from the existing native BFS reference.
Production buffer-bank reservation/event binding, full scheduler wiring and
overlap profiling remain separate gates. CPU/TCP tests exercise concurrent
admission, staggered consumers and pre-launch failure; this hardware fixture
does not yet exercise multiple simultaneously active receive banks.
