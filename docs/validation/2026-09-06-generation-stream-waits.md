# Generation-bound downstream stream waits

`NativeEvent::wait` enqueues `cudaStreamWaitEvent(stream, event, 0)` after
validating the recorded generation. The host need not first observe completion.
Unrecorded, stale and retired generations reject before submitting native work;
a native wait error poisons the event. Multiple consumer streams may wait on
the same active generation. Waiting does not release a payload lease or mark
the producer complete.

This follows the [CUDA stream-wait contract](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html):
future work on the destination stream depends on work captured by the event.
The wrapper adds generation validation and requires caller-owned buffer and
consumer lifetimes. Graph capture is outside this wrapper's contract.

RED: three tests failed because `EventGeneration::wait` was absent. GREEN:
`cargo test --locked -p mgbfs-runtime --test event_generation` passes all eight
tests, including wait-before-host-completion and native-error propagation.

The native scatter fixture now enqueues two D2D consumers on separate
nonblocking streams immediately after each NCCL event record, before polling
either transfer event. Each consumer has a separate completion event and a
PayloadLease token; exact bytes are checked only after that consumer completes.
Two payload banks remain live. Setup uploads and final test readbacks still
synchronize. This changes the fixture, not the production BFS dispatcher.

Hardware validation is pending. Running Kaggle v37 uses the preceding source
`d226dc9` and does NOT contain these new cross-stream waits. Do not restart that
run or attribute its eventual result to this change.

## Hardware result

Kaggle v38 completed at `93cdf30f41e842e5ff04e7637267022dfb3b6132`
(package `d6a37a1`). Two distinct Tesla T4s, 15360 MiB each:
`GPU-725193e5-01aa-1dc6-ecd5-d9889c18a1cf` and
`GPU-c009cfd5-fe86-6d83-fc97-df036b5fafbf`.

The cross-stream fixture passes plain and all four sanitizer modes. All
sanitizer error summaries are zero, including race warnings/hazards. Exact
bytes match after both independent consumer events; empty epochs and source
switching also pass. This establishes the tested event dependencies and
lifetimes, not a measured overlap speedup or production BFS integration.

Raw reconciliation on 2026-09-06:

```text
python test_results/audit_sanitizer_v30.py test_results/distributed-sanitizer-v38/distributed-sanitizer test_results/distributed-sanitizer-v38-summary/distributed-sanitizer/summary.json 93cdf30f41e842e5ff04e7637267022dfb3b6132
```

Result: `RAW_GATE_RECONCILED`, 40 tool logs, 36 measured rank results,
36 warmup results and 36 archive verifiers. All 24 profile selections preserve
S4 layers `[1,3,5,6,5,3,1]`; existing runtime, macro and archive regressions
also pass. Only logs and metadata were downloaded locally. S13 was not rerun.
