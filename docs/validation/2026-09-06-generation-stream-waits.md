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
