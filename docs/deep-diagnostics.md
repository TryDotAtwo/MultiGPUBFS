# Comprehensive opt-in diagnostics

See [the one-run CUDA build contrast](diagnostic-build-contrasts.md) for the
static/shared and native-SM120 comparison carried by the extended package.

Run `multigpubfs doctor --deep` in a fresh process inside the assigned GPU job.
The execution budget defaults to 1800 seconds. `--total-timeout` changes it;
`--stage-timeout` bounds an individual synthetic stage. Metadata finalization
and diagnostic process cleanup require a small additional bounded allowance.

The ordinary doctor and production BFS use their existing behavior. Deep mode
changes only its own child process environments. It never changes allocation
visibility, replaces driver files, resets GPUs, restarts real BFS, requests sudo,
or runs large/full graphs. EAGER/LAZY contrasts are ctypes-only synthetic tests;
the native CLI's EAGER admission remains unchanged.

Coverage: loader order before/after context, module/data EAGER/LAZY/inherited,
file-polling control thread, direct memory and attributes, native properties
wrapper with inner entry/return checkpoints, successor/readback, pinned memory,
streams/events, empty CUDA Graph, persistent inventory, NCCL transport contrasts,
small CPU-oracle permutation/repeated/matrix/wide/LRX graphs, hash/sorted/GEMM,
parent/full/radix/shards, persistent jobs and tuning, single/multiple visible GPUs.

Timeouts capture only owned diagnostic process groups: /proc thread state,
wchan/syscall/stack when permitted, loader maps and resource limits. Optional
strace is used only when already installed. No administrator installation is
required. Missing access and untested/budget-skipped cases are explicit.

Outputs: report.json, deep-summary.txt, deep-analysis.json, logs, checksummed
diagnostics.zip. ZIP excludes raw case inputs, full environment and BFS state
payloads. It can contain server/library paths and NCCL connection metadata;
return it privately rather than publishing raw archives. No single finite test
can cover every GPU API parameter, graph, driver bug or security policy.
Mode correlations are evidence, not proof of a unique root cause. This suite
is correctness/startup diagnostics, not a throughput benchmark or multi-node
acceptance test.
