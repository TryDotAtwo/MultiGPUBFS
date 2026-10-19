# One-shot UT(4,47) eight-GPU diagnostic (dev29)

Run once inside one allocated Linux x86-64 job with exactly eight visible GPUs. Do not start eight copies with srun/mpirun: this controller owns all eight ranks. Allow at least 16 CPU threads to match the reported server setup.

This is an isolated diagnostic, not a new computational release or a claim that CUDA999 is fixed. It installs the pinned dev29 wheel using the current Python and --no-deps, preserves the shared native cache, and does not replace Torch dependencies.

## Matrix

- Initial driver/context/memory queries for all eight GPUs, native identity, dependencies and topology.
- Eight-rank NCCL exchange and complete UT(4,3), checked against an independent CPU oracle (729 states).
- UT(4,47), original capacity/batch, fresh 256-shard processes.
- One persistent pool with transitions 1 → 128 → 256 → 1, checked against prefix [1,6,26,110,414].
- Real default autotuning with an empty profile cache, then ordinary UT(4,47) without a depth limit or fixed capacity.
- Health check after each stage; unsafe cleanup or unhealthy GPUs suppress further work.

Run the isolated comparisons with the inherited baseline, with only NCCL_NET_GDR_LEVEL=LOC added, and with only NCCL_IB_DISABLE=1 added. All tests use eight GPUs together, not individual-card catalog runs. Low-risk controls precede known failing geometries.

## Bounds and evidence

The work/diagnostic budget is 30 minutes including installation, with a cleanup/archive reserve. Production trials have 180-second BFS budgets. Cleanup and ZIP finalization can add wall time. Incomplete graphs are not complete; budget/health skips are explicit.

The embedded SHA-verified x86-64 NCCL tracer needs no compiler; its required GLIBC symbols are at most 2.17. Its C source is included and a compiler fallback exists if loading fails. It records API entry/return, ranks, PIDs/TIDs, counts, datatypes, peers, communicators, streams and asynchronous-status changes. Before each outer group ends, it also retains per-peer send/receive totals, message counts and an ordered count/datatype signature, even if verbose traces rotate. In these Contract records, count is the total element count and datatype is the number of messages. It forwards original arguments unchanged. Diagnostic trace timings are not benchmarks. Production autotuning does not use the interposer.

Failures retain native layer/call phase files, rank and NCCL logs, rotating API traces, bounded /proc thread/wait/stack/maps/FD snapshots, and available kernel/driver evidence. Denied access is recorded as a gap. No device reset, global server change, or automatic restart of a failed calculation is performed.

One ZIP contains metadata/logs/checksums, not BFS state arrays, job-command files, or a full environment dump. It is never uploaded automatically. Hostnames and filesystem paths may appear; review before public sharing. Oversized logs are explicitly truncated to their tails and listed in archive-omissions.json.

## Results

- REPORT.txt: stages, observations, last reached calls, and an accepted launch configuration if found.
- report.json and each stage.json: detailed results, cleanup and trace-observation status.
- completed.json and run-*/report.json: actual layers, plans, timings and autotuning evidence.
- manifest.json: archived file SHA-256 values.

Recommendation requires the full small oracle, collective, fresh-256 probe, persistent transitions and measured production autotuning to pass in the same mode. Failed tuner candidates or an unmeasured fallback do not qualify. FULL_GRAPH_COMPLETE additionally requires COMPLETE and 47**6 = 10,779,215,329 states. AUTOTUNE_AND_BOUNDED_BFS_VERIFIED means only a bounded computation, not a completed graph or validation of other families.

For an accepted configuration, launch-verified.sh runs the graph in the same allocation with the verified transport setting and a finite 900-second work limit. If no configuration qualifies, no working command is claimed. Different nodes require validation. The diagnostic cannot guarantee identification of a root cause when the driver is unhealthy or necessary evidence is inaccessible.

## Validation

Run python scripts/test_ut47_deep_diagnostic.py on Linux for supervisor controls. These are separate from physical eight-GPU acceptance, whose receipt identifies exact source, hardware and limitations. The installed library remains dev29; all new diagnostic behavior is in the standalone script.

NCCL asynchronous state 7 is ncclInProgress, not a standalone failure. The trace records state changes so repeated polling cannot overwrite the last submitted send/receive/group operations. See the [NCCL result types](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/api/types.html) and [transport settings](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/user-guide/docs/env.html).
