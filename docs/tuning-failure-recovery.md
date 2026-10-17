# Autotuning failures and measured selection

Use `NativeOptions(autotune=True)` or the default. A verified optional CUDA999 candidate failure excludes that candidate, preserves its primary error and durable evidence, closes the failed group, refreshes admission and measures remaining candidates. It never substitutes an unmeasured session-wide baseline. After a captured failure, fewer than two sufficiently measured surviving candidates raises `AUTOTUNE_NO_MEASURED_PROFILE_AFTER_FAILURE`.

Short successful probes are validated and recorded in `excluded_prefix_candidates`. They do not shrink the comparison window of sufficient candidates. Matching completed prefixes, with warmup excluded, determine the measured winner. `selection_measured` and `size_profiles_measured` describe different coverage; the package does not claim an unmeasured frontier tier was benchmarked.

Only exact recognized capacity refusals of additional generator plans can be rejected without cancelling a previously admitted measured choice. They remain visible in `size_generator_rejections`. Python verifies submitted command and response identity; Rust permits resident reuse only for read-only plan commands without a recorded native fault. The original error response remains intact. A failed BFS, real CUDA/NCCL fault, unknown error or dead process is never reused or automatically restarted by this rule.

[Physical GPU receipt and scope](measured-tuning-acceptance-2026-10-17.md). The original eight-GPU server CUDA999 root cause is unknown; a missing graph definition prevents exact reproduction.
