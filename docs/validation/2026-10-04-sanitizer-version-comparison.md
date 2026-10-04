# Full runtime sanitizer version comparison

The retained real-BFS failures use compilerCUDA12.9 but host Compute
Sanitizer2025.1.0.0. This is a discriminating version comparison, not proof
of incompatibility or a runtime fix.

`typed_sanitizer_version_gate` runs the same production typed runtime
DENSE/HASH_FIRST under host and CUDA12.9 tool packages, all four tools,
with independent ranks, full-state/archive oracle and unfiltered error
reporting. No alternative runtime, altered device resources or suppression.

Official component inspected:
[NVIDIA redistrib12.9.1](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json),
`cuda_sanitizer_api`12.9.79 linux-x86_64, archive SHA256
`e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf`.
Remote archive bytes were independently checked against this SHA. Archive
includes the actual instrumenter at compute-sanitizer/compute-sanitizer
and a launcher under bin. It is not merely sanitizer API headers.

Host executable is resolved before SDK/bin changes PATH; each replay
explicitly selects the proper executable directory. The wrapper in SDK/bin
must not silently turn both sides into the pinned tool. Actual --version
outputs are retained; the pinned binary's SHA is recorded. A failed old
tool remains a failed old-tool result even if the new tool passes.

Local preparation: selection tests RED then GREEN; scripts63 OK/3 skipped,
Python238 OK/8 skipped. Hardware results remain pending.
