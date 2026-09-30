# Boundary v40: real-rank host-debugger replay

Runtime: `c7488ca6d901afc1545edba4e25ea28f915f8206`, unchanged.
Runner: `173d99c`. Real two T4s with bidirectional P2P; NCCL_LSA/DENSE/CUCO_RANK.

Host GDB followed actual independent `mgbfs` processes 2839 and 2840, rather
than the CUDA debugger agent encountered in v39. Both application inferiors
eventually exited `01`. Origin rank 0 logged the injected owner error, abort
begin and abort end code 0, then completed owner and stream drains. Rank 1
reported `CUDA_STATUS_7` and completed its drains. This label comes from the
shared wrapper status formatter; it does not prove a CUDA hardware fault.

The original unresolved abort did not reproduce in this instrumented run.
There is no runtime fix and no untraced bounded-termination acceptance.
The ten-second sample preceded application discovery. The twenty-second sample
failed to obtain registers from an application that had exited by the stack
request. Parent debugger supervision timed out at about 27 seconds; the
reported host stack acceptance is NOT_PROVEN. Scheduled sample records refer
to mutable thread-group dictionaries, so their eventual exit-code fields are
not independent time-stamped snapshots of when the sample started.

Both actual mapped NCCL libraries have:

- SHA-256 `19d9851b65feef08fb78280ffc0d3e9067d23bc1c637e4c93e3a6dd6e20b06e6`;
- ELF build-id `51943fef80521fbce0dafcd0e685cabf68fbd538`;
- runtime version `2.29.7+cuda12.9` and reported source suffix `stable b81d6a5a3`.

The full reported build commit resolves through the NVIDIA GitHub API to
`b81d6a5a3d2fa95ad11f6453c51cd6a6ba19f9b8`. The public `v2.29.7-1` tag currently
points to `b91894bd5b190c874d98a017f93f5daa515b65d0`; the compare API reports
diverged histories. Source analysis for this wheel must use the reported build
commit, not assume equal source trees from equal version labels.

Follow-up authoritative source-byte verification resolves the apparent `init.cc`
discrepancy: actual build commit → tree
`d61e2c86cfc507969000a963782b2e2231ff1475` → `src` tree
`2cbc91316842b12bc6000778e74eb910a4ad57c6` → `init.cc` Git blob
`b1d092d5e0f5068df67905ccc4ca9d28b6614a32`. Git API base64 bytes total 132,628;
SHA-1 of `blob <byte-length>\0<bytes>` matches that blob ID. The release-tag
commit's tree also resolves this file to the **same blob**, despite different
overall histories. Its abort entry is line 2697. The browser's 2877-line raw
view is inconsistent with these verified bytes and must not supply line anchors.
This proves equality of this file only, not of the whole NCCL source trees.

Raw evidence: `test_results/kaggle_nonblocking_boundary_v40/lsa-bfs-gate/`,
including both `/proc` maps, diagnostic metadata and complete debugger output.
NCCL stdout is buffered: interleaving with stderr markers does not establish
cross-stream chronological order. A cleanup-order/lifetime race remains a
hypothesis, not an established cause. No early free, internal abort-field
override, healthy host barrier or hidden transport fallback has been added.
