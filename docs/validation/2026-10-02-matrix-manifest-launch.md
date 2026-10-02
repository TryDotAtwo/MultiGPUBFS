# Matrix manifest input connected to the existing native launcher

The CLI now accepts:

```
mgbfs bench --manifest matrix.json batch bootstrap archive-prefix output-dir
```

This is a connected benchmark input contract, not production run/preflight/
calibrate or full RunConfigV1 support. Existing environment selections and
explicit archive/search-only contracts remain. No configuration field is
silently advertised as honored by a production dispatcher.

The file is decoded and MatrixGroup::validate is applied inside the existing
rank preparation closure, after bootstrap and before communicator/archive
admission. File/parse/inverse/range errors participate in the existing group
configuration vote rather than being returned by the CLI before rendezvous.
Graph identity is SHA256 of validated canonical serialization, not file path
or whitespace. The same graph enters owner, LSA, retirement and archive;
no second scheduler or CPU BFS fallback is added. Single-rank macro input
uses the same loader, while multi-rank macro remains explicitly unsupported.
Manifest input uses canonical matrix_u8, not an inferred compact codec.

Evidence:

- RED: the new real CLI invocation initially returned CLI_USAGE instead of
  entering the existing launcher/build check. GREEN: CPU and CUDA CLI suite
  accepts the manifest option without CPU fallback; 7 bench tests pass.
- Real-file CPU tests preserve nonidentity start/inverse moves, canonical
  identity across whitespace/paths, changed-start identity, and reject malformed
  or mathematically invalid manifests without substituting a reference graph.
- Full ordinary cargo test --locked passes again after final admission/help
  edits, session75952 exit0; no serial override or narrowed suite.
- Actual one-GPU sm86 execution, pinned existing diagnostic NCCL/CUDA stack:
  manifest_nonidentity_start_runs_both_profiles_with_full_state_archive passes
  DENSE and HASH_FIRST. Mod3 2x2 start [1,1,0,1], depth0 exactly that state;
  depth1 exactly [1,0,0,1] and [1,2,0,1]. Archive structural checks and each
  state's recomputed Hash128 match; group COMPLETE exists.
- Existing unrelated unused_mut warning in reference_bench.rs remains.

The new peer_malformed_manifest_cancels_valid_rank_before_archive_admission
fixture runs two independent CLI processes. Rank0 receives the valid manifest;
rank1 receives truncated JSON. Both terminate naturally in0.39s under the15s
test bound; rank0 reports REMOTE_CONFIGURATION_FATAL and rank1 reports
MATRIX_MANIFEST_PARSE. Neither rank creates an archive, per-rank result or
group COMPLETE. This uses one local physical GPU: rank1 fails preparation
before device admission. It is pre-communicator control evidence, not a
physical two-GPU search or post-communicator cancellation gate.

Full Linux CUDA/library-owner CLI suite passes:15 tests,2 explicit ignored
hardware fixtures; the new admission fixture was executed separately above.
This does not prove physical2T4 acceptance, sanitizer-complete status,
multi-GPU macro or performance.
No paid resource was created. Fresh 2T4 quote only returned machine28909,
the previously failed peer-read/write host; it was not rented again.
