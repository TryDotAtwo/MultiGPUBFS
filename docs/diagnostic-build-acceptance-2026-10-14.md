# CUDA runtime and SM120 diagnostic acceptance

Package 0.2.0.dev15: [GitHub Release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261014-v15).
Wheel SHA256: 2b9863cf25533e547c76f330057ee7d01f89f7e9f5bf0be2a7d02fc7a7505604.
Python source: 1cabb5582bbde847ebe24aea40925aa872f02acc.
Normal native source: ec67761ed67e2ced650b600c2d0a2cc232e840d1.

The four same-source runtime/architecture configurations compiled successfully;
ELF runtime linkage and cubin/PTX inventory were verified. Builds:
[38608137281](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38608137281).
Clean pip install, all variant loader paths, 88 CPU regression tests and exact
production executable/default-library byte identity passed during
[assembly 38614283539](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38614283539).

The installed package completed the full deep suite on two physical T4 GPUs:
197 stages, counts {"PASS": 191, "SKIPPED": 6}, 1014.28 seconds.
All 24 four-build API, persistent and CPU-oracle S3 comparisons passed.
Inner CUDA call markers, real native-worker library mappings, archive hashes,
and owned-group cleanup were asserted. The healthy baseline stayed selected
for subsequent diagnostic coverage. Four additional actual multi-GPU
generic/matrix/specialized oracle cases passed on shared-sm120.
Ordinary doctor afterwards passed. A one-second execution budget produced
INCOMPLETE, exit code 1 and a finalized metadata archive.

This validates finite single-host synthetic diagnostics on SM75. Native SM120
cubin compilation is verified, but there was no physical SM120 execution or
acceptance on the original server. The original hang remains a target-server
question. No throughput or multi-node claim is made. GPU unsupported and
uncompiled modes remain explicit skipped entries. No raw collaborator archive
or graph-state payload was published.

The [one-run guide](diagnostic-build-contrasts.md) explains observed call
boundaries, controlled pairs, diagnostic-only continuation and uncertainty.
The release includes a small sanitized acceptance receipt; the server owner
returns their diagnostics.zip privately.
