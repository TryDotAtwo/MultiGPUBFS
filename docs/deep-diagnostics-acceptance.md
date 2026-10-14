# Comprehensive diagnostic acceptance

Final package: 0.2.0.dev14, release native-pip-20261014-v14.
Wheel SHA256: 5fa2adb9d048e2025b873f3b616cb1530c01ad4a5cb8877348e7da993181fa69.
Native checkpoint source ec67761ed67e2ced650b600c2d0a2cc232e840d1;
final Python package source a91ca695ea69f1b74296d06529ad4c294fbd11d8.

62 CPU diagnostic regression tests passed (Actions 38604597893).
Portable native compile and clean pip install: 38598052642.
Final package and clean pip install: 38604602994; upload: 38605056800.

Full installed v13 suite on two T4 GPUs: 171 stages, 150 PASS, 14 SKIPPED,
7 FAIL, elapsed 962.91 seconds. The failures were unsupported
SM75 GEMM launched without capability checking and specialized backends receiving
generic options. v14 corrected only those diagnostic scenarios. An additional
GPU acceptance run checked all changed cases, additional finite puzzle successor
probes, ordinary doctor afterwards, and total-budget termination. Its receipt
status is PASS; the budget-limited report correctly returns INCOMPLETE.
Unchanged successful cases were not rerun as an identical full suite.

Native inner hardware entry/return checkpoints and metadata-only ZIP checksums
were asserted in the full suite. Owned process groups were checked after tests.
No algorithmic BFS, tuner, worker, buffer planning or dedup changes were made.
Raw user diagnostic archives and graph state payloads were not published.
The source of the original Blackwell server hang remains unresolved until its
owner runs the final package. This is not performance or multi-node acceptance.
The default command is `multigpubfs doctor --deep` with a 30-minute execution
budget, explicit unsupported/timeout/skipped entries and one diagnostic archive.
