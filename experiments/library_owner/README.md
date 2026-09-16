# Library owner adapters and characterization

This directory contains cuDF and cuCollections owner adapters, their common
C ABI, and isolated characterization fixtures. The adapters are integrated
as explicit `CUDF_RELATIONAL` / `CUCO_INDEXED` reference selections in the
native distributed BFS. The production native owner remains available unchanged.

Fixtures check Hash128 distinct/anti-join or indexed membership, stable source
provenance, history, accepted-next exclusion, empty input, and fixed-pool OOM.
Host synchronization remains explicit; integration is not evidence of overlap
or competitive performance. See `docs/plans/library-first-bfs.md` for the
versioned evidence: v43 passed full BFS and four sanitizer tools on two T4s,
plus six actual torchrun scenarios and twelve archive verifications. Full
performance/peak-memory acceptance is still pending.

Inspected source pins:

- cuDF v26.04.00: `f9c3cf195768647ac39d98674a614ca414cd1baa`
- RMM v26.04.00: `48b36cc6714bef05d555372ae778b6702cb26858`

Use the matching installed cuDF/RMM 26.04 CMake packages on Linux, CUDA 12,
with `-DCMAKE_CUDA_ARCHITECTURES=75`. No FetchContent/source build or backend
fallback is performed. Record package versions and wheel hashes separately:
the inspected Git SHA is not proof of a binary wheel's source provenance.

Run `ctest --output-on-failure`, then the executable under each of Compute
Sanitizer memcheck/racecheck/initcheck/synccheck on T4. A successful fixture is
only a prerequisite for an adapter; it does not certify BFS or NCCL correctness.
