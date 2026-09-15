# libcudf boundary characterization

This executable is a test fixture, **not an integrated BFS implementation**.
It checks Hash128 distinct/anti-join, stable source provenance, cached history,
accepted-next exclusion, empty input, and fixed-pool OOM. Host synchronization
in this fixture is deliberate and is not a pipeline/performance claim.

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
