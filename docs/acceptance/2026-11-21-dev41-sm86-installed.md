# dev41 SM86 installed-wheel validation

These are results for a **test-only SM86 wheel**, not an acceptance claim for
all CUDA targets or a published release. Production package assembly and its
own installed-wheel check are separate.

The wheel, executable and SM86 CUDA component were built from source commit
`12916819cab0f9ddec62cc50a4c4c48e39b9b5d6`. Hardware was eight RTX 3060
devices on one host. The installed package was `0.2.0.dev41`; the native
executable SHA-256 was
`10cb2e53915b9e8936582889445eed263cd6c370ab877aae0ee628fd8f2da687`,
and the actual loaded CUDA library SHA-256 was
`ca4fba4419120aaa4a37123e56faedbb91c8e4b58e263481e9a15b517c43d908`.
The bundle was prepared from checksum-verified files, then tested offline.
Native device discovery reported eight devices.

On two of the eight GPUs, the installed package passed all five acceptance
stages: hash full-layer multisets including forced collisions and resource
limits; bitmap atomic/warp exact-layer fixtures; bitmap batch exact-layer
fixtures; both memory presets with persistent reuse and resource stops; and
default CayleyPy activation with the six backend/strategy combinations and a
small autotuned job. The latter checked that persistent worker processes loaded
the tested library. See the complete source-pinned commands and records in
[installed acceptance evidence](2026-11-21-dev41-sm86-installed.tgz)
(SHA-256 `fd9ba51c8975eda1f78ae38b83a4d00af4a1fe42600591e2684c75cf1a5e0aa0`).

On all eight GPUs, another run checked healthy persistent reuse across different
graphs, injected finish-write, finish-publish, dropped-communicator, after-seed,
and count-to-payload failures, and checked that each failed group was closed
and that a new group could compute. The fault test compares layer counts and
terminal states for its small examples; the full per-layer multiset oracle is
the separate two-GPU gate. [Eight-GPU fault evidence](2026-11-21-dev41-sm86-epoch8.tgz)
has SHA-256 `dc30aa025ea24493641363063c52d9311ff1e7c9a3bdea5a30115212c2eb518d`.

The forced collision fixture, 2^32 boundary limits, B200, and earlier large
benchmark scope remain as described in the
[integrated follow-up](2026-11-21-integrated-followup.md). These tests do not
prove B200 behavior or physical allocation of 2^32 rows on one GPU.
