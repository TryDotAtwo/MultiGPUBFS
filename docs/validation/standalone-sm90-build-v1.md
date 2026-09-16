# Standalone SM90 build: Kaggle v1

Notebook: `trydotatwo/mgbfs-standalone-sm90-build`, version 1,
script version 350391789. Kaggle status COMPLETE, build status BUILT.
Source: `cbc092bd980f7600e14e90719db84f82025d4aa3` (clean checkout).
Wrapper: `d8e73e1`. Logs saved locally under
`test_results/standalone-sm90-v1/standalone-build/`.

The new `scripts/remote_build.py` completed its full path, not a mocked
orchestration test. The kernel printed BUILT at 269.2 s, including checkout,
downloads, dependency installation and compilation. This is a single build
observation, not a Vast provisioning-time bound or BFS measurement.

- CUDA architecture 90; nvcc 12.9.86, cudart 12.9.79, CCCL 12.9.27,
  nvrtc 12.9.86; every redistributed archive SHA256 verified.
- cuCollections `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4`.
- CUTLASS `ffa119a1255d78998536107466cc7097ecefa393`.
- Rust 1.75.0; locked Python dependency manifest SHA256
  `048466576b06ec3f3c9ee0e33ffde22dc6cd6d3733612819ff4ec1f40bf1d745`.
- Native CUDA shared library linked; library-owner/cuCollections targets built.
- Release `library_multi_gpu` test executable built; Cargo build-finished true.
- Release `mgbfs` CLI built (23.89 s for the final Cargo stage).

No SM90 binary was executed on the T4 host. This proves compile/link with the
Kaggle host's NCCL development environment, not Vast compatibility, H200 runtime
correctness, eight-rank NCCL, sanitizer results, or LRX13 completion. Those gates
remain required. Only small logs/manifests were downloaded; no GPU SDK, binary
tree or state dataset was copied to the user's computer. No Vast instance was
rented for this validation.
