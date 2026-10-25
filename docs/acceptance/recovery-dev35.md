# dev35 first-run cluster setup

`python -m multigpubfs setup` prepares an owned versioned venv, installs CayleyPy and CPU PyTorch for graph construction, selects all visible GPUs for native BFS, performs recovery acceptance and verifies a generated persistent API example. It writes an activation command and one setup.zip. No scheduler submission, driver changes or production retries. See docs/cluster-setup.md.

90 focused tests passed. A fresh environment setup completed on 2x3060 in 254.58 seconds, including recovery and generated example. Final published-wheel acceptance is recorded separately after publication. Native components unchanged. Unrelated repository-wide Rust formatting failures remain; this is not a claim that every CI check passed.


Installed release acceptance: exact GitHub wheel SHA256 73d5d21bafcd9ec05a982ac9d06705f67c7faacc92732bbd98cd4b8e774a5ce7. Fresh managed environment reached READY in 258.38 seconds; all visible GPUs [0,1] selected, measured native tuning, CPU-oracle API checks and generated persistent example passed. Subsequent prefetch with CUDA_VISIBLE_DEVICES empty and MGBFS_NATIVE_OFFLINE=1 reused the environment and verified existing SM86 cache. No GPU acceptance claimed for that prefetch. See recovery-dev35-installed.json.
