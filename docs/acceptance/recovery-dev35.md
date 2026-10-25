# dev35 first-run cluster setup

`python -m multigpubfs setup` prepares an owned versioned venv, installs CayleyPy and CPU PyTorch for graph construction, selects all visible GPUs for native BFS, performs recovery acceptance and verifies a generated persistent API example. It writes an activation command and one setup.zip. No scheduler submission, driver changes or production retries. See docs/cluster-setup.md.

90 focused tests passed. A fresh environment setup completed on 2x3060 in 254.58 seconds, including recovery and generated example. Final published-wheel acceptance is recorded separately after publication. Native components unchanged. Unrelated repository-wide Rust formatting failures remain; this is not a claim that every CI check passed.
