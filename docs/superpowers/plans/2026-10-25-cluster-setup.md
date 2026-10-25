# Cluster first-run setup

Goal: one command inside an allocated single-host Linux NVIDIA GPU job prepares an isolated Python environment, verifies recovery on all visible GPUs, registers the recipe, and produces a ready API example and metadata bundle.

No multi-node implementation, scheduler submission, GPU resets, production restart or silent GPU subset. Login-node prefetch is a separate explicit mode; missing GPU is not success. Existing user environments are never replaced. Runtime installation and diagnosis have separate bounded budgets. Full graph runs remain unlimited.

- Add failing tests for allocation, environment ownership, locking, READY gates and CLI.
- Implement setup command using an owned venv and pinned release/dependencies, original environment evidence and local scratch.
- Exercise source package and installed GitHub wheel from a fresh managed environment on 2x3060.
- Publish source/package/receipt, preserve metadata and delete exact rental.

Use executing-plans in this session. No subagents.
