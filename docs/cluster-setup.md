# First run on an unfamiliar cluster (dev35)

Supported target: one allocated Linux x86_64 NVIDIA node, one parent task, all GPUs visible to that task. This is not a multi-node launcher or a way to repair broken drivers. Request GPU resources through the cluster scheduler first. Do not use eight parent tasks for eight GPUs.

Install the small package and run preparation:

```bash
python -m pip install --upgrade "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261025-v35/multigpubfs-0.2.0.dev35-py3-none-any.whl" && python -m multigpubfs setup
```

Setup uses a separate versioned venv under ~/.local/share/multigpubfs/environments. Existing user environments are not replaced. It installs pinned CPU PyTorch for CayleyPy graph construction, pinned CayleyPy, and this release; native BFS still runs on GPUs with its own verified CUDA components. This environment is intended for native BFS, not PyTorch GPU training. Package repositories must be reachable for the initial installation. If Python lacks venv, the report identifies that prerequisite.

It discovers all visible logical GPUs, tests communication/autotuning/BFS, searches supported recovery policies, installs a verified node-specific recipe, and runs a generated CayleyPy persistent example in a separate process. READY requires native recovery and real API acceptance. Failed production runs are never retried automatically. The diagnostic budget is 30 minutes; installation has a separate 20 minute budget; production has no artificial time limit.

The output prints a source activation.sh command and the managed Python command. Source that activation in each subsequent job, then run your Python program normally with enable_native() and persistent_graph_workers(). The setup.zip contains preparation metadata/logs and diagnostic archives, including failure evidence. Paths and logs may identify your environment; inspect before sharing publicly.

For login-node preparation without GPU access:

```bash
python -m multigpubfs setup --prefetch-only --sm 100,120 --cache /shared/my-mgbfs-cache
```

Use --sm all to fetch every production architecture. This returns PREFETCHED_NOT_GPU_VERIFIED, never READY. Use the same --environment and --cache in the GPU job. Downloaded release wheels/components and installed dependencies are reused. A different node/driver/package must be verified again before inheriting a workaround. No full catalog is run separately on every GPU; recovery checks the selected set jointly.

--wheel /path/multigpubfs-0.2.0.dev35-py3-none-any.whl --sha256 HASH accepts a pre-downloaded matching wheel. This does not supply missing Python dependencies; prepare the managed environment on a networked node first. A stale .setup-lock is never silently removed: confirm its former process is gone before removing it.

No promise is made that every unknown cluster fault can be bypassed. A BLOCKED result preserves the tested failure and remaining operator action.
