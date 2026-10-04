# Kaggle v92: harness import failure, no rank probe

- Notebook: `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, v92; worker ERROR.
- Pinned source: `d195bab1f395bd0e2fb11aa00dc9f65d91615e66`.
- Two physical T4s; P2P admitted in both directions.
- Experimental NCCL library built successfully; SHA256:
  `13fd0884c6eb768e038c41528f5ce92b2cc1c9e061bfad05518e88daf54b756c`.
- Summary SHA256:
  `ee91f4ec68961c8fb30978f4137a814a497b1e13419cf7221d3848c4b952a838`.
- `window_runs` is empty. `ModuleNotFoundError: process_scope` occurred
  before either rank process was started. No sanitizer result or BFS result.

Cause: the supervisor used the uploaded script's `__file__` to locate scripts.
Kaggle relocates that script to `/kaggle/src/script.py`, separate from the
pinned checkout. The supervisor must use its source-checkout `cwd` instead.

Regression: a fresh Python interpreter loads the actual notebook, relocates
its `__file__`, runs two actual bounded child processes from the source
checkout, and requires both registration markers. It failed with the same
missing-module exception before the fix and passed afterwards. Full scripts
suite: 56 tests, 3 skipped. This is harness evidence, not GPU acceptance.

All available small summaries/logs were downloaded into
`build/kaggle-v92-observation` before any subsequent submission.
