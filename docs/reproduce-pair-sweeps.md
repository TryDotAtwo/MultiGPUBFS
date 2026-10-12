# Reproduce the GPU pair sweeps

Validation status (2026-10-12): the pinned native package passed independent CPU-layer comparisons on physical 2×T4 for LRX and M2. This new multi-pair notebook is undergoing private Kaggle validation; it has not yet been accepted end to end. Its Kaggle page remains private until the gate finishes.

[Open the Kaggle notebook](https://www.kaggle.com/code/trydotatwo/multigpubfs-pair-sweep), choose **Copy & Edit**, enable **GPU (2×T4)** and **Internet**, and Run All. No compilation or manual package installation is needed: the attached public `trydotatwo/multigpubfs` utility installs the immutable checksummed native release.

The [notebook source](../examples/cayleypy_pair_sweep.ipynb) is also in this repository. Its default grid is intentionally small (`n=2..5`, `r=1..n`). Change `FAMILY` to `lrx` or `rapaport_m2`, the range, per-pair seconds and total work seconds. All visible GPUs are selected automatically. There is no provisioning or billing control in the notebook; on rented machines the platform must enforce a separate spending/deletion limit.

## What we compute

For each pair the canonical start is `[0,...,n-r] + [n-r]*(r-1)`: one symbol appears `r` times. Generator definitions come from the actual CayleyPy family, not a replacement generator implementation. `device="cpu"` only constructs the graph definition; the native traversal executes on GPUs. `multigpubfs.enable_native()` explicitly installs the native `graph.bfs()` adapter; import alone does not patch CayleyPy.

A single `persistent_graph_workers()` scope owns the entire sweep. Healthy resident ranks and compatible allocations/profile decisions can be reused across pairs; hardware/resource/geometry checks still apply. Tiny complete-orbit bounds skip expensive pilots, while larger graphs use bounded measured profiles. These are bounded choices, not proof of a globally fastest profile. Changed geometries can require new admission or tuning; zero-cost transitions are not promised.

## Files saved

| File | Contents |
|---|---|
| `results.csv` | n, r, COMPLETE/INCOMPLETE/PENDING, canonical-start eccentricity when complete, completed depth, visited count, API/BFS/tuning seconds |
| `layers.csv` | Every committed layer's depth, state count, cumulative count and expansion time (no expansion for depth zero) |
| `nN-rR/report.json` | Full native status/reason, layer vectors, chosen profile, memory plan, devices and runtime hashes |
| `nN-rR/states.json` and native receipts | Compact terminal state data and exact runtime evidence; not every state of every layer |
| `configuration.json`, `checksums.json` | Reproducible grid settings and SHA256 for every saved file |

Current compact retention samples the last committed layer up to 1000 states and retains a previous layer when it is strictly smaller than 1000. It does not archive all small intermediate layers, all graph states, edges or paths. COMPLETE means the frontier exhausted; INCOMPLETE preserves a committed prefix and reason. Its completed depth must not be advertised as an exact eccentricity. Even for COMPLETE, eccentricity of the selected start should only be called graph diameter when the graph's symmetry justifies it (repeated-symbol quotients need that distinction).

Same-directory reruns reuse valid completed/partial report files and reconstruct CSVs. They do not resume an unfinished BFS frontier. Use a new directory for changed settings. Unknown errors stop the notebook rather than silently turning into successful rows. Total work budgeting leaves pending pairs explicit; initialization and optional upload also take time.

## Optional Hugging Face publication

Create/choose a dataset you own, set `HF_REPO`, and add a **Kaggle Secret named HF_TOKEN** with dataset write permission. The final cell uploads the output folder and reads back every uploaded payload at the returned immutable commit to verify SHA256. `HF_PRIVATE=True` creates a private new dataset; it does not change visibility of an existing dataset. Set it false only when you intend a new public dataset. Never paste a token into public code. State data stays on the GPU host/HF; no download to your desktop is necessary.

## Hardware and provenance

The pinned utility release is source `419fa34f`, CUDA12.9, SM75/SM86. It was verified on physical 2×T4 and 2×RTX3060. It is **not a B300 binary**; Blackwell requires a separately compatible native artifact and actual hardware acceptance. Large production timing and memory coverage must be established on the target hardware; a small successful notebook does not demonstrate that a large graph fits.

Our earlier large rented-host campaigns used the repository's `scripts/run_auto_tail.py` orchestration, deadline/lease guards, saved ledgers and HF readback. Those scripts require a matching source/native runtime environment and campaign configuration. This notebook publishes a simpler current CayleyPy API workflow with the same canonical pair definition and compact-output purpose; it is not a promise of identical historical production scheduling or timing.
