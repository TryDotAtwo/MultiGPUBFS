# Single-command GitHub pip acceptance — 2026-10-13

The published `multigpubfs-0.2.0.dev2-py3-none-linux_x86_64.whl` installs directly from GitHub Releases. CUDA runtime and NCCL are private wheel contents; no bootstrap, HF dependency, Kaggle Inputs, manual chmod, build toolchain or library-path setup is required.

## Artifact

- [Release](https://github.com/TryDotAtwo/MultiGPUBFS/releases/tag/native-pip-20261013-v2)
- SHA256: `ddcbca720876f87fc048c107fb8227d42bc3f6e3ebb9894e87d34db0e405ffac`
- Package/native source: `7046577446e546b86317f6e2e24cf319552b29cd`.
- [Remote Ubuntu22.04 rebuild and packaging](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38335771377).
- [Fresh Ubuntu22.04 direct public URL install with CayleyPy extra](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38338201218): successful import, explicit activation, graph construction, private library resolution and `pip check`. This Actions runner has no GPU.

## Actual GPU acceptance

[Private Kaggle gate, version4](https://www.kaggle.com/code/trydotatwo/multigpubfs-single-pip-check-20261013) completed successfully on physical 2×T4. The private notebook is not publicly readable; the machine-readable receipts are included alongside this report.

| Check | Result |
|---|---|
| One direct pip installation with `[cayleypy]` | Passed |
| Existing Torch/NVIDIA Python distributions unchanged | Passed |
| LRX n2..5, all14 pairs, 2GPU | COMPLETE; every layer matched independent CPU oracle |
| RapaportM2 n2..5, all14 pairs, 2GPU | COMPLETE; every layer matched independent CPU oracle |
| Fresh isolated Linux venv, direct wheel installation | Passed; no source checkout, Torch or CayleyPy required for core API |
| LRX5 core API on1/2GPU | COMPLETE120states; layer vectors and terminal states matched oracle |
| Matrix SL2(3) core API on1/2GPU | COMPLETE24states; layer vectors and terminal states matched oracle |
| Deliberately insufficient capacity on2GPU | Valid INCOMPLETE resource stop; completed prefix and snapshot matched oracle |
| Installed CLI on2GPU | COMPLETE; layer vector matched oracle |
| Fresh venv `pip check` | Passed |

Kaggle Python3.13 retained Torch2.11.0+cu128, nvidia-nccl-cu12 2.28.9 and nvidia-cuda-runtime-cu12 12.8.90. Native subprocesses used bundled CUDA12.9.79/NCCL2.30.7. Native activation remains explicit: `multigpubfs.enable_native()`.

## Limits and earlier test failures

The wheel targets Linux x86_64, Python3.10+, Ubuntu22.04 ABI and SM75/SM86. This acceptance does not establish Blackwell/B200/B300 or multi-host support. SSH is a transport: installation and execution were verified in a fresh remote Linux shell; no additional paid SSH host was rented for this rebuilt artifact.

Gatev1 passed the28 GPU pairs but failed while Kaggle's `ensurepip` created an extra test venv. Gatev2 then failed after the test artificially removed the host's standard `LD_LIBRARY_PATH`, hiding driver library discovery. Gatev4 uses the normal host environment and ordinary pip, and passed the full test. The package was unchanged between these gates. No user needs to set a library path.

GPU result receipts contain metadata and layer counts, not state payloads. Public basic/sweep notebook publication and execution are recorded separately; the private gate already validates their calculation path.
