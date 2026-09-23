# Device-count owner pack: physical 2xT4 primitive gate

- RED source `6ae0884143b1ef39a99ba672c73c5833d1c290b3`, private Kaggle `trydotatwo/mgbfs-device-count-owner-pack-gate` v1: compilation failed because `mgbfs_exchange_pack_device_n` was undefined. This was the expected missing-contract failure, not a CUDA runtime failure.
- GREEN source `a272a17d47320136d8288f4dbfada6732f03c851`, the same private Kaggle kernel v2: status `COMPLETE`; each of two distinct Tesla T4 GPUs passed plain, memcheck, racecheck, initcheck and synccheck.
- Test covers device-count zero, partial (3/6), full (6/6), over-capacity (7/6), untouched output tail, owner counts and packed source-row identity. The existing host-count and invalid-reference fixtures also ran in the same executable.
- Downloaded logs and summary remain in ignored `test_results/device-count-red-v1/` and `test_results/device-count-green-v2/`.

This validates a single-GPU CUDA primitive on each T4, not multi-rank NCCL, full owner/transport/retirement, or speed. The full small-graph BFS regression is a separate gate.
