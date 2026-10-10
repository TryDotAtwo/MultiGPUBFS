# Exact GEMM generator selection

Both shared-host and external-rank startup tuning can compare CUDA against a preallocated CUTLASS uint8/int32 Tensor Core matrix generator. The matched BFS prefix includes packing, nonlinear hashing, routing, exchange, history and exact deduplication. A five percent switching margin retains CUDA for marginal results. This is a bounded sample, not a global optimum guarantee. Generator choice is recorded in the profile and serialized native memory plan, propagated into production and included in cache/configuration identity.

The initial exact domain is matrix rows 8..64 divisible by 8, columns 1..64, at most 64 generators, starting values and coefficients 0..255, and each generator modulus 1..256. Maximum accumulation is 64*255*255=4,161,600, safely within signed int32. Modular outputs preserve the byte domain. Unsupported graphs retain CUDA. Automatic GEMM admission also requires SM80-or-newer hardware and a native capability receipt; old binaries cannot silently masquerade as a GEMM measurement. Permutations are left on CUDA after the preceding stage panel showed their dense GEMM encoding loses.

Application device allocations happen before BFS loops. Admission includes a conservative GEMM workspace reserve, padding and bounded launch geometry. Products are consumed through the original immutable child origin; the existing nonlinear hash and exact payload comparisons are unchanged. Full transport reuses GEMM products for source payload materialization; parent transport continues its exact GPU origin regeneration at destination. This cost is included in real BFS pilot timing. The separate linear hash-projection experiment is not installed as a replacement hash.

Validation on two RTX3060: 30 native boundary cases checked hashes and every regenerated payload element against ordinary CUDA/CPU arithmetic, including odd batch sizes, rectangular states and moduli 2/251/256. Eight complete 40,320-state BFS runs matched CPU layers and terminal states for one/two GPUs, HASH/SORTED histories and full/parent transport. Shared-host and independent external-rank collective choosers executed the GEMM candidate and completed the same graph. 504 Python tests passed with one conditional skip; Rust library tests passed. Native source compiled for SM86. Blackwell, physically separate nodes and 8/128-rank throughput are not hardware-validated by these tests.

Default use is unchanged: run_graph(graph, output). MGBFS_GENERIC_GENERATOR=gemm forces the bounded exact path; cuda forces its native implementation during a manually configured run. Startup auto may compare the alternative for an eligible graph. The shipped SM86 wheel below contains this feature; the previous portable wheel is not retroactively changed.

## Accepted SM86 wheel

```bash
python -m pip install --force-reinstall "https://huggingface.co/datasets/TryDotAtwo/multigpubfs-native-releases/resolve/21597679aac67c9f98be3ef242ab37f177bc5752/linux-x86_64-sm86-gemm-cuda13.2/8337f111caef0ed5eedda380d34ec92096299a27/multigpubfs-0.2.0.dev0-py3-none-linux_x86_64.whl" "nvidia-cuda-runtime==13.2.86" "nvidia-nccl-cu12==2.30.7"
```

Wheel SHA256: `cc90131cbebfc69774b8e4c39b8acaccb8d9ed636dac532ab50ac0acccf1bfdb`. This wheel targets SM86; use the separately documented portable release or a current source build for other architectures.
