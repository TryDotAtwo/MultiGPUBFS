# NCCL architecture admission candidate (not accepted)

Baseline RED: current production single-sm86 DENSE/CUB LSA memcheck completed with 12 CUDA API errors in ncclInitKernelsForDevice; the full canonical S4 archive still verified. No sanitizer filtering or suppression is accepted.

An isolated dependency candidate is recorded in patches/nccl-2.29.7-minimum-arch.patch (SHA-256 1af3a5df99c2b3b9a4f66ca4c33cf66e513e208f73480e58b4183c8f0e05b26b). Upstream base: NVIDIA/nccl b91894bd5b190c874d98a017f93f5daa515b65d0, tag v2.29.7-1.

The existing standard/symmetric generators emit a minimum-architecture array derived from their own required_cuda metadata. Initialization skips cudaFuncGetAttributes only below that minimum. It does not null kernel pointers, change kernel selection, remove collective families or suppress CUDA API errors. Family-specific restrictions still use the existing probe when above the minimum; the lower bound is not claimed as a complete capability predicate.

The patch is not wired into any production build or Kaggle wrapper. Its reverse apply check against the modified source passes. Matching clean and patched source builds are running sequentially in the isolated existing Linux build volume: CUDA 12.8.93, NVCC_GENCODE=-gencode=arch=compute_86,code=sm_86, NVTX=0, make -j2 src.build. No ONLY_FUNCS restriction is used. Clean source is cloned from the pinned repository HEAD; candidate source includes exactly the five-file patch. No paid rental or GPU notebook is involved.

Next acceptance: run the same actual CLI/oracle and unfiltered memcheck with both resulting libraries and distinguish an architecture-build effect from the patch effect. Then test the other tools/profiles. A local success would not close two-T4 registration, inter-rank faults, timeline or A/B requirements. The stock library and its failed raw log remain preserved.
