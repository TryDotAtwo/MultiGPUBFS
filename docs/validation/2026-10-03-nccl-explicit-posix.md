# Explicit single-node NCCL allocator policy candidate

The unfiltered physical T4 full-BFS memcheck from7ed52a3 returns97 because
NCCL2.29.7 allocator.cc60 probes FABRIC with cuMemCreate and gets800 before
retrying POSIX. Neither that run nor the earlier two-A4000 gate is closed.

Candidate a527013 changes only the pinned allocator's FABRIC inclusion:
when the existing NCCL_MNNVL_ENABLE flag is explicitly0, do not request
FABRIC. Auto/default2 and enabled1 retain the original vendor behavior.
VMM, POSIX exportability, allocation granularity, mapping/access, window
registration and ncclMemFree are unchanged. No runtime allocator clone,
legacy cudaMalloc, CUDA API reporting suppression or hot-path change.

Pinned upstream: b91894bd5b190c874d98a017f93f5daa515b65d0.
Patch SHA256: e73af6f263bb0eebb22904a20251c8b5da0dec2b463fdeb3c5a9c4fbc88b3072.
Existing minimum-architecture patch applies first, then this patch.
`kaggle/lsa-bfs-gate/kernel.py` explicitly selects the candidate through
`minimum_arch_guard_posix`, sets MNNVL_ENABLE=0 before child startup, checks
both patch digests and runs the allocation policy regression. Wheel and
minimum_arch_guard remain separately selectable, not automatic fallbacks.

Regression compiles the actual pinned ncclMemAlloc function against
controlled CUDA API boundaries. It is not a GPU/window correctness test.
RED: disabled-MNNVL success/error cases each perform2 cuMemCreate calls,
including the forbidden FABRIC probe. GREEN:5 cases cover explicitoff,
auto/on behavior, absent FABRIC capability and fatal POSIX allocation error.
b85d3b7 adds two RED/GREEN admission tests: explicit source or compiler
absence must fail rather than skip. Full Python discovery in scripts now
passes33 tests. Patch reverse-check against the actual modified source and
Python compilation pass. Full pinned NCCL build has compiled allocator.o;
complete link and physical unfiltered sanitizer acceptance remain pending.

Update: clean Linux checkout (both exact patches applied) successfully built
the complete pinned NCCL sm75 library using CUDA12.9, NVTX and -j4. Library
SHA256: dba3edbfdff05229051fd3e7f4d88a3cc88054456380fafd145bd744e4464b68.
Actual local RTX3070Laptop VMM allocation/memset/synchronize/free via this
library passes unfiltered memcheck with0errors, NCCL_CUMEM_ENABLE=1 and
NCCL_MNNVL_ENABLE=0. This isolated one-GPU allocation test does not create
a communicator/window and is not physical2T4 BFS acceptance.

The CUDA-boundary doubles do not establish remote mapping, process failure
handling, full BFS correctness, two-T4 overlap or performance. Those require
the existing integrated runtime and hardware gates, not a policy-unit pass.

NVIDIA2.29.7 documentation requires VMM-compatible memory for windows:
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2297/user-guide/docs/usage/bufferreg.html
The controlled opt-out is a local source patch, not an upstream NCCL fix.
