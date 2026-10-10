# Optional bounded sort-unit compilation

MGBFS_SORTED_FAST_COMPILE is a CMake cache option with default 0 (unchanged optimization behavior). min, mid and max require a compiler that actually advertises --Ofast-compile; invalid values or unsupported compilers fail during configuration. The option applies only to generic_sorted_native.cu, not every CUDA kernel. nvcc describes this as a compile-time/runtime-performance tradeoff. It is not an execution-speed optimization claim.

On the current CUDA13.2 eight-target build, the original compiler remained live for a long time in the SM100 front end after26 other objects had completed. A mid candidate reuses those byte-identical completed objects and recompiles just the sorted unit. The original live build is explicitly cancelled, not mistaken for a failed/completed job after an observation timeout. Physical exactness and matched runtime timing remain required before release selection. Other architecture compilation is not physical GPU acceptance.

The configuration gate observed the requested option absent before implementation, present only on the selected translation unit after implementation, and rejected an invalid option. Runtime benchmarks, clean installation and immutable artifact verification remain separate gates.
