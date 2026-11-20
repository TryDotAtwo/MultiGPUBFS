# Experimental segmented BFS dev38

This prerelease retains the configurable 32-bit segment row ABI and exact
permutation bitmap. It reclaims unclaimed persistent-session buffers and idle
CUDA-pool allocations before querying free memory for segment sizing.

A GPU regression test allocated 256 MiB in a parked session buffer: the old
sizing order left that buffer parked and failed; the new sizing function
released it and recovered GPU memory. An exact two-rank GPU BFS tested both
memory backends with automatic and manual shard settings, and a CayleyPy
persistent API test exercised segment maxima of 64 and 2^32 on both backends.
See `docs/acceptance/segmented32-20261120/cachetrim-2gpu.json` for evidence.

The CUDA components are reused from dev37 after exact CUDA-source identity
verification; the Rust executable is rebuilt from this release source. This is
an experimental prerelease. B200 runtime acceptance, S14 on 8x B200, and a
real traversal with more than 2^32 resident states on one rank remain untested.
