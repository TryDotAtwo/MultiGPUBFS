# Segmented memory backends: prerelease

This release adds opt-in `segmented24` and `permutation_bitmap` storage.
The first is the primary new general-purpose path; the existing `legacy`
backend remains the default for compatibility. Select explicitly using
`NativeOptions(memory_backend="segmented24")` or
`NativeOptions(memory_backend="permutation_bitmap")`.

The package uses a small Python wheel plus independently checksummed GitHub
release components. The shared CUDA/NCCL archive is identical to the previous
release and can be reused from its content-addressed cache. Users can prefetch
one or all targets without a GPU and use the same cache offline in GPU jobs.

Evidence: source tests on 2x RTX 3060 and 8x RTX 3090 include exact small
oracles, complete S12 and segmented S13. The installed candidate was also
checked on 2x RTX 3060 through CayleyPy persistent workers. CUDA components
for SM75/80/86/89/90/100/103/120 and universal PTX have been compiled and
their inventories checked. This is not a B200 or B300 hardware run.

Limitations: the segmented path scans a shard's segments sequentially;
neither complete S14 nor more than 2^32 resident states on a single GPU was
verified. Bitmap supports permutation actions only and allocates for the
complete multiset domain. Both are experimental; an INCOMPLETE report is not
a completed graph or diameter. See
[the detailed design](https://github.com/TryDotAtwo/MultiGPUBFS/blob/5e81b7991da0566b27d973aa6ed83ec49062a727/docs/experimental-memory-backends.md)
and source-pinned acceptance manifest.
