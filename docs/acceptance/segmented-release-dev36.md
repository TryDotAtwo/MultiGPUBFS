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
[the detailed design](../experimental-memory-backends.md)
and source-pinned acceptance manifest.

## Public package verification

The published dev36 wheel was tested on 2x RTX 3060. CayleyPy persistent
S3-S6 matched independently computed layer sizes for all three backends.
A separate installation from the GitHub working branch passed S4 with both
new backends. With 32 rows per bank, S6 used
two segments on each GPU. The production diagnostic passed on those two
GPUs, including NCCL and BFS. A second prefetch reused the cached archives
without transferring data. See [installed verification](segmented-dev36-installed.json).

The [Native CPU contracts workflow](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/44611559377)
passed its CPU contracts and type checks, but its Rust formatting gate failed.
A checked parent already had formatting differences in 55 files; this branch
had differences in 60. The release pipeline passed its separate checks.
