# Batch hash integration checkpoint — 2026-11-20

This is experimental, gated work. `batch` remains unavailable without the private test gate, and no large-graph or B200 acceptance is claimed.

On 2 × RTX 3060 with rental source f4be4199e5d908ccbf061fb45b62e6a60edcf504:

* Batch hash: S4, directed C6 and repeated-symbol graph matched the complete state sets at every layer against a CPU oracle, including switching backend/strategy in one persistent worker pool.
* Batch hash S5: forced equal hashes and zero merge-workspace cap matched every layer's exact state set. Deferred runs stayed searchable; a subsequent normal-hash persistent job completed. The normal case recorded actual run merges.
* Matrix, signed i64, directed and 25-element wide permutation cases matched all layer sizes and final saved state sets.
* Both batch backends passed balanced/max-capacity public CayleyPy API, deliberate rank-cap resource stop, complete-layer snapshot, and next persistent job on two GPUs.
* Both batch backends wrote local row 2^32-1 and spilled the next candidate under CUDA virtual mappings on both GPUs. Only three 2 MiB pages per GPU were physically mapped; this verifies address/overflow logic, not allocation of 2^32 states.
* Standalone hash directory-growth and collision tests passed on both GPUs and compute-sanitizer memcheck reported zero errors.

Still outstanding: integration fault/cleanup tests, larger-layer A/B/C measurements, complete accounting and target B200 validation. One mistakenly generated test ran bitmap instead of hash; its output was rejected and the corrected hash test was rerun.
