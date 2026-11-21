# Experimental segmented strategies dev40

This prerelease adds an explicit atomic, warp or batch strategy for the
segmented hash and permutation bitmap backends, plus retry of a smaller
segment when owner metadata cannot be admitted. Atomic remains the default.

Source-level and two- and eight-RTX-3060 results, exact correctness scope,
timings, memory receipts and limitations are documented in the repository's
acceptance reports. GPU acceptance is separate from compiling native
components. This prerelease has not been tested on B200 or B300 and does not
prove a physical allocation of over 2^32 rows on a single GPU. The full S14
result and capacity on arbitrary graphs remain unverified.
