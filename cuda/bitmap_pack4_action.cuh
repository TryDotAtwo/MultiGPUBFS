#pragma once
#include "bitmap_pack4.cuh"
#include <cuda_runtime.h>
#include <cstdint>

// Experimental accessor only: the caller must validate before ranking.
// Error is thread-owned (not a shared error accumulator). Packed storage,
// descriptor, generators and input pointers must remain immutable for the
// duration of a kernel.
struct BitmapPack4Action {
    uint32_t elements, generators, count;
    const uint64_t* parents;
    const uint32_t* permutations;
    MgbfsPack4Dictionary dictionary;

    __device__ __forceinline__ bool valid(uint32_t child, uint32_t* error) const {
        bool ok = elements == dictionary.elements &&
                  mgbfs_pack4_valid(dictionary) && generators && parents &&
                  permutations && uint64_t(child) < uint64_t(count) * generators;
        if (ok) {
            const uint64_t word = parents[child / generators];
            ok = mgbfs_pack4_valid_word(dictionary, word);
            uint32_t used = 0;
            const uint64_t start = uint64_t(child % generators) * elements;
            for (uint32_t e = 0; ok && e < elements; ++e) {
                const uint32_t src = permutations[start + e];
                if (src >= elements || (used & (uint32_t(1) << src))) {
                    ok = false;
                } else {
                    used |= uint32_t(1) << src;
                }
            }
        }
        if (!ok && error) *error |= 8u;
        return ok;
    }

    __device__ __forceinline__ int64_t value(uint32_t child, uint32_t element) const {
        if (!parents || !permutations || !generators || element >= elements ||
            elements != dictionary.elements || !elements || elements > 16 ||
            !dictionary.distinct || dictionary.distinct > elements ||
            uint64_t(child) >= uint64_t(count) * generators) return 0;
        const uint32_t source = permutations[uint64_t(child % generators) * elements + element];
        if (source >= elements) return 0;
        const uint64_t word = parents[child / generators];
        const uint32_t ordinal = uint32_t((word >> (source * 4)) & 15u);
        if (ordinal >= dictionary.distinct) return 0;
        return dictionary.symbols[ordinal];
    }
};
