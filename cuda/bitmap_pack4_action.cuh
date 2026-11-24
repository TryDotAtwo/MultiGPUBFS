#pragma once
#include "bitmap_pack4.cuh"
#include "permutation_bitmap.cuh"
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
            uint64_t used = 0;
            const uint64_t start = uint64_t(child % generators) * elements;
            for (uint32_t e = 0; ok && e < elements; ++e) {
                const uint32_t src = permutations[start + e];
                if (src >= elements || (used & (uint64_t(1) << src))) {
                    ok = false;
                } else {
                    used |= uint64_t(1) << src;
                }
            }
        }
        if (!ok && error) *error |= 8u;
        return ok;
    }

    __device__ __forceinline__ int64_t value(uint32_t child, uint32_t element) const {
        if (!parents || !permutations || !generators || element >= elements ||
            elements != dictionary.elements || !elements || elements > 64 ||
            !dictionary.distinct || dictionary.distinct > elements ||
            uint64_t(child) >= uint64_t(count) * generators) return 0;
        const uint32_t source = permutations[uint64_t(child % generators) * elements + element];
        if (source >= elements) return 0;
        const uint64_t word = parents[child / generators];
        const uint32_t width = mgbfs_pack4_width(dictionary);
        const uint32_t ordinal = uint32_t((word >> (source * width)) & ((1u << width) - 1));
        if (ordinal >= dictionary.distinct) return 0;
        return dictionary.symbols[ordinal];
    }

    // The launch ABI has already validated the dictionary and shape. For a
    // full distinct domain the packed ordinals are Lehmer digits' inputs.
    // Product domains and multisets still use the checked value-based rank.
    __device__ __forceinline__ bool rank_candidate(const BitmapDeviceDomain& domain,
                                                    uint32_t child, uint64_t* out) const {
        if (elements <= 16 && elements == dictionary.distinct && domain.distinct == elements) {
            const uint64_t word = parents[child / generators];
            if (elements < 16 && (word >> (4 * elements)) != 0) return false;
            const uint32_t* permutation = permutations + uint64_t(child % generators) * elements;
            uint32_t used = 0;
            uint64_t rank = 0, total = 1;
            for (uint32_t element = 0; element < elements; ++element) {
                const uint32_t source = permutation[element];
                if (source >= elements || domain.counts[element] != 1) return false;
                const uint32_t ordinal = (word >> (4 * source)) & 15u;
                if (ordinal >= elements) return false;
                const uint32_t bit = 1u << ordinal;
                if (used & bit) return false;
                if (dictionary.symbols[ordinal] != domain.symbols[ordinal])
                    return valid(child, nullptr) && bitmap_rank(domain, *this, child, out);
                const uint32_t remaining = elements - element;
                rank = rank * remaining + ordinal - __popc(used & (bit - 1));
                total *= remaining;
                used |= bit;
            }
            if (total != domain.total) return false;
            *out = rank;
            return true;
        }
        return valid(child, nullptr) && bitmap_rank(domain, *this, child, out);
    }
};
