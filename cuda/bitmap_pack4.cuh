#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define MGBFS_PACK4_HD __host__ __device__ __forceinline__
#else
#define MGBFS_PACK4_HD inline
#endif

// Validated once per job. All symbols remain in their original signed domain;
// the ordinal is merely a lossless resident representation.
struct MgbfsPack4Dictionary {
    uint32_t elements;
    uint32_t distinct;
    int64_t symbols[16];
};

MGBFS_PACK4_HD bool mgbfs_pack4_valid(const MgbfsPack4Dictionary& d) {
    if (d.elements == 0 || d.elements > 16 ||
        d.distinct == 0 || d.distinct > d.elements) return false;
    for (uint32_t i = 1; i < d.distinct; ++i) {
        if (d.symbols[i - 1] >= d.symbols[i]) return false;
    }
    return true;
}

MGBFS_PACK4_HD bool mgbfs_pack4_valid_word(const MgbfsPack4Dictionary& d, uint64_t word) {
    if (!mgbfs_pack4_valid(d)) return false;
    // Guard the shift: shifting a 64-bit integer by 64 is undefined.
    if (d.elements < 16 && (word >> (4 * d.elements)) != 0) return false;
    for (uint32_t i = 0; i < d.elements; ++i) {
        if (((word >> (4 * i)) & uint64_t(15)) >= d.distinct) return false;
    }
    return true;
}

MGBFS_PACK4_HD bool mgbfs_pack4_encode(const MgbfsPack4Dictionary& d,
                                       const int64_t* row, uint64_t* out) {
    if (!row || !out || !mgbfs_pack4_valid(d)) return false;
    uint64_t result = 0;
    for (uint32_t i = 0; i < d.elements; ++i) {
        uint32_t ordinal = 0;
        while (ordinal < d.distinct && d.symbols[ordinal] != row[i]) ++ordinal;
        if (ordinal == d.distinct) return false;
        result |= uint64_t(ordinal) << (4 * i);
    }
    *out = result;
    return true;
}

MGBFS_PACK4_HD bool mgbfs_pack4_decode(const MgbfsPack4Dictionary& d,
                                       uint64_t word, int64_t* row) {
    if (!row || !mgbfs_pack4_valid_word(d, word)) return false;
    for (uint32_t i = 0; i < d.elements; ++i) {
        row[i] = d.symbols[(word >> (4 * i)) & uint64_t(15)];
    }
    return true;
}

// Generator convention: output position e reads source position perm[e].
// Reject a malformed mapping rather than permitting a non-permutation state.
MGBFS_PACK4_HD bool mgbfs_pack4_apply(const MgbfsPack4Dictionary& d,
                                      uint64_t word, const uint8_t* perm, uint64_t* out) {
    if (!perm || !out || !mgbfs_pack4_valid_word(d, word)) return false;
    uint32_t used = 0;
    uint64_t result = 0;
    for (uint32_t i = 0; i < d.elements; ++i) {
        const uint32_t source = perm[i];
        if (source >= d.elements || (used & (uint32_t(1) << source))) return false;
        used |= uint32_t(1) << source;
        result |= ((word >> (4 * source)) & uint64_t(15)) << (4 * i);
    }
    *out = result;
    return true;
}

#undef MGBFS_PACK4_HD
