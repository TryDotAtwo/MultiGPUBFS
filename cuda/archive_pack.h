#pragma once
#include "state_commit.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Losslessly encode canonical one-hot n*n u8 matrices as n-byte row-to-column
 * permutations. Invalid input poisons the shared run ring with fatal 18. */
int mgbfs_archive_pack_permutation_u8(uint32_t n, uint32_t stride,
    const uint8_t* states, uint32_t count, uint8_t* permutations,
    MgbfsStateRingControl* ring, void* stream);
#ifdef __cplusplus
}
#endif
