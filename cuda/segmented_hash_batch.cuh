#pragma once
#include <cstdint>
// Flat SoA state storage and exactly two index columns (12 bytes/live row).
// Keys and rows are indexed [bank][row], with each bank at most 2^32 rows.
struct MgbfsBatchHashSegment {
 void* states;
 uint64_t* keys;
 uint32_t* rows;
 uint64_t capacity;
 uint32_t banks;
};
struct MgbfsBatchHashRun {
 uint32_t segment,bank;
 uint64_t begin,length,min_hash,max_hash;
};
