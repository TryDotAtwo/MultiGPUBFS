#pragma once
#include <cstdint>
extern "C" int mgbfs_generic_seed_i64(uint32_t elements,const int64_t* states,uint32_t state_stride,uint32_t state_count,
 uint64_t* slots,uint32_t slot_capacity,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);
extern "C" int mgbfs_generic_expand_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutation_tables,
 const int64_t* matrix_tables,const uint32_t* moduli,uint64_t* slots,uint32_t slot_capacity,
 int64_t* visited,uint32_t visited_capacity,uint32_t* visited_count,uint32_t* future,uint32_t future_capacity,
 uint32_t* future_count,uint64_t seed,uint32_t hash_bits,uint32_t* error,void* stream);
extern "C" int mgbfs_generic_gather_i64(uint32_t elements,const int64_t* source,uint32_t source_stride,
 const uint32_t* indices,uint32_t count,int64_t* output,uint32_t output_stride,void* stream);
