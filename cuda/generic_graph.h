#pragma once
#include <cstdint>
// Source/output are preallocated element-major int64 SoA planes. A selected
// child index encodes parent*generator_count+generator. No host state work.
extern "C" int mgbfs_generic_generate_i64(uint32_t kind,uint32_t elements,uint32_t rows,uint32_t cols,uint32_t generators,
 const int64_t* parents,uint32_t parent_count,uint32_t parent_stride,const uint32_t* permutation_tables,
 const int64_t* matrix_tables,const uint32_t* moduli,const uint64_t* selected_children,uint32_t output_count,
 int64_t* output,uint32_t output_stride,uint32_t* device_error,void* stream);
