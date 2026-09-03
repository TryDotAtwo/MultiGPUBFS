#pragma once
#include "mgbfs_cuda.h"
// Internal geometry component; public queries also interrogate compiled GEMM.
int generation_shape(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,uint32_t variant,MgbfsGenerateBytes* out);
int hash_shape(uint32_t bytes,uint32_t capacity,MgbfsHashBytes* out);
