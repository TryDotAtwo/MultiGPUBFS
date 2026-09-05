#include "bounded_owner.h"
#include <climits>
static_assert(sizeof(MgbfsBoundedOwnerBytes)==32);
extern "C" int mgbfs_bounded_owner_query(uint32_t i,uint32_t j,uint32_t k,
    uint32_t backend,uint32_t refinement_capacity,uint32_t tile_limit,
    MgbfsBoundedOwnerBytes* out) {
  if(!out)return 1;
  *out={};
  if(!i||i>INT_MAX||!j||j>i||!k||k>INT_MAX||uint64_t(j)*k>SIZE_MAX/16)return 1;
  if(backend>1||(backend==1&&(refinement_capacity<i||!tile_limit||tile_limit>256)))return 1;
  *out={i,uint64_t(i)*4,uint64_t(j)*k*16,backend==1?uint64_t(j)*4:0};
  return 0;
}
