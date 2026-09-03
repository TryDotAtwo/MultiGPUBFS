#include "allocation_shape.h"
#include <climits>
static_assert(sizeof(MgbfsGenerateBytes)==48);
static_assert(sizeof(MgbfsHashBytes)==40);
int generation_shape(uint32_t n,uint32_t moves,uint32_t modulus,uint32_t capacity,uint32_t variant,MgbfsGenerateBytes* out) {
  if(!out)return 1;
  *out={};
  if(variant>4||(variant==4&&n!=4)||n==0||uint64_t(n)*n>33025||moves==0||moves>65535||modulus<2||modulus>256||capacity==0||uint64_t(capacity)*n>INT32_MAX-3||uint64_t(n)*(modulus-1)*(modulus-1)>INT32_MAX)return 1;
  MgbfsGenerateBytes q{};
  q.k=(n+15)&~15u;q.stride=(n*n+15)&~15u;
  q.rows=variant?((moves*n+3)&~3u):moves*n;
  q.columns=(capacity*n+3)&~3u;
  // Validated factors above bound every product below UINT64_MAX.
  q.generators=uint64_t(q.rows)*q.k;
  q.packed_parents=uint64_t(q.columns)*q.k;
  q.products_s32=uint64_t(q.rows)*q.columns*4;
  *out=q;return 0;
}
int hash_shape(uint32_t bytes,uint32_t capacity,MgbfsHashBytes* out) {
  if(!out)return 1;
  *out={};
  if(bytes==0||bytes>33025||capacity==0||capacity>INT32_MAX/16)return 1;
  MgbfsHashBytes q{};q.stride=(bytes+15)&~15u;
  q.weights=uint64_t(q.stride)*16;q.offsets=16;q.partials_s32=uint64_t(capacity)*64;
  *out=q;return 0;
}
