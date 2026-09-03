#include "allocation_shape.h"
#include <cassert>
#include <cstdio>
#include <climits>
#include <initializer_list>
#ifdef PUBLIC_QUERY
#define generation_shape mgbfs_generate_query
#define hash_shape mgbfs_hash_query
#endif
#undef assert
#define assert(condition) do { if (!(condition)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#condition); return 1; } } while (0)
int main() {
  MgbfsGenerateBytes g{};
  for(unsigned variant=0;variant<5;++variant) {
    assert(generation_shape(4,6,256,2,variant,&g)==0);
    assert(g.generators==384 && g.packed_parents==128 && g.products_s32==768);
    assert(g.k==16 && g.stride==16 && g.rows==24 && g.columns==8);
  }
  assert(generation_shape(3,2,3,3,0,&g)==0);
  assert(g.rows==6 && g.columns==12 && g.products_s32==288 && g.generators==96);
  assert(generation_shape(3,2,3,3,1,&g)==0);
  assert(g.rows==8 && g.products_s32==384 && g.generators==128);
  for(unsigned variant : {4u,5u}) {
    assert(generation_shape(3,2,3,3,variant,&g)!=0);
    assert(g.products_s32==0 && g.rows==0);
  }
  assert(generation_shape(0,6,3,2,0,&g)!=0);
  assert(generation_shape(182,6,3,2,0,&g)!=0);
  assert(generation_shape(4,65536,3,2,0,&g)!=0);
  assert(generation_shape(4,6,3,UINT_MAX,0,&g)!=0);
  assert(generation_shape(4,6,257,2,0,&g)!=0);
  assert(generation_shape(4,6,3,2,0,nullptr)!=0);
  MgbfsHashBytes h{};
  assert(hash_shape(9,12,&h)==0);
  assert(h.stride==16 && h.weights==256 && h.offsets==16 && h.partials_s32==768);
  assert(hash_shape(17,2,&h)==0);
  assert(h.stride==32 && h.weights==512 && h.partials_s32==128);
  assert(hash_shape(33026,2,&h)!=0 && h.weights==0);
  assert(hash_shape(16,UINT_MAX,&h)!=0);
  assert(hash_shape(0,2,&h)!=0);
  assert(hash_shape(16,0,&h)!=0);
  assert(hash_shape(16,1,nullptr)!=0);
  std::puts("ALLOCATION_SHAPE_PASS");
}
