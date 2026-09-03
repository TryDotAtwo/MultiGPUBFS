#include "mgbfs_cuda.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)
int main() {
  static_assert(sizeof(MgbfsRouteBytes)==64);
  MgbfsRouteBytes q{};
  CHECK(mgbfs_route_query(0,nullptr)!=0);
  for (uint32_t c : {0u,0x80000000u,0xffffffffu}) {
    std::memset(&q,0xff,sizeof(q));
    CHECK(mgbfs_route_query(c,&q)!=0);
    MgbfsRouteBytes zero{};
    CHECK(std::memcmp(&q,&zero,sizeof(q))==0);
  }
  for (uint32_t c : {1u,17u,4097u}) {
    CHECK(mgbfs_route_query(c,&q)==0);
    CHECK(q.sorted==uint64_t(c)*16 && q.refs==uint64_t(c)*8);
    CHECK(q.indices==uint64_t(c)*4 && q.selected==uint64_t(c)*4 && q.flags==c);
    CHECK(q.sort_scratch>0 && q.select_scratch>0);
    CHECK(q.scratch==std::max(q.sort_scratch,q.select_scratch));
    MgbfsRouteBytes again{};
    CHECK(mgbfs_route_query(c,&again)==0);
    CHECK(std::memcmp(&q,&again,sizeof(q))==0);
  }
  std::puts("ROUTE_QUERY_PASS");
}
