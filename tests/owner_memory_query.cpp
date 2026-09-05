#include "bounded_owner.h"
#include <cstdio>
#include <climits>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); return 1; } } while (0)
int main() {
  MgbfsBoundedOwnerBytes q{};
  CHECK(mgbfs_bounded_owner_query(17,3,11,0,0,0,&q)==0);
  CHECK(q.flags==17 && q.indices==68 && q.merged==528 && q.refinement_errors==0);
  CHECK(mgbfs_bounded_owner_query(17,3,11,1,17,8,&q)==0);
  CHECK(q.flags==17 && q.indices==68 && q.merged==528 && q.refinement_errors==12);
  CHECK(mgbfs_bounded_owner_query(17,3,11,1,16,8,&q)!=0);
  CHECK(q.flags==0 && q.indices==0 && q.merged==0 && q.refinement_errors==0);
  CHECK(mgbfs_bounded_owner_query(17,3,11,1,17,257,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(17,3,11,2,17,8,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(0,3,11,0,0,0,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(2,3,11,0,0,0,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(17,3,0,0,0,0,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(17,3,UINT_MAX,0,0,0,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(INT_MAX,INT_MAX,INT_MAX,0,0,0,&q)!=0);
  CHECK(mgbfs_bounded_owner_query(17,3,11,0,0,0,nullptr)!=0);
  std::puts("BOUNDED_OWNER_QUERY_PASS");
}
