#include "bitmap_pack4_owner.cuh"
#include "bitmap_pack4_route.cuh"
#include <cstdint>

// Experimental opt-in ABI. A matching Rust caller must check all struct sizes
// and the version before passing values by value. Test barrier builds must
// never advertise this production-shaped ABI.
#ifndef MGBFS_PACK4_OWNER_TEST_BARRIER
extern "C" uint32_t mgbfs_pack4_bfs_abi_query(uint32_t field) {
 switch(field){
  case 0:return 1;
  case 1:return sizeof(MgbfsPack4Dictionary);
  case 2:return sizeof(MgbfsPack4RouteArgs);
  case 3:return sizeof(MgbfsPack4OwnerArgs);
  case 4:return sizeof(MgbfsPack4SeedArgs);
  default:return 0;
 }
}
#endif
