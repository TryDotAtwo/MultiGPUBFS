#include "../cuda/owner_partition.h"
#include <array>
#include <cassert>
#include <cstdint>
int main() {
  // Eight prefix bins: counts 2,0,1,0,0,1,0,2. Include both endpoints.
  const std::array<uint32_t,24> keys{
    0,0,0,0, 0,0,0,0x1fffffff,
    0,0,0,0x40000000, 0,0,0,0xa0000000,
    0,0,0,0xe0000000, 0,0,0,0xffffffff};
  const std::array<uint32_t,9> expected{0,2,2,3,3,3,4,4,6};
  for(uint32_t b=0;b<=8;++b)
    assert(mgbfs_owner_boundary(keys.data(),6,b,8)==expected[b]);
  assert(mgbfs_owner_boundary(keys.data(),6,1,2)==3);
  assert(mgbfs_owner_boundary(keys.data(),6,1,1)==6);
  for(uint32_t w: {1u,2u,4u,8u})
    for(uint32_t b=0;b<=w;++b)
      assert(mgbfs_owner_boundary(nullptr,0,b,w)==0);
}
