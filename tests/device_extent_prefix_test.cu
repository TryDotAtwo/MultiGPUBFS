#include "../cuda/device_extent_prefix.cuh"
#include <cstdint>
#include <iostream>
#include <vector>
#define CHECK(x) do { if(!(x)) { std::cerr<<"FAIL line "<<__LINE__<<"\n"; return 1; } } while(0)

int main() {
  // Empty extents can appear anywhere, including the beginning and end.
  for (unsigned code=0;code<4096;++code) {
    unsigned x=code;std::vector<uint64_t> prefix{0},counts;
    for(unsigned i=0;i<6;++i) {
      const unsigned count=x%4;x/=4;counts.push_back(count);
      prefix.push_back(prefix.back()+count);
    }
    for(uint64_t ordinal=0;ordinal<prefix.back();++ordinal) {
      uint64_t e=0,row=0,expected=ordinal,want=0;
      while(want<counts.size()&&expected>=counts[want])expected-=counts[want++];
      CHECK(mgbfs_resident::locate_parent_extent(prefix.data(),counts.size(),ordinal,&e,&row));
      CHECK(e==want&&row==expected);
    }
    uint64_t e=0,row=0;
    CHECK(!mgbfs_resident::locate_parent_extent(prefix.data(),counts.size(),prefix.back(),&e,&row));
  }
  const uint64_t huge[]={0,0,4294967295ull,4294967297ull,UINT64_MAX};
  uint64_t e=0,row=0;
  // A bank of 2^32 rows includes UINT32_MAX; the next row belongs
  // to the following extent, with aggregate offsets wider than u32.
  const uint64_t full_bank[]={0,4294967296ull,4294967297ull};
  CHECK(mgbfs_resident::locate_parent_extent(full_bank,2,UINT32_MAX,&e,&row)&&e==0&&row==UINT32_MAX);
  CHECK(mgbfs_resident::locate_parent_extent(full_bank,2,4294967296ull,&e,&row)&&e==1&&row==0);
  CHECK(mgbfs_resident::locate_parent_extent(huge,4,4294967295ull,&e,&row)&&e==2&&row==0);
  CHECK(mgbfs_resident::locate_parent_extent(huge,4,4294967296ull,&e,&row)&&e==2&&row==1);
  CHECK(mgbfs_resident::locate_parent_extent(huge,4,UINT64_MAX-1,&e,&row)&&e==3&&row==UINT64_MAX-1-4294967297ull);
  CHECK(!mgbfs_resident::locate_parent_extent(huge,4,UINT64_MAX,&e,&row));
  std::cout<<"PASS extent prefix: empty extents and 64-bit boundaries\n";
}
