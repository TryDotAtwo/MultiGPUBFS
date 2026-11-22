#include "device_mailbox_layout.h"
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#define CHECK(expr) do { if(!(expr)) { \
  std::fprintf(stderr,"CHECK failed at %s:%d: %s\n",__FILE__,__LINE__,#expr); \
  std::exit(1); \
} } while(0)
using Layout=MgbfsDeviceMailboxLayoutV1;
static int plan(uint32_t world,uint32_t lanes,uint32_t slots,uint32_t bytes,uint64_t budget,Layout& out) {
  return mgbfs_device_mailbox_layout_v1(world,lanes,slots,bytes,budget,sizeof(out),&out);
}
int main() {
  static_assert(sizeof(Layout)==72, "stable mailbox layout ABI");
  const auto max=std::numeric_limits<uint64_t>::max();
  for(uint32_t world=1;world<=8;++world) for(uint32_t lanes=1;lanes<=3;++lanes)
    for(uint32_t slots=1;slots<=5;++slots) for(uint32_t bytes=0;bytes<=65;++bytes) {
      Layout v{};
      CHECK(plan(world,lanes,slots,bytes,max,v)==MGBFS_MAILBOX_LAYOUT_OK);
      CHECK(v.abi_version==1 && v.struct_bytes==sizeof(v));
      CHECK(v.world==world && v.lanes==lanes && v.slots_per_ring==slots);
      CHECK(v.requested_payload_bytes==bytes && v.payload_stride_bytes>=bytes);
      CHECK((v.payload_stride_bytes%16)==0 && v.payload_stride_bytes-bytes<16);
      CHECK(v.ring_count==uint64_t(world)*lanes);
      CHECK(v.slots_offset%64==0 && v.payload_offset%64==0 &&
             v.close_offset%64==0 && v.total_bytes%64==0);
      std::vector<std::pair<uint64_t,uint64_t>> intervals;
      for(uint64_t r=0;r<v.ring_count;++r) {
        intervals.push_back({v.slots_offset+r*slots*64, v.slots_offset+(r+1)*slots*64});
        if(v.payload_stride_bytes)
          intervals.push_back({v.payload_offset+r*slots*v.payload_stride_bytes,
                               v.payload_offset+(r+1)*slots*v.payload_stride_bytes});
        intervals.push_back({v.close_offset+r*64,v.close_offset+(r+1)*64});
      }
      std::sort(intervals.begin(),intervals.end());
      for(size_t i=0;i<intervals.size();++i) {
        CHECK(intervals[i].first<intervals[i].second);
        CHECK(intervals[i].second<=v.total_bytes);
        if(i) CHECK(intervals[i-1].second<=intervals[i].first);
      }
      Layout exact{};
      CHECK(plan(world,lanes,slots,bytes,v.total_bytes,exact)==MGBFS_MAILBOX_LAYOUT_OK);
      CHECK(!std::memcmp(&v,&exact,sizeof(v)));
      Layout preserved;std::memset(&preserved,0xa5,sizeof(preserved));Layout before=preserved;
      CHECK(plan(world,lanes,slots,bytes,v.total_bytes-1,preserved)==MGBFS_MAILBOX_LAYOUT_BUDGET);
      CHECK(!std::memcmp(&preserved,&before,sizeof(before)));
    }
  Layout result{};
  CHECK(plan(0,1,1,0,max,result)==MGBFS_MAILBOX_LAYOUT_INVALID);
  CHECK(plan(1,0,1,0,max,result)==MGBFS_MAILBOX_LAYOUT_INVALID);
  CHECK(plan(1,1,0,0,max,result)==MGBFS_MAILBOX_LAYOUT_INVALID);
  CHECK(plan(1,1,1,UINT32_MAX,max,result)==MGBFS_MAILBOX_LAYOUT_OVERFLOW);
  CHECK(plan(UINT32_MAX,UINT32_MAX,UINT32_MAX,64,max,result)==MGBFS_MAILBOX_LAYOUT_OVERFLOW);
  CHECK(mgbfs_device_mailbox_layout_v1(1,1,1,1,max,sizeof(result)-1,&result)==MGBFS_MAILBOX_LAYOUT_INVALID);
  CHECK(mgbfs_device_mailbox_layout_v1(1,1,1,1,max,sizeof(result),nullptr)==MGBFS_MAILBOX_LAYOUT_INVALID);
  // No physical allocation: arithmetic must not narrow the total to 32 bits.
  CHECK(plan(65536,65536,1,0,max,result)==MGBFS_MAILBOX_LAYOUT_OK);
  CHECK(result.ring_count==(uint64_t(1)<<32));
  CHECK(result.total_bytes==(uint64_t(1)<<39));
  std::puts("PASS mailbox layout checked arithmetic, exact budget and disjoint rings");
}
