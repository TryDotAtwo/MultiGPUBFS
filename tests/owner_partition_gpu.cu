#include "../cuda/mgbfs_cuda.h"
#include "../cuda/directories.h"
#include <cuda_runtime.h>
#include <array>
#include <stdexcept>
#include <cstdio>
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool ok){if(!ok)throw std::runtime_error("OWNER_PARTITION_GPU");}
int main(){
  cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  uint32_t *keys,*states,*packed,*counts,*n,*fatal,*window_begin,*window_rows;uint64_t* refs;MgbfsOwnerRange* dir;
  ck(cudaMalloc(&keys,96));ck(cudaMalloc(&states,96));ck(cudaMalloc(&packed,96));
  ck(cudaMalloc(&counts,36));ck(cudaMalloc(&refs,48));ck(cudaMalloc(&n,4));
  ck(cudaMalloc(&fatal,4));ck(cudaMalloc(&dir,4*sizeof(MgbfsOwnerRange)));
  ck(cudaMalloc(&window_begin,4));ck(cudaMalloc(&window_rows,4));
  std::array<uint32_t,24> input{};
  const uint32_t high[6]={0,0x1fffffff,0x40000000,0xa0000000,0xe0000000,0xffffffff};
  for(unsigned i=0;i<6;++i)input[4*i+3]=high[i];
  const std::array<uint64_t,6> order{5,0,4,1,3,2};
  std::array<uint32_t,24> payload{};for(unsigned i=0;i<24;++i)payload[i]=i+71;
  ck(cudaMemcpyAsync(keys,input.data(),96,cudaMemcpyHostToDevice,s));
  ck(cudaMemcpyAsync(states,payload.data(),96,cudaMemcpyHostToDevice,s));
  ck(cudaMemcpyAsync(refs,order.data(),48,cudaMemcpyHostToDevice,s));
  for(uint32_t w:{1u,2u,4u,8u}){
    ck(cudaMemsetAsync(counts,0xff,36,s));
    require(!mgbfs_exchange_pack_n(w,16,6,(uint8_t*)states,6,keys,refs,6,(uint8_t*)packed,counts,s));
    std::array<uint32_t,9> got{};std::array<uint32_t,24> actual{};
    ck(cudaMemcpyAsync(got.data(),counts,36,cudaMemcpyDeviceToHost,s));
    ck(cudaMemcpyAsync(actual.data(),packed,96,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));
    std::array<uint32_t,8> want{};
    for(auto h:high)++want[(uint64_t(h)*w)>>32];
    for(unsigned i=0;i<w;++i)require(got[i]==want[i]);
    require(got[w]==UINT32_MAX);
    for(unsigned i=0;i<6;++i)for(unsigned j=0;j<4;++j)require(actual[4*i+j]==payload[4*order[i]+j]);
    unsigned begin=0;
    for(unsigned owner=0;owner<w;++owner){
      uint32_t rows=want[owner];ck(cudaMemcpyAsync(n,&rows,4,cudaMemcpyHostToDevice,s));
      ck(cudaMemsetAsync(fatal,0,4,s));
      require(!mgbfs_owner_bucket_directory_n(keys+4*begin,n,6,4,owner,w,dir,fatal,s));
      std::array<MgbfsOwnerRange,4> ranges{};uint32_t error;
      ck(cudaMemcpyAsync(ranges.data(),dir,sizeof(ranges),cudaMemcpyDeviceToHost,s));
      ck(cudaMemcpyAsync(&error,fatal,4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));require(!error);
      unsigned offset=0;
      for(unsigned b=0;b<4;++b){unsigned count=0;
        for(unsigned i=begin;i<begin+rows;++i)if(((uint64_t(high[i])*w*4)>>32)==owner*4+b)++count;
        require(ranges[b].begin==offset&&ranges[b].count==count);offset+=count;
      }
      begin+=rows;
    }
    require(!mgbfs_exchange_pack_n(w,16,6,(uint8_t*)states,6,keys,refs,0,(uint8_t*)packed,counts,s));
    ck(cudaMemcpyAsync(got.data(),counts,36,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));
    for(unsigned i=0;i<w;++i)require(!got[i]);
  }
  // The producer publishes only a device count. Packing must not require a
  // host readback before launch, including zero, partial and overflow tails.
  for (uint32_t valid:{0u,3u,6u,7u}) {
    ck(cudaMemcpyAsync(n,&valid,4,cudaMemcpyHostToDevice,s));
    ck(cudaMemsetAsync(counts,0xff,36,s));
    ck(cudaMemsetAsync(packed,0xcd,96,s));
    require(!mgbfs_exchange_pack_device_n(4,16,6,(uint8_t*)states,6,
        keys,refs,n,(uint8_t*)packed,counts,s));
    std::array<uint32_t,9> got{};std::array<uint32_t,24> actual{};
    ck(cudaMemcpyAsync(got.data(),counts,36,cudaMemcpyDeviceToHost,s));
    ck(cudaMemcpyAsync(actual.data(),packed,96,cudaMemcpyDeviceToHost,s));
    ck(cudaStreamSynchronize(s));
    if(valid==7){require(got[0]==UINT32_MAX);continue;}
    std::array<uint32_t,4> want{};
    for(unsigned i=0;i<valid;++i)++want[(uint64_t(high[i])*4)>>32];
    for(unsigned i=0;i<4;++i)require(got[i]==want[i]);
    require(got[4]==UINT32_MAX);
    for(unsigned i=0;i<valid;++i)for(unsigned j=0;j<4;++j)
      require(actual[4*i+j]==payload[4*order[i]+j]);
    for(unsigned i=valid*4;i<24;++i)require(actual[i]==0xcdcdcdcdu);
  }
  // A route window is derived entirely from device counts. A reversed
  // rank-to-logical-owner assignment must not change the packed offset.
  const std::array<uint32_t,4> partition{1,2,0,3};
  ck(cudaMemcpyAsync(counts,partition.data(),16,cudaMemcpyHostToDevice,s));
  uint32_t expected_total=6;
  ck(cudaMemcpyAsync(n,&expected_total,4,cudaMemcpyHostToDevice,s));
  for(uint32_t owner:{3u,1u,2u,0u}){
    require(!mgbfs_owner_window_from_counts(4,6,owner,counts,
        n,window_begin,window_rows,s));
    uint32_t begin=99,rows=99;
    ck(cudaMemcpyAsync(&begin,window_begin,4,cudaMemcpyDeviceToHost,s));
    ck(cudaMemcpyAsync(&rows,window_rows,4,cudaMemcpyDeviceToHost,s));
    ck(cudaStreamSynchronize(s));
    const uint32_t expected_begin[4]={0,1,3,3};
    require(begin==expected_begin[owner]&&rows==partition[owner]);
  }
  const std::array<uint32_t,4> over_capacity{1,2,1,3};
  ck(cudaMemcpyAsync(counts,over_capacity.data(),16,cudaMemcpyHostToDevice,s));
  require(!mgbfs_owner_window_from_counts(4,6,1,counts,n,
      window_begin,window_rows,s));
  uint32_t bad_rows=0;
  ck(cudaMemcpyAsync(&bad_rows,window_rows,4,cudaMemcpyDeviceToHost,s));
  ck(cudaStreamSynchronize(s));require(bad_rows==UINT32_MAX);
  const std::array<uint32_t,4> failed_source{UINT32_MAX,0,0,0};
  ck(cudaMemcpyAsync(counts,failed_source.data(),16,cudaMemcpyHostToDevice,s));
  require(!mgbfs_owner_window_from_counts(4,6,3,counts,n,
      window_begin,window_rows,s));
  ck(cudaMemcpyAsync(&bad_rows,window_rows,4,cudaMemcpyDeviceToHost,s));
  ck(cudaStreamSynchronize(s));require(bad_rows==UINT32_MAX);
  const std::array<uint32_t,4> short_partition{1,2,0,2};
  ck(cudaMemcpyAsync(counts,short_partition.data(),16,cudaMemcpyHostToDevice,s));
  require(!mgbfs_owner_window_from_counts(4,6,3,counts,n,
      window_begin,window_rows,s));
  ck(cudaMemcpyAsync(&bad_rows,window_rows,4,cudaMemcpyDeviceToHost,s));
  ck(cudaStreamSynchronize(s));require(bad_rows==UINT32_MAX);
  uint64_t invalid=6;ck(cudaMemcpyAsync(refs,&invalid,8,cudaMemcpyHostToDevice,s));
  require(!mgbfs_exchange_pack_n(8,16,6,(uint8_t*)states,6,keys,refs,6,(uint8_t*)packed,counts,s));
  uint32_t marker;ck(cudaMemcpyAsync(&marker,counts,4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));require(marker==UINT32_MAX);
  uint32_t six=6;ck(cudaMemcpyAsync(n,&six,4,cudaMemcpyHostToDevice,s));ck(cudaMemsetAsync(fatal,0,4,s));
  require(!mgbfs_owner_bucket_directory_n(keys,n,6,4,0,8,dir,fatal,s));
  ck(cudaMemcpyAsync(&marker,fatal,4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));require(marker==32);
  for(void* p:{(void*)keys,(void*)states,(void*)packed,(void*)counts,(void*)refs,(void*)n,(void*)fatal,(void*)dir,(void*)window_begin,(void*)window_rows})ck(cudaFree(p));
  ck(cudaStreamDestroy(s));std::puts("OWNER_PARTITION_GPU_PASS");
}
