#pragma once
#include "generic_action.cuh"
#include <cstdint>

// One canonical hash/owner formula for the single- and multi-CTA schedulers.
// Equal packed states decode through the same immutable job dictionary.
namespace mgbfs_device_hash_key {
template<class Word,class Config>
__device__ inline uint64_t hash(const Word& word,const Config& config) {
  uint64_t value=config.seed;
  for(uint32_t e=0;e<word.dictionary.elements;++e)
    value=mix64(value^uint64_t(word.value(e))^uint64_t(e));
  return finish_hash(value,config.bits);
}
template<class Args,class Config>
__device__ inline unsigned owner(uint64_t hash,const Args& a,const Config& config) {
  const uint64_t high=hash>>32;
  if(!config.owner_cuts)return unsigned((high*a.world)>>32);
  if(config.owner_cuts[0]!=0||config.owner_cuts[a.world]!=(uint64_t(1)<<32))return a.world;
  uint32_t low=0,upper=a.world;
  while(low+1<upper){
    const uint32_t mid=(low+upper)/2;
    if(high<config.owner_cuts[mid])upper=mid;else low=mid;
  }
  return (config.owner_cuts[low]<=high&&high<config.owner_cuts[low+1])?low:a.world;
}
}
