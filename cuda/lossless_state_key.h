#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define MGBFS_KEY_HD __host__ __device__
#else
#define MGBFS_KEY_HD
#endif
namespace mgbfs_lossless {
struct Key {uint64_t lo,hi;};
MGBFS_KEY_HD inline uint64_t diffuse(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
// Two Feistel rounds: bijection on all 128-bit words, without assumptions on diffuse.
MGBFS_KEY_HD inline Key mix(Key k){k.lo^=diffuse(k.hi^0x9e3779b97f4a7c15ULL);k.hi^=diffuse(k.lo^0xd1b54a32d192ed03ULL);return k;}
MGBFS_KEY_HD inline Key unmix(Key k){k.hi^=diffuse(k.lo^0xd1b54a32d192ed03ULL);k.lo^=diffuse(k.hi^0x9e3779b97f4a7c15ULL);return k;}
MGBFS_KEY_HD inline bool shape(uint32_t n,uint32_t bits){return n>0&&bits>0&&bits<=8&&uint64_t(n)*bits<=128;}
// Caller proves all values fit bits and the entire immutable state fits 128 bits.
MGBFS_KEY_HD inline Key pack(const uint8_t* state,uint32_t n,uint32_t bits){Key k{0,0};for(uint32_t j=0;j<n;++j){uint32_t at=j*bits;uint64_t x=state[j];if(at<64){k.lo|=x<<at;if(at+bits>64)k.hi|=x>>(64-at);}else{k.hi|=x<<(at-64);}}return mix(k);}
MGBFS_KEY_HD inline uint8_t element(Key encoded,uint32_t j,uint32_t bits){Key k=unmix(encoded);uint32_t at=j*bits;uint64_t x;if(at<64){x=k.lo>>at;if(at+bits>64)x|=k.hi<<(64-at);}else x=k.hi>>(at-64);return uint8_t(x&((1u<<bits)-1));}
}
#undef MGBFS_KEY_HD
