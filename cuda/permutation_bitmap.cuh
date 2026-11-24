#pragma once
#include <stdint.h>

// A descriptor is built from the validated host multiset. Symbols are sorted,
// counts are positive and sum to elements, and total is the exact multinomial.
struct BitmapDeviceDomain {
 const int64_t* symbols;
 const uint32_t* counts;
 uint32_t distinct;
 uint32_t elements;
 uint64_t total;
};

// Computes floor(ways * fraction / remaining) without overflowing uint64_t.
// The required fraction is <= remaining; the result is therefore <= ways.
__device__ __forceinline__ uint64_t bitmap_scale(uint64_t ways,uint32_t fraction,uint32_t remaining){
 return (ways/remaining)*fraction+((ways%remaining)*fraction)/remaining;
}

__device__ __forceinline__ uint32_t bitmap_symbol_ordinal(const BitmapDeviceDomain& domain,int64_t value){
 uint32_t low=0,high=domain.distinct;
 while(low<high){
  uint32_t mid=low+(high-low)/2;
  if(domain.symbols[mid]<value)low=mid+1;else high=mid;
 }
 return low<domain.distinct&&domain.symbols[low]==value?low:domain.distinct;
}

// With unique symbols the rank is a Lehmer code. A register bitmask counts
// lower unused symbols without the multiset scan or per-symbol division.
template<class A>
__device__ __forceinline__ bool bitmap_rank_distinct(const BitmapDeviceDomain& domain,const A& a,uint32_t row,uint64_t* out){
 uint32_t used=0;
 uint64_t rank=0,total=1;
 for(uint32_t e=0;e<domain.elements;e++){
  if(domain.counts[e]!=1)return false;
  const uint32_t ordinal=bitmap_symbol_ordinal(domain,a.value(row,e));
  if(ordinal==domain.distinct)return false;
  const uint32_t bit=uint32_t{1}<<ordinal;
  if(used&bit)return false;
  const uint32_t remaining=domain.elements-e;
  rank=rank*remaining+ordinal-uint32_t(__popc(used&(bit-1)));
  total*=remaining;
  used|=bit;
 }
 if(total!=domain.total)return false;
 *out=rank;
 return true;
}

// The accessor supplies a.value(row, element) in the original state order.
// Exactly one bijective 64-bit rank for each distinct multiset permutation.
// No hash identity assumption is used; invalid inputs return false.
template<class A>
__device__ bool bitmap_rank_component(const BitmapDeviceDomain& domain,const A& a,uint32_t row,uint64_t* out){
 if(!out||!domain.symbols||!domain.counts||!domain.distinct||domain.distinct>64||
    !domain.elements||domain.elements>64||!domain.total)return false;
 // A two-position orbit with distinct symbols contributes one exact bit.
 // Verify both symbols so malformed rows cannot alias a valid rank.
 if(domain.elements==2&&domain.distinct==2&&domain.total==2){
  const int64_t low=domain.symbols[0],high=domain.symbols[1];
  if(low>=high||domain.counts[0]!=1||domain.counts[1]!=1)return false;
  const int64_t first=a.value(row,0),second=a.value(row,1);
  if(first==low&&second==high){*out=0;return true;}
  if(first==high&&second==low){*out=1;return true;}
  return false;
 }
 // 20! is the largest distinct permutation domain representable in u64.
 if(domain.distinct==domain.elements&&domain.elements<=20)
  return bitmap_rank_distinct(domain,a,row,out);
 uint8_t counts[64];
 uint32_t sum=0;
 for(uint32_t i=0;i<domain.distinct;i++){
  uint32_t c=domain.counts[i];if(!c||c>64)return false;counts[i]=uint8_t(c);sum+=c;
 }
 if(sum!=domain.elements)return false;
 uint64_t ways=domain.total,rank=0;
 for(uint32_t e=0;e<domain.elements;e++){
  uint32_t low=bitmap_symbol_ordinal(domain,a.value(row,e));
  if(low==domain.distinct||counts[low]==0)return false;
  uint32_t less=0;for(uint32_t j=0;j<low;j++)less+=counts[j];
  uint32_t remaining=domain.elements-e;
  uint64_t delta=bitmap_scale(ways,less,remaining);
  if(delta>=domain.total||rank>domain.total-1-delta)return false;
  rank+=delta;
  ways=bitmap_scale(ways,counts[low],remaining);
  counts[low]--;
 }
 if(rank>=domain.total||ways!=1)return false;
 *out=rank;return true;
}


// Product-domain metadata is immutable for one BFS job. Components are
// positional orbits; every component's symbols/counts describe its own exact
// multiset. Full-domain descriptors retain the original binary layout.
template<class A> struct BitmapComponentAccessor {
 const A& source;
 const uint32_t* positions;
 __device__ __forceinline__ int64_t value(uint32_t row,uint32_t element) const {
  return source.value(row,positions[element]);
 }
};
template<class A>
__device__ bool bitmap_rank(const BitmapDeviceDomain& domain,const A& a,uint32_t row,uint64_t* out){
 if(!(domain.distinct&0x80000000u))return bitmap_rank_component(domain,a,row,out);
 if(!out||!domain.symbols||!domain.counts||!domain.elements||domain.elements>64||!domain.total)return false;
 const uint32_t components=domain.distinct&0x7fffffffu;
 if(!components||components>domain.elements)return false;
 uint64_t rank=0,product=1;uint32_t element_sum=0;
 for(uint32_t i=0;i<components;i++){
  const uint32_t* m=domain.counts+uint64_t(i)*7;
  const uint32_t elements=m[0],distinct=m[1];
  const uint64_t radix=uint64_t(m[5])|(uint64_t(m[6])<<32);
  if(!elements||!distinct||distinct>elements||elements>domain.elements-element_sum||!radix||
     m[2]<components*7||m[4]<components*7||product>domain.total/radix)return false;
  BitmapDeviceDomain component{domain.symbols+m[3],domain.counts+m[4],distinct,elements,radix};
  BitmapComponentAccessor<A> accessor{a,domain.counts+m[2]};
  uint64_t digit=0;if(!bitmap_rank_component(component,accessor,row,&digit)||digit>=radix)return false;
  if(rank>(UINT64_MAX-digit)/radix)return false;
  rank=rank*radix+digit;product*=radix;element_sum+=elements;
 }
 if(element_sum!=domain.elements||product!=domain.total||rank>=domain.total)return false;
 *out=rank;return true;
}
