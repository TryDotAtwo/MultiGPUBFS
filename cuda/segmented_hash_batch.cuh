#pragma once
#include <cstdint>
// Flat SoA state storage and exactly two index columns (12 bytes/live row).
// Keys and rows are indexed [bank][row], with each bank at most 2^32 rows.
struct MgbfsBatchHashSegment {
 void* states;
 uint64_t* keys;
 uint32_t* rows;
 uint64_t capacity;
 uint32_t banks;
};
struct MgbfsBatchHashRun {
 uint32_t segment,bank;
 uint64_t begin,length,min_hash,max_hash;
 // Runs below 256 rows use four words to cover low-byte buckets, tagged
 // by word four. Older runs of at most 32 rows used only word zero.
 // Runs of at least 256 rows store all seven quantile fences.
 uint64_t hash_fences[7];
};
static_assert(sizeof(MgbfsBatchHashRun)==96,"batch hash run ABI v2");

struct MgbfsHashPathParentV1 {
 uint64_t parent;
 uint32_t source,generator;
};
static_assert(sizeof(MgbfsHashPathParentV1)==16,"batch hash path parent ABI v1");

struct MgbfsHashPathHitV1 {
 uint32_t segment,bank,row,found;
};
static_assert(sizeof(MgbfsHashPathHitV1)==16,"batch hash path hit ABI v1");

struct MgbfsHashPathMeetingV1 {
 uint32_t source_range, source_row, opposite_segment, opposite_bank, opposite_row, found;
};
static_assert(sizeof(MgbfsHashPathMeetingV1) == 24 && alignof(MgbfsHashPathMeetingV1) == 4,
 "batch hash frontier meeting ABI v1");

struct MgbfsHashWindowEdgeV1 {
 uint64_t parent;
 uint32_t segment,bank,row,source,generator,found;
};
static_assert(sizeof(MgbfsHashWindowEdgeV1)==32,"batch hash window edge ABI v1");

// Logical references for an index spanning physical extents. The descriptor
// ID resolves both segment and bank. A directory must retain that ID until
// all its series and in-flight readers have dropped the reference.
struct MgbfsHashRowRef {
 uint64_t value;
};

#if defined(__CUDACC__)
#define MGBFS_HASH_REF_HD __host__ __device__
#else
#define MGBFS_HASH_REF_HD
#endif

struct MgbfsHashCompactRefLayout {
 // Each descriptor addresses at most 2^row_bits rows. Using 64-bit
 // arithmetic also supports the single-descriptor, 32-row-bit case.
 uint32_t row_bits;

 MGBFS_HASH_REF_HD constexpr bool try_encode(uint32_t descriptor,uint32_t row,uint32_t& index) const {
  if(row_bits>32)return false;
  if(uint64_t(row)>=(uint64_t{1}<<row_bits))return false;
  const uint64_t ref=(uint64_t(descriptor)<<row_bits)|row;
  if(ref>UINT32_MAX)return false;
  index=uint32_t(ref);
  return true;
 }

 // Precondition: row_bits <= 32, established when admitting the layout.
 MGBFS_HASH_REF_HD constexpr uint32_t descriptor(uint32_t index) const {
  return uint32_t(uint64_t(index)>>row_bits);
 }

 MGBFS_HASH_REF_HD constexpr uint32_t row(uint32_t index) const {
  return uint32_t(uint64_t(index)&((uint64_t{1}<<row_bits)-1));
 }
};

// Admit the entire geometry before choosing a column width. Checking only
// the currently allocated extents would allow pool growth to overflow it.
MGBFS_HASH_REF_HD constexpr bool mgbfs_hash_compact_ref_layout(
 uint64_t extent_count,uint64_t rows_per_extent,MgbfsHashCompactRefLayout& layout){
 constexpr uint64_t word_space=uint64_t{1}<<32;
 if(!extent_count||!rows_per_extent||rows_per_extent>word_space)return false;
 uint32_t bits=0;
 uint64_t rounded_rows=1;
 while(rounded_rows<rows_per_extent){
  rounded_rows<<=1;
  ++bits;
 }
 if(extent_count>word_space/rounded_rows)return false;
 layout={bits};
 return true;
}

MGBFS_HASH_REF_HD constexpr MgbfsHashRowRef mgbfs_hash_wide_ref(uint32_t descriptor,uint32_t row){
 return {uint64_t(descriptor)<<32|row};
}
MGBFS_HASH_REF_HD constexpr uint32_t mgbfs_hash_ref_descriptor(MgbfsHashRowRef ref){
 return uint32_t(ref.value>>32);
}
MGBFS_HASH_REF_HD constexpr uint32_t mgbfs_hash_ref_row(MgbfsHashRowRef ref){
 return uint32_t(ref.value);
}

// The remap is indexed by the old descriptor ID. A removed descriptor
// cannot occur in a live series. Keep the output unchanged on failure so
// the caller can validate an entire series before any in-place rewrite.
MGBFS_HASH_REF_HD constexpr bool mgbfs_hash_remap_ref(
 MgbfsHashCompactRefLayout layout,uint32_t ref,const uint32_t* old_to_new,uint64_t extent_count,uint32_t& result){
 if(layout.row_bits>32)return false;
 const uint32_t old=layout.descriptor(ref);
 if(old>=extent_count||old_to_new[old]==UINT32_MAX)return false;
 return layout.try_encode(old_to_new[old],layout.row(ref),result);
}

MGBFS_HASH_REF_HD constexpr bool mgbfs_hash_remap_ref(
 MgbfsHashRowRef ref,const uint32_t* old_to_new,uint64_t extent_count,MgbfsHashRowRef& result){
 const uint32_t old=mgbfs_hash_ref_descriptor(ref);
 if(old>=extent_count||old_to_new[old]==UINT32_MAX)return false;
 result=mgbfs_hash_wide_ref(old_to_new[old],mgbfs_hash_ref_row(ref));
 return true;
}

// A searchable snapshot replaces a prefix of the legacy run directory.
struct MgbfsHashSeriesView {
 const uint64_t* keys;
 const void* references;
 const uint32_t* prefix_offsets;
 uint32_t length;
 uint32_t replaced_runs;
 uint32_t reference_bytes;
 MgbfsHashCompactRefLayout layout;
};

// Snapshots cover consecutive, disjoint portions of the catalog prefix.
// In rolling history this lets each live bank retire independently.
struct MgbfsHashSeriesDirectory {
 MgbfsHashSeriesView series[3];
 uint32_t count;
 uint32_t replaced_runs;
};

static_assert(sizeof(MgbfsHashSeriesView)==40,"hash snapshot view ABI");
static_assert(sizeof(MgbfsHashSeriesDirectory)==128,"hash snapshot directory ABI");

// Resolve a tile of the original raw SoA shard span after all of its accept
// and spill passes. Each original record gives one result in input order.
// Keep the span and the committed one-layer catalog alive until the stream
// drains. A nonzero device error invalidates the entire output tile.
struct GenericRouteRecord;
extern "C" int mgbfs_hash_batch_resolve_window_v1(uint32_t state_bytes, uint32_t elements,
 const void* states, const GenericRouteRecord* records, uint32_t rows, uint32_t begin,
 uint32_t count, uint32_t source, uint32_t shard, uint32_t generators, uint32_t expected_bank,
 const MgbfsBatchHashSegment* segments, uint32_t segment_count, const MgbfsBatchHashRun* runs,
 const uint32_t* run_count, uint32_t run_capacity, MgbfsHashWindowEdgeV1* edges,
 uint32_t output_capacity, uint32_t* error, void* stream, const MgbfsHashSeriesDirectory* view,
 uint32_t max_series);

// Each series uses the same high-bit prefix directory. The terminal entry
// is the row count, avoiding overflow for the highest hash prefix.
constexpr uint32_t MGBFS_HASH_SERIES_PREFIX_BITS=8;
constexpr uint32_t MGBFS_HASH_SERIES_PREFIXES=1u<<MGBFS_HASH_SERIES_PREFIX_BITS;
constexpr uint32_t MGBFS_HASH_SERIES_SHIFT_INDEX=MGBFS_HASH_SERIES_PREFIXES+1;
constexpr uint32_t MGBFS_HASH_SERIES_DIRECTORY_WORDS=MGBFS_HASH_SERIES_PREFIXES+2;

MGBFS_HASH_REF_HD constexpr uint32_t mgbfs_hash_series_shift(uint64_t first,uint64_t last){
 uint64_t differing=first^last;
 uint32_t span=0;
 while(differing){++span;differing>>=1;}
 return span>MGBFS_HASH_SERIES_PREFIX_BITS?span-MGBFS_HASH_SERIES_PREFIX_BITS:0;
}

MGBFS_HASH_REF_HD constexpr uint32_t mgbfs_hash_series_prefix(uint64_t hash,uint32_t shift){
 return uint32_t(hash>>shift)&(MGBFS_HASH_SERIES_PREFIXES-1);
}

MGBFS_HASH_REF_HD constexpr uint32_t mgbfs_hash_series_prefix_begin(
 const uint64_t* keys,uint32_t length,uint32_t prefix,uint32_t shift){
 if(prefix==MGBFS_HASH_SERIES_PREFIXES)return length;
 uint32_t lo=0,hi=length;
 while(lo<hi){
  const uint32_t mid=lo+(hi-lo)/2;
  if(mgbfs_hash_series_prefix(keys[mid],shift)<prefix)lo=mid+1;
  else hi=mid;
 }
 return lo;
}

enum class MgbfsHashSeriesMatch { absent,present,bad_reference };

template<class State,bool ReturnHit=false,class Ref,class Query>
MGBFS_HASH_REF_HD MgbfsHashSeriesMatch mgbfs_hash_series_probe(
 const uint64_t* keys,const Ref* references,const uint32_t* prefix_offsets,uint32_t length,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,MgbfsHashCompactRefLayout layout,
 uint64_t hash,Query query,uint32_t elements,MgbfsHashPathHitV1* hit=nullptr){
 if constexpr (sizeof(Ref)==sizeof(uint32_t)){
  if(layout.row_bits>32)return MgbfsHashSeriesMatch::bad_reference;
 }
 // Empty prefix ranges reject misses without touching the keys array.
 if(!length)return MgbfsHashSeriesMatch::absent;
 const uint32_t shift=prefix_offsets[MGBFS_HASH_SERIES_SHIFT_INDEX];
 if(shift>64-MGBFS_HASH_SERIES_PREFIX_BITS)return MgbfsHashSeriesMatch::bad_reference;
 const uint32_t prefix=mgbfs_hash_series_prefix(hash,shift);
 uint32_t lo=prefix_offsets[prefix],hi=prefix_offsets[prefix+1];
 if(lo>hi||hi>length)return MgbfsHashSeriesMatch::bad_reference;
 while(lo<hi){
  const uint32_t mid=lo+(hi-lo)/2;
  int cmp=hash<keys[mid]?-1:hash>keys[mid]?1:0;
  if(cmp==0){
   uint32_t descriptor,row;
   if constexpr (sizeof(Ref)==sizeof(uint32_t)){
    descriptor=layout.descriptor(references[mid]);row=layout.row(references[mid]);
   }else{
    descriptor=mgbfs_hash_ref_descriptor(references[mid]);row=mgbfs_hash_ref_row(references[mid]);
   }
   const uint32_t segment=descriptor/3,bank=descriptor%3;
   if(segment>=segment_count)return MgbfsHashSeriesMatch::bad_reference;
   const auto& extent=segments[segment];
   if(!extent.states||!(extent.banks==1||extent.banks==3)||bank>=extent.banks||
      extent.capacity>(uint64_t{1}<<32)||row>=extent.capacity)return MgbfsHashSeriesMatch::bad_reference;
   const auto* states=static_cast<const State*>(extent.states);
   const uint64_t stride=extent.capacity*extent.banks,offset=extent.capacity*bank+row;
   for(uint32_t e=0;e<elements;e++){
    const State stored=states[uint64_t(e)*stride+offset];
    const int64_t candidate=query.value(e);
    if(candidate<int64_t(stored)){cmp=-1;break;}
    if(candidate>int64_t(stored)){cmp=1;break;}
   }
    if(cmp==0){
     if constexpr(ReturnHit)*hit={segment,bank,row,1};
     return MgbfsHashSeriesMatch::present;
    }
  }
  if(cmp>0)lo=mid+1;
  else hi=mid;
 }
 return MgbfsHashSeriesMatch::absent;
}

template<class State>
struct MgbfsHashArrayQuery {
 const State* states;
 MGBFS_HASH_REF_HD int64_t value(uint32_t element) const { return states[element]; }
};

template<class State,class Ref>
MGBFS_HASH_REF_HD MgbfsHashSeriesMatch mgbfs_hash_series_lookup(
 const uint64_t* keys,const Ref* references,const uint32_t* prefix_offsets,uint32_t length,
 const MgbfsBatchHashSegment* segments,uint32_t segment_count,MgbfsHashCompactRefLayout layout,
 uint64_t hash,const State* query,uint32_t elements){
 return mgbfs_hash_series_probe<State>(keys,references,prefix_offsets,length,segments,segment_count,
  layout,hash,MgbfsHashArrayQuery<State>{query},elements);
}

#undef MGBFS_HASH_REF_HD
