#pragma once
#include "segmented_history_exact.cuh"
#include "generic_action.cuh"
#include <cuda/atomic>
#include <cstdint>

// Internal primitive for a future GPU-resident hash scheduler. Not a public
// BFS entry. The caller must supply the SAME full-state hash for equal states,
// enforce one stable rank/shard per key, and hold an exclusive shard lease
// through lookup, row publication and replay. A table has no concurrent
// readers/writers outside that lease. Descriptor ranges, counts and storage
// for different extents must not overlap and remain live through the job.
//
// A shared pool lease may be returned only when all banks are empty and the
// table contains only EMPTY/RETIRED slots after checked retirement. An extent_full outcome may be transient while another
// shard holds an unused lease; it must be retried after other owners drain
// before it is promoted to a global resource stop. No input has been consumed
// by extent_full/cap_full/table_full. On invalid metadata the whole job fails.
//
// Segment rows are local u32 values, including UINT32_MAX; segment counts and
// indices remain u64. The existing MgbfsSegment layout and exact row compare
// are used, so a future scheduler can retain the ordinary archive format.
namespace mgbfs_device_hash {
using namespace mgbfs_segmented_exact;
struct Extent { MgbfsSegment segment; uint64_t* counts; };
struct Owner {
    const Extent* extents;
    uint64_t extent_count;
    uint32_t* extent_owner; // UINT32_MAX denotes a drained, reset extent
    uint64_t* rank_count;   // optional rank-wide target/history hard cap
    // Optional job-owned index. The shard lease exclusively protects its
    // head/cursor; each leased extent owns one next entry.
    uint64_t* shard_head;
    uint64_t* extent_next;
    uint64_t* free_cursor;
    uint64_t rank_limit;
    uint32_t elements,shards,shard,bank;
};
enum class Status:uint32_t { accepted,duplicate,extent_full,cap_full,table_full,invalid };
struct Outcome { Status status; uint64_t extent; uint32_t row; };
struct Probe { bool valid,duplicate; uint64_t slot,row; };
template<class Input> struct InputView {
    const Input& input;uint32_t elements;
    __device__ int64_t value(uint32_t,uint32_t e) const { return input.value(e); }
};
__device__ inline bool valid_extent(const Extent& e,uint32_t bank,uint32_t elements,uint64_t type_size) {
    const MgbfsSegment& s=e.segment;
    if(!e.counts||!s.slots||!s.states||!s.slots_count||
       (s.slots_count&(s.slots_count-1))||!s.capacity||s.capacity>local_rows||
       !(s.banks==1||s.banks==3)||bank>=s.banks||
       (s.banks==3&&!s.positions)||
       !(s.position_item_bytes==0||s.position_item_bytes==4||s.position_item_bytes==8)||
       (s.position_item_bytes==4&&s.slots_count>local_rows)||!elements)return false;
    const uint64_t stride=s.capacity*s.banks;
    return stride<=UINT64_MAX/elements && stride*elements<=UINT64_MAX/type_size;
}
template<class T,class Input>
__device__ inline Probe probe(const Extent& e,uint32_t bank,uint32_t elements,uint64_t hash,const Input& input) {
    const MgbfsSegment& s=e.segment;
    uint64_t count=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(e.counts[bank])
        .load(cuda::memory_order_acquire);
    if(count>s.capacity)return {false,false,0,0};
    const uint64_t start=mgbfs_shared_table_bucket(hash,s.slots_count);
    const uint64_t wanted=hash&prefix;
    uint64_t first=s.slots_count;
    for(uint64_t step=0;step<s.slots_count;++step) {
        const uint64_t slot=(start+step)&(s.slots_count-1);
        const uint64_t seen=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(s.slots[slot])
            .load(cuda::memory_order_acquire);
        if(seen==retired) { if(first==s.slots_count)first=slot; continue; }
        if(seen==empty) { if(first==s.slots_count)first=slot; break; }
        if(seen&pending)return {false,false,0,0};
        const uint64_t payload=seen&payload_mask;
        const uint32_t row_bank=uint32_t((payload>>32)&3u);
        const uint64_t row=payload&row_mask;
        if(row_bank>=s.banks||row>=s.capacity||
           row>=cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(e.counts[row_bank])
                    .load(cuda::memory_order_acquire))return {false,false,0,0};
        if((seen&prefix)==wanted) {
            const InputView<Input> view{input,elements};
            const Comparison c=compare_row<T>(view,0,s,payload);
            if(c==Comparison::invalid)return {false,false,0,0};
            if(c==Comparison::equal)return {true,true,0,count};
        }
    }
    return {true,false,first,count};
}
template<class T,class Input>
__device__ inline Outcome accept(const Owner& a,uint64_t hash,const Input& input) {
    static_assert(sizeof(T)==1||sizeof(T)==8,"only existing u8/i64 rows are supported");
    const Outcome bad{Status::invalid,0,0};
    if(!a.extents||!a.extent_count||!a.elements||!a.shards||a.shard>=a.shards||
       (!a.rank_count&&a.rank_limit))return bad;
    // A failed conversion must not create a stored state unequal to its input.
    for(uint32_t e=0;e<a.elements;++e)
        if(int64_t(static_cast<T>(input.value(e)))!=input.value(e))return bad;

    uint64_t chosen=a.extent_count,chosen_slot=0,chosen_row=0;
    bool any_table_full=false;
    const bool indexed=a.shard_head!=nullptr;
    if(indexed&&(!a.extent_next||!a.free_cursor||!a.extent_owner))return bad;
    uint64_t i=indexed?a.shard_head[a.shard]:0;
    uint64_t traversed=0;
    while(i<a.extent_count) {
        if(++traversed>a.extent_count)return bad;
        const uint64_t next=indexed?a.extent_next[i]:i+1;
        if(a.extent_owner&&cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.extent_owner[i])
             .load(cuda::memory_order_acquire)!=a.shard) {
            if(indexed)return bad;
            i=next;continue;
        }
        const Extent& e=a.extents[i];
        if(!valid_extent(e,a.bank,a.elements,sizeof(T)))return bad;
        const Probe p=probe<T>(e,a.bank,a.elements,hash,input);
        if(!p.valid)return bad;
        if(p.duplicate)return {Status::duplicate,i,0};
        if(p.row<e.segment.capacity) {
            if(p.slot==e.segment.slots_count)any_table_full=true;
            else if(chosen==a.extent_count) { chosen=i;chosen_slot=p.slot;chosen_row=p.row; }
        }
        i=next;
    }
    if(indexed&&i!=UINT64_MAX)return bad;
    bool new_lease=false;
    if(chosen==a.extent_count&&a.extent_owner) {
        const uint64_t start=indexed?a.free_cursor[a.shard]:0;
        if(start>=a.extent_count)return bad;
        for(uint64_t step=0;step<a.extent_count;++step) {
            // Compute (start+step) modulo extent_count without u64 overflow.
            const uint64_t i=step<a.extent_count-start?start+step:step-(a.extent_count-start);
            auto lease=cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.extent_owner[i]);
            uint32_t expected=UINT32_MAX;
            if(!lease.compare_exchange_strong(expected,a.shard,cuda::memory_order_acq_rel,cuda::memory_order_acquire))
                continue;
            const Extent& e=a.extents[i];
            if(!valid_extent(e,a.bank,a.elements,sizeof(T)))return bad;
            for(uint32_t b=0;b<e.segment.banks;++b)
                if(cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(e.counts[b])
                      .load(cuda::memory_order_acquire)!=0)return bad;
            const Probe p=probe<T>(e,a.bank,a.elements,hash,input);
            if(!p.valid||p.duplicate||p.row!=0||p.slot==e.segment.slots_count)return bad;
            chosen=i;chosen_slot=p.slot;chosen_row=0;new_lease=true;
            if(indexed)a.free_cursor[a.shard]=(i+1<a.extent_count)?i+1:0;
            break;
        }
    }
    if(chosen==a.extent_count)return {any_table_full?Status::table_full:Status::extent_full,0,0};
    if(a.rank_count) {
        cuda::atomic_ref<uint64_t,cuda::thread_scope_device> total(*a.rank_count);
        uint64_t count=total.load(cuda::memory_order_acquire);
        while(count<a.rank_limit)
            if(total.compare_exchange_weak(count,count+1,cuda::memory_order_acq_rel,cuda::memory_order_acquire))break;
        if(count>=a.rank_limit) {
            // A newly claimed but unused segment is still fully empty.
            if(new_lease) {
                uint32_t expected=a.shard;
                if(!cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(a.extent_owner[chosen])
                    .compare_exchange_strong(expected,UINT32_MAX,cuda::memory_order_release,cuda::memory_order_relaxed))
                    return bad;
            }
            return {count>a.rank_limit?Status::invalid:Status::cap_full,0,0};
        }
    }
    if(indexed&&new_lease) {
        a.extent_next[chosen]=a.shard_head[a.shard];
        a.shard_head[a.shard]=chosen;
    }
    const Extent& e=a.extents[chosen];
    const MgbfsSegment& s=e.segment;
    const uint64_t local=uint64_t(a.bank)*s.capacity+chosen_row;
    const uint64_t stride=s.capacity*s.banks;
    auto* states=static_cast<T*>(s.states);
    for(uint32_t k=0;k<a.elements;++k)states[uint64_t(k)*stride+local]=static_cast<T>(input.value(k));
    if(s.positions) {
        if(s.position_item_bytes==4)reinterpret_cast<uint32_t*>(s.positions)[local]=uint32_t(chosen_slot);
        else s.positions[local]=chosen_slot;
    }
    const uint64_t token=(hash&prefix)|(uint64_t(a.bank)<<32)|chosen_row;
    cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(s.slots[chosen_slot]).store(token,cuda::memory_order_release);
    cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(e.counts[a.bank]).store(chosen_row+1,cuda::memory_order_release);
    return {Status::accepted,chosen,uint32_t(chosen_row)};
}
}
