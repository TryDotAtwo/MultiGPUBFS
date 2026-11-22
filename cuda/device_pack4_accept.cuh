#pragma once
#include "bitmap_pack4.cuh"
#include "bitmap_owner_page.cuh"
#include "generic_action.cuh"
#include "permutation_bitmap.cuh"
#include <cuda/atomic>
#include <cstdint>

// Internal resident bitmap acceptance primitive; not a public BFS entry.
// One scheduler lane owns a logical shard for the entire transaction, including
// retries. Different shards may update different bits of the same word. Each
// extent is allocated before launch and remains live until every packet and
// export reader is drained. Host admission must validate aligned pointers,
// extents and exact owner mapping on every rank before launching the scheduler.
//
// A result of extent_full leaves the input retryable and requires a preadmitted
// extent or an orderly resource stop. A result of cap_full is a resource stop;
// every outstanding rank-cap reservation already has an exclusive physical row
// and must publish or fail the job. Bad state/metadata or an owner race is a
// fatal error, never an INCOMPLETE resource result.
namespace mgbfs_device_pack4 {
struct Extent {
    uint64_t* rows;
    uint64_t* counts; // one 64-bit count per bank
    uint64_t capacity; // per bank, in [1, 2^32]
    uint32_t banks;
    uint32_t reserved;
};
struct Owner {
    BitmapDeviceDomain domain;
    MgbfsPack4Dictionary dictionary;
    const Extent* extents;
    uint64_t extent_count;
    uint64_t* bits;
    uint64_t owner_len;
    uint64_t* rank_count; // null means no rank limit
    uint64_t rank_limit;
    uint64_t owner_boundary;
    uint32_t world,owner,shards,shard,bank,owner_page;
};
enum class Status:uint32_t { accepted,duplicate,extent_full,cap_full,invalid,owner_race };
struct Outcome {Status status;uint64_t extent;uint32_t row;};
struct WordAccessor {
    MgbfsPack4Dictionary dictionary;
    uint64_t word;
    __device__ __forceinline__ int64_t value(uint32_t,uint32_t e) const {
        return dictionary.symbols[(word>>(4*e))&15u];
    }
};
__device__ __forceinline__ Outcome accept(const Owner& a,uint64_t claimed_id,uint64_t packed) {
    const Outcome invalid{Status::invalid,0,0};
    if(!a.world||a.world>128||a.owner>=a.world||!a.shards||a.shard>=a.shards||
       !a.bits||!a.domain.total||a.domain.elements!=a.dictionary.elements||
       !mgbfs_pack4_valid_word(a.dictionary,packed)||(!a.extents&&a.extent_count)||
       (!a.rank_count&&a.rank_limit))return invalid;
    const uint64_t expected_len=a.domain.total/a.world+(a.owner<a.domain.total%a.world);
    if(a.owner_len!=expected_len)return invalid;
    const uint64_t page_span=uint64_t(a.owner_page)*a.world;
    if((!a.owner_page&&a.owner_boundary)||(a.owner_page&&
       ((a.owner_page&(a.owner_page-1))||a.owner_page>(1u<<20)||!page_span||
        a.owner_boundary!=a.domain.total-a.domain.total%page_span)))return invalid;
    const WordAccessor word{a.dictionary,packed};
    uint64_t id=0;
    if(!bitmap_rank(a.domain,word,0,&id))return invalid;
    id=mgbfs_bitmap_permute_owner_id(id,a.world,a.owner_page,a.owner_boundary);
    if(id!=claimed_id||id>=a.domain.total||id%a.world!=a.owner||
       (uint64_t(uint32_t(mix64(id)))*a.shards)>>32 != a.shard)return invalid;
    const uint64_t local=id/a.world;
    if(local>=a.owner_len)return invalid;
    const uint64_t mask=uint64_t(1)<<(local%64);
    cuda::atomic_ref<uint64_t,cuda::thread_scope_device> bits(a.bits[local/64]);
    if(bits.load(cuda::memory_order_acquire)&mask)return {Status::duplicate,0,0};

    uint64_t chosen=0,row=0;
    bool found=false;
    for(uint64_t i=0;i<a.extent_count;++i) {
        const Extent& x=a.extents[i];
        if(!x.rows||!x.counts||!x.capacity||x.capacity>(uint64_t(1)<<32)||
           !x.banks||a.bank>=x.banks||x.banks>3||x.reserved)return invalid;
        cuda::atomic_ref<uint64_t,cuda::thread_scope_device> count(x.counts[a.bank]);
        const uint64_t n=count.load(cuda::memory_order_acquire);
        if(n>x.capacity)return invalid;
        if(n<x.capacity){chosen=i;row=n;found=true;break;}
    }
    if(!found)return {Status::extent_full,0,0};
    if(a.rank_count) {
        cuda::atomic_ref<uint64_t,cuda::thread_scope_device> count(*a.rank_count);
        uint64_t current=count.load(cuda::memory_order_acquire);
        while(current<a.rank_limit) {
            if(count.compare_exchange_weak(current,current+1,
                  cuda::memory_order_acq_rel,cuda::memory_order_acquire))break;
        }
        if(current>=a.rank_limit)return {Status::cap_full,0,0};
    }
    const Extent& x=a.extents[chosen];
    x.rows[uint64_t(a.bank)*x.capacity+row]=packed;
    const uint64_t old=bits.fetch_or(mask,cuda::memory_order_release);
    if(old&mask)return {Status::owner_race,chosen,uint32_t(row)};
    cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(x.counts[a.bank])
        .store(row+1,cuda::memory_order_release);
    return {Status::accepted,chosen,uint32_t(row)};
}
}
