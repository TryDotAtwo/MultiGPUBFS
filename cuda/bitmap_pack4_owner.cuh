#pragma once
#include "bitmap_pack4.cuh"
#include "generic_route.h"
#include <cstdint>

// Experimental, single-shard atomic owner only. This is not the production ABI.
// The caller owns both incoming buffers and every spill origin until replay
// finishes. The same shard must not be processed by two streams concurrently.
// Different shards can share visited words because bit changes are atomic.
// Optional rank-wide quota is shared by all shard streams. A CAP_RETRY
// flag requires the caller to drain all streams and replay retained origins;
// it is not itself proof that the rank's final limit has been reached.
struct MgbfsPack4OwnerArgs {
    MgbfsPack4Dictionary dictionary{};
    uint32_t elements=0,state_bytes=0,packed=0;
    const void *local=nullptr,*remote=nullptr;
    const GenericRouteRecord *local_meta=nullptr,*remote_meta=nullptr;
    const uint32_t *local_counts=nullptr,*remote_counts=nullptr;
    uint32_t rank=0,world=0,owner=0;
    uint64_t total=0,owner_len=0;
    uint64_t* bits=nullptr;
    uint32_t shard=0,shards=0,q=0,visible=0;
    const uint32_t* selection=nullptr;
    uint32_t selected=0;
    uint64_t capacity=0;
    uint32_t banks=0,bank=0;
    uint64_t *rows=nullptr,*accepted=nullptr;
    uint64_t* global_count=nullptr;
    uint64_t global_limit=0;
    uint32_t *spill=nullptr,*spill_count=nullptr,*error=nullptr;
#ifdef MGBFS_PACK4_OWNER_TEST_BARRIER
    uint32_t* test_gate=nullptr;
#endif
};
extern "C" int mgbfs_pack4_owner_accept(MgbfsPack4OwnerArgs args,void* stream);
// Single rebased input queue. The bitmap still uses the actual rank, world,
// owner and shard geometry. Selection and spill origins are packet-local.
extern "C" int mgbfs_pack4_owner_accept_packet_v1(MgbfsPack4OwnerArgs args,void* stream);
// The same rebased queue, with 16-byte rank/nibble pairs at local_meta.
extern "C" int mgbfs_pack4_owner_accept_wire_v1(MgbfsPack4OwnerArgs args,void* stream);

// Seed requires exclusive access to its bitmap word and an initialized
// accepted counter. A malformed or duplicate seed publishes nothing.
struct MgbfsPack4SeedArgs {
    MgbfsPack4Dictionary dictionary{};
    const int64_t* input=nullptr;
    uint64_t *row=nullptr,*bits=nullptr,*accepted=nullptr;
    uint32_t* error=nullptr;
    uint64_t local=0,owner_len=0;
};
extern "C" int mgbfs_pack4_seed(MgbfsPack4SeedArgs args,void* stream);
