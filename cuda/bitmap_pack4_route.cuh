#pragma once
#include "bitmap_pack4.cuh"
#include "permutation_bitmap.cuh"
#include "generic_route.h"
#include <cstdint>

// Experimental producer, not the production ABI. Output is the existing
// GenericRouteRecord and either embedded u8 values or a separate i64 SoA.
// Input parent rows stay alive through the whole route kernel. On queue
// overflow output is invalid and the caller must collectively retry the input.
struct MgbfsPack4RouteArgs {
    MgbfsPack4Dictionary dictionary{};
    BitmapDeviceDomain domain{};
    const uint64_t* parents=nullptr;
    const uint32_t* permutations=nullptr;
    uint32_t count=0,generators=0,world=0,source=0,shards=0,capacity=0;
    uint32_t state_bytes=0,grouped=0;
    const uint32_t* map=nullptr;
    GenericRouteRecord* records=nullptr;
    void* payload=nullptr;
    uint32_t *counts=nullptr,*error=nullptr;
    uint32_t owner_page=0;
    uint64_t owner_boundary=0;
};
extern "C" int mgbfs_pack4_route(MgbfsPack4RouteArgs args,void* stream);
