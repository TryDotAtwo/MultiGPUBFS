#pragma once

#include "generic_route.h"

constexpr uint32_t packed_route_max_elements = 24;

// The packed route ABI stores up to 24 exact bytes after the identity.
struct PackedRouteState {
    uint64_t low = 0;
    uint64_t middle = 0;
    uint64_t high = 0;

    __device__ __forceinline__ void append(uint32_t index, uint64_t value) {
        if (index < 8) low |= value << (8 * index);
        else if (index < 16) middle |= value << (8 * (index - 8));
        else high |= value << (8 * (index - 16));
    }

    __device__ __forceinline__ GenericRouteRecord record(uint64_t identity) const {
        return {identity, low, uint32_t(middle), uint32_t(middle >> 32),
                uint32_t(high), uint32_t(high >> 32)};
    }
};

template<class Action>
__device__ __forceinline__ GenericRouteRecord pack_route_state(
    const Action& action, uint32_t child, uint64_t identity) {
    PackedRouteState packed;
    for (uint32_t index = 0; index < action.elements; ++index) {
        packed.append(index, uint64_t(action.value(child, index)));
    }
    return packed.record(identity);
}
