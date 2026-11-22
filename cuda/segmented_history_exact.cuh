#pragma once
#include "segmented_history.cuh"
#include <cstdint>

// Shared full-state comparison for existing stream and resident owners.
// The hash prefix only avoids some row reads; it is never proof of equality.
namespace mgbfs_segmented_exact {
constexpr uint64_t empty=~uint64_t(0);
constexpr uint64_t retired=empty-1;
constexpr uint64_t pending=uint64_t(1)<<63;
constexpr uint64_t prefix=0x7ffffffc00000000ULL;
constexpr uint64_t row_mask=(uint64_t(1)<<32)-1;
constexpr uint64_t payload_mask=(uint64_t(1)<<34)-1;
constexpr uint64_t local_rows=uint64_t(1)<<32;
enum class Comparison:uint32_t { different,equal,invalid };
template<class T,class Input>
__device__ inline Comparison compare_row(const Input& input,uint32_t child,
        const MgbfsSegment& seg,uint64_t payload) {
    const uint64_t row=payload&row_mask;
    const uint32_t bank=uint32_t((payload>>32)&3u);
    if((payload&~payload_mask)||bank>=seg.banks||row>=seg.capacity)
        return Comparison::invalid;
    const auto* states=static_cast<const T*>(seg.states);
    const uint64_t stride=seg.capacity*seg.banks;
    const uint64_t local=uint64_t(bank)*seg.capacity+row;
    for(uint32_t e=0;e<input.elements;++e)
        if(input.value(child,e)!=int64_t(states[uint64_t(e)*stride+local]))
            return Comparison::different;
    return Comparison::equal;
}
}
