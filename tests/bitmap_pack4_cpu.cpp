#include "../cuda/bitmap_pack4.cuh"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>

int main() {
    MgbfsPack4Dictionary d{};
    d.elements = 8;
    d.distinct = 8;
    const int64_t symbols[] = {-9000000000LL, -31, -1, 0, 9, 25, 789, 40000000000LL};
    std::copy(symbols, symbols + 8, d.symbols);
    assert(mgbfs_pack4_valid(d));
    std::array<int, 8> order{};
    std::iota(order.begin(), order.end(), 0);
    uint64_t count = 0;
    do {
        int64_t original[16]{}, restored[16]{};
        for (int i = 0; i < 8; ++i) original[i] = symbols[order[i]];
        uint64_t word = 0;
        assert(mgbfs_pack4_encode(d, original, &word));
        assert(mgbfs_pack4_decode(d, word, restored));
        assert(std::equal(original, original + 8, restored));
        for (int i = 0; i < 7; ++i) {
            uint8_t perm[16]{};
            for (int j = 0; j < 8; ++j) perm[j] = uint8_t(j);
            std::swap(perm[i], perm[i + 1]);
            uint64_t changed = 0, expected = 0;
            int64_t independent[16]{};
            for (int j = 0; j < 8; ++j) independent[j] = original[perm[j]];
            assert(mgbfs_pack4_apply(d, word, perm, &changed));
            assert(mgbfs_pack4_encode(d, independent, &expected));
            assert(changed == expected);
        }
        ++count;
    } while (std::next_permutation(order.begin(), order.end()));
    assert(count == 40320);

    MgbfsPack4Dictionary wide{};
    wide.elements = 16;
    wide.distinct = 16;
    for (int i = 0; i < 16; ++i) wide.symbols[i] = int64_t(i) * 1000000000LL - 7000000000LL;
    int64_t row[16]{};
    for (int i = 0; i < 16; ++i) row[i] = wide.symbols[i];
    uint64_t last = 0;
    assert(mgbfs_pack4_encode(wide, row, &last));
    assert((last >> 60) == 15);
    int64_t got[16]{};
    assert(mgbfs_pack4_decode(wide, last, got));
    assert(std::equal(row, row + 16, got));

    MgbfsPack4Dictionary repeated{};
    repeated.elements = 5;
    repeated.distinct = 2;
    repeated.symbols[0] = -4;
    repeated.symbols[1] = 17;
    int64_t rep[] = {17, -4, 17, -4, 17};
    uint64_t repword = 0;
    assert(mgbfs_pack4_encode(repeated, rep, &repword));
    assert(mgbfs_pack4_decode(repeated, repword, got));
    assert(std::equal(rep, rep + 5, got));
    assert(!mgbfs_pack4_decode(repeated, repword | (uint64_t(1) << 20), got));
    assert(!mgbfs_pack4_decode(repeated, (repword & ~uint64_t(15)) | uint64_t(2), got));
    int64_t unknown[] = {17, -4, 99, -4, 17};
    assert(!mgbfs_pack4_encode(repeated, unknown, &repword));
    uint8_t invalid_perm[] = {0, 1, 1, 3, 4};
    assert(!mgbfs_pack4_apply(repeated, 0, invalid_perm, &repword));
    MgbfsPack4Dictionary bad = repeated;
    bad.symbols[1] = -4;
    assert(!mgbfs_pack4_valid(bad));
    bad = repeated; bad.elements = 17;
    assert(!mgbfs_pack4_valid(bad));
    bad = repeated; bad.distinct = 0;
    assert(!mgbfs_pack4_valid(bad));
    std::cout << "PACK4_CPU_EXACT_PASS permutations=" << count << " edges=" << count*7 << "\n";
}
