# Experimental pack4 codec leaf — CPU only

This branch contains an isolated prototype of a 4-bit-per-symbol resident
permutation codec. It is **not wired into BFS, the public API, the wheel, the
CUDA build or any default**. It has not been compiled with nvcc or run on a
GPU. In particular, this commit provides no S14/S15 capacity claim.

On the remote eight-RTX-3060 host, the standalone CPU test was compiled with:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror tests/bitmap_pack4_cpu.cpp -o /workspace/mgbfs-pack4-cpu
/workspace/mgbfs-pack4-cpu
```

It reported `PACK4_CPU_EXACT_PASS permutations=40320 edges=282240`. It covers
all S8 permutations and adjacent swaps, signed symbols, one repeated-symbol
case, the sixteenth nibble, and several invalid inputs. It is a codec leaf
test, not a comparison of every BFS layer or a GPU performance result.

The CPU prototype and logs were saved in an archive with SHA-256
`ae3de5d7eadbda4b20ea775025943433554a4d036aba4e3122a47b6ef9f6b979`.
Follow the architecture and verification plan before integrating it.
