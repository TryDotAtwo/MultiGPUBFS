# Hardware-derived generic BFS planning

The general GPU planner reads free VRAM, SM count, L2 size and compute capability from each selected CUDA device. In external-rank launches each rank queries its own device; hardware metadata is carried into the shared launch receipt.

State capacity comes from byte-exact admission, with max(1 GiB, 10% free VRAM) reserve. The previous 2^28 state cap is removed. Table slot counts and saved slot positions are 64-bit. Canonical row indices remain 32-bit: the three-bank arena is bounded by its row-address range, not a device-independent throughput limit. Serialized plans are revalidated before launch.

Hash tokens preserve full 32-bit row indices, a 31-bit tag and a pending-origin flag. Tag equality never substitutes for full state equality. Wide table symbols use a separate native ABI; old libraries cannot satisfy the new symbols accidentally.

Candidate shard counts derive from admitted state bytes, L2 working-set scale and SM count. Each candidate must pass native memory admission. Bounded same-graph GPU pilots compare identical completed layer prefixes, including compatible history, transport and CUDA/GEMM alternatives. The pilot buffer limit limits startup work only: production is admitted again against physical VRAM. Candidate selection is heuristic plus measurement, not proof of a global optimum.

Small/large frontier thresholds scale with selected SM count. Online chunk measurements refine batch and compatible generator choices at layer boundaries. Shard/history geometry is chosen at startup and is not rebuilt within a live layer.

Actual Blackwell acceptance requires a GPU test on that hardware; large-memory synthetic admission and SM86 correctness do not establish B200 throughput.
