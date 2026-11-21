# S13 Coxeter: 8 × RTX 3060, bitmap warp

Status: COMPLETE. This is one full run; the other strategy cells are still being measured. Identity start, adjacent transpositions, 8 PCIe GPUs across two NUMA nodes without NVLink. Graph contains 6,227,020,800 states, 79 nonempty layers and eccentricity 78.

BFS: 1483.764569 s. Completed target depths 34–44: 703.619193 s for 3,052,643,942 new states and 36,412,333,584 generated edges. One run, with two route retries. NVML 1 s sampled maximum: 2023 MiB per GPU; it can miss brief peaks. Worst rank peak live rows 111,623,259; exact accounted peak segment storage 1,635,779,520 bytes, excluding bitmap, queues, scratch, CUDA and NCCL.

All layer counts match the independent Mahonian oracle. The final two layers were compared by full states, with cross-rank duplicates checked; earlier layers were checked by counts. The corresponding compressed evidence contains metadata and memory samples, not graph states. Source checkout d067734f3e9f012e3e394ccf712f123867c00e0c, CUDA library SHA256 c567135bde4bdf014b6137c90956f6ad44fd7f7fea8a2c075e3425213fdf970e, executable SHA256 82c40088a24bb8620a9fdc3e8272646546a41d49773739433bb6faa19c1198e3. Geometry: 8 shards, 4 owner lanes, batch 8192, exact exchange, segment cap 2^32 rows, allocated segment size 1,048,576, balanced preset, autotune disabled.

A separate server-to-server transfer was attempted during this run; it may have affected shared host resources. Treat timing as a single potentially affected observation, not a B200 or scaling claim.

Evidence SHA256: `70f9c48e53c846d425b5f79c07125e75fa2b50148f3e24d8db178fc8aea90e1b`.
