# Existing HF streaming launcher: rank-owner integration

The existing `streamed_bfs_launcher.py` previously rejected CUCO_RANK and
could not express LSA, physical source-bank count, completion epoch window,
independent StateRing capacity or extent-descriptor capacity. It therefore
could not launch the current target path with an explicit archive/HF plan.

The launcher now accepts CUCO_RANK and forwards these settings through the
existing native runtime environment ABI, before process creation. Physical
payload banks and K completion credits remain independently configurable.
Legacy defaults remain host-sized NCCL, with unchanged native defaults for
omitted capacities. No buffer padding, runtime fallback or new transport was
introduced. CUCO_INDEXED+LSA and CUCO_RANK+HASH_FIRST+host-sized NCCL are
rejected before launch, matching ReferenceSelection::with_transport.

Two behavioral RED/GREEN tests establish forwarding and early rejection.
Negative cases cover unsupported banks, bool-as-int, insufficient credits,
zero StateRing and oversized descriptors. Full scoped Python suite:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests scripts -x -q
-p no:cacheprovider`: 261 passed, 14 skipped, 13.64 seconds.

This establishes launcher configuration integration only. Linux FIFO test
is skipped on Windows. No actual multi-GPU/HF run, target-hardware failure
gate, CUDA sanitizer gate or timeline was performed for this change.
Native device admission remains authoritative for VRAM/NCCL constraints.
