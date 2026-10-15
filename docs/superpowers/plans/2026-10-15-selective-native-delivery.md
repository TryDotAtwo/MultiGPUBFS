# Selective native delivery

Approved user workflow: one small pip package, automatic GPU/driver discovery, download only compatible precompiled native components from GitHub, check SHA256, cache, and oracle-verified doctor command. No separate user loader, build toolkit, HF, paid rental, local product development/build or subagents.

The CayleyPy API and BFS algorithms remain unchanged. Import/activation must remain usable without a GPU (for graph definitions and preinstalled container images); the first actual native request and doctor prepare missing native components. An explicit executable or installed full native bundle remains supported.

## Delivery

Pure Python wheel/source pin an immutable GitHub component index. A common archive supplies the accepted Rust CLI, private cudart/NCCL and licenses. Eight shared-runtime libraries compile the unchanged CUDA inputs separately for native SM75/80/86/89/90/100/103/120. A same-source universal shared library with ordinary compute80 PTX handles mixed/new compatible GPUs. Static/legacy diagnostic components are fetched only on selection. No compiler runs on the user's GPU host.

Independent CUDA Driver API discovery runs in a bounded child, preserves CUDA_VISIBLE_DEVICES and uses logical device IDs. Selection requires compatibility with every visible GPU. Lack of GPU access is reported; management count is not treated as a CUDA proof. Unsupported platforms/drivers are explicit, not silent CPU fallback.

## Cache integrity

Use XDG_CACHE_HOME or user cache, optional MGBFS_NATIVE_CACHE. Pin URLs, SHA256, byte sizes and per-file hashes in the installed package. Lock owned cache entries across processes, stream downloads to .part files, verify before atomic publication, prohibit archive path escapes/symlinks/devices, and reuse common content. Complete cached payload works offline; corruption is detected. Download errors identify component/phase and do not become BFS query timeouts. Diagnostic alternatives do not invalidate a verified production manifest/profile.

## Acceptance

CPU negative/positive tests: exact/compatible/mixed selection, driver visibility, timeout isolation, wrong hashes/sizes/URLs/archive paths, interruption/atomic cache publication, concurrent callers, no repeated fetch on warm cache, deferred diagnostics and unchanged installed-bundle/explicit override behavior. Full existing CPU suite; clean wheel/source installs without CUDA/GPU.

Private 2xT4 acceptance: install small public GitHub wheel, cold prepare must fetch only common+SM75; complete tiny oracle BFS, ordinary doctor and its actual READY command, deferred static probe, warm/offline new process, persistent S11/S12 with all Mahonian layers and terminal states. Record actual bytes/timings/component identities. Other SM builds have inventory evidence only until run on real hardware.

## Plan

1. Add CPU target-selection regression, then scoped build helper and eight-way CPU-only native build.
2. Add downloader/cache/hardware tests before implementation; integrate central native root resolution and diagnostic bootstrap.
3. Package common, per-SM, universal and optional controls from CI/accepted v18; publish immutable components and pin index.
4. Build/publish small Python wheel; real installed GPU acceptance; update source/docs/examples and publish sanitized receipts.
