# One-run CUDA startup investigation

Install the [pinned GitHub package](install-native-package.md) and read the
[measured diagnostic acceptance](diagnostic-build-acceptance-2026-10-14.md).

The small package carries a pinned index, not four complete CUDA libraries. Ordinary native preparation fetches the common component and one compatible production library. Optional alternatives download only when their diagnostic case is reached; a declaration without installed bytes is explicitly `PINNED_DECLARATION_NOT_INSTALLED`, not verified execution.

| Diagnostic build | CUDA runtime linkage | Native cubins |
|---|---|---|
| shared-sm120 (historical production ID) | Shared | Selected per-SM image, or universal when needed |
| shared-universal | Shared | SM75/80/86/89/90/100/103/120 + compute80 PTX |
| static-sm120 | Static | SM75/80/86/89/90/100/103/120 + compute80 PTX |
| baseline-static-legacy | Static | SM75, SM86 |
| shared-legacy | Shared | SM75, SM86 |

Runtime-only comparisons use **shared-universal versus static-sm120**, keeping architecture coverage identical. The compact per-SM production image is a separate selection/native-code contrast. Older full v18 wheels retain their original four-entry inventory. All components use the same CUDA kernel/helper fingerprint. Only diagnostic child environments select alternatives; downloading them leaves the production manifest, library and profile identity unchanged. No compiler or administrator access is required on the target server.

The Rust executable links shared libcudart in every comparison. This matrix
checks whether outcomes depend on the library's runtime linkage, native multi-architecture
images or their combination. Separate static runtimes are supported; finding
two instances does not itself prove the cause. Explicit shared linkage in
our library also does not guarantee every vendor library uses that instance.

Package assembly checks ELF NEEDED/local symbols, cubin/PTX inventory,
SHA256 of every tracked CUDA input except CMakeLists.txt, actual complete source trees, build-recipe identity and artifact SHA256. The legacy recipe differs only in default CMake settings; runtime linkage and architecture flags are explicit and independently checked in the resulting ELF/cubin/PTX inventories. Actual tree IDs remain distinct in each receipt. Each diagnostic child verifies hashes and
the library paths actually resolved by its loader before making GPU calls.
Production direct API/kernel, resident inventory and exact S3 BFS run first
on every selected GPU. `production_bfs_acceptance` records that finite gate.
The normal loading matrix, modes, multi-GPU tests and finite catalogue follow;
legacy build contrasts run last so a problematic JIT/startup cannot prevent
the earlier production checks. The full report retains legacy failures;
the production gate and each actual BFS result remain separately visible.
There is no silent production fallback.
For each selected GPU it covers direct EAGER API/kernel probes, real resident
native inventory (two requests to one worker) and tiny S3 BFS with an
independent CPU layer/terminal-state oracle.

The C++ hardware wrapper emits separate flushed entries and returns around
cudaSetDevice and cudaGetDeviceProperties. A timed-out process therefore has
a call boundary, an API log, loaded driver/runtime/JIT libraries and owned
process/thread snapshots where permitted. Missing access is reported rather
than interpreted as a denial. Native CLI EAGER policy stays unchanged;
EAGER/LAZY loading contrasts in the broader suite use direct API probes.

Run from a fresh process inside the assigned GPU job:

```sh
python -m multigpubfs.diagnostics --deep
```

The execution budget is 1800 seconds. Cleanup and metadata finalization have
a small additional bounded allowance. GPU visibility/allocation is preserved.
Only diagnostic-owned process groups are stopped; real BFS and unrelated
processes are not restarted or killed. Untested and budget-skipped cases stay
explicit. Unsupported GEMM or uncompiled LSA is not a successful execution.

If `working_configuration: READY` is present, use its verified command and preserve the proof directory. When no working recipe is found, return the single `diagnostics.zip` privately. It includes `report.json`,
`deep-summary.txt`, `deep-analysis.json`, build evidence, API/worker/native
logs, call boundaries and process snapshots, with a checksum manifest. It
excludes graph-state payloads, raw case inputs and full job environment.
Library/server paths and NCCL connection metadata may still be present.

Interpretation:

* ALL_PASS: this finite synthetic scenario did not reproduce the hang.
* VARIANT_DEPENDENT: some builds pass and others fail; inspect controlled
  runtime/native-fatbinary pairs and exact call boundaries.
* ALL_FAILED: none of the applicable builds passed that scenario.
* INCOMPLETE: missing, unavailable or budget-skipped comparisons prevent a
  complete contrast.

These outcomes establish observations, not a unique root cause. The test
preserves the inherited JIT cache, so cold/warm or transient effects remain
possible. Native matching cubins avoid requiring PTX JIT for those matching
packaged kernels; they do not eliminate every vendor library's initialization
or JIT. The finite suite cannot guarantee coverage of every graph/API
parameter, multi-node transport, security policy or inaccessible kernel state.
It is a startup/correctness investigation, not a throughput benchmark.

Both ordinary and deep doctor can recover the visible logical GPU set from agreeing independent driver/runtime probes after a native-count failure. They retain the original failure and execute the same real-BFS/printed-command gate before declaring a recipe READY. See [current release acceptance](selective-native-acceptance-2026-10-15.md).
