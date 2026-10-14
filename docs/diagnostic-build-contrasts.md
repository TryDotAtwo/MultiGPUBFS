# One-run CUDA startup investigation

The extended `doctor --deep` package carries four CUDA libraries from the
same CUDA source tree. Only isolated diagnostic child processes select them.
The ordinary executable and default CUDA library remain byte-identical to
the accepted v14 baseline. No build tool, administrator access or GPU reset
is required on the target server.

| Diagnostic build | Runtime linked by CUDA library | Native cubins |
|---|---|---|
| baseline-static-legacy | Static | SM75, SM86 |
| shared-legacy | Shared | SM75, SM86 |
| static-sm120 | Static | SM75, SM86, SM120 |
| shared-sm120 | Shared | SM75, SM86, SM120 |

The Rust executable links shared libcudart in every comparison. This matrix
checks whether outcomes depend on the library's runtime linkage, native SM120
images or their combination. Separate static runtimes are supported; finding
two instances does not itself prove the cause. Explicit shared linkage in
our library also does not guarantee every vendor library uses that instance.

Package assembly checks ELF NEEDED/local symbols, cubin/PTX inventory,
source-tree identity and SHA256. Each diagnostic child verifies hashes and
the library paths actually resolved by its loader before making GPU calls.
The comparison section is prioritized before the broader synthetic suite.
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

Return the single `diagnostics.zip` privately. It includes `report.json`,
`deep-summary.txt`, `deep-analysis.json`, build evidence, API/worker/native
logs, call boundaries and process snapshots, with a checksum manifest. It
excludes graph-state payloads, raw case inputs and full job environment.
Library/server paths and NCCL connection metadata may still be present.

Interpretation:

* ALL_PASS: this finite synthetic scenario did not reproduce the hang.
* VARIANT_DEPENDENT: some builds pass and others fail; inspect controlled
  runtime/SM120 pairs and exact call boundaries.
* ALL_FAILED: none of the four builds passed that scenario.
* INCOMPLETE: missing, unavailable or budget-skipped comparisons prevent a
  complete contrast.

These outcomes establish observations, not a unique root cause. The test
preserves the inherited JIT cache, so cold/warm or transient effects remain
possible. Native SM120 cubins avoid requiring PTX JIT for those matching
packaged kernels; they do not eliminate every vendor library's initialization
or JIT. The finite suite cannot guarantee coverage of every graph/API
parameter, multi-node transport, security policy or inaccessible kernel state.
It is a startup/correctness investigation, not a throughput benchmark.
