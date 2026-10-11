# Resident profile / allocation audit — 2026-10-11

Base: c7500d7d5ea168a0dd91b5b86be4c9d844f1ef04. Validation: eight physical RTX 3060 12 GiB on one host, SM86 native library d1033ed77b025491484fff37fb1d5f9237e5783b81a18dd4ff03c62da6bad2a5.

## Obligation, change, evidence, scope

| Obligation | Implementation | Evidence | Limit |
|---|---|---|---|
| Avoid repeated admission | One-use launch transaction tied to worker generation, job sequence, graph, options and dependency stamps | transaction negative unit tests, real eight-rank launches | incompatible geometry uses real admission |
| Reuse actual allocations | Constructor allocation multiset, exact sizes and multiplicities; retain requested buffers, release others; reject incomplete consumption | n9→n9r2→n14→n9; wide matrix full/parent/radix regression | generic HASH; SORTED keeps original layout checks |
| Prove cached capacity exists | All-rank successful retired job, parked communicator, raw manifest equality, required device streams/events; native proof recheck before reuse | missing-proof GPU rejection and failed-group no-restart test | GEMM private workspace freshly checks free VRAM and may allocate |
| Fast compatible pair selection | Same LRX action and state width, orbit-bound gate; admitted target resource proof, source provenance with measured=false | different n9r2 target, repeats, final clean timings | source performance reuse does not establish target global optimum |
| Correct layer-size cache | Persist and match max-rank frontier bucket; validate unique 0..31 buckets independently of ordered global base tiers | repeated full n9 CPU oracle; duplicate global boundary native regression | missing tiers retain conservative policy / bounded online probes |
| Avoid repeated handle setup and polling | Device-bound quiescent stream/event reuse; inotify job/bootstrap wakeup with bounded fallback | same worker PIDs, handle hits, Rust wait tests | Python receipt polling and private GEMM setup remain |
| Avoid per-rank repeated identity lookup | Select group once, bind generation/workers/native/devices; retain process-liveness checks | group-selection negative tests, final eight-rank run | no claimed isolated speedup |
| Preserve content correctness | Exact layer counts and retained current/previous states independently checked | small LRX and wide matrix CPU oracles; same processes throughout | n14 is capacity-limited, not a complete graph |

## Timing boundaries

Clean request→latest rank BFS timestamp includes all launch preparation; profile timing is a subset. CPU oracles run after timed GPU jobs. No states are transferred to the Windows computer.

| Pair | Profile ms | Request→BFS ms | Whole job ms |
|---|---:|---:|---:|
| startup9 | 18402.734 | 18485.143 | 18598.962 |
| different9r2 | 2.554 | 47.251 | 142.433 |
| repeat9r2 | 5.363 | 46.924 | 146.795 |
| large14 | 134.943 | 180.681 | 314.348 |
| return9 | 27.431 | 80.430 | 184.324 |
| repeat9 | 3.146 | 52.821 | 164.050 |

Cold startup includes tuning and workers, about 18.4 s; this is not the warmed switch cost. Compatible warmed transitions are 47–53 ms; profile choice 2.6–5.4 ms. First n14 geometry transition about 181 ms, then returning to n9 about 80 ms. Baseline c750 warm n9 profile ~30.94 ms and request→BFS ~106.37 ms from prior evidence; these are individual observations, not a statistically matched heavy-graph speedup claim. The bounded n14 prefix must not be compared to a full or larger-capacity run.

A separate clean pre-group-selection run measured 2.2–2.5 ms selection and 35–49 ms transitions. Host/tuning differences prevent attributing the final timing difference to that small edit. cProfile identifies repeated file polling and metadata work in the launcher; it adds overhead and is not acceptance timing.

## Verification and failures

Final selected Python suite: 50 passed; additional resident launcher/CLI tests: 3 passed. Earlier comprehensive selected suite: 50 passed. Runtime CUDA unit suite: 61 passed; earlier combined CLI/runtime suite: 64 passed. Release CLI compiled with --features cuda and locked dependencies.

The weighted BMMA test previously exercised an explicitly SM75-only backend on SM86; the test now runs that backend only on hardware reporting SM75, retaining all CUB correctness cases on this host. No SM86 BMMA acceptance claim. An earlier accidental compiled-off CLI was detected and rebuilt correctly, with failed receipts retained. Missing physical-proof rejection is deliberate validation, not a successful BFS run.

Helper consultation: existing chat “Следить за работой и помогать” independently reviewed five initial patch scripts against constructors. Its concerns about constructor failure, device-bound handles and extent validation were addressed. Newer source changes received our additional audit and tests; no claim of independent review of the entire final diff.

## Deployment limits

No B200/B300 rental or test. No acceptance claim for 128 GPU or a globally optimal profile. Docker image and SM103/SM120 release package have not been updated by this source audit. First-ever startup and incompatible layouts still require preparation. This work improves resident sweep transitions; it does not prove increased BFS kernel throughput on a saturated B300 graph.
