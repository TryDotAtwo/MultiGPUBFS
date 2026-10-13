# Fast tuning and geometry acceptance вЂ” 2026-10-13

Package0.2.0.dev4, Python sourcec4ce3ebfc3921e58f76f31a3222ee797697d7e0a. The CUDA/Rust binaries are unchanged verified portable artifacts from7046577446e546b86317f6e2e24cf319552b29cd. Release `native-pip-20261013-v4` includes private CUDA/NCCL; no local build or paid rental was used.

## Changes

- Recognize the actual adjacent-transposition generator set, not a graph name. For distinct-label Coxeter graphs, the coefficients of product_(i=1..n)(1+x+...+x^(i-1)) give exact layer sizes. Their maximum is a safe global bound for each rank, independent of hash balance. Automatic capacity is capped by this bound and live native VRAM admission. Explicit capacity is preserved. Unknown families are not assigned this bound.
- Skip replayed cold-prefix pilots for certified Coxeter and modest permutation orbits. Admission remains live. Batch selection samples actual disjoint GPU chunks online. This seed is explicitly conservative, not reported as a measured optimum. Forced history/transport/sort/generator settings and matrix CUDA/GEMM selection preserve their existing selector.
- Keep one persistent worker pool around multi-graph examples. The pair-sweep example already had this scope; the basic example now demonstrates it explicitly.

## Default API in a persistent pool, physical2Г—T4

| Graph | Full API s | BFS s | Cold selector s | Hash table/GPU |
|---|---:|---:|---:|---:|
| S11 |2.773|2.708|0.0158|128MiB|
| S12 |35.994|35.912|0.0151|2GiB|

S11/S12 complete counts, all individual layer sizes and diameters match the exact inversion generating polynomial. Earlier cold selectors consumed10.80/7.90seconds and tables occupied2/8GiB respectively. The first graph still pays process/CUDA startup (S3:3.85seconds). Pool reuse is essential for many tiny graphs.

These fixes reduce startup work and memory; the automatic BFS itself did not establish a speed gain against the prior persistent example (S11~2.75s,S12~35.82s). A separate fixed-batch run measured1.90/25.76seconds, but it was a different Kaggle execution, not a same-host old/new paired benchmark. Do not advertise that difference as a proven code speedup.

## Batch experiment on the same2Г—T4 execution

| Graph | Original batch BFS s |4Г—batch BFS s |
|---|---:|---:|
| S11 |1.901|1.984|
| S12 |25.755|26.127|

Identical full layer vectors and table geometry; zero source retries. The larger batch lost and was not made the default. Earlier1/4/16-shard S12 comparison also favored1shard. These are bounded single-run comparisons, not proof of a universal optimum.

## CayleyPy catalogue0.2.0

All48 exposed factories were exercised through the actual `CayleyGraph.bfs()` adapter and native2GPU engine:

-36 `PermutationGroups` factories.
-3 `MatrixGroups` factories.
-9 `Puzzles` factories.

The bounded adapter test injects only `capacity=256` at the adapter launcher boundary; generator definitions, CUDA execution, result conversion and snapshots are real. CPU independently generates reference layers and checks every completed layer plus sampled-state membership/uniqueness.39 bounded representatives completed;12 produced correct prefixes, including3 additional nonmodular matrix cases. All3 modular matrix families also completed through the unmodified API with ordinary automatic admission. This covers representative parameters, not every parameter combination or full huge/infinite groups. Finite matrix examples use n3/modulo3; nonmodular matrices also passed bounded-prefix correctness, but full infinite-group performance is not claimed.

Cataloguev2 originally failed only because the koltsov3 fixture used an invalid n4. n6 subsequently passed; final cataloguev3 and v4 completed with no failures. Initial v1 failed before execution because the test notebook lacked kernel metadata. No algorithm was restarted or changed in response to a timeout.

## Evidence

- [Packaging/clean Linux install](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38386130807).
- [39 tuning/geometry regression checks](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38386144682), all passed.
- Metadata receipts and reproducible catalogue notebook accompany this report. No state payloads were downloaded or committed.
- This wheel targets Linux x86_64/SM75/SM86. Actual validation used2Г—T4; this is not B200/B300 or multi-host acceptance.

## Same-GPU automatic/fixed check

After one persistent-pool warmup, the two modes ran consecutively on the same2Г—T4 host with identical admitted plans and layer vectors:

| S12 mode | BFS s | Full call s |
|---|---:|---:|
| Automatic/live tuning |35.852|35.932|
| Fixed batch/no tuning |35.782|35.843|

The observed difference is0.2%, not the ~40% suggested by separate executions. The cause of the25.8s versus35.8s variation between executions/session histories is not isolated. No code-speedup claim is made from that comparison.

Canonical modular matrices now also use the certified finite entry-space upper bound. Mixed moduli, noncanonical starts, unbounded matrices and bounds outside the supported capacity range retain ordinary native admission. On hardware where exact GEMM is available, the existing CUDA/GEMM measurement is retained. On the tested T4s, the3 finite matrix API calls completed in0.069вЂ“0.080s, with selectors around0.015вЂ“0.016s and capacity19683/rank. These are representative small finite cases, not large-matrix throughput measurements.

[Clean direct GitHub URL installation with CayleyPy extras](https://github.com/TryDotAtwo/MultiGPUBFS/actions/runs/38386799492) passed for the final wheel. SHA256: `7b1a2de550bdd268c1628c314c01fc2f8fc09d315c0503e48de1fdff5a0d944c`.

## Final public examples

The final0.2.0.dev4 wheel passed actual public Kaggle Run All:

- [Basic version7](https://www.kaggle.com/code/trydotatwo/cayleypy-multigpubfs):6 complete native2GPU runs, including the persistent Coxeter3..7 loop.
- [Pair sweep version5](https://www.kaggle.com/code/trydotatwo/multigpubfs-pair-sweep):14 complete LRX pairs n2..5 in a shared pool.

Both use the direct checksummed GitHub pip requirement, with no attached Inputs. Machine-readable public receipts are included in `docs/acceptance/geometry-20261013/public-kaggle.json`.
