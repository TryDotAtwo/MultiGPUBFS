# Macro operator compilation: arbitrary source correction

`MacroGeneratorSet::compile` previously enumerated from `graph.start`.
For a nonidentity source A, the stored operator for word G was G*A,
so applying it to a state X computed (G*A)*X instead of G*X.
The matrix manifest permits any invertible start; this was a correctness bug,
not a restriction of that manifest. Compilation now begins at the identity,
independently of the BFS source. BFS itself still begins at the requested A.

Regression: all 27 starts of UT(3,3), macro depths 1, 2 and 10.
The operator library must be source-independent and weighted full-state layers
must equal the ordinary successor BFS oracle from each source.
The new test failed before the fix at depth 1 and a nonidentity start.

Verification on Windows CPU:

```
cargo test --locked -p mgbfs-core --test macro_generators
8 passed; 0 failed (63.71 s)
```

Existing S8/S12 frozen macro digests also remain unchanged. This validates
the shared CPU compiler and weighted oracle, not native CUDA execution from
nonidentity sources or multi-rank macro scheduling. Those hardware gates
remain required. Identity-start results are not changed by this correction.

## Native source regression: Kaggle v21

Launcher pins cdde19980e41de6aa55ffe991929465435848cc5 and adds
`native_macro_nonidentity_source_preserves_original_layers`.
The single-rank CUDA fixture starts UT(3,3) at two successive original moves,
uses parent batch 7, K=1/2/3/10 and pre-dedup OFF/ON, and compares every
full-state original-depth layer to the ordinary matrix BFS oracle.

Downloaded `macro-plain.log`, `macro-memcheck.log`, `macro-racecheck.log`,
`macro-initcheck.log`, `macro-synccheck.log` all report one passed test.
All four sanitizer summaries are zero, including race hazards and warnings.
Inventory identifies two distinct real T4s; this macro fixture uses one GPU,
not a multi-rank weighted scheduler. Logs reside in
`test_results/distributed-sanitizer-v21/distributed-sanitizer/`.

## CPU CI restored

GitHub run 33990041597 at d6a374dd07d22c453867822c67a34ee00bb7a9e8
passed all steps: Rust 1.75 formatting, full default CPU tests, Kaggle guard
tests, allocation geometry, query adapter and descriptor ABI.
Earlier runs stopped at formatting and then an obsolete allocation test
which rejected the now-supported generation5 variant. The replacement checks
the compact geometry explicitly (n=3, moves=2, batch=3: rows=8, columns=4,
generators=128, packed parents=64, products=128 bytes). The old test's failure
was reproduced locally; the corrected test prints ALLOCATION_SHAPE_PASS.
No production capacity validation or CI gate was disabled.
