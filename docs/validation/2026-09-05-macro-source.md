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
