# Bounded native startup and execution

Native graph inventory and admission queries now have a 60-second cold timeout. `MGBFS_NATIVE_QUERY_SECONDS` accepts a finite positive override in the native environment. The direct single-GPU native execution path has the same seconds+180 completion grace as distributed workers. Timeout terminates the child and raises an error; it is never COMPLETE, an INCOMPLETE snapshot, or permission to restart BFS.

Three real-child CPU tests verify public/profile query termination, execution termination and invalid limits. A two-process HTTP control fixture injects a hanging query at rank1; both ranks fail in approximately 0.52 seconds, the query child is absent, and no native BFS ranks or completed reports exist. This is a one-host control-path fault test, not physical multi-node GPU acceptance.

The declared Rust1.75 minimum was tested directly: 438 non-CUDA core/runtime integration tests passed, including archive and topology contracts, and the CUDA-feature CLI compiled with Rust1.75. CUDA compilation and physical GPU acceptance remain separate gates. A new eight-target native build with expensive ptxas optimizations disabled is in progress; no build or hardware acceptance is implied here.
