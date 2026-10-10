# Catalog entry and resolved native identity

The public API and CLI resolve named CayleyPy constructors with explicit positional/keyword parameters, without evaluated source. Six CPU interface tests passed; a real pinned CayleyPy0.2.0 CPU wrapper/oracle test and named 120-state LRX constructor also passed. Named CLI GPU and clean installed gates remain pending.

Runtime reports now hash the library resolved by the actual ELF loader. A valid bundle manifest supplies source/toolkit/architecture provenance only when the resolved library bytes match its declared artifact. An overridden library retains its actual hash but does not inherit the old package source commit. A real compiled Linux ELF fixture reproduced the former false provenance and now passes with two distinct library overrides; tampered bundle artifacts remain rejected.

The current 54-test startup/profile CPU suite passed. Its first discovery command mistyped the network-control test filename; zero discovered tests was a failure and was corrected, rather than counted as acceptance. These CPU checks and the Rust1.75 checks do not establish physical GPU acceptance of the new portable build.
