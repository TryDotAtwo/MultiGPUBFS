# Artifact storage and cleanup

- Do not place large build caches, datasets, archives or profiling artifacts on C:.
- Build and local scratch belong on D:; explicitly set Cargo target/temp and HF cache paths there when launching tools. Do not redefine HOME or CODEX_HOME.
- Large graph states and results stay on the GPU host and are published directly to the appropriate Hugging Face dataset repository. Preserve its existing privacy; do not download large datasets through this computer.
- Before deleting unique local results, verify remote availability, byte sizes and checksums, and retain manifests, source/configuration and provenance. Avoid creating another large archive copy on C:.
- Reproducible inactive build/test caches may be removed with explicit validated paths. Preserve source snapshots and active-run files.
- Coordinate ownership with the archive worktree's chat before cleaning its artifacts. Never run competing cleanup against another active checkout.
