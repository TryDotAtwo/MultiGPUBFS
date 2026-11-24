# MultiGPUBFS

Exact breadth-first traversal on NVIDIA GPUs for implicit Cayley graphs and supported matrix actions. The runtime distributes state ownership across ranks and checks exact equality on hash collisions.

## Run

- [Install the Linux CUDA runtime](docs/install-native-package.md)
- [Reuse a session for traversal and exact route searches](docs/bfs-session.md)
- [Launch a graph locally or with allocated ranks](docs/unified-launch.md)
- [Use the CayleyPy adapter](docs/cayleypy-native-integration.md)
- [Read the segmented backend setup and reproducible examples](docs/experimental-segmented.md)

## Code

| Area | Entry point |
| --- | --- |
| Python session and admission | [api.py](multigpubfs/api.py), [memory_admission.py](multigpubfs/memory_admission.py) |
| Python local and cluster launch | [launch.py](multigpubfs/launch.py), [cluster.py](multigpubfs/cluster.py) |
| Rust CLI and request validation | [mgbfs-cli](crates/mgbfs-cli), [mgbfs-core](crates/mgbfs-core) |
| Segmented traversal | [rank runtime](crates/mgbfs-runtime/src/segmented_distributed_native.rs), [packet exchange, hash index and layer commit](crates/mgbfs-runtime/src/segmented_distributed_native) |
| Exact routes and window output | [native path runtime](crates/mgbfs-runtime/src/generic_path_native.rs), [route output](multigpubfs/path_search.py), [rank-owned CSR](multigpubfs/path_window_files.py) |
| CUDA FFI and kernels | [mgbfs-cuda](crates/mgbfs-cuda), [cuda](cuda) |
| Contracts and integration tests | [tests](tests), [native CI](.github/workflows/native-contracts.yml) |

Build CPU contracts with `cargo test --locked` and see the [reference CLI](docs/reference-cli.md) for the reference benchmarks. Native GPU execution requires the matching executable and CUDA library.

Default session results, tuning diagnostics and benchmark outputs go to the user state directory. Set `MGBFS_ARTIFACT_ROOT` to choose a base directory. Explicit output paths and `MGBFS_TUNING_EVIDENCE_ROOT` take precedence.

[Development history, reports and experiments](https://github.com/TryDotAtwo/MultiGPUBFS-archive)
