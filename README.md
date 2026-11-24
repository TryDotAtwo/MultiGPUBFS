# MultiGPUBFS

Exact breadth-first traversal on NVIDIA GPUs for implicit Cayley graphs and supported matrix actions. The runtime distributes state ownership across ranks and checks exact equality on hash collisions.

## Run

- [Install the Linux CUDA runtime](docs/install-native-package.md)
- [Launch a graph locally or with allocated ranks](docs/unified-launch.md)
- [Use the CayleyPy adapter](docs/cayleypy-native-integration.md)
- [Read the segmented backend setup and reproducible examples](docs/experimental-segmented.md)

## Code

| Area | Entry point |
| --- | --- |
| Python launch and CLI | [multigpubfs](multigpubfs) |
| Rust CLI and request validation | [mgbfs-cli](crates/mgbfs-cli), [mgbfs-core](crates/mgbfs-core) |
| GPU scheduling, ownership and transport | [mgbfs-runtime](crates/mgbfs-runtime) |
| CUDA FFI and kernels | [mgbfs-cuda](crates/mgbfs-cuda), [cuda](cuda) |
| Contracts and integration tests | [tests](tests), [native CI](.github/workflows/native-contracts.yml) |

Build CPU contracts with `cargo test --locked` and see the [reference CLI](docs/reference-cli.md) for the reference benchmarks. Native GPU execution requires the matching executable and CUDA library.

[Development history, reports and experiments](https://github.com/TryDotAtwo/MultiGPUBFS/blob/72a3e2a0b0521985229823cc6c4c959ec22d81fd/README.md)
