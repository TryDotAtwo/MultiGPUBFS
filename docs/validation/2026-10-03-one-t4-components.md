# Physical one-T4 component gates

Source: `7949759b839a7b51edb2fa7e798317a3028ec5d4`, clean public checkout.
Owned diagnostic instance 53923881, machine 136299, one Tesla T4,
GPU UUID `GPU-ed10a20a-5c90-cedf-1aa9-60bb9748da68`, sm75, 16384 MiB.
CUDA 12.9.1; Compute Sanitizer 2025.2.1.0, build 35969825.

Both existing component fixtures passed plain execution and **unfiltered**
memcheck, racecheck, initcheck and synccheck with error-exitcode 97.
Non-race tools reported zero errors; racecheck reported zero hazards,
zero errors and zero warnings. No NCCL registration occurs in these fixtures.

| Fixture | Verified scope |
| --- | --- |
| `hash_first_generate`, `MGBFS_TEST_HASH_TC=1` | Tensor generation/hash versus CPU arithmetic oracle, n=1,3,4,8,9,16,17; modulus=2,5,256; frozen hash, origins, device counts, zero rows, capacity failure |
| `bounded_owner`, `MGBFS_TEST_BMMA=1` | BMMA compare/commit, stable source indices, persistent immutability, cross-batch duplicates, random seeds, capacity failures, compact/history/window layouts |

Binary SHA256:

- TC: `be009ba4d7080f7590ed5030aeda20c3c4d7e1c39010f71f2031983f61c12fcb`
- BMMA: `b301d3f5030466982b711c3eae0cd2194cd560e80d2847e9b76dd36467519050`

Small raw-log package downloaded and hash-verified locally:
`build/one-t4-20261003/one-t4-primitives.tar.gz`, SHA256
`9375e6270bc42b86596b8e9cf0e4943dc26dbf911efaaf7578a005719c7b1539`.
It contains build/plain/eight sanitizer logs, not graph archives.

These results do **not** close full-BFS sanitizer gates, NCCL registration
initcheck, owner/transport/retirement multi-rank acceptance, physical 2xT4
correctness, overlap or paired performance measurements. Integrated native
build and execution are separate gates.

## Integrated execution on the same physical T4

Pinned NCCL `b91894bd5b190c874d98a017f93f5daa515b65d0` and native sm75
library built successfully. CPU contract tests for core/runtime/CLI and CUDA
Rust check passed. These do not include the legacy root GPU wrapper package.

- `tensor_generation_hardware_admission_and_layer_counts`: PASS, real Tensor
  HASH_FIRST S4, full-state/hash oracle and verified durable archive.
- `manifest_nonidentity_start_runs_both_profiles_with_full_state_archive`:
  PASS, both DENSE/HASH_FIRST, literal nonidentity matrix start and layer sets.
- Native BMMA S4 and U4(mod2), both profiles and pre-dedup OFF/ON: all eight
  independent one-rank executions PASS full canonical state/depth oracle.
  S4 has 24 states in 7 layers; U4(mod2) has 64 states in 9 layers.
- Whole S4 HASH_FIRST+INT_MMA_SM75+BMMA_BUCKET+NCCL_LSA+archive:
  racecheck/initcheck/synccheck return 0 and each passes full-state oracle.
  Unfiltered memcheck returns 97 with one CUDA API error 800 at NCCL
  `allocator.cc:60`, FABRIC `cuMemCreate` probe. The program subsequently
  completes using the vendor POSIX retry. This is **not** a passing memcheck
  gate. No suppression, allocator patch or backend fallback was introduced.

One-GPU initcheck success does not close the prior independent two-GPU NCCL
registration failure 719 or prove remote receive-slot correctness.

Downloaded integrated evidence, including small S4/U4(mod2) archives:
`build/one-t4-20261003/integrated-evidence.tar.gz`, SHA256
`5fd23739edd1125cd90b830e539d4fd9c31742ce4e4009bbf5d346c53ca95f9a`.
Both remote and local hashes match. Instance53923881 deleted early after
retrieval; exact identity check, DELETE acceptance and subsequent GET ABSENT
confirmed. Source and downloaded results remain; remote checkout/builds were
removed with the disposable instance.
