# Two A4000: reserve admission and 16-byte archive evidence

Reports were recovered from HF after instance 54041651 was deleted. Local
remote paths did not contain reports; that absence did not imply failed jobs.
Only small report JSON was read on the user's computer, never GPU states.

## Native admission with corrected 32-batch Graph bridge

Source 0959e4b, CUCO_RANK, NCCL_LSA, two ranks, batch 32768, (15,4).
Each run has a 30-second search deadline and finishes depth 33. These are
INCOMPLETE traversals, not complete enumeration or long-duration stability tests.
Reported device total is 16,752,181,248 bytes per GPU. Independent sampled peaks:

| Startup reserve | Admitted rows per rank | Peak per GPU, bytes | Peak / device total |
| --- | ---: | ---: | ---: |
| 256 MiB | 80,731,013 | 16,505,634,816 | 98.528% |
| 128 MiB | 81,407,525 | 16,644,046,848 | 99.355% |
| 64 MiB | 81,745,776 | 16,709,058,560 | 99.743% |

Every profile publishes three Parquet shards totaling 416,592,321 bytes.
The remote readback reports all checksums, the case manifest and sweep ledger
verified. Observed peaks are sampled values, not guaranteed continuous maxima.
The native low-reserve ABI creation/destruction gate also passes.

[Reserve report](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261003-vram-reserve-abi/report.json)

Payload and manifest revisions:

- 256 MiB: 7226c1499689fe61664f001a92a95bea64c8520d
- 128 MiB: 9a45fc341939bdc5b93ebe0c1ecc0db39ee68409
- 64 MiB: 90b7b41f9404ca692f091b59b700c262f30fa8d8

## Real wide packed states

(17,14), two ranks, CUCO_RANK/NCCL_LSA, batch 2, Graph window 32,
64 MiB native pool. The remote CPU oracle compares all 4,080 states over
30 layers. Actual archive packing is 16 bytes per state. Publication uses
one 47,559-byte Parquet payload; remote readback verifies its checksum,
manifest and sweep ledger at 48946a7f7e2af6a8a2f6844879d68a91d11eb203.

[Wide-state report](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/main/evidence/20261003-wide-state-graph/report.json)

## Remaining gates

Automatic production configurations still default to a 1 GiB reserve and
Graph disabled. These successful explicit profiles do not establish a safe
universal reserve or automatic transport/Graph performance selection. Existing
timing gates show both a small slowdown and a larger-workload improvement.
Automatic profile selection and an end-to-end production sweep using it remain
required. A sampled 99.743% peak alone does not prove maximum capacity,
zero slowdown, or reliable long-running allocator headroom.

The provider DELETE succeeded and a subsequent GET returned no instance.
The local watchdog stopped reporting before its deadline, so it did not enforce
that deadline. Exact billed charges cannot be recovered with the current key.
