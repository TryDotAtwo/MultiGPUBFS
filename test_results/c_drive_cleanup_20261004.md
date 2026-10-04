# Coordinated C: cleanup, 2026-10-04

Archive owner chat 01a0f968-27be-75e1-a464-0ea088904945 independently owns
physical-tail-stress-20261002 and tail-vast-sweep-20261002 in its worktree.
The main runtime chat did not modify those results or current remote runs.

Removed only inactive reproducible Cargo build caches:

| Explicit Temp target | Removed logical bytes |
|---|---:|
| mgbfs-cross-check | 2,035,681,320 |
| mgbfs-cpu-contracts | 1,656,139,089 |
| mgbfs-sideband-287db6d48c5947bf9237bbbe696d56af/target | 599,826,549 |

Total removed logical bytes: 4,291,646,958 (~4.00 GiB).
Observed C: free before: 213,966,848 bytes; after: 4,238,299,136 bytes.
Net observed free-space increase: 4,024,332,288 bytes (~3.75 GiB), not
identical to logical deletion size because of allocation and concurrent use.

All absolute targets were resolved within the explicitly named Temp root;
reparse roots/descendants were rejected. Process inventory found no matching
non-shell process. Last modification dates were September28/30. Cache roots
contained Cargo debug/target structures. Removal is not trash-recoverable;
these caches can be rebuilt from preserved source.

Sideband crates and Cargo.lock were retained (roughly286KB). No unique result
was deleted, and no HF upload was performed by this chat during this cleanup.
The archive owner is responsible for remote verification before its own
unique-result removal. Current Kaggle jobs and D: runtime artifacts remain.
The storage rule is recorded in root AGENTS.md.
