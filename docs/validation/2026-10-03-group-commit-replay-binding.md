# Replay completion marker binding

Base: `793f92f`. The GPU runtime and NCCL/event protocol are unchanged.

The independent-rank replay previously required only that
`result/group-complete.json` exist. It checked canonical archive states and
per-rank COMPLETE/counts, but did not check the marker schema, rank hashes or
the marker's archive/config binding. This was a verifier coverage gap, not
evidence that the runtime had published a false COMPLETE.

The same existing archive verifier now additionally requires:

- `mgbfs-group-run-commit-v1`, COMPLETE, the requested world size;
- `file_fsync` scope, because this replay explicitly disables FIFO/search-only;
- group bootstrap digest matching the canonical archive config digest;
- SHA-256 of the exact bytes of each rank result, in rank order;
- each rank's identity, world size, scope and bootstrap matching that group.

No archived data is rewritten and no GPU readback is introduced. Tests first
failed because omitted/mismatched markers and invalid rank metadata were
accepted; the subsequent implementation rejected them. Another RED test caught
the missing unconditional rank-bootstrap/archive comparison in reference runs.
Full project Python suite: 249 passed, 14 skipped. The previously documented
bare-pytest duplicate vendored module collection problem remains unchanged.

This strengthens the acceptance gate for archive-finalize/asymmetric failures;
it is not a substitute for running those failures on two independent GPU ranks.
No hardware run, A/B, Graph merge or new publication of graph data is claimed.
