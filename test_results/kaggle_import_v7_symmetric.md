# Symmetric VMM v7 diagnostic: PASS, no BFS waiver

Source bd29407. Two independent T4 rank processes with P2P enabled. All30
local/import × initialization/layout variants × plain/four-tool pairs pass.
Sixty original rank logs retained at `build/kaggle-import-v7-observation`.
Root summary SHA256
`3a53da4a9d27364ec7ff806ebc7b9ae2c7c4794db525bc284a043e3464c5121e`.

The NCCL-shaped 4GiB-rounded rank stride/512MiB-aligned virtual reservation
and runtime async memset after import did not reproduce the full-runtime
activation failure. This is not a proof that every imported/aliased
allocation works under initcheck, and does not close the BFS initcheck gate.
NCCL window/shadow objects and runtime's other allocations remain different.
The next test compares two instrumenter versions on the unchanged full BFS,
retaining old failures rather than hiding them.
