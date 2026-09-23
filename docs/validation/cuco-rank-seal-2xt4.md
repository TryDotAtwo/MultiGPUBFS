# cuCO rank owner FinalizeDepth seal: two-T4 gate

Private `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v10 was RED at
`0b7adf1cd50961331434cc5a19ac6aef9b5623bd`: the owner probe reached
linking and failed on the missing `mgbfs_library_rank_seal_v1`.

Version 11 built `4241d9ddded08d48d29c138247b59080dd61460b` and
reported PASS on two physical T4s. The test rejects seal while the first
epoch still has borrowed readers. After the final stream drain and completion,
seal releases the cuCO membership tables (RMM live allocation decreases),
then the previous-history GPU buffer is overwritten. The accepted key planes
still export the correct keys. Plain plus memcheck, racecheck, initcheck and
synccheck owner probes passed on both cards with zero errors/hazards. Raw
outputs: `test_results/kaggle_cuco_rank_v10_red/` and
`test_results/kaggle_cuco_rank_v11/`.

The test uses a fixture stream drain at FinalizeDepth. It does not establish
the production Rust scheduler's event/lifetime ordering, complete BFS-layer
correctness, or a CPU-free NCCL/retirement path.
