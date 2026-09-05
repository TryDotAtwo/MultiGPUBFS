# Shared one/two-rank engine: T4 gate

Kaggle `trydotatwo/mgbfs-distributed-sanitizer` version 29, script version
347583042, completed on two physical T4 GPUs. Source:
`b520a788e1ad6de3202e77dadf7bf59662f941c3`.

Downloaded selected evidence is in
`test_results/distributed-sanitizer-v29/distributed-sanitizer/`: summary,
plain, memcheck, racecheck, initcheck, synccheck and owner-query logs, plus
the notebook event log in the parent directory. The download completed before
any subsequent version of this kernel was submitted.

All 12 runtime tests passed plain and under each of the four sanitizers.
Memcheck, initcheck and synccheck each report zero errors; racecheck reports
zero hazards, errors and warnings. This includes full-state archive comparison
for the one-rank profile fixture and the coordinated VRAM rejection fixture.

The completed summary contains 24 distinct process smoke configurations:
world sizes 1 and 2, each with DENSE, scalar HASH_FIRST and IMMA HASH_FIRST,
both owner backends and pre-dedup ON/OFF. Every entry is PASS, with global
S4 layer counts `[1,3,5,6,5,3,1]`. The pinned launcher requires full warmup,
reported allocation/reserve fields and successful CLI verification per rank.
The subsequent selected download completed and its inventory was checked:
72 rank JSON files (36 measured and 36 warmup) and 36 CLI verification logs.
Every verification log reports VERIFIED. All measured records report COMPLETE,
warmup completed, positive explicit payload no larger than aligned allocation,
and exactly 1 GiB untouched reserve. Every measured rank's layer counts match
its warmup record. No archive state payload was downloaded locally.

This is a correctness/sanitizer gate, not performance evidence or proof of
the final overlapped architecture. Conservative one-rank peer allocations,
the host-synchronized reference schedule and the other documented architecture
gaps remain.

After this gate, `trydotatwo/mgbfs-distributed-bench` version 10 was submitted
with source `b02bf3f2526c5bce0f5331cb30c5317c2c80b91a`: one active T4,
S11, all 12 native profiles, five repetitions, pinned CayleyPy single-GPU
baseline batch calibration, mandatory lossless permutation-u8 archives while
search states remain matrices. Measurements are pending. The separate
two-T4 S11 version 2 run was not restarted.
