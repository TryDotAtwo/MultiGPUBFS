# Expanded production owner DAG capture fixture

Existing experiments/library_owner/cuco_owner_probe.cu now captures the
device-count AoS bridge, rank compare, shard counts, reservation, owner
commit, state materialization and next-extent publication.
Host-only lease completion occurs after graph launch and stream drain; it is
not represented as a captured GPU node or asynchronous lease proof.
All extra scratch, AoS and directory storage is allocated and initialized
before capture. CUB/cuCO/runtime production functions are used directly.

After instantiate/launch/drain it checks original KEEP_FIRST states/keys,
accepted counts and the actual published extent count/begin/count/ready.
Success marker RANK_FULL_OWNER_DAG_CAPTURE_PASS is printed only after these
checks. Subsequent original transactions and capacity tests remain intact.

Pending CUDA build and real hardware execution. This is an acceptance
fixture change, not a production algorithm change or a capture PASS claim.
No NCCL/LSA/rank fault or Nsight timeline result follows from owner capture.
v33 pins older e8f8fa6 and does not contain the expanded fixture.
