# S13 compact capacity v2: incomplete

Source 971ab50ec529711e0d9707242468e1493e9675c3, physical 2xT4,
160M records/rank for state ring and layer capacity; archive disabled.
Peak sampled device consumption: 12743 MiB/rank.

Expansion of depth 48 completed, producing depth 49 with local counts
152539815 and 152539613 (305079428 globally). Expansion of depth 49
failed with NATIVE_OWNER_FATAL_11; peer reported REMOTE_OWNER_BATCH_FATAL.
CUDA state_commit.cu maps 11 to state ring capacity exhaustion, including
live extent occupancy and wrap padding. This is not cudaMalloc OOM and does
not establish the graph peak or maximum feasible n.

No final layer manifest or state archive was produced. Kaggle worker COMPLETE
means diagnostic script completed; embedded search status is FAILED.
S14 feasibility remains unconfirmed. Archive/HF streaming cannot remove the
live-frontier VRAM requirement.

Evidence: test_results/s13-capacity-v2/s12-capacity-probe/summary.json and
s13-capacity-160000000.log. Rank stderr lines are interleaved.

## v3: 178M records/rank

Source f2482c4db18c4c6a2596a7344f857d667b442b3a. Archive disabled.
Peak 14129 MiB/rank on 15360 MiB T4s (1231 MiB unconsumed).
Depth 51 reached with rank counts 172245371 and 172258148, totaling
344503519. Expansion failed with ring capacity code 11 on rank 0:
head=1374433896, tail=1552432168, capacity=178000000,
requested survivors=4474. Existing logical occupancy=177998272;
only 1728 records remained before the request. Peer exited with remote fatal.

This confirms live ring exhaustion, not a diagnostic parsing error. Increasing
capacity further has little room while preserving the 1 GiB reserve. The
graph peak remains unknown; no S14 feasibility claim is supported.
Evidence: test_results/s13-capacity-v3/s12-capacity-probe/.
