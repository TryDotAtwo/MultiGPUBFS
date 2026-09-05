# HF commit batching

Implementation: `7f027f37553b2d9cfadd9e20e0794228fff2c4fa`.
Use documented `CommitOperationAdd` + `HfApi.preupload_lfs_files` with
`free_memory=True`, retaining only the mutated operation after upload.
Only then recycle the fixed RAM slot. If the API keeps the payload (for
example, regular Git rather than LFS classification), fail rather than
commit bytes from a reused slot.

After complete archive validation and all upload receipts, commit metadata
in batches of at most 256 operations. Upload the rank manifest only after
all batches succeed. Global promotion remains separate. Approximately
6228 state files need 26 staging batch commits across two balanced ranks,
plus two rank manifests and one global promotion, rather than 6228 commits.
Payload slots remain bounded; file and operation metadata scale with shard
count, as the existing file inventory already does.

Local tests: 24 HF/stream tests pass, including 257 files -> [256, 1]
commit batches, released payloads, non-LFS rejection, upload failure and
commit failure without a complete rank marker. These tests use a fake HF
transport and real Parquet serialization, not a remote throughput proof.

Kaggle `mgbfs-s11-hf-stream` v17 is an S8 live gate of this source:
40320 expected states, 100000 capacity/rank, 96 pinned slots/rank,
262144 archive rows, real 2xT4. The editor launcher overrides only the
module configuration; credentials remain bound through Kaggle Secrets.
Gate v17 stopped before BFS at create_branch: HTTP 429, repository commit
limit still exhausted, approximately one hour cooldown requested at
2026-09-05 11:39 UTC. No preupload/commit behavior was tested remotely.
Do not retry before 12:40 UTC; recheck the live service then. S13 must not
be relaunched until the remote publication gate passes. Pinned-ring
throughput remains a separate unverified constraint.

API contract: https://huggingface.co/docs/huggingface_hub/guides/upload

## Live gate v18 after cooldown

PASS: S8 40320 states, two Parquet shards, one live upload slot/rank.
Search 0.357291257s, native archive completion 1.28542754s, runner wall
11.370417459s (not global promotion time), 437 MiB/device on physical 2xT4.
Source remains 7f027f37553b2d9cfadd9e20e0794228fff2c4fa.
HF run `s8-native-2xt4-20260905-124255` promoted at commit
`d00f33b3a5ffde8ae8efdf182c823c8bc33190b6`.
Evidence: `test_results/hf-editor-v18/s11-hf-stream/summary.json` and
`promote.log`. CLI downloaded these successfully but failed encoding its
final combined notebook log on Windows; this is not a remote run failure.
This validates actual preupload and promotion, not S13 sustained throughput.
S13 editor reconfiguration was attempted but browser timed out: verify the
draft and saved versions before launching, to avoid duplicate runs.
