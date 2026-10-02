# BFS tail archive integration contract

Branch: `codex/bfs-tail-archive`. The active BFS checkout is not modified.

`scripts/bfs_tail_archive.py` implements SSD retention and publication ordering.
`scripts/run_tail_bfs.py` wires it to the existing CUDA/NCCL archive FIFO and
the existing global layer completion boundary. `tail_wire.py` validates the
native archive chain, removes routing hashes and packs words on CPU. A layer
is admitted only after every rank has both a checked archive layer commit
and the existing completed-advance marker. Candidate states are not admitted.

Packing uses little-endian words: symbol i occupies bits 4*i; padding is zero.
Width is 8 bytes for n<=16 and 16 for 17<=n<=32, provided symbols fit [0,15].
Larger alphabets are rejected; do not silently truncate symbols.

Retention uses decimal GB. The local tail retains at least three entire layers
and at least 10 GB, or all layers back to depth zero. Each layer updates counts,
whole-layer wall duration, per-GPU observed VRAM peaks, and an INCOMPLETE snapshot
of at most 1 GB. Its oldest selected layer may be partial; offsets are explicit.
COMPLETE snapshots contain only whole layers. All layer statistics survive tail
eviction. `COMPLETE` must be requested only after graph exhaustion; archive
success alone cannot establish search completion.

The producer must supply the exact start, L/R/X definitions, program commit and
full launch configuration. Time is measured at its existing completion boundary.
VRAM peaks must come from a separate monitor whose interval is recorded; they
are sampled peaks rather than guaranteed physical maxima. No batch-level CUDA
synchronization is introduced by this module.

`tail_upload.Publisher` is one background worker with a 25 GB pending-byte
limit. It pins files with hard links immediately when enqueued, verifies
checksums, batches snapshot payloads into a single commit, and publishes the
manifest in a following commit. Only the newest not-yet-started snapshot remains
queued; the in-flight snapshot stays pinned. Every local completed-layer
snapshot and all completed-layer statistics are still generated. Failed uploads
retain pinned inputs and propagate an error. Run paths are
`tail-runs/<run_id>/...`; use a fresh run_id. Remote obsolete files may remain
but are not referenced by the latest manifest. Live HF publication was validated
on 2026-10-02 for COMPLETE (9,4) and an INCOMPLETE capacity failure; see the
validation report for remote manifests and checksum readback evidence.

The separate nvidia-smi monitor requests samples every 50 ms. The manifest
uses host receipt timestamps and the earliest rank BEGIN/latest rank END
window. A short layer with no observation records a null peak explicitly;
it does not invent an exact physical maximum. Raw CUDA kernel/duplicate/path
metrics are not part of the tail manifest.

Recorded integration validation:

- An early launcher deadline cancellation passed on the isolated two-GPU
  host: INCOMPLETE, depth -1, zero admitted files. This tests cancellation
  before any layer completes. Native capacity failure separately retained
  completed layers 0..3 in the two-GPU validation.
- Physical SSD stress wrote 13 decimal GB, retained four full layers totaling
  10.4 GB, evicted the oldest layer and verified retained SHA-256 by readback.
  The INCOMPLETE suffix was exactly 1 GB and its checksum also matched.
  This synthetic payload check is not a >=10 GB real BFS/GPU result.
- Archive on/off was compared on (15,4) on two RTX 3090 GPUs with three
  repetitions per owner and mode. All twelve runs share 26 completed layers;
  comparisons use that identical prefix, not a full graph completion.
  Median whole-layer totals were 0.462 s / 0.647 s for CUCO_RANK without/with
  archiving and 0.521 s / 0.742 s for CUCO_INDEXED. This selects CUCO_RANK
  for this measured configuration; it does not establish a universal winner.

Validation: `python -m unittest discover -s tests -p 'test_*tail*.py'`.
GPU full-word oracle: `lrx_multiset_two_rank_cuco_full_state_oracle`.
Archived full-word oracle: `python scripts/verify_tail_oracle.py <saved-root>`.
Matched benchmark: `scripts/tail_remote_panel.py` (isolated rental paths).

Finite grid driver: `scripts/sweep_tail_bfs.py`, where `m` maps to native `r`.
No range is required: by default the driver considers every `(n,r)` with
`2<=n<=32` and `1<=r<=n`, visiting increasing n. Packing/alphabet and u64 orbit
limits are classified before any GPU work. Explicit range filters remain optional.
Pass a positive `--deadline-seconds` and the same config/source/runtime-env
arguments as the single-run driver. `--plan-only` needs no GPU configuration.
After a resource stop at `(n,r)`, larger n at that same r are recorded as
unattempted INCOMPLETE with `pruned_by`; other r branches continue independently.
This is an operational heuristic, not proof that skipped graphs cannot fit.
Recognized stops include allocation OOM, capacity/ring limits and SSD exhaustion.
CUDA status 2 requires a diagnostic whose source line is a CUDA allocation,
so unrelated NCCL errors with the same numeric status do not prune a branch.
HF upload errors, archive worker I/O errors and timeouts do not trigger pruning.
Native sticky layer/request capacity code 16 is recognized specifically;
other rank-depth codes and remote cancellation alone do not prove a resource
stop. The policy fingerprint is v2. The automatic 527-pair GPU pruning test
on two RTX 3060 is recorded in `validation/2026-10-02-auto-pruning.md`.
Resume preserves the policy and prior exclusions; changing resource limits or
retrying pruned pairs requires a fresh root/run ID.
Each pair gets its own run directory and HF run ID. `sweep.json` records
COMPLETE/INCOMPLETE, last completed layer and reason per attempted pair,
unsupported pairs, and the remaining unstarted pairs. Restarting with the
identical configuration resumes pending pairs without overwriting earlier
runs. The alphabet must fit four bits and the native orbit count must fit u64;
unsupported pairs are explicitly recorded, never claimed as completed.
The finite GPU grid for 2<=n<=15 and 1<=m<=n executed all 119 pairs on two
RTX 3090 GPUs: 93 COMPLETE and 26 INCOMPLETE with 256 archive ring slots.
Rerunning those 26 pairs with 2048 host archive ring slots completed eight
additional graphs, leaving 101 COMPLETE and 18 INCOMPLETE among unique pairs.
These remaining graphs retain explicit native stop reasons; they are not
claimed complete. The baseline 2591 payload files (2,161,864,688 bytes) and
119 final manifests were published directly from the GPU host in HF commit
`745f6dfff88013ae13d32806c606fed11d9406f1`. Remote streamed checksum verification
passed for both cohorts: 3646 files, 8,374,561,856 bytes. The larger-ring cohort
is frozen at HF commit `da05ec7af6f871560adf5253e7b8831ece21275d`; the combined
verified report is at `evidence/20261002-directhf/final-report.json` in the same
dataset.
Required metadata checks passed for all 145 baseline/rerun manifests, including
exact starts, actions, packing, program/binary identifiers, complete launch
configuration, contiguous layer counts/times, and per-GPU sampled observations.

GPU-host publication for a finite sweep can use `scripts/publish_tail_batch.py`.
Run the grid without per-case `repo_id` while HF commits are throttled, retaining
the states on that GPU host's SSD, then run on the same host:

```sh
python scripts/publish_tail_batch.py --sweep-root /root/my-sweep \
  --repo-id TryDotAtwo/multigpubfs-bfs-results --deadline-unix <absolute-deadline>
```

This verifies every staged checksum before making payload commits. All case
payloads share one commit; case manifests and the sweep ledger follow in another.
The CLI waits on repository commit quota responses until its deadline and leaves
inputs intact on failure. It does not download states to the user's computer.
Direct GPU-host uploads and remote checksum readback passed for both owners at
(7,4), and for 16-byte states at (17,16), (32,31), and (32,32). The exact native
source and separately deployed publisher versions are recorded in HF evidence.
If a terminated upload exhausts retries with an Xet network error, retain the
GPU-host inputs and retry publication with `HF_HUB_DISABLE_XET=1`. This uses
ordinary HF LFS upload without repeating BFS. Do not start a competing uploader
while the original handle is still active.

Storage stress: `python scripts/stress_tail_archive.py <fresh-directory>`;
requires at least 18 decimal GB free. Recorded physical validation is in
`docs/validation/2026-10-02-physical-tail-stress.json`. Payloads are explicitly
synthetic, not claimed as reachable graph layers or search-completion proof.

For an existing Windows DPAPI-encrypted token, the saved-run publisher accepts
`--token-dpapi <path>`; the credential requires the original Windows account.
Keep this file outside Git. Linux GPU runs use the standard HF_TOKEN secret
environment or HF login cache. On this Windows host, set HF_HUB_DISABLE_XET=1
before invoking the publisher to bypass a local Xet cache access failure.
