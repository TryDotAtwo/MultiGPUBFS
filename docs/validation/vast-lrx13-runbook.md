# Vast LRX13 hardware procedure

Status: preparation only; no server rented. Total project cap is USD 100,
including compute, storage and network. Existing account credit is not the cap.

## Before renting

- Pin a published source commit containing the eight-rank test (not only the
  earlier T4-tested runtime), dependency commits and build configuration.
- Build CUDA for SM90, including the cuCollections library. The SM75/SM90
  library build passed in [T4 v13](n-peer-t4-v13.md); H200 execution is still
  unverified and must use the pinned toolchain/dependencies.
- Prepare access using the existing authorized Vast session; do not create new
  credentials implicitly. Confirm a usable terminal/SSH path before spending.
- Record the selected offer ID, hourly compute/storage rates, upload/download
  charges, and billable units. The observed $28.915/hour listing is a snapshot,
  not a guaranteed price or an all-inclusive quote.
- Reserve money for transfer and teardown; derive the maximum rental deadline
  from the actual quote. Killing the BFS process does not stop rental billing.
- Prepare a way to stop billing and preserve small logs on failure. Do not start
  if the stopping procedure or total-cost bound is unresolved.

## Physical hardware gate

Record `nvidia-smi -L`, `nvidia-smi topo -m`, driver/CUDA/NCCL versions and free
GPU/host/disk memory. Require eight distinct H200 devices; verify the actual
interconnect rather than assuming NVLink from the model name.

After setting library paths to the built native and library shared objects:

```sh
cargo test -p mgbfs-runtime --features cuda,library-owner --test library_multi_gpu --release --no-run
cargo test -p mgbfs-runtime --features cuda,library-owner --test library_multi_gpu --release library_eight_rank_layers_and_archives_match_oracle -- --ignored --exact --nocapture
```

The test covers native CUB and cuCollections, pre-dedup ON/OFF, DENSE/HASH_FIRST,
identity/permuted ownership, S4 and U3m3. It uses eight device threads. It must
actually execute: an ignored-test count is not PASS evidence.

Run all four Compute Sanitizer tools on the produced test executable with the
same exact test filter. Save exit statuses and complete logs. Then separately
test the CLI with eight torchrun processes; the device-thread fixture does not
prove process bootstrap. Use a fresh rendezvous/output path for every case.

## LRX13 acceptance

The platform-independent streaming entry point is now
`python scripts/streamed_bfs_launcher.py --config <explicit-config.json>
--run-dir <new-directory> --source <pinned-checkout>`.
It creates one FIFO/consumer per rank, monitors consumer failures during BFS,
uses a shared search/drain/publication timeout, and requires reference-checked
HF promotion. It does not provision the machine or stop billing.
Local Linux container tests cover real FIFO creation and process supervision;
they do not prove NCCL, GPU performance, or successful remote HF publication.
Production capacity/pool settings and an independent rental teardown deadline
must still be selected before renting.

- Identity start, L/R/X (cycle, inverse cycle, swap), full group, no coset.
- Set `MGBFS_STATE_CODEC=permutation_u8` and matching archive codec explicitly.
- Set capacities explicitly: 13! exceeds the CLI's u32 default capacity range.
- Choose CUB/cuCollections from matched bounded H200 measurements, including
  full-device memory and mandatory archive. No automatic backend fallback.
- Preserve archive output. Current non-streaming reference code reserves the
  full graph payload *per rank*; do not size disk by dividing that reservation
  by eight. Configure and validate streaming before using a smaller disk.
- Budget pinned archive plus Parquet/upload staging for all eight ranks. A
  fluid-queue replay is a diagnostic estimate, not a hard capacity guarantee.
- Collect all rank summaries, per-depth times, allocation plans, external VRAM
  sampling and build/config manifests. Require all ranks COMPLETE.
- Run `python scripts/verify_lrx_layers.py RANK_RESULTS --world 8`: expected
  6,227,020,800 states, 79 depth entries, diameter 78. This checks counts only.
- Verify archive commits/checksums and publication receipts separately. Upload
  large artifacts directly from the remote machine, never through the laptop.
  Promote with `scripts/promote_hf_stream.py --repo-id
  TryDotAtwo/multigpubfs-bfs-results --world-size 8 --reference
  data/reference/lrx13-layers.json <all-eight-rank-stream-commits>`.
  The reference flag checks archive layer counts before any Hub access, including
  publication reconciliation. It is a histogram gate, not a full-state oracle.
- Record actual billed cost and terminate the rental after durable outputs are
  secured. Report partial/capacity/deadline failures as INCOMPLETE.

Unresolved before rental: bounded launch and
teardown procedure, streaming/archive capacity configuration, final offer quote.
