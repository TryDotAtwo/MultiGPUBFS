# Startup diagnostic v11 acceptance

Source: 8458e883db60761eb5e3919653a0f27ee21a54df.
Release: native-pip-20261014-v11, package 0.2.0.dev11.

- GitHub Actions 38574783202: 47 diagnostic regression tests and 3 startup
  checkpoint/thread tests passed.
- GitHub Actions 38574613919: full portable native compile, wheel assembly and
  clean Linux installation passed. Upload 38577840941 passed.
- Private Kaggle startup-trace-acceptance v2: COMPLETE on two T4 GPUs.
  Required stages PASS; two loader fault-only probes correctly SKIPPED.
  Both independent memory-info and hardware-info probes, persistent workers,
  known S3 BFS, NCCL communicator/all-reduce and multi-GPU S3 BFS passed.
  Full healthy diagnostic took 71.22 seconds.
- All six inventory checkpoints and standalone graph-info hardware checkpoint
  were asserted in actual native-startup.log files. ZIP member checksums were
  verified. Timed-out, invalid-device and missing-native diagnostic scenarios
  produced expected failures and no surviving test process groups.
- v1 completed the healthy diagnostic successfully, then its acceptance harness
  used the wrong directory name. v2 fixed only that test path; wheel unchanged.

Sanitized acceptance receipt is doctor-acceptance.json in the Release.
The RTX PRO 6000 Blackwell server-specific hang remains unconfirmed until its
owner runs v11. T4 acceptance does not establish Blackwell acceptance.
