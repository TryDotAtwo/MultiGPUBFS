# v90 follow-up: incomplete verification, real registration failures

Notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version90,
scriptVersionId355137539; source749c691363007836969536c8629e9b30bc92e842.
Worker COMPLETE, report INCOMPLETE. Two physical P2P-enabled T4s.

The new notebook mode omitted pyarrow from its private verifier environment.
Successful rank searches therefore reached `verify_process_archives` and failed
on importing export_hf_dataset. Do not interpret all twenty replay failures as
GPU failures. The archive verifier dependency is now installed for every typed
mode and imported before the long NCCL build.

Downloaded small JSON/log artifacts show three of eighteen initcheck pairs
produced both COMPLETE rank records: DENSE/banks2/repeats0,2 and
HASH_FIRST/banks4/repeat0. Remaining fifteen pairs produced no rank records.
Their zero sanitizer error count is not acceptance. A representative failed
DENSE/banks3/repeat0 log reports `LSA_ACTIVATE_GROUP: CUDA_STATUS_8`, with
NCCL dev_runtime.cc:1037/611 warnings `unspecified launch failure`; registration
inserted the window before device-communicator creation failed. Root cause
remains OPEN; this is not proof of a vendor-only defect.

Both timeline cases produced two COMPLETE rank records, group completion
markers and rank nsys-rep files. Full-state archive verification and timeline
analysis still need recovery using retained Kaggle outputs. No full gate or
performance claim follows. Local files: build/kaggle-v90-observation.
State archives were not downloaded to the workstation.
