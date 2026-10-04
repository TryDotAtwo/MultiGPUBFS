# Native rank gate v4: single-GPU pass, multi-GPU unsupported

Source b47c2707bc3dd346ae0cdc8c2b6df7ff45198443.
Notebook trydotatwo/mgbfs-lsa-typed-stress-t4 v4, worker COMPLETE.
Report NATIVE_SINGLE_GPU_PASS_MULTI_GPU_UNSUPPORTED. Both physical T4
GPUs reported cudaDeviceCanAccessPeer allowed0. No two-rank gate ran.

All24 independent single-GPU runs pass canonical full-state S4 oracle:
physical GPU0/1 x CUCO_RANK/CUB_SORT_MERGE/BMMA_BUCKET x DENSE/HASH_FIRST
x local pre-dedup ON/OFF. Each has24 states and layer sizes
[1,3,5,6,5,3,1], checksummed rank/group commit and durable archive.
These use the reference_bench entry point, not typed production.
No source-bank reuse is exercised by these single-GPU cases.

CUB_SORT_MERGE and BMMA_BUCKET bounded owner leaves pass memcheck,
racecheck, initcheck and synccheck (eight leaf gates total).
The host NCCL wrapper test passes API-boundary doubles; the Tensor
generation hardware-conditional counts/archive gate also passes.
Neither substitutes for full two-rank sanitizers or timeline.

Root summary retained in build/kaggle-native-v4-summary. Selected small
JSON/log evidence download is in progress under build/kaggle-native-v4-observation.
Do not republish the notebook until that download completes.
Future native_rank_gate admission requires P2P before building/running;
this avoids repeatedly spending a build plus24 cases on hosts incapable
of the required two-rank LSA tests. It does not waive any test on admitted hosts.
