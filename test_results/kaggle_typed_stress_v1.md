# Updated runtime: 72-case U4(F3) full-state gate

Notebook `trydotatwo/mgbfs-lsa-typed-stress-t4`, v1, worker COMPLETE.
Source `5a4ea73f5dd58a5ed58e0178caedc60b991514e6`.
Report TYPED_STRESS_PASS: all 72 expected configurations pass, no timeout.
Two independent processes on two physical Tesla T4 GPUs, sm75,
bidirectional P2P admitted. This is full BFS correctness, not performance.
Summary SHA256:
`3371a7509b2be344e601947138dcb5617364b662220d1195146ea9aeac7d6640`.

Cross-product: CUCO_RANK / CUB_SORT_MERGE / BMMA_BUCKET;
DENSE / HASH_FIRST; local pre-dedup OFF / ON; logical rank maps01 /10;
source banks2 /3 /4. Completion epoch window3. Each replay requires
source-bank reuse and verifies archived canonical states at every depth
against the independent CPU oracle; 729 unique states of U4 over F3.
Layer counts: [1,6,20,56,116,208,268,52,2].

The observed BMMA DENSE/preOFF/map10/banks4 detail shows zero exits for
both ranks, group COMPLETE, no forced cleanup, and26 source-bank reuses
per rank. Its process wall time is not search time and is not an A/B result.

Main summary retained in build/kaggle-typed-stress-v1-summary-only.
Detailed summaries and rank logs are being retained separately in
build/kaggle-typed-stress-v1-observation; download must finish before replacing
this notebook version. No large state archive was downloaded to this machine.

Still open: updated full asymmetric-failure matrix, four full-runtime
sanitizers (especially initcheck), full-BFS timeline, paired A/B, production
macro-depth integration and remaining agreed publication/configuration gates.
v97 in the other notebook is the live full-runtime acceptance job.
