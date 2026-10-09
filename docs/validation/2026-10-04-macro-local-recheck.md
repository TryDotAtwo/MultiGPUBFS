# Current macro GPU recheck

Source HEAD: `9830889`; native libraries unchanged from the current local
route-bank build. Hardware: one RTX 3070 Laptop (sm86), Linux Docker CUDA
runtime. No rental, Kaggle launch or new remote publication in this check.

`cargo test -p mgbfs-runtime --features cuda,library-owner --test macro_native
-- --nocapture --test-threads=1`: seven passed, zero failed, 0.54 seconds.

Coverage: full-state unitriangular U3/F2, U3/F3 and U4/F2 layers for K=1/2/3;
nonidentity U3/F3 source for K=1/2/3/10; compact S4/S5 layers for K=1/2/3;
pre-dedup OFF/ON and partial batches; complete verifiable archive; sticky
capacity failure; preflight before buffer allocation; two-bank allocation.

`cargo test -p mgbfs-runtime --features cuda,library-owner --lib
macro_produce_captures_and_runs_without_host_count_readback -- --nocapture
--test-threads=1`: one passed, 0.14 seconds. The actual producer is captured,
launched and settled against S4 depth-two oracle, with device counts checked
against conservative bounds after capture execution.

This is not multi-rank macro acceptance. `prepare_production` still rejects
macro_depth != 1 with RUN_MACRO_DISPATCH_UNAVAILABLE. The reference macro
dispatcher remains single-rank. Settlement and complete GPU/NCCL DAG overlap
are not proved by producer capture. These are correctness tests, not A/B
measurements; neither time is a throughput or production search-time claim.
