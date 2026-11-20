# Batch bitmap integration checkpoint, 2026-11-20

Experimental checkpoint: batch remains gated behind `MGBFS_SEGMENT_TEST_BATCH=1` and is not a supported production setting. Hash batch is a standalone prototype, not yet connected to BFS. Do not interpret this commit as completion or a performance claim.

The native bitmap batch path has passed two-GPU exact full-layer comparison against a CPU oracle on S4, a directed C6 and a repeated-symbol graph. The test switched batch -> hash warp -> bitmap atomic -> batch in one persistent worker session, with 3-row segments and 2-entry tiles. This establishes correctness for those cases only, not large-graph or production acceptance.

The route test verifies wide-state payload, overflow without partial scatter and retry. The hash prototype adds seed and bank-retirement checks. Further integration, collision/pressure testing, complete resource accounting and A/B/C benchmarks remain outstanding.

Rental source checkpoint: e3312346d13bcdfd99ed232134051cd2fd6008b9.
