# DENSE lookahead hardware gate and comparison launch

Runtime `013ed5c979f4225db273e0015fa9ed72fd230c90`.
Kaggle `trydotatwo/mgbfs-dense-lookahead-t4`, v1: COMPLETE.
Downloaded source-sha and summary match the runtime. The downloaded notebook
log contains 13 passed / 0 failed for each full archive fixture invocation:
plain, memcheck, racecheck, initcheck and synccheck. All reported sanitizer
summaries are clean, including zero racecheck warnings. The summary reports
all 24 reference profile smoke selections PASS. Individual raw-file
reconciliation is still pending; do not conflate this with that stronger audit.

Evidence: `test_results/dense-lookahead-v1-summary` (summary, source SHA,
notebook log); raw download target `test_results/dense-lookahead-v1-raw`.

Raw acceptance completed later on 2026-09-15: the separate raw-file auditor
returned `RAW_GATE_RECONCILED` for 40 tool logs, 36 measured rank records,
36 warmup records and 36 verifier outputs. It required exactly 13 archive
tests and three scatter tests per tool, checked each profile selection and
global layer counts, and matched the source SHA. Its backward-compatibility
check still accepts the old 12-test v46 gate; supplying 13 for that old gate
is rejected with `FIXTURE_RESULT_MISMATCH`. The pending raw-audit statement
above records the launch-time evidence boundary and is now superseded.

After this gate, two private v1 comparison notebooks were submitted and their
RUNNING states confirmed:

- `trydotatwo/mgbfs-lookahead-s11-comparison`: two active T4 ranks.
- `trydotatwo/mgbfs-lookahead-s11-one-t4`: one active T4 rank.

Both pin the same runtime, preserve the existing five-repeat profile panel,
mandatory native archive and immutable CayleyPy baseline. No new timing or
memory result exists yet. The gate does not complete the production dispatcher
or establish a performance benefit from lookahead. S13 was not rerun.
