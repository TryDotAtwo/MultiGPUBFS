# Production warmup replay gate

The existing independent-process replay accepts `--bench-warmup`. It enables
the production runtime's shared warmup/measure orchestration and verifies
the measured archives against the independent full-state/depth oracle.
Both authenticated rank records must contain boolean `warmup_completed=true`.
Missing, false, string and integer lookalikes fail verification. Default
replays now explicitly verify `warmup_completed=false`.

The existing typed gate adds `typed_warmup_gate`, two CUCO_RANK configurations
(DENSE/HASH_FIRST, three physical source banks, three completion credits).
Each uses real `mgbfs run`, healthy archive verification and the existing
asymmetric startup/constructor/owner/archive/capacity fault matrix.
No additional runtime or process framework is introduced.

Local validation: oracle rejection test RED then GREEN; 238 Python tests
OK/8 skipped, 60 script tests OK/3 skipped. Both warmup configurations pass
the actual CLI offline admission. GPU execution remains pending.
Warmup replays now also flip the requested setting on each rank separately
in two additional failure cases. A launcher-only control is removed before
starting the rank; the runtime sees ordinary contradictory warmup settings.
Both ranks must exit nonzero within the bounded deadline, report the real
configuration fatal, and produce neither measured group nor per-rank COMPLETE.
The active v103 uses the earlier immutable source without these new cases;
their hardware execution remains pending. The launcher tests cover both
flip directions and preservation of the unselected rank/environment.
