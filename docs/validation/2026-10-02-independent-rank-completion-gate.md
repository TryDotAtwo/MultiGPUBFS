# Independent-rank failure completion gate

The existing replay now rejects failure cases that leave any rank COMPLETE,
not only a group-complete marker. Malformed or unknown rank reports fail
closed; absent reports and explicit INCOMPLETE remain allowed. The check is
used by the actual existing replay verdict after both child exits, not by a
parallel harness. No runtime hot-path change, NCCL call or payload padding.

Real temporary-file regression fixtures reproduced two RED acceptance bugs:
rank COMPLETE without group marker and truncated rank JSON were accepted.
All five fixtures are GREEN after the fix. Full Python discovery reports
22 tests, zero failures, one Windows skip (real Linux process cleanup).

Kaggle admission was rechecked this turn: the existing private
trydotatwo/mgbfs-native-rank-owner-t4 worker was terminal COMPLETE; push of
the prepared source-pinned gate was rejected for the weekly 30-hour GPU
quota. CLI exit 0 is not admission. No new worker or second notebook started.

This proves replay-verdict behavior, not two-T4 runtime correctness. Fresh
asymmetric rank processes, T4 registration/initcheck, full timeline and
paired A/B remain open. Existing colleague output provenance was requested;
no foreign GPU process, source checkout or lease was modified.
