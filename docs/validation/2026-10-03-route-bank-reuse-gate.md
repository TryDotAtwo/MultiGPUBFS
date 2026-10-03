# Within-depth physical route-bank reuse gate

Runtime base: `83099a1`; this follow-up adds a submission counter, not another
scheduler, allocation or synchronization. `route_bank_reuses` increments only
when an already used physical bank is assigned another parent batch within the
same depth; it does not count reuse across depth boundaries or imply overlap.

Why: S4 has a peak layer of six states. Split over two ranks, it cannot reliably
force four source banks to be reused within a layer. U4 over F2 has exact layer
sizes `[1,3,5,8,11,13,13,8,2]` (64 states). With batch size one and two ranks,
a 13-state layer forces at least seven parent batches on some rank, independent
of the hash partition. This exercises actual within-depth reuse for all bank
counts 2/3/4. The gate requires a positive aggregate reuse count, while allowing
one rank to have zero because ownership may be asymmetric.

## Changes and evidence

- Runtime exports the actual counter in each rank result, without GPU readback.
- Replay rejects missing, negative, non-integer and all-zero reuse evidence when
  `--require-bank-reuse` is requested; ordinary legacy replay remains compatible.
- The existing typed hardware gate adds six U4/F2 configs (both profiles,
  banks2/3/4) to the 48 S4 configs. No second runtime or receive bank is added.
- RED: GPU fixture failed to compile because the counter API was missing;
  Python tests failed because the reuse config builder and verifier option were
  missing. GREEN: exact per-depth counter, full states, hashes and archive passed
  on the real local RTX3070Laptop for all six CUCO_RANK cases. Total within-depth
  bank reassignments were 47/40/34 for banks2/3/4 respectively in each profile.
  HASH_FIRST generation was scalar on this local sm86 stack.
- All-target CUDA/library-owner check passed; full CPU Rust suite passed;
  project Python suite passed 246 tests, skipped14. Existing unused-mut warnings
  remain. Bare pytest's duplicate vendored CayleyPy collection problem is
  unchanged and was recorded in the preceding receipt.

No new Kaggle notebook or own Vast rental was launched. The new two-rank U4
gate has not run yet. Four sanitizer and Nsight results in the preceding receipt
belong to the earlier binary; they are not silently relabelled as this revision.
