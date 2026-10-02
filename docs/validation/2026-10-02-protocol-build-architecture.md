# Physical protocol build architecture

Runtime source: bb97d5a7674d1cc315b840e75690d4844b128082.
Own diagnostic lease: 53907608, machine 149819, two RTX A4000.
This is not the mandatory T4 acceptance.

The first seven replay invocations failed their initial healthy case. They
are not completed sanitizer or fault-injection gates. Both independent ranks
exited 1; initial DENSE case took 1.656 seconds, without forced cleanup or
group COMPLETE. A healthy-only replay with NCCL_DEBUG=INFO exposed
enqueue.cc:1703 CUDA failure `named symbol not found`.

The notebook incorrectly built NCCL, native CUDA and library owner with sm75
even for the explicitly admitted A4000 diagnostic (sm86). Build target now
follows the admitted hardware, rejects unknown targets before provisioning,
and is recorded in the report. T4 and RTX2070 remain sm75, A4000 uses sm86.
No runtime fallback, payload-padding or synchronization change is introduced.

Regression was RED for missing hardware selection, then GREEN. Python suites:
scripts discovery 23 tests, 1 skipped; tests discovery 203 tests, 8 skipped.
Existing unrelated worktree changes were preserved.

Physical rebuild of all three dependencies under sm86 is in progress. The
architecture hypothesis is not a confirmed resolution until a healthy
two-process replay passes. Initial NCCL variant remains explicitly experimental
minimum_arch_guard, with the original wheel and prior sm75 build retained.
