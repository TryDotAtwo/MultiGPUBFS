# dev33: one-shot recovery diagnostics

`python -m multigpubfs doctor --recover --devices 0,1,2,3,4,5,6,7 --total-timeout 1800`

Safe-first communication probes, fresh-process health gates after failures, exact CPU-oracle BFS, measured UT(4,47) tuning, loading-order/API and BFS-mode checks, identity-bound launch commands, and one checksummed metadata archive. Diagnostic children remain in the supervisor process group. Production BFS keeps its unlimited default and is never automatically restarted.

Source validation: complete PASS/READY runs on 2 x RTX 3060 and 8 x RTX 5060 Ti (SM120), including the public profile launcher. UT(4,3) completed with exact CPU layers and terminal states; UT(4,47) was resource-limited and is not a full traversal. The all-peer exchange used every selected GPU. The original RTX PRO 6000 servers and B200 were not available for acceptance.

Python validation: 882 existing/current tests passed and 210 subtests passed (one compiler-dependent test passed on rerun after installing g++); one test was skipped. The added failure-metadata archive test also passed, taking the cumulative count to 883. The final focused suite passed 41 tests. Native binary and architecture components are unchanged, pinned and checksummed.

See [source receipts](recovery-dev33.json) and [usage and limits](../recovery-diagnostics.md). Installed release acceptance is recorded separately after the GitHub wheel is available. Failed experiments remain visible even when a verified workaround is READY. A working small/prefix test cannot prove every graph fits or that the kernel driver will never fail later.
