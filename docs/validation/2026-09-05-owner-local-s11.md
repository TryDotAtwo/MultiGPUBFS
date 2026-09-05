# Owner-local buckets: S11 v8

Source f13ef1e514b680337177ad3b81962a8050681123, physical 2xT4.
All 56 S11 layer counts agree with v7; archive-on and off counts agree.
Peak memory 945 MiB/rank versus 1075 MiB/rank before local bucket storage.
Search with archive 2.180935756s; without archive 1.915728666s;
durable completion 9.675757338s. Single runs; no timing speedup claimed.

Directory tests validate local ranges for both logical owners and reject the
wrong owner. memcheck/racecheck/initcheck/synccheck all clean on both T4s.
These are primitive checks, not full-runtime sanitizer or full-set equality.
Swapped rank-map end-to-end gate is still outstanding for this change.

Evidence: test_results/s11-owner-local-v8/s11-distributed-probe/summary.json
and owner-directory-gpu*-*.log. S13 is not yet complete.
