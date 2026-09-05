# S13 owner-local v4: time limit, not observed capacity failure

Source f13ef1e514b680337177ad3b81962a8050681123. Capacity 220M/rank,
batch 262144, archive disabled. Physical 2xT4 peak 13985 MiB/rank.
Harness terminated at its 1800-second process time limit (TIMEOUT, -9).
Depth 51 expansion completed; depth 52 contained 179272026 + 179268874
= 358540900 states. No ring fatal was reported before timeout.
Depth 50 took about 237s; depth 51 about 263s. These traces show progress,
not proof of a deadlock. Graph exhaustion, peak frontier and complete
cardinality remain unknown. No graph publication.

Evidence: test_results/s13-owner-local-v4/s12-capacity-probe/.
Next run may extend time without changing batch/capacity, to separate the
configured time limit from the memory boundary.
