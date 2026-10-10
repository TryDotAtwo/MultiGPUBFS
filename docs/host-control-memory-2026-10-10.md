# Host control memory checkpoint

The public launcher reads Linux MemAvailable and visible cgroup v1/v2 current/maximum memory, including mount-relative ancestors, before native launch. Unknown inventory remains unknown. A conservative compact-terminal and graph/control-copy estimate is checked against half the available budget. This is a cold control estimate, not a guarantee for all native pinned allocations or externally changing memory pressure. The external JSON payload limit is checked before calculation.

External ranks acknowledge reading/writing each phase result before rank zero releases that phase's rendezvous values. Pilot terminal snapshots no longer accumulate until the end of the sweep. Output evidence on disk is retained. No checks were added to the GPU iteration path.

Forced SORTED owner lanes above one exposed an independent inventory bug: the inventory query used one shard while inheriting multiple owner lanes. Inventory now uses neutral HASH geometry; actual SORTED memory admission still executes independently on each local GPU with the requested geometry.

Four synthetic memory tests cover nested parent v2 limits, v1 unlimited sentinels and zero headroom, unknown inventory and insufficient-memory rejection. The control test covers phase release and preservation of unrelated keys. The two-GPU external suite verified complete720/40320-state permutations, a matrix, resource-limited exact retention and configuration mismatches. An isolated-device0 test verified forced SORTED four lanes and720 states. Physical multi-host and8/128 GPU configurations remain unverified.
