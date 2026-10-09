# Graph integration boundary after physical route banks

Main inspected revision: `ac24a0d` (runtime bank implementation `ce1e0e7`).
The isolated `bfs-tail-archive` worktree was inspected read-only. Its Graph
runtime is based before the current physical route-bank and cancellation work.

Comparing its `distributed_native.rs` against main removes `route_banks`,
`epoch_window`, and `state_descriptor_capacity` from `DistributedConfig`, the
`RouteBank` storage and bank-local events, mapped terminal-word admission checks,
and the within-depth bank reuse counter. These are ancestry differences, not
proof that the isolated branch itself introduced each regression.

Do not replace main's runtime wholesale with that branch. Integration must
preserve the current allocation accounting, bounded completion credits, one LSA
receive slot, device-fatal admission, bank last-reader dependencies, and
FinalizeDepth. Captured source pointers and producer event lifecycle must refer
to the actual physical bank; a successful single-source-bank Graph fixture does
not prove the 2/3/4-bank contract.

The Graph owner was sent the concrete differences and asked for the integration
revision/event contract and a bounded, noncompeting two-rank test window.
No code in that worktree or remote GPU processes was modified by this check.

Read-only remote observations on owner-managed instance 54052996:
`/root/tail-integration-gate.pid` contained 3617, but `/proc/3617` was absent;
`report.json` was `INCOMPLETE` with `RuntimeError: background publication failed`.
A subsequent process lookup found no matching integration, automatic-tail,
replay, or reference processes. This is terminal evidence for that observed job,
not authorization to take over the owner's rental or proof that a repair will
not start later. No main two-rank acceptance run has been launched here.

The full goal remains open: target two-T4 failures/sanitizers/timeline, paired
A/B, and the remaining agreed profiles/archive/HF gates are not closed by this
compatibility inspection.
