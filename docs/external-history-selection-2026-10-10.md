# External-rank history selection checkpoint, 2026-10-10

Two physical RTX3060 on one host, independent worker directories and TCP/NCCL bootstrap. Five policy tests passed. The external-rank GPU suite passed complete720-state rolling graph, resource-stop retention, directed matrix graph, graph mismatch rejection, history mismatch rejection and collective tuning/full40320-state CPU oracle.

Collective tuning now compares HASH transport/order/batch variants plus SORTED_RUNS owner lanes1/2/4/8 at common admitted capacity and common completed layer prefix. Production receives the selected history and lane geometry. Explicit history/candidate-order constraints exclude incompatible profiles. Graph/native/parameter identity now includes history and lanes before native workers are started.

Local sorted-memory planning is now collective: each rank invokes graph-local-plan on its own physical device, initial plans negotiate a common batch, each rank re-queries that batch locally, and the coordinator combines distinct capacities with weighted owner cuts. Shared transport geometry and graph identity are checked before native workers launch. Ten policy/aggregation tests passed, including logical128-rank differing capacities/scratch and negative identity/geometry cases. The full external GPU suite passed again.

The isolated-visibility GPU gate also passed720 states and exact terminal retention: each process exposes only its own physical card as local device0; the resulting rank device vector is[0,0]. This establishes that local GPU IDs need not be globally unique and that coordinator access to peer devices is unnecessary. It is one physical host, not a multi-host throughput result.

Still required: release/package acceptance for this updated CLI/Python protocol, private evidence readback and delivery. Physical multi-host8/128GPU and Blackwell remain unverified.
