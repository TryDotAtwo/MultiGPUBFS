# External-rank history selection checkpoint, 2026-10-10

Two physical RTX3060 on one host, independent worker directories and TCP/NCCL bootstrap. Five policy tests passed. The external-rank GPU suite passed complete720-state rolling graph, resource-stop retention, directed matrix graph, graph mismatch rejection, history mismatch rejection and collective tuning/full40320-state CPU oracle.

Collective tuning now compares HASH transport/order/batch variants plus SORTED_RUNS owner lanes1/2/4/8 at common admitted capacity and common completed layer prefix. Production receives the selected history and lane geometry. Explicit history/candidate-order constraints exclude incompatible profiles. Graph/native/parameter identity now includes history and lanes before native workers are started.

Still incomplete: sorted CUB allocation probes for remote rank devices currently execute on the coordinator's local GPU IDs. Constructor revalidation rejects an under-admitted shape, but rejection does not fulfill automatic heterogeneous multi-host planning. Each rank must query its own native shape and participate in common-batch negotiation before the final plan. Physical multi-host8/128 and Blackwell acceptance remain unverified.
