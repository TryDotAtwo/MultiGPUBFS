# Current universal BFS acceptance audit

The current public entry points accept CayleyPy graph definitions and JSON graph definitions. Public generic runs retain compact terminal states. Existing specialized full archives remain a separate mode. Exactness evidence must precede throughput claims.

| Requirement | Evidence and boundary |
| --- | --- |
| Exact permutation and integer matrix semantics | CPU graph-definition tests, independent GPU layer oracles, byte and int64 states, directed/all-visited and inverse-verified rolling history. Catalog coverage is representative constructors, not exhaustive puzzle parameters. |
| One and two physical GPUs | Fresh installed public API and external-rank tests, including 120/257/720/40320-state exact cases. |
| Rank-local admission | Actual isolated device visibility and weighted capacities. Logical 128-rank policy tests do not establish physical 128-GPU operation. |
| HASH and sorted owner lanes | Actual two-GPU oracles and bounded conditional owner retry. Sorted mode remains explicitly selectable; profile selection is bounded, not a proof of global optimality. |
| Cold host admission | RAM/cgroup and output/temp disk estimates. Rank-local refusal propagates through the collective control service before GPU workers start. Estimates exclude native pinned transfer allocations. |
| Runtime provenance | Actual binary, Python source and resolved CUDA library hashes; package source commit accepted only from a verified bundle manifest. Unbundled source commit is unknown, not inferred. |
| Control lifecycle | Phase results released after every rank acknowledges consumption; host refusal fault terminates both ranks without launching GPU workers. |
| Physical multi-node, 8/128 GPUs and Blackwell | Not accepted by the available two-RTX3060 evidence. Multiarchitecture compilation is not physical GPU acceptance. |

## Fixed-work scaling result

Six alternating complete runs processed the same 3,628,800-state graph (10 moving and 15 fixed positions), with three runs on each of one and two RTX3060. HASH, parent transport, four shards and batch 65536 were fixed; aggregate state capacity was equal. Completed layer counts and canonical compact terminal states agreed.

Median completed-layer time was 0.047940733 s on one GPU and 0.568540260 s on two GPUs: two GPUs were approximately 11.86 times slower for this configuration. BFS wall timing also favored one GPU. This is a negative scaling result, not an acceleration claim. A communication bottleneck is a hypothesis until profiling identifies it. The result does not establish performance at larger frontiers or on other hardware.

## Immutable delivered evidence

Previously delivered host/control package: source a37480a9eb92e479b9eee5d9bc98063c5736e5e9; public release revision 77903224f739334802e3fac139ab3c3ec6487aef; private proof revision b774e2dc8117dc56d485ea58c811e8c9567d9cf0. Local-rank proof revision 7e49f8e5e5866724dfb509cb4435a5257d2329cf and bounded-retry proof revision 3e780ddbe74ade61e35f57ae02f87dd6fa8b668b preserve earlier gates. Current cold disk/provenance extensions and portable native build require a new package receipt before being described as delivered.

Only one GitHub branch remains: codex/integrated-bfs-best-practices-delivery. Remaining acceptance work includes clean installed verification of the latest changes, proof publication, artifact delivery and verified rental teardown. The broad goal remains active.

## Bounded-transfer follow-up

Live-chunk queue transfer preserves each allocated receiver stride and transfers full-state SoA planes separately. Parent SoA planes are compacted at the live chunk stride used by every reader. No new child-count readback is introduced. Twenty two-RTX3060 exact-layer gates cover HASH/SORTED, rolling/all-visited, wide byte/int64 states, parent/full transport, collisions and resource-stop preservation.

Three alternating matched two-GPU runs on the same 3,628,800-state graph gave median completed-layer times 0.566575350 s before and 0.395965297 s after, a ratio of 1.430871. Background CPU compilation was present for both variants. This is not maximum-frontier, 8/128-rank or Blackwell throughput acceptance; two-GPU scaling still underperforms one GPU on this fixed case.

The attempted eight-target CUDA build hit its explicit time limit. The changed NCCL object was linked with unchanged accepted SM86 CUDA objects; other architecture acceptance remains pending.
