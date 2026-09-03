# Native matrix BFS: архитектура исполнения v2

Дата: 2026-09-03. Режим: автономная проработка по запросу пользователя.
Это единый актуальный design contract первого backend. История:
[ARCHITECTURE_NEED](../ARCHITECTURE_NEED.md). Hash wire algorithm:
[GEMM_U8_P32X4_V1](gemm-hash-v1.md).

Статус: спецификация и самостоятельный статический аудит; production runtime
не реализован. Новые конкретизации ниже — решения этой проектной ревизии,
а не утверждение об их ранее полученном одобрении или измеренном ускорении.
Переход к коду — только после замыкания перечисленных contract gates.

Первые исполняемые проверки контрактов и оставшиеся gates:
[architecture-models](validation/2026-09-03-architecture-models.md).
Они не являются реализацией production scheduling.

## 1. Что произошло и что запрещено повторять

Проверочный `DenseDeviceStepper` был использован для большого performance test
до появления шардового owner, StateRing и архива. Это не допустимая подмена
согласованной архитектуры. Тест полезен как отрицательный результат прототипа.

Подтверждено кодом `cuda/owner.cu:106-140`:

- каждый parent batch повторно сливает целиком prev/current/accepted;
- два массива 32-byte Record имеют размеры `3*layer_capacity + candidates`
  и `3*layer_capacity`, дополнительно три index planes и два flag planes;
- два CUB Select обходят bound с запасом, а не только живые кандидаты;
- проверка sorted старых слоёв повторяется на каждом батче.

В измеренной конфигурации m24 эти owner arrays занимают 7.1085 GiB без CUB
scratch из 11.0273 GiB native allocation delta. Две state arenas — 0.9537 GiB.
Это объясняет перерасход памяти; точный вклад стадий во время ещё не измерен.
Источник чисел: [large comparison](validation/2026-09-03-large-single-gpu-result.md).

Запрещено переносить этот owner в production с переименованием в shard.
Ни один owner job не получает указатель и count целого rank-local hash layer.
Его вход — только заранее ограниченные bucket ranges. Scratch не зависит от
общей capacity слоя. Отдельные primitives остаются тестовыми reference tools.

## 2. Семантика и границы

Single-node Linux, один native Rust процесс на GPU, CUDA/C++ через C ABI.
`torchrun --standalone --nproc-per-node=W --no-python mgbfs run ...` — launcher,
не data plane. Первый hardware gate: реальные два T4. Multi-node, пути,
bidirectional/target search, permutations, recovery, migration вне v1 runtime.

Manifest: квадратные n*n canonical row-major u8 matrices, modulus q=2..256,
invertible start A, normalized generators G, inverse map. Действие однозначно:
`child = G[move] * parent mod q`, как в существующем matrix oracle.
Нормализация/дедуп генераторов происходит до config digest; порядок фиксирован.
Каждый опубликованный parent применяется к каждому generator ровно один раз.

Проверяется inverse closure, включая произведения с обеих сторон. Тогда
`Fnext = unique(children(Fcur)) \ (Fprev union Fcur)`.
До FinalizeDepth дети принятого next не порождаются. Пустой глобальный next
после drain означает COMPLETE; capacity/fatal — INCOMPLETE.

Identity — Hash128, не full-state comparison. Полные state bytes нужны для
следующего шага и обязательного архива. Условная корректность при отсутствии
коллизии, seeded SHAKE assumption и p^-4 bound — строго по hash spec.
Смена seed создаёт независимую проверку, а не исправляет старый архив.
Hash не объявляется криптографически collision-resistant.

OwnerCommit окончательно принимает состояние. До commit проверены old layers,
текущий incoming и ранее committed next. Ни финализация, ни поздний кандидат
не удаляют принятое. Первый committed представитель сохраняется; внутри job
tie-break `(source_rank, source_batch_seq, candidate_ordinal)`.
При коллизии представитель может зависеть от admission order; trace сохраняется.

## 3. Реестр решений

Сохранено из согласованного: два фиксированных профиля DENSE/HASH_FIRST,
pre-dedup ON/OFF, CUB_SORT_MERGE/BMMA_BUCKET, prefix ownership, flat buckets,
потоковая материализация, StateRing, async обязательный архив, fail-fast,
один semantic cut между глубинами, отсутствие fallback.

Конкретизации v2:

- owner работает с microbucket, shard — единица writer exclusion/планирования;
- old membership не создаёт merged copy старых слоёв;
- accepted update: отдельный bounded output, затем copy-back, НЕ in-place
  parallel backward merge без доказанной безопасности;
- accepted store хранит только Hash128; постоянные StateRef для каждого hash
  не нужны при независимых порядках state/hash архивов и отсутствии paths;
- один transport sequence для candidates, requests, responses и receipts;
- HASH_FIRST возвращает terminal receipts для каждого source batch/owner,
  включая zero survivors; родители не удерживаются до конца всей глубины;
- target extents HASH_FIRST резервируются в response order, до OwnerCommit;
- все gather и дополнительные GEMM materializations оплачены в byte ledger.

Отвергнуто: whole-layer per-batch merge, giant owner scratch, StateRef в каждой
промежуточной копии без потребителя, late dedup, global equality matrix,
дополнительная перестановка ownership bits, заполнение GPU бесполезным GEMM.

Открыто только для измерений: tiles, число одновременно активных lanes, batch
и microbucket capacity, выбор профиля/backend. Они фиксируются в RunConfig
до allocation; auto-selection внутри BFS запрещён.

## 4. Конфигурация и размеры

Все размеры/произведения вычисляются checked u64/size_t. Библиотечные API с
32-bit counts получают только job-local count, проверенный на диапазон.

| Symbol | Определение |
|---|---|
| D, S | n*n logical bytes; S=align_up(D,16) storage stride |
| W, H, B | ranks; shards/rank; total microbuckets/rank |
| P, C | parent microbatch; C=P*move_count |
| K | максимальный count одного bucket в любом resident hash layer |
| L | capacity compact hash arena, L<=B*K |
| R | state ring records, не capacity каждого из двух фронтиров |
| E | candidate records одного transport offer/receiver budget |
| I, J | incoming records и touched-bucket descriptors одного owner job |
| Ng,Nr,Nx,No,Nm | число generation,route,receive,owner,materialization lanes |
| A,Q | pinned archive slots и bytes/slot |

Defaults только как старт calibration: H=64, B=64*256, W=2, 1 GiB untouched
reserve на rank. P,K,L,R,E,I,J и числа lanes обязательны в конфиге, не угаданы
из полного числа вершин. `expected_max_unique_states` — collision/archive
bound, не основание аллоцировать столько записей в каждом GPU scratch.

W,H,B/H — степени двойки. Owner/shard/bucket — successive high prefix bits
числового Hash128; `logical_owner_to_rank` — проверенная permutation.
Rank map, GPU UUID map, seed128, all capacities, backend policy/build digest,
timeouts, archive extents, optional fusion mode входят в canonical config.

RunConfig wire: schema u32, затем ordered field-wise LE encoding, counted
arrays (u64 length), fixed enum numeric IDs, no unknown fields/reserved nonzero.
SHA256 полного manifest+config+policy manifest проверяется всеми ranks.
Фактические tile/scratch queries и compiler/library versions сохраняются;
перестановка одинаковых GPU не меняет logical owner function.

## 5. ABI и идентичность записей

Host/wire encoding field-wise LE. Device planes align>=256 B, records не
переносятся через сортировку полными состояниями. Padding всегда zero.

| Record | Поля | Bytes/alignment |
|---|---|---|
| Hash128 | четыре u32 residue, word3 most significant | 16/16 |
| StateRef | absolute allocation record sequence u64 | 8/8 |
| OriginRef | source_rank u32, move u16, zero u16, parent StateRef | 16/16 |
| PayloadRef | receive_slot u32, row u32 | 8/8 |
| Range | begin u64, count u64 | 16/8 |
| BucketJob | bucket u32, lane u32, incoming Range, prev Range, curr Range, accepted_count u32, generation u32 | 64/64 |
| StateExtent | seq,physical_begin,count,logical_begin u64; depth,state u32; outstanding,archive_lease,reserved u64 | 64/64 |
| MessageHeader | magic,schema,kind,flags u32; run_tag,transport_seq,source_batch_seq u64; depth,src,dst,count u32; payload_bytes u64 | 64/64 |

StateRef resolves through checked monotonic allocation sequence and ring modulo;
wrap padding consumes sequence numbers. A stale ref outside live ranges is fatal.
No descriptor ID reuse aliases an old origin. Sequence overflow is fatal.
Frozen size/offset assertions and byte vectors required before CUDA use.

Candidate planes: hash16 + DENSE state S OR HASH_FIRST origin16.
Internal tie provenance is batch header + per-row original ordinal u32; it
survives source sort and unique. A batch fragment carries its original batch
sequence; header schema explicitly versions the ordinal plane.
Request: origin16; response: state S in request order. Expected hash stays at
owner in materialization slot, never inferred from response position alone.
Receipt: source batch seq u64, owner u32, flags u32, emitted u64, accepted u64
(32 B). Control framing adds length and full run/session validation.

## 6. Память и сроки жизни

`a(x)=align_up(x,256)`. Каждая строка — отдельный pool; никакая память за C ABI
не может отсутствовать в плане. Scratch sizes — реальные startup queries
закреплённых implementations; отсутствующий query означает PREFLIGHT_FAIL.

| Pool | Bytes | Writer / reader / release |
|---|---:|---|
| StateRing | a(R*S) | materializer / parent pack + archive / leases closed, FIFO head |
| Two compact hash arenas | 2*a(16*L) | finalizer / owner + archive / final drain + D2H |
| Accepted buckets | a(16*B*K) | sole shard writer / owner + finalizer / depth rotation |
| Bucket counts | a(4*B) | owner / builder+finalizer / reset after compact |
| Two hash directories | 2*a(8*(B+1)) | finalizer / builder / arena lifetime |
| Extent/obligation metadata | a(64*Nextent)+query_obligations | coordinator / pack+archive+receipts / seq-safe retire |
| Parent banks | Ng*a(P*S) | pack / generate / generated event |
| Generation banks | Ng*(a(C*S)+a(16*C)+query_gen_temp(P)+query_hash_temp(C)) | producer / router / route copy complete |
| Route banks | Nr*(2*a(16*C)+2*a(4*C)+query_sort(C)+query_select(C)+query_wire(E)) | route / comm / send complete |
| Receive banks | Nx*query_wire(E) | comm / owner+materialize / all job refs retired |
| Owner lanes | No*Qowner(I,J,K) | one owner lane / commit / all copies complete |
| Materialization | Nm*Qmat(M,S) | request builder+response / target copy / response verification |
| Tables/control/events | query_fixed | preflight/coordinator / all / run end |
| Archive device staging | query_archive_gpu | archive gather / D2H / copy event |
| Archive pinned | A*align_up(Q,4096)+query_control_pinned | D2H / disk writer / write consumed |

Generation bank state C*S exists in both profiles for the two-GEMM path;
HASH_FIRST releases it after hashing/route metadata copy, not after owner reply.
An epilogue-fused implementation may reduce temp, only with separate policy
query and oracle gate. No claim that GEMM inherently reduces persistent memory.
Current implementation's int32 generator output and 64*C hash partial bytes
must appear in queries until a verified epilogue actually removes them.

Owner scratch concrete upper bound (SoA, no old-layer Record copies):
`Qowner = 2*(a(16*I)+a(8*I)+a(4*I)) + a(4*I) + 4*a(I)
          + a(16*J*K) + a(64*J) + a(32*J) + a(64)
          + query_merge(I) + query_select(I) + query_scan(J)`.
28I accounts hash16+payload8+ordinal4, each field plane aligned independently.
The J*K hash output is for accepted copy-back; old membership is read-only.
32*J metadata accounts four u32 category counts, survivor count u32, new accepted
count u32 and output offset u64 per bucket; 64 bytes are job totals/fatal/control.
Each library query returns named allocations, not one opaque
unexplained allowance. No term `No*L` or `No*B*K` is permitted.

Qmat includes both request order buffers, expected hashes16*M, origin16*M,
permutation4*M, response M*S, reconstruction temp and target range descriptors.
Wire query includes 64-byte headers per fragment, provenance ordinals, offset
tables and alignment. E bounds total receive records from ALL peers, not E/peer.

No automatic overlays initially. An overlay requires identical pool owner and
nonoverlapping lifetimes proved by the release event; finalization cannot reuse
archive staging or generation buffers with live NCCL/D2H leases.
Memory report sums all above + runtime/NCCL observed overhead + declared margin;
after CUDA/NCCL/module warmup and allocations cudaMemGetInfo must still show
untouched reserve. NCCL internal late allocation is a gate to measure, not a
fictional byte-exact native allocation. Report native owned bytes separately.

State bound: `R >= max_t(unconsumed_current + reserved_next + wrap_padding)`.
DENSE current releases after its independent parent copies and archive D2H;
HASH_FIRST additionally needs terminal receipts and served responses.
Predicting frontier shape is not a capacity proof: configured R exhaustion is
fatal. Small graph simulator supplies exact trace high-water; large runs supply
measured high-water and a conditional capacity envelope, never a completion promise.

Example, NOT production allocation choice: B=16384,K=4096,L=32M => accepted
1 GiB, old hash arenas 0.954 GiB. No=2,J=32 gives only 4 MiB accepted output
scratch across lanes, versus multi-GiB global merge records. Candidate/workspace,
state, archive, NCCL and bucket skew costs remain to be added. K overflow fails
even when other buckets have unused space.

## 7. Jobs и producer/consumer ownership

Each pool has its OWN state machine; a RouteSlot is not an OwnerScratch.
`FREE -> RESERVED -> WRITING -> READY -> READING -> FREE`, generation tags
prevent stale events. Slot references retired only after all consumers finish.
An occupied slot is not an allocation failure; dispatch admits a job only when
its fixed output credit exists. Once admitted, exceeding its reserved capacity
is fatal. No host/disk wait is injected into a GPU stream.

| Job | Ready condition | Completion / released input |
|---|---|---|
| Pack | current extent + free parent bank | ParentReady; DENSE enumeration lease |
| GenerateHash | parent ready + generation bank | HashReady; parent bank |
| SortRoute | HashReady + route bank | RouteReady; generation bank after payload copy |
| Exchange | globally admitted typed ticket + receive credit | ReceiveReady / send credit |
| BuildOwner | receive range + shard idle + owner lane | bounded descriptors; receive lease retained |
| OwnerCompare | descriptors and frozen counts | survivor mask/count, no persistent mutation |
| ReserveCommit | counts + target/materialize credits | committed hashes and target obligations |
| Materialize | DENSE survivors OR HASH_FIRST request ready | StateReady; payload/source leases |
| ArchiveCopy | immutable state/hash range + pinned credit | D2HDone; GPU archive lease |
| FinalizeDepth | global drain proof | next frontier published |

Separate pack, generate, route, comm, owner lanes, materialize, D2H, finalize
streams. One host dispatcher polls CUDA events and bounded completion mailboxes;
small async D2H metadata copies are explicit, not cudaDeviceSynchronize.
Disk worker never owns dispatcher progress. Host control metadata size/copy rate
is measured. Single-node TCP control is not candidate/hash data plane.

Graph templates capture a bounded job, never an entire layer. Actual counts
are parameters/metadata. Dynamic CUB num_items jobs are launched by dispatcher
after count publication, or graph variants use exact admitted count. No whole
layer bound scan to avoid a small metadata transfer. NCCL stays outside graphs.
No promise that unrelated streams have independent SM or bandwidth resources.

## 8. Generate, sort, route

Generate P parents at a time; all G children of each parent are generated.
Logical child ordinal parent_local*G+move. Matrix accumulation requires
`n*(q-1)^2 <= INT32_MAX`; hash requires `D*255^2 <= INT32_MAX`.
Zero pad canonical input, never hash uninitialized storage.

Two dependent Tensor Core jobs: exact matrix action with modular epilogue,
then seeded hash projection. Selected tiles/layouts fixed before allocation.
Output int32 intermediates may be streamed through bounded buffers; cross-job
fusion requires keeping modular reduction before hash. Multiplying algebraic
projections through a nonlinear modular reduction is NOT a valid fusion.

One full Hash128 radix sort per source batch, index payload only. Stable unique
optional, keep original ordinal. Build destination/bucket boundaries once.
DENSE gathers state payload once into contiguous destination planes; HASH_FIRST
sends origins. Sorted fragments remain sorted after bounded E splitting.
No owner-side full radix re-sort of all received layers.

Source batches may span extents; parent source mappings are retained per batch
until receipts in HASH_FIRST. Source pre-dedup discarded ordinals are accounted
locally; source batch emitted count, not raw C, determines receipt obligations.

## 9. Owner: ограниченная работа, необратимый commit

One writer per shard; separate shards can run concurrently. Shard queue orders
candidate ticket sequence; explicit empty entries advance its watermark.
Long epoch split into jobs <=I incoming and <=J touched buckets. Very large
incoming bucket split into consecutive jobs, not truncated. No job requires
all peers to produce nonempty ranges.

1. Merge sorted source segments using bounded fan-in pairwise passes on I rows;
   select deterministic equal-key representative. Segments/provenance retained.
2. For each touched bucket get prev/curr ranges from directories once. Enforce
   old counts<=K. Merge-path set-difference streams candidate and read-only old
   ranges; writes only candidate flags, never materializes union with old keys.
3. Compare remaining candidates to committed accepted[0:count], count<=K.
   All equality categories have priority: incoming duplicates, prev, curr, next.
4. Compact survivors <=I; scan per-bucket counts. Check accepted+new<=K, layer
   budget, target StateRing, descriptor and profile-specific materialization
   credits BEFORE persistent mutation. Failure stops run; no partial success.
5. Reserve StateRing ranges in desired state output order. DENSE output order is
   compacted winners; HASH_FIRST is (source, parent StateRef, move) request order.
6. Merge each touched accepted range with new sorted hashes into independent
   lane output <=J*K records. Bulk copy back after merge finishes; then publish
   counts and OwnerCommit event. No concurrent reader/writer of that shard.
7. DENSE materializer copies survivors, HASH_FIRST publishes reserved request
   obligations. StateReady is distinct from HashCommitted. Later owner jobs
   may reject duplicates immediately; next generation waits finalization.

Accepted store contains hashes only. Winners' state identity survives in job
payload/origin until materialization; no global hash->state lookup is needed
in single-source exhaustive BFS. Full-state archive verification checks that
materialized hashes equal the committed multiset. Production audit mode checks
every response hash before StateReady; no empty placeholder states are published.

Counts only visible after all matching stores/events. Startup and Finalize
validate old sortedness once. Hot owner validates descriptor generation/ranges,
not whole immutable old layers on every batch.

Work bound per job: O(merge_fanin*I + J*K + I) key operations for CUB path;
accepted update bytes depend on touched bucket occupancy, never rank layer size.
Across epochs, buckets can still be reread. Track amplification explicitly;
sharding is not a proof of linear total work. Increase aggregation/bucket count
only in a separately configured run after measuring occupancy and traffic.

BMMA_BUCKET shares owner commit/reservation path. Candidate/reference ranges
are refined by additional hash prefixes into bounded tile tasks. XOR popcount
zero means equal Hash128; OR matches into per-candidate flags. Pairwise results
never stored as I*K matrix. Same-incoming ties honor ordinal; identical hashes
at full prefix collapse by adjacent unique. Descriptor pool/work cap exhaustion
fails, never falls back to CUB. Refinement changes descriptors, not persistent
bucket allocator. BMMA cannot bypass comparison to accepted next.

## 10. StateRing и HASH_FIRST termination receipts

One flat R*S pool; FIFO extent descriptors, no per-state allocation/list.
Allocation coordinator scans ready jobs' requested ranges once per wave,
checks monotonic head/tail distance, reserves disjoint output, emits wrap marker
if needed. Each extent is a contiguous immutable state range after StateReady.
No parent is freed by merely reading its hash.

DENSE: Pack closes enumeration lease only after independent parent copy exists.
HASH_FIRST: parent live until every emitted child batch has terminal receipts
from every destination that received its fragments, and all accepted requests
have been served from that parent. A receipt with accepted=0 is mandatory.

Protocol for each (source_batch_seq,owner): source announces total emitted
fragments/count; owner tracks all processed fragments. Owner enqueues accepted
requests AND terminal receipt only after all decisions for that batch/owner.
Receipt carries accepted request count. Source may see requests before receipt;
it tracks actual served count and seals only when receipt total matches. A zero
survivor case closes without requests. Missing/duplicate fragments or receipt
counts are fatal. Local pre-dedup obligations already removed at source.

Requests are sorted by source parent ref and move before target reservation;
responses preserve that order, so owner writes contiguous target ranges, not
per-state random scatter. Batch header/range map connects each response to its
reserved target. Source can release parent after response send finishes and
all receipts close; owner need not retain source VRAM until disk completion.

Ring FIFO may delay reclaim of later extents behind a leased head. This is
accounted high-water, not secretly movable storage. Full ring -> fatal. Requests,
responses and receipts have separately reserved credits and dispatch priority
over admission of new generation. Thus progress does not depend on freeing a
candidate receive bank that is itself waiting for materialization credit.
The same partition applies to ticket/descriptor metadata, including zero-payload
tickets: candidate metadata cannot consume the reserved response/request/receipt
ticket pools. Data-plane buffers alone do not establish this progress property.
No global request-count barrier at end of depth.

## 11. Multi-GPU transport

Bootstrap file schema includes run UUID, full config digest, rank0 TCP endpoint,
NCCL unique ID and expected world size. Rank0 creates via temp+atomic rename;
other ranks validate freshness/run identity. Dedicated progress thread per rank;
timeouts and duplicate rank/GPU rejected before BFS. Bootstrap secrets never
included in public performance reports.

One communicator and ONE monotonically increasing transport_seq covers all
message kinds: CANDIDATE, REQUEST, RESPONSE, RECEIPT, FINALIZE, FATAL.
Separate kind queues do not authorize independent NCCL issue orders.
Sequencer round-robins ready source offers, prioritizing response/request/receipt
drain; source batch or entire depth completion is not needed to issue a ticket.

Ticket: counts/planes for each src,dst,kind, depth, fragment ID. Control phase
pins output inputs and checks receiver credits globally BEFORE data launch.
Every rank receives same ticket and issues matching NCCL calls in sorted
(src,dst,plane) order, including a control entry for zero payload. Self routes
become receive views with leases, not extra NCCL copies. Data P2P grouped;
GroupEnd/async-error contract respected before recording stream completion.

Offer bound ensures aggregate incoming<=E; oversized logical batch is fragmented
before ticket admission. Receive banks are pinned to tickets until owner jobs
retire all references. Future ticket can enqueue without waiting for previous
data completion if it has independent credits, but issue ordering is global.
There is transport rendezvous, not a layer semantic barrier. No claim of zero
synchronization between dependent communication operations.

Rank with empty frontier still progresses tickets. All control headers checked
against full-session identity, depth, allowed kind, seq and byte bounds.
Conservation tracks offered/sent/received/consumed separately.

## 12. Finalize, archive, failure

Finalize ticket only after source closed on all ranks, no candidate/owner jobs,
no in-flight transport, all HASH_FIRST receipts/requests/responses closed,
all committed target extents StateReady. Counter snapshots alone are unsafe:
sequencer closes admission then drains ticket watermarks and rechecks queues.

Scan accepted bucket counts; sum<=L and equals committed/materialized totals.
Copy sorted unique buckets in prefix order into old Hprev arena after its readers
AND archive lease complete. This produces sorted Hnext without sort or dedup.
Build directory; adjacent equality is invariant failure, never deletion.
Rotate Hprev/Hcurr. Clear accepted counts only after copy readers finish.
Register state/hash archive obligations before FrontierPublish; layer0 follows
same owner/hash/archive rules with global count1.

Archive device range -> pinned buffer via D2H -> independent disk worker.
GPU lease ends at D2HDone, pinned lease at consumed write. Full pinned/descriptor
queue or reclaim attempt on unclosed lease -> fatal, not disk backpressure.
Existing busy slots can have bounded queued descriptors; no unbounded host queue.
Disk extent physically preallocated before depth0, successful ftruncate alone
is insufficient. Short writes handled until complete or explicit error; no lost
bytes. flush failure gives INCOMPLETE. State and hash plane orders independent.

Archive schema revision must encode: file magic/version/full config digest,
rank/W, state/action/hash IDs, allocated bytes; 4096-aligned chunks with 64-byte
header (seq,kind,depth,begin,count,payload size, checksum tag), full BLAKE3-256
digests in LayerCommit. Field offsets/frozen vectors are contract gate, not
memcpy of a C struct. RunCommit only after all rank LayerCommits and durable
flush of final manifest. GPU search_complete time separate from durable time.

Capacity and fault codes include specific pool/job/bucket, required/available,
depth/rank and first fatal. No new semantic admissions after fatal. Healthy
transport gets ordered FATAL ticket; broken NCCL/control uses out-of-band
abort/watchdog and process-group termination, not another collective that may
hang. Partial current layer receives no LayerCommit; prior durable layers valid.

Conservation, with disjoint priority categories:
`generated=frontier*G=source_duplicates+emitted`;
`global emitted=owner incoming`;
`incoming=epoch_duplicates+prev_hits+curr_hits+accepted_hits+committed`;
`committed=materialized=next_count`; HASH_FIRST requests=responses=committed.
Accounting checks do not replace full state-set oracle.

## 13. Capacity/performance proof obligations

Per-job ledger reports bytes read/written for packing, generator intermediates,
hash partials, radix passes, routed states, old bucket membership, accepted
copy-back, reconstruction, D2H and disk. Arithmetic model generation roughly
2*P*G*n^3 integer operations; hash 2*C*D*16 limb multiply-add operations plus
modular epilogues. Padding/tile underfill and extra memory traffic reported.
GEMM speedup alone does not prove BFS throughput or memory superiority.

Owner read amplification = total old/accepted keys read / owner incoming.
Report per bucket/depth/rank, plus max occupancy, job-size histogram and empty
launch fraction. A job scanning unreferenced bucket/layer ranges is a contract
violation irrespective of a small-graph speed result.

No production winner until correctness + whole-run comparison with tuned
baseline. Archive is mandatory native output and baseline difference explicit.
Both profiles/pre-dedup/backends remain separate runs. Measured finite P/K
calibration cannot claim optimality over all graph shapes.

## 14. Статический аудит и gates до implementation

| Finding | Разрешение в спецификации | Обязательное доказательство |
|---|---|---|
| Global owner merge/scratch | bounded I,J,K jobs, immutable old views | allocation ledger + touched-range trace |
| Unsafe in-place bucket merge | out-of-place lane output then copy-back | overlap/race tests |
| HASH_FIRST parent deadlock | per-batch terminal receipts + dedicated credits | permuted-event simulator incl zero survivors |
| Sorted requests scatter targets | sort before target extent reservation | origin/state/hash oracle |
| Independent NCCL epoch namespaces | single typed total order | asymmetric/empty two-rank trace |
| Hidden matrix/hash intermediates | per-policy allocation queries | byte ledger vs cudaMemGetInfo |
| StateRef ABA / ring wrap | monotonic absolute sequence + live-range check | wrap/exhaustion/stale ref fixtures |
| Archive overwrites old Hprev | independent arena D2H lease | deliberately delayed archive schedule |
| Tiny epochs reread all buckets | amplification metric + bounded bucket/job | workload-scale profile, no speed guarantee |
| Config/archive serialization underspecified in code | versioned field ledger and vectors | frozen byte tests before runtime |

Self-audit covers semantics, data, ownership, concurrency, transport, memory,
failure and output. This is NOT a claim that pending simulator/hardware gates
passed. In particular final byte-exact query-backed plan and frozen new wire/
archive vectors must exist before production scheduling code.

Implementation dependency gates (not permission to skip architecture):

1. ABI/config/archive vectors, checked memory ledger, owner and StateRing models.
2. Whole-run CPU schedule simulator: both profiles, arbitrary readiness, empty
   ranks, overload, failure at every reservation/publication, archive lag.
3. GPU bounded owner and materialization full-state oracle, four sanitizer modes.
4. Complete one-rank runtime INCLUDING shard scheduling, StateRing and archive.
5. Real 2xT4 NCCL protocol: asymmetric traffic, rank map swap, zero payload,
   delayed ranks, all profiles/backends; no two-independent-GPU substitution.
6. Full layer sets m2..6/seeds0,1,20260828, baseline m5..8; forced collision
   demonstrates probabilistic boundary. Full bytes required, not counts only.
7. Nsight Systems confirms overlap and no layer-wide owner work; Compute
   Sanitizer memcheck/racecheck/initcheck/synccheck. Counter permissions absent
   -> explicit limitation, not fabricated occupancy claims.
8. Five fresh-process tuned A/B m12,16; capacity m20,24 and larger common graph.
   Median/MAD, all repetitions, native allocation plan/cudaMemGetInfo/SMI,
   PyTorch allocated/reserved, search/durable times, complete/fatal outcomes.

## 15. Implemented / designed / unverified

Implemented primitives: exact matrix generation, seeded hash, source sort,
test-only global owner, test-only two-slot feedback, CPU contract models.
Measured: two independent T4 primitive checks, generator variants, experimental
unarchived one-GPU comparisons. Production architecture above is designed,
not implemented; neither real multi-rank runtime nor production Pareto win exists.
No performance kernel changes or new GPU launches are part of this revision.

Official runtime references checked 2026-09-03:
[NCCL group ordering](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html),
[CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).
Deployment pins actual toolchain versions separately; current documentation
does not silently upgrade the tested CUDA/CUTLASS/NCCL build.

## Appendix A. Wire/archive field ledger

No host `sizeof` is the serialization format. New production structures use
schema2 and do not silently reinterpret existing prototype schema1. The ABI
tables above describe device records; the following fixes persisted framing.

MessageHeader offsets: magic0:u32, schema4:u32, kind8:u32, flags12:u32,
run_tag16:u64, transport_seq24:u64, source_batch_seq32:u64, depth40:u32,
src44:u32, dst48:u32, count52:u32, payload_bytes56:u64. Magic=0x4d474232.
kind: candidate_dense1, candidate_hash2, request3, response4, receipt5.
flags currently zero. Frames >u32 records must be fragmented before admission.
Control TCP envelope: length u64 + run UUID16 + seq u64 + message bytes;
length bounded before reading/allocating, peer/run/seq validated.

Each wire frame payload is ordered planes, starts at 256-byte offsets with
zero padding: DENSE hash16*N, ordinal4*N, stateS*N; HASH hash16*N, ordinal4*N,
origin16*N; request origin16*N; response stateS*N; receipt32*N.
Payload bytes=sum of align_up(size,256) per plane. Max frames/offer Fwire is
explicit config; staging query includes Fwire*64 headers, 16*Fwire directory,
and all planes. Count sums checked against E, never trusted from headers.
Large states do not change Hash128 or OriginRef size.

FileHeader schema2: 4096 bytes, zero reserved. Offsets:
magic0:8 bytes ASCII MGBFSAR2; schema8:u32=2; header_bytes12:u32=4096;
config_digest16:32 bytes; run_uuid48:16 bytes; rank64:u32; W68:u32;
logical_bytes72:u32; storage_bytes76:u32; action_id80:u32;
hash_id84:u32; checksum_id88:u32; zero92:u32; extent_bytes96:u64;
used_bytes104:u64 (initially4096); header_digest112:32 bytes.
Digest BLAKE3-256 of whole header with bytes112..143 zero. Updating used_bytes
is not the durability authority; valid commits define readable prefix.

ChunkHeader offsets (64 B): seq0:u64, kind8:u32, depth12:u32,
record_begin16:u64, record_count24:u64, payload_bytes32:u64,
payload_digest_low12840:16 bytes, flags56:u32, zero60:u32.
Chunk begins at 4096-aligned file offset; occupied bytes are
align_up(64+payload_bytes,4096). No unbounded individual chunk: <=Q pinned
capacity minus framing, with splitting at record boundaries.
kind: states1,hashes2,shard_directory3,bucket_directory4,counters5,
layer_index6,layer_commit7,run_commit8. Hash/state order independent.
Per-record state file stride S (padding zero), hash stride16, directories u64.

LayerIndex payload: chunk entries (seq u64, file_offset u64, full_digest32),
48 bytes each. Index may span bounded chunks; each links previous index digest
in a 32-byte prefix. LayerCommit fixed payload: depth u32, zero u32,
state_count u64, hash_count u64, last_index_seq u64, last_index_digest32,
config_digest32 =>96 bytes. Commit includes indexes for all plane/counter chunks.
RunCommit payload: config_digest32, last_layer_commit_digest32, total_states u64,
nonempty_layers u64, status u32=COMPLETE, zero u32 =>88 bytes.
Cross-rank manifest contains every rank's final commit digest and is written
temp->flush->rename->directory flush. A crash before this does not yield a
globally COMPLETE run. INCOMPLETE diagnostic summary is not a RunCommit.

Disk plan conservatively allocates Narchive*(S+16) plus chunk padding,
directories per depth, index entries, commits and configured counter budgets.
Narchive, max_depths, max_chunks and counter bytes are explicit limits; graph
size estimate alone does not bound format overhead. Overflow of any one fails.
Runtime writer has no dynamically growing in-memory list of all chunk hashes:
LayerIndex is streamed, only rolling digest + bounded pending entries retained.

## Appendix B. Self-audit counterexamples

- Two peers emit same hash in different tickets: sole shard writer publishes
  accepted before next job; second sees next hit. No finalization dedup needed.
- Owner rejects every child: terminal receipt accepted0 still closes source
  batch; no request event is needed to reclaim parent.
- Accepted requests arrive before receipt: source serves them, retains count;
  closing requires receipt total AND completed sends, not arrival order.
- Response delayed while generation continues: target was reserved before
  commit; next cannot publish until StateReady; capacity stop is explicit.
- Owner update needs to overwrite entries another block reads: merge goes into
  independent J*K output; copy-back only after kernel completion.
- All source hashes target one bucket: jobs split at I, accepted+survivors>K
  fails before mutation; unused capacity elsewhere is not silently borrowed.
- One rank has no frontier: it still progresses all typed tickets and final
  drain; local emptiness cannot skip NCCL operations.
- Archive disk is stalled: already copied GPU memory can retire; pinned slots
  eventually exhaust and fail rather than blocking a producer on disk.
- Full-state ring wraps with outstanding request: absolute StateRef stays live
  and prevents overwrite; no descriptor slot reuse disguises stale origins.
- Very small incoming batches touch every bucket: bounded scratch is still
  correct, but repeated old reads can be expensive. This is an exposed metric,
  not a solved performance theorem; batching/prefix choice requires measurement.

All cases require executable model fixtures before implementation promotion.
