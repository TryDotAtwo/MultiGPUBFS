# Multi-GPU BFS для графов Кэли — архитектурный контракт v1

> **2026-09-03: актуальный контракт — [архитектура matrix runtime v2](docs/matrix-runtime-architecture-v2.md).**
> Она заменяет нормативные требования этого документа для первого matrix backend.
> Текст ниже сохранён как история решений, не как альтернативная спецификация.
> Single-bucket `DenseDeviceStepper` — проверочный прототип, не реализация этой
> архитектуры. Аудит и запреты на его перенос включены в v2.

## Уточнение первого backend, 2026-09-02

Утверждённый implementation plan уточняет этот контракт. При расхождении с
ранними slices и вариантами ниже применяются следующие решения:

- standalone native runtime в этом репозитории; интеграция в CayleyPy отдельно;
- первый `MatrixGroupManifestV1`: square canonical row-major `u8`, modulus
  `2..256`, inverse-closed матричные генераторы; permutation backend вне v1;
- `GEMM_U8_P32X4_V1` вместо GF(2)/AND.POPC hash и Zobrist baseline из §7.2;
  точная спецификация: [gemm-hash-v1.md](docs/gemm-hash-v1.md);
- генерация через CUTLASS unsigned INT8 MMA и modular epilogue, второй GEMM
  вычисляет hash; владельцы `CUB_SORT_MERGE` и экспериментальный `BMMA_BUCKET`
  с XOR.POPC; Tensor Core ускорение требуется доказать end-to-end замерами;
- `DENSE` / `HASH_FIRST`, local pre-dedup ON/OFF выбираются до аллокаций;
- обязательный hardware gate — реальные 2×T4; RTX 3070 используется только
  для локальной разработки и не заменяет этот gate;
- baseline: неизменённый CayleyPy `feature/bfs-torchrun-distributed@f0f2b8e`;
  основной A/B — m=12,16, пять запусков; capacity — m=20,24;
- исходники и доказательства реализации:
  [native-matrix-implementation.md](docs/native-matrix-implementation.md).

SHAKE256 из 128-битного seed является псевдослучайным разворачиванием, а не
информационно-теоретически независимой выборкой всех коэффициентов. Равенство
вероятности коллизии `1/p^4` доказано для независимых равномерных affine
коэффициентов; перенос этой оценки на seeded реализацию использует допущение
о SHAKE256. Hash-only BFS остаётся вероятностным, не exact full-state BFS.

Статус: рабочая архитектура перед реализацией. Здесь зафиксированы семантика,
владение данными, горячий конвейер, память, протокол между рангами и критерии
приёмки. Параметры производительности ещё требуют измерений; скрытых fallback и
смены алгоритма во время запуска нет.

## 1. Что строим

Первая версия решает одну задачу:

```text
single-source exhaustive BFS из состояния A
по конечному графу Кэли
полными слоями
до пустого следующего слоя либо явного capacity/fatal stop
```

В v1:

- один процесс владеет ровно одной GPU;
- полный текущий фронтир находится в VRAM владельцев;
- владелец состояния определяется его `Hash128`;
- межкарточный data plane состоит из маршрутизации кандидатов, а в
  `HASH_FIRST` ещё из финальной материализации состояний;
- каждый опубликованный слой выгружается через pinned RAM на диск;
- все большие GPU, pinned-host и disk extent allocations выполняются до
  глубины 0;
- внутри глубины работают независимые готовые jobs; единственный обязательный
  глобальный semantic cut — `FinalizeDepth`;
- профиль запуска неизменяем.

Не входят в v1:

- bidirectional BFS;
- поиск цели и ранняя остановка;
- parent DAG, подсчёт путей и восстановление пути;
- directed Cayley graph без доказанной границы обратного слова;
- смена числа рангов, hash seed, владельцев или профиля на ходу;
- динамическое перераспределение памяти, spill из VRAM в RAM, retry и fallback;
- Python/Torch в горячем пути.

Python остаётся тонким API/launcher-слоем для CayleyPy. Host runtime — Rust,
GPU hot path — C++/CUDA с CUTLASS/CuTe, CCCL/CUB и NCCL. `torchrun` допустим как
внешний launcher: каждый rank делает `exec` одного native worker и дальше Torch
не участвует.

## 2. Семантический контракт

Пусть `F_d` — точный слой расстояния `d` от `A`, а `B_d` — шар до `d`.
В начале глубины `d` система обязана иметь:

```text
CurrentStates = F_d
PrevHashes    = H(F_(d-1))     // пусто при d = 0
CurrHashes    = H(F_d)
```

Каждый `x in F_d` порождает ровно по одному логическому ребёнку на каждый
генератор `g in S`:

```text
child(x,g) = Action(x,g)
logical_child_id = (depth, source_rank, parent_ref, move_id)
```

Набор генераторов v1 обязан быть inverse-closed. Тогда для любого ребра из
`F_d` глубина второго конца лежит только в `{d-1,d,d+1}`. Поэтому новый слой
получается как

```text
F_(d+1) = unique(children(F_d)) \ (F_(d-1) union F_d).
```

Именно поэтому в VRAM нужны только два опубликованных hash-слоя и строящийся
третий. Более старые слои не участвуют в membership после закрытия глубины.
Это правило запрещено применять, если inverse-closed preflight не прошёл.

Авторитетная операция над ребёнком выполняется только его владельцем:

```text
U_(shard,epoch) = unique(C_(shard,epoch))
                  \ (Hprev union Hcurr union Hnext_accepted)
Hnext_accepted <- Hnext_accepted union U_(shard,epoch).
```

`Hnext_accepted` монотонно растёт. Публикация `OwnerCommit` означает, что hash
прошёл дедупликацию не только внутри входной пачки и против старых слоёв, но и
против всех ранее принятых owner epochs этой глубины. После этого решение
необратимо: логическое состояние уже принадлежит `F_(d+1)` и никакая поздняя
финализация не имеет права его удалить. Source-side pre-dedup может только
сократить трафик; семантической силы у него нет.

`OwnerCommit` и `FrontierPublish` — разные события. Первое окончательно решает
принадлежность состоянию к `F_(d+1)` и разрешает его materialization/archive.
Но порождать из него детей можно только после общего `FinalizeDepth`, когда
доказано, что весь `F_d` исчерпан: иначе смешались бы соседние глубины.

### 2.1. Что значит равенство

Горячий путь считает два состояния равными по `Hash128`. Полные состояния при
обычной дедупликации не сравниваются. Значит, v1 является:

```text
layer-exact при условии отсутствия коллизии Hash128;
probabilistic относительно полной семантики состояния.
```

Для `n` различных состояний и идеально случайного 128-битного семейства
верхняя birthday-оценка:

```text
P(collision) <= n(n-1) / 2^129.
```

`hash_seed_lo` и `hash_seed_hi` задаются пользователем и входят в идентичность
запуска. Повтор с другим seed — независимая проверка результата, но не
исправление первого запуска. Архив хранит полные состояния, поэтому результаты
разных seed можно позже сравнить уже по каноническим state bytes.

Будущий truly-exact профиль с инъективным кодом либо разрешением коллизий по
полному состоянию оставлен отдельным направлением. Он не должен незаметно
менять ABI или смысл текущего профиля.

### 2.2. Каноническое действие

`StateTraits` полностью определяет вершину и действие:

```cpp
template<class Puzzle>
struct StateTraits {
    static constexpr uint32_t kLogicalBytes;
    static constexpr uint32_t kStorageBytes;
    static constexpr uint32_t kAlignment;
    static constexpr uint32_t kMoveCount;

    using StateStorage = /* fixed-size, trivially copyable */;

    static void canonicalize(StateStorage&);
    static bool validate(const StateStorage&);
    static StateStorage reference_apply(const StateStorage&, uint16_t move);
    static bool validate_inverse_manifest();
};
```

Padding всегда канонически нулевой и участвует в frozen test vectors. Для
перестановочного профиля действие фиксируется буквально:

```text
child[p] = parent[generator[move][p]].
```

Scatter, обратная перестановка и другая сторона умножения являются другим
`action_id`. Для матричного профиля manifest отдельно фиксирует
`parent*generator` либо `generator*parent`, поле/модуль, точное накопление и
каноническую редукцию.

## 3. Неизменяемый профиль запуска

```cpp
enum class FrontierProfile : uint8_t {
    DENSE,       // состояние + hash порождаются сразу
    HASH_FIRST,  // сначала hash + origin, состояние только для победителей
};

enum class LocalPreDedup : uint8_t {
    OFF,
    ON,
};

enum class GenerateHashBackend : uint8_t {
    CUDA_ZOBRIST,
    TC_HASH,
    MATRIX_TC_HASH,
};

enum class OwnerDedupBackend : uint8_t {
    CUB_SORT_MERGE,
    BMMA_BUCKET,
};
```

Все четыре значения выбираются до memory plan и не меняются во время BFS.
`BMMA_BUCKET`, `TC_HASH` и `MATRIX_TC_HASH` считаются экспериментальными, пока
не выиграют end-to-end benchmark у базового пути с тем же результатом.

Обязательная конфигурация делится на четыре группы.

```text
Semantic:
  state_profile_id, action_id, canonicalization_id
  start_state, generator_manifest, inverse_manifest
  hash_algorithm_id, hash_seed_lo, hash_seed_hi

Topology:
  world_size, rank_to_gpu_uuid[]
  owner_bits, shard_bits, bucket_bits
  logical_owner_to_rank[]
  ownership_epoch

Algorithm:
  frontier_profile, local_pre_dedup
  generate_hash_backend, owner_dedup_backend
  parent_batch, route_slot_records, route_slot_count
  exchange_trigger_records, owner_stream_count
  owner_job_candidate_records, owner_job_bucket_descriptors
  next_bucket_capacity_records
  control_progress_timeout_ms, nccl_progress_timeout_ms

Capacity:
  state_ring_records, state_extent_descriptors
  layer_hash_records_per_arena, next_bucket_store_records
  next_bucket_count_records, final_bucket_scan_scratch_bytes
  send/receive/materialize slot capacities
  CUB/CUTLASS scratch bytes
  pinned_archive_slots, pinned_archive_slot_bytes
  disk_extent_bytes_per_rank
  untouched_vram_reserve_bytes
```

Canonical serialization всех полей, start state, generator tables и rank map
даёт `ConfigDigest256 = SHA-256(serialized_config)`. Каждый rank обязан получить
тот же digest до первой GPU-работы.

Hash policies v1 полностью воспроизводимы:

```text
ZOBRIST128_V1:
  table[position][byte_value] получена из SHAKE256(domain, seed, position, value)
  state hash = XOR table[position][canonical_state_byte]

GF2_128_V1:
  бинарный basis[state_bits][128] получен из SHAKE256(domain, seed, dimensions)
  state hash = canonical_state_bits * basis mod 2
```

Domain strings, integer byte order, XOF byte order и frozen vectors входят в
отдельный versioned hash spec до реализации kernel. Нельзя назвать две разные
таблицы одним `hash_algorithm_id`.

### 3.1. Owner и shard без лишней перестановки битов

В v1 `world_size`, `shards_per_rank` и `buckets_per_shard` — степени двойки:

```text
world_size        = 2^owner_bits
shards_per_rank   = 2^shard_bits
buckets_per_shard = 2^bucket_bits
```

Из уже равномерного `Hash128` берётся один непрерывный prefix:

```text
logical_owner = top owner_bits
local_shard   = next shard_bits
microbucket   = next bucket_bits
physical_rank = logical_owner_to_rank[logical_owner]
```

`logical_owner_to_rank` — задаваемая пользователем перестановка физических
рангов. Поэтому 128 GPU и 64 shard на GPU означают всего 7+6 prefix bits; это
несколько shift/mask и одна маленькая таблица рангов. Отдельного
`BijectivePermute` нет.

Полная сортировка по `Hash128` автоматически группирует owner, shard, bucket и
одинаковые ключи. Таблица bucket boundaries не читается каждым кандидатом:
один job builder читает offsets и передаёт kernel уже готовые
`pointer + count` диапазоны.

При 64 shard и 256 базовых buckets на GPU требуется 16 384 диапазона, а не
`2^128` корзин. `BMMA_BUCKET` может создавать дополнительные только непустые
refinement descriptors; их число ограничено capacity и заранее выделенным
пулом.

## 4. Форматы данных

Все большие наборы — SoA-плоскости. Нет device `malloc`, linked lists,
указателя на каждый state и AoS-сортировки полного состояния.

```cpp
struct alignas(16) Hash128 {
    uint64_t lo;
    uint64_t hi;
};                                  // 16 B

struct StateRef {
    uint32_t extent_id;
    uint32_t slot;
};                                  // 8 B

struct alignas(16) OriginRef {
    uint32_t source_rank;
    uint16_t move_id;
    uint16_t reserved_zero;
    StateRef parent;
};                                  // 16 B

struct TargetExtentDesc {
    uint32_t extent_id;
    uint32_t first_slot;
    uint32_t record_count;
    uint32_t response_begin;
};                                  // 16 B

struct alignas(64) WireBatchHeader {
    uint32_t magic;
    uint32_t schema;
    uint32_t kind;
    uint32_t header_bytes;
    uint64_t config_digest_low64;
    uint64_t ownership_epoch;
    uint64_t exchange_epoch;
    uint32_t depth;
    uint32_t source_rank;
    uint32_t destination_rank;
    uint32_t record_count;
    uint64_t payload_bytes;
};                                  // 64 B

struct alignas(64) OwnerBucketJobDesc {
    uint64_t incoming_begin;
    uint64_t prev_begin;
    uint64_t curr_begin;
    uint32_t incoming_count;
    uint32_t prev_count;
    uint32_t curr_count;
    uint32_t accepted_count_snapshot;
    uint32_t local_shard;
    uint32_t bucket;
    uint32_t flags;
    uint32_t reserved_zero32;
    uint64_t reserved_zero64;
};                                  // 64 B

struct alignas(64) StateExtentDesc {
    uint64_t physical_begin;
    uint64_t record_count;
    uint64_t allocation_seq;
    uint64_t logical_order_begin;
    uint32_t depth;
    uint32_t extent_id;
    uint32_t state;
    uint32_t archive_lease;
    uint64_t reserved_zero[2];
};                                  // 64 B
```

`StateStorage` имеет compile-time размер и выравнивание профиля. Реализация
обязана иметь `static_assert` на каждый размер и нулить reserved/padding.

### 4.1. Wire planes

Каждый wire batch имеет один маленький header и отдельные плотные device
плоскости:

```text
DENSE_CANDIDATE:
  Hash128[count]
  StateStorage[count]

HASH_CANDIDATE:
  Hash128[count]
  OriginRef[count]

MATERIALIZE_REQUEST:
  OriginRef[count]
  TargetExtentDesc[extent_count]   // диапазоны, не адрес на каждую запись

MATERIALIZE_RESPONSE:
  StateStorage[count]
```

Порядок response равен порядку request. Owner принимает response прямо в один
или несколько заранее зарезервированных непрерывных target extents. Поэтому
нет random scatter по `target_state_idx` на каждое состояние.

Header содержит:

```text
schema, kind, depth, ownership_epoch, exchange_epoch
source_rank, destination_rank, record_count
config_digest_low64, payload_bytes
```

Полный digest проверяется в preflight; короткое поле в header ловит случайную
порчу/смешение запуска.

## 5. Память

### 5.1. `StateRing`: один физический пул для current и next

Полные состояния находятся в одном заранее выделенном кольцевом массиве:

```text
StateStorage state_ring[state_ring_records]
StateExtentDesc extent_ring[]
```

Легальные состояния extent:

```text
FREE -> RESERVED
RESERVED -> MATERIALIZED                 // DENSE owner copy
RESERVED -> PENDING_MATERIALIZATION      // HASH_FIRST OwnerCommit
PENDING_MATERIALIZATION -> MATERIALIZED  // response + hash verification
MATERIALIZED -> CURRENT                  // только на чистом depth rotation
CURRENT -> ENUMERATED -> RECLAIMABLE -> FREE
```

Archive lease может задержать только `RECLAIMABLE -> FREE`; он не меняет
семантическое состояние записи.

Это не список страниц. Данные каждого extent непрерывны; descriptors лежат в
плотном кольце; allocation и reclamation идут FIFO. Один owner job резервирует
целый диапазон после scan, а не делает `atomicAdd` на каждого survivor.

Если диапазон не помещается до конца физического кольца, allocator ставит один
wrap marker и начинает следующий extent с нуля. Tail никогда не пересекает
голову ещё живого current extent. Нарушение capacity — fatal, не ожидание
расширения.

Несколько owner streams не соревнуются за tail. Они публикуют survivor counts
в `AllocationWave`; coordinator делает один scan по jobs, один checked advance
кольцевого tail и выдаёт каждому job непересекающийся extent. Поэтому цена
allocation — на job wave, а не на state, и порядок `allocation_seq`
детерминирован.

В `StateRing` попадают только записи после `OwnerCommit`, поэтому каждый extent
плотный и целиком финальный: поздних duplicate holes и `live_bits` нет.
Следующая глубина читает ровно `record_count` последовательных записей. Когда
extent полностью скопирован в parent jobs, закрыты его materialization
obligations и archive lease, весь extent освобождается.

Следствие для профилей:

```text
DENSE:
  parent extent можно освобождать потоково после того, как все его записи
  скопированы в независимые ParentSlot и закрыт archive lease;
  освобождённые записи сразу принимают next extents;
  capacity проверяет max(unconsumed_current + committed_next).

HASH_FIRST:
  OriginRef ссылается на parent extent до потоковой materialization;
  current extent удерживается до owner decisions по всем его детям и
  обслуживания всех принятых запросов;
  target extent резервируется перед каждым OwnerCommit;
  capacity проверяет current + next + wrap slack.
```

Это реальная разница в памяти. `HASH_FIRST` экономит candidate states, но не
имеет права преждевременно убить родителей.

### 5.2. Hash arenas и owner accepted store

На rank существуют:

```text
Hprev   — уникальные, полностью отсортированные owned hashes F_(d-1)
Hcurr   — уникальные, полностью отсортированные owned hashes F_d
HnextBuckets — уникальные, уже принятые owner'ом hashes F_(d+1)
```

`Hprev/Hcurr` лежат компактно и имеют shard/microbucket offsets.
`HnextBuckets` — одна плоская заранее выделенная SoA:

```text
Hash128  accepted_hash[next_bucket_store_records]
StateRef accepted_state_ref[next_bucket_store_records]
uint32  accepted_count[next_bucket_count_records]
```

Rank-local `bucket_id = (local_shard << bucket_bits) | microbucket`, поэтому
`next_bucket_count_records = shards_per_rank * buckets_per_shard`, а начало
bucket равно `bucket_id * next_bucket_capacity_records`. Никаких per-record
pointers и списков страниц нет. Каждый bucket — фиксированный непрерывный
диапазон, содержащий `accepted_count[bucket]` отсортированных уникальных
records.

В `DENSE` `StateRef` указывает на уже записанное финальное состояние. В
`HASH_FIRST` target extent и `StateRef` резервируются перед commit, а отдельный
`MaterializeSlot` уже владеет соответствующим `OriginRef`; до завершения ответа
extent имеет состояние `PENDING_MATERIALIZATION` и не может быть опубликован как
frontier input.

`next_bucket_store_records` вычисляется ровно как
`next_bucket_count_records * next_bucket_capacity_records`. Переполнение даже
одного bucket — fatal с его фактическими `required/available`, даже если в
других buckets осталась пустая память. Это сознательная цена прямой адресации и
bounded contiguous owner lookup без динамического allocator.

На `FinalizeDepth` bucket records только уплотняются в освобождаемый старый
layer arena в порядке `bucket_id`. Дедупликации там нет: одинаковый hash в
published buckets означает нарушение `OwnerCommit` и fatal. Обе compact layer
arenas обязаны вместить рассчитанный worst case, а scan temp полностью входит в
`final_bucket_scan_scratch_bytes`; неучтённой временной копии не появляется.

Все radix/merge temp sizes запрашиваются у CCCL/CUB на startup и входят в один
статический scratch plan. Scratch разных несовместимых фаз может overlay, но
live вход и выход одной операции никогда не alias.

### 5.3. Транзитные slots

Заранее создаются независимые кольца:

```text
ParentSlot[Np]
GenerateSlot[Ng]
RouteSlot[Nr]     // raw + sorted ping-pong
ReceiveSlot[Nx]
OwnerScratch[No]
MaterializeSlot[Nm]
ArchivePinnedSlot[Na]
```

У каждого slot — фиксированная capacity, device counters и CUDA events.
Большой буфер не появляется в depth loop. Состояние slot меняется только по
конечному автомату и одному владельцу-стадии.

### 5.4. Memory-plan equation

Preflight печатает и сохраняет точную сумму:

```text
VRAM_required =
    StateRing
  + Hprev + Hcurr + HnextBuckets(hash + StateRef)
  + layer/bucket directories and counts
  + max(live phase scratch layouts, final bucket compaction layout)
  + route/receive/materialize slots
  + generator/hash tables
  + CUDA/NCCL reserve
  + untouched_vram_reserve_bytes.
```

Номинальную память GPU нельзя использовать как capacity. План обязан пройти на
самом маленьком rank budget и учитывать skew; среднее `global/world_size` не
является безопасной границей.

## 6. Один слой как потоковый DAG

```text
Current StateExtent
        |
        v
PackParents -> GenerateHash -> SourceSort/Route -> Exchange -> OwnerDedupCommit
     ^              |               |                 |            |
     |              +--------------- slot events -----+            +-> HnextBuckets
     +---- StateRing reclamation                                     +-> DENSE state archive
                                                                    |
                          state archive <- HASH_FIRST: Materialize <-+

all parent/route/exchange/owner/materialize obligations drained
        |
        v
CompactAcceptedBuckets -> HashArchiveEnqueue -> PublishNext/Rotate
```

Пока один slot генерирует, другой сортируется, третий пересылается, а owner
streams разбирают ранее принятые shards. Dispatcher запускает любой готовый
job — это и есть roulette. Внутри повторно используемого slot вход/выход
переключаются A/B — это ping-pong.

Нет общей фазы «сначала породить всех детей». Материализуется только
`parent_batch * move_count`, после чего slot немедленно уходит дальше.

### 6.1. Slot state machine

```text
FREE
 -> PACKING
 -> GENERATING
 -> ROUTING
 -> READY_FOR_EXCHANGE
 -> IN_EXCHANGE
 -> OWNER_READY
 -> OWNER_DEDUP
 -> OWNER_COMMITTING
 -> FREE
```

Переход разрешает только CUDA event предыдущей стадии. Host не делает
`cudaDeviceSynchronize()` в depth loop. Занятый slot не останавливает другие
slots или другие shards.

Stream independence разрешает overlap, но не гарантирует его: sort, BMMA и
generate kernels конкурируют за SM/register/shared-memory ресурсы. Число
streams и job shape принимаются только по Nsight timeline и end-to-end depth
time. Формулировка «все блоки обязаны всегда работать» не является инвариантом;
инвариант — отсутствие искусственного глобального ожидания при наличии ready
job и свободной downstream capacity.

### 6.2. Liveness и резервирование

Перед запуском producer job уже существует его output slot. Перед candidate P2P
зарезервирован receive slot. Owner сначала вычисляет точный survivor count по
каждому bucket, затем единым checked reservation подтверждает одновременно:

```text
свободные slots в HnextBuckets
target StateRing extents
HASH_FIRST MaterializeSlot и response ranges
owner scratch для commit.
```

Только после успешной local checked reservation owner пишет state/payload и
публикует новые bucket counts. Нехватка любой persistent capacity немедленно
даёт fatal; уже принятый hash никогда не ждёт ресурса, необходимого для его
materialization.

Свободный output slot — launch predicate, а не allocation внутри kernel. Если
`MaterializeSlot` временно занят, owner job остаётся `OWNER_READY`, а roulette
запускает другие shards. Все его persistent ranges резервируются только перед
переходом в `OWNER_COMMITTING`; циклическое ожидание запрещено watchdog'ом.

### 6.3. Streams

Минимальная раскладка на rank:

```text
pack_stream
generate_stream[GEN_CONCURRENCY]
route_stream[ROUTE_CONCURRENCY]
comm_stream
owner_stream[OWNER_CONCURRENCY]
materialize_stream       // потоковые HASH_FIRST jobs после OwnerCommit
archive_copy_stream
finalize_stream
```

Фиксированные job shapes захватываются в CUDA Graph templates. Dispatcher,
выбор ready job, NCCL schedule и depth cut остаются снаружи graph.

## 7. Стадии горячего пути

### 7.1. `PackParents`

Вход — один или несколько плотных current extents. Kernel:

1. читает extent линейно;
2. копирует ровно `record_count` записей в последовательные `ParentSlot`
   ranges, разрезая только хвост физического batch;
3. сохраняет `StateRef` каждого parent для `HASH_FIRST`.

Здесь нет filter/scan: owner никогда не публикует дырявый extent. На job
выполняется одна резервация диапазона, не один atomic на запись.

### 7.2. `GenerateHash`

Логический layout всегда parent-major:

```text
candidate_ordinal = parent_local * move_count + move_id.
```

Ровно `parent_count * move_count` обязанностей создаются один раз.

`DENSE` выводит:

```text
child_state[candidate]
child_hash[candidate]
```

`HASH_FIRST` выводит:

```text
child_hash[candidate]
OriginRef(source_rank, parent_state_ref, move_id)
```

Полный child в `HASH_FIRST` может жить в registers/shared внутри kernel, но не
записывается в persistent VRAM.

#### CUDA baseline

Для перестановочных состояний базовый kernel совмещает gather ребёнка и
Zobrist-128. Он является correctness/performance baseline. Принудительно
представлять перестановку плотной матрицей и умножать её GEMM запрещено без
измеренного выигрыша: это добавляет данные и вычисления к обычному gather.

#### Tensor Core hash

`TC_HASH` представляет канонические state bits как строки `X`, а seed-defined
128-битный бинарный базис как `R`. Hash получается как GF(2) projection:

```text
H = X * R mod 2.
```

Binary MMA считает `AND.POPC`, epilogue берёт parity. Для fixed seed CPU и GPU
обязаны совпадать на frozen vectors. Это полезная тяжёлая работа для Tensor
Cores, но профиль принимается только после сравнения с fused Zobrist по полному
уровню, включая packing bytes и occupancy.

#### Matrix groups

`MATRIX_TC_HASH` делает один logical `GenerateHash` job:

```text
batched/grouped exact matrix product
-> canonical modular epilogue
-> binary hash projection
-> DENSE state store либо HASH_FIRST hash-only store.
```

Это может быть один custom CUTLASS kernel либо два зависимых graph nodes без
host sync. Tensor Core путь допустим только если manifest доказывает отсутствие
rounding/accumulator overflow до точной редукции. В противном случае такой
backend не существует для данного `StateTraits`, и preflight его отвергает.

### 7.3. `SourceSort/Route`

Каждый source batch один раз radix-сортируется по полному `Hash128`. Payload
переставляется через compact indices; полные states не таскаются через каждый
radix pass.

Эта единственная сортировка одновременно даёт:

- одинаковые локальные hash рядом;
- непрерывные owner ranges;
- внутри owner — shard и microbucket ranges;
- отсортированные streams для дальнейшего merge на владельце.

При `LocalPreDedup::ON` adjacent equal hash сжимаются после sort. При `OFF`
scan/compaction пропускается, но сортировка остаётся, потому что она нужна для
крупных транзакций. Оба режима сохраняют одну BFS-семантику и сравниваются
end-to-end. В обоих случаях весь owner-side dedup выполняется безусловно;
`LocalPreDedup` не может отключить ни одну owner-проверку.

`DENSE` делает одну индексную gather-copy state payload в уже сгруппированный
send buffer. Чтение переставляемых states не идеально последовательное, но
производится один раз; все записи и wire ranges плотные.

### 7.4. `Exchange`

Data plane использует grouped NCCL send/recv. Все ranks выпускают операции в
одинаковом `(exchange_epoch, peer_rank, plane_kind)` порядке. Нулевой payload
тоже имеет control entry, поэтому порядок не расходится.

Один коммуникационный epoch:

1. rank с готовым route slot посылает control-plane `READY` sequencer'у;
2. sequencer выдаёт всем ranks один `BEGIN(exchange_epoch)`; rank закрепляет
   один готовый slot либо участвует нулевым offer;
3. ranks выполняют один ordered all-gather точных `send_count[peer]`;
4. каждый receiver проверяет свою заранее выделенную capacity;
5. глобальный fatal check запрещает data launch при overflow;
6. по общей матрице все ranks независимо строят один и тот же список только
   ненулевых пар;
7. `ncclGroupStart/End` ставит recv/send в фиксированном peer/plane порядке;
8. receive events публикуют готовые sorted segments owner scheduler'у.

Comm epoch синхронизирует только `comm_stream`; generate, route и owner jobs
продолжают работать. Это транспортный rendezvous, а не semantic barrier слоя.
Sequencer не выбирает данные и не ждёт завершения GPU jobs: он лишь нумерует
готовые rendezvous. При drain он продолжает выдавать epochs, пока все ranks не
объявили source closed и общая ready/in-flight сумма не стала нулём.

Локальный owner range не копируется через NCCL: он входит как ещё один sorted
segment в тот же owner job.

### 7.5. `OwnerDedupCommit`

Owner — единственное место, где принимается решение о novelty. Для одного
`local_shard` одновременно работает не более одного writer job; разные shards
обрабатываются параллельно roulette-пулом. Внутри shard jobs коммитятся по
возрастанию `exchange_epoch`; готовность более позднего epoch не позволяет
обойти более ранний.

Пусть `A` — уже опубликованные `HnextBuckets` данного shard, а `C` — sorted
segments очередного owner epoch от всех источников. Базовый путь:

```text
k-way merge source segments
-> adjacent unique within owner epoch
-> set-difference with matching Hprev bucket ranges
-> set-difference with matching Hcurr bucket ranges
-> set-difference with matching committed Hnext bucket ranges
-> scan survivor counts
-> capacity check for every touched bucket and every downstream resource
-> reserve final StateRing extents
-> DENSE state copy or HASH_FIRST materialization enqueue
-> merge new Hash128+StateRef into fixed Hnext buckets
-> publish OwnerCommitEvent and new accepted_count values.
```

После сети полного radix sort нет: уже отсортированные source segments именно
merge'ятся. Это ответ на вопрос «не придётся ли сортировать и до, и после
отправки»: до отправки — одна radix sort, после — линейный merge.

Каждый touched bucket читается и переписывается только как непрерывный диапазон.
Начало вычисляется арифметически; count читается один раз на bucket. Старый
sorted range и новый sorted survivor range сливаются в свободный хвост
фиксированного bucket обратным merge либо эквивалентным block-wide merge.
Записи идут по плотным адресам. `accepted_count[bucket]` публикуется только после
завершения всех stores, а следующий job того же shard ждёт `OwnerCommitEvent`.
`accepted_count_snapshot` в descriptor обязан совпасть с текущим count; иное
значение означает ошибку scheduler/видимости и даёт `OWNER_COMMIT_INVARIANT`.

Выбор представителя детерминирован: уже committed record всегда остаётся;
среди равных records текущего epoch остаётся минимальный
`(source_rank, parent allocation_seq, parent slot, move_id)`. После commit
принятый record никогда не заменяется и не удаляется.

При отсутствии `Hash128` collisions результат по полным состояниям не зависит
от разбиения на epochs: разные origins одного hash материализуют одно и то же
каноническое состояние. При настоящей коллизии committed представитель может
зависеть от порядка epochs; это находится за заявленной probabilistic гарантией
v1 и не маскируется словом «детерминизм». Replay одного сохранённого
communication trace остаётся детерминированным.

В `DENSE` owner одним диапазоном резервирует final `StateRing` extent и плотно
копирует туда только survivors. В `HASH_FIRST` он резервирует такой же target
extent и materialization slot; в bucket сразу записывается его final `StateRef`,
а `OriginRef -> StateRef` job выполняется на отдельном stream. В обоих профилях
hash уже окончательно принят; различается только момент появления его state
bytes.

Корректность необратимости следует по индукции по owner epochs. До первого
epoch accepted set пуст. Текущий commit добавляет множество, уникальное внутри
себя и не пересекающееся с `Hprev`, `Hcurr` и прежним accepted set. Поэтому
новый accepted set снова уникален и не пересекается со старыми слоями. Эти три
набора только читаются последующими jobs; поздний кандидат может быть отвергнут
ранним, но ранний уже никогда не может быть отвергнут поздним.

### 7.6. Два owner backends

#### `CUB_SORT_MERGE`

Это обязательный baseline:

- CCCL/CUB radix sort на source;
- merge-path/k-way merge на owner;
- adjacent unique;
- линейный set-difference по совпадающим `Hprev`, `Hcurr` и committed `Hnext`
  bucket ranges;
- scan capacities и block-wide update фиксированных owner buckets.

Старые и accepted ranges могут читаться несколькими owner epochs. Поэтому
`exchange_trigger_records`, `bucket_bits` и `next_bucket_capacity_records`
выбираются калибровкой; production запрещает настолько маленькие epochs или
крупные buckets, при которых повторные bounded reads становятся главным
bottleneck.

#### `BMMA_BUCKET`

Это второй, экспериментальный exact-relative-to-Hash128 backend. Radix prefix
partition остаётся: Tensor Cores не сортируют.

Для bounded candidate/reference bucket binary MMA сравнивает 128-битные hashes
плитками через `XOR.POPC`; нулевой popcount означает equality. Если bucket
больше лимита, он детерминированно разбивается следующими hash bits. Хранятся
только flat nonempty range descriptors. При исчерпании всех 128 bits одинаковые
ключи сводятся обычным segmented reduction, а не квадратной матрицей.

После tile compare warp держит mask в регистрах:

```text
keep_mask
survivor_count = popc(keep_mask)
lane_offset = prefix popc/shfl
```

Отвергнутый lane не пишет. Выживший lane знает точный плотный offset. BMMA
сравнивает кандидатов также с committed `Hnext` bucket, поэтому его keep-mask
имеет ту же необратимую семантику `OwnerCommit`, что и CUB baseline. Финального
cross-run merge не существует.

Backend разрешён только если exhaustive bucket fixtures показывают полную
эквивалентность CUB baseline, включая boundary splits и одинаковые 128-bit
ключи. Если kernel не собран для фактического SM, запуск отклоняется; runtime
fallback нет.

### 7.7. Аудит обращений к памяти

Нормальный путь состоит из sequential/coalesced reads, scan и dense writes.
Допущены только явно ограниченные irregular reads:

```text
DENSE route: один gather StateStorage по отсортированным candidate indices;
DENSE owner: один gather победивших state payloads из входных sorted segments;
HASH_FIRST materialize: parent reads после сортировки OriginRef по StateRef.
Owner accepted lookup: один arithmetic jump на touched bucket, затем bounded
                       contiguous range read.
```

У первых двух destination writes всегда плотные. HASH_FIRST parent reads
превращаются в монотонный проход по parent extents. Owner/shard/bucket mapping
не вызывает per-candidate directory chase: prefix bits уже находятся в sort
key, а descriptor строится один раз на непустой range.

Запрещены random per-record writes в persistent frontier, hash layer или wire
buffer. Адрес каждой записи получается из warp/block mask и одного
job-level prefix sum. Cache locality может ускорить чтение, но корректность и
capacity никогда не опираются на предположение «оно, наверное, в L2».

## 8. Почему owner accepted set — фиксированные prefix buckets

Необратимый `OwnerCommit` требует проверить новый hash против **всех** ранее
принятых hash этой глубины. Оставить независимые dirty runs до финализации
нельзя: тогда раннее состояние ещё могло бы оказаться поздним дублем и не было
бы финальным. С другой стороны, схема

```text
clean <- unique(merge(clean, every_small_incoming_batch))
```

перечитывала бы и переписывала весь растущий слой и стала бы квадратичной.

Поэтому owner accepted set разбит достаточным числом hash-prefix buckets.
Каждый bucket имеет фиксированную contiguous capacity и остаётся малым bounded
unit of work. Incoming записи уже стоят по `bucket_id`; job builder создаёт один
`pointer + old_count + incoming_count` descriptor на touched bucket. Один
block/несколько cooperative warps сравнивают и обновляют только этот диапазон.
Directory read и reservation выполняются один раз на bucket, не на state.

Это не случайный hash-table probe: кандидаты заранее отсортированы, соседние
jobs обходят возрастающие bucket ids, а чтение и запись внутри bucket плотные.
Irregular здесь только выбор начала редкого touched range. Его стоимость и
повторное чтение bounded bucket должны быть измерены; корректность на cache не
опирается.

`bucket_bits` выбирается вместе с `next_bucket_capacity_records`, чтобы
максимальный ожидаемый bucket оставался дешёвым, а суммарный slack помещался в
memory plan. Фактический overflow запрещает commit и завершает run. Никакого
перехода на runs, linked pages, global hash table или CPU fallback нет.

После drain каждый bucket уже уникален, а разные buckets имеют разные prefixes.
Scan counts и копирование buckets в возрастающем порядке сразу дают полностью
отсортированный уникальный `Hnext`; equality при контрольном adjacent check —
fatal invariant violation, а не ещё один дедуп.

## 9. Два frontier profiles

### 9.1. `DENSE`

```text
GenerateHash: state + hash
Route:        state + hash
OwnerCommit:  reject epoch/Hprev/Hcurr/already-accepted duplicates
              write one dense final state extent and commit its StateRefs
Finalize:     compact already-unique hash buckets; never remove a state
Next:         dense final state extents already materialized
```

Плюсы:

- выживший ребёнок не вычисляется второй раз;
- parent extents освобождаются потоково;
- при survival 90–95% почти все переданные state bytes всё равно нужны;
- в `StateRing` нет provisional states и duplicate holes.

Цена:

- state bytes отправляются до окончательного owner dedup;
- fixed `HnextBuckets` требуют заранее оплаченного bucket slack;
- accepted bucket update повторно читает bounded старый диапазон.

### 9.2. `HASH_FIRST`

```text
GenerateHash: hash + OriginRef
Route:        hash + OriginRef
OwnerCommit:  reject epoch/Hprev/Hcurr/already-accepted duplicates,
              reserve final StateRefs and enqueue only accepted OriginRefs
Materialize:  regenerate committed states потоково, параллельно следующим jobs
Finalize:     wait for materialization; compact already-unique hash buckets
Next:         dense all-live state extents
```

После каждого owner commit принятые записи попадают в заранее выделенный
`MaterializeSlot`. Один или несколько commits агрегируются до крупной
транзакции. Затем владельцы:

1. группируют accepted origins по `source_rank` и parent extent;
2. закрывают глобальный request-count exchange, после чего каждый source знает
   полное число ссылок на каждый свой parent extent;
3. отправляют requests большими непрерывными ranges;
4. source обслуживает parent extents по `allocation_seq`, применяет move и
   уменьшает их точный outstanding-origin count;
5. responses возвращаются в исходном request order и сразу попадают в уже
   зарезервированные target
   extents;
6. target extent публикует `MATERIALIZED`, после чего его можно архивировать;
7. головной parent extent освобождается, когда закрыты owner decisions по всем
   его детям, обслужены все accepted origins и
   закрыт archive lease.

Request/response использует тот же sequencer и отдельное пространство
`materialize_epoch`, поэтому candidate и materialization NCCL calls не могут
перемешаться. В audit mode owner повторно хэширует response и сравнивает его с
committed hash до публикации target extent.

Плюсы:

- state traffic и persistent state writes пропорциональны победителям;
- особенно выгоден при большом old-hit/duplicate ratio.

Цена:

- второй apply-move для каждого победителя;
- дополнительный request/response exchange;
- current states живут дольше;
- до materialization нужно одновременно вместить current, accepted next и
  кольцевой wrap slack.

Профиль выбирает отдельная calibration run. Во время production BFS он не
переключается.

## 10. `FinalizeDepth`

Финализация начинается только после глобально согласованного drain:

```text
all CurrentState extents enumerated
all parent obligations generated
all route slots retired
all candidate exchange epochs completed
all receive slots consumed
all OwnerDedupCommit jobs completed
all HASH_FIRST materialize slots and responses completed
all accepted target extents are MATERIALIZED
no kernel/collective capable of creating another Hnext accepted record
```

Затем на каждом owner rank:

1. `accepted_count` всех buckets сканируются в порядке `(shard,bucket)`;
2. их сумма обязана совпасть со счётчиком `owner_committed_records` и
   поместиться в arena, чей старый `Hprev` больше не нужен;
3. уже sorted unique bucket ranges копируются по scan offsets; concatenation
   сразу даёт полностью отсортированный `Hnext`;
4. adjacent-equality kernel подтверждает отсутствие duplicate; equality —
   `OWNER_COMMIT_INVARIANT`, а не повод удалить одну из записей;
5. все `StateRef` обязаны указывать на плотные materialized extents той же
   глубины, а их число обязано равняться `next_count`;
6. строятся shard/microbucket offsets нового compact hash layer;
7. проверяются локальные conservation equations и capacities;
8. проверяется регистрация всех state archive obligations; запускается D2H
   compact hash/directory chunks и ещё не стартовавших state tails;
9. fixed-order collectives сводят fatal, `next_count`, counters и fingerprints;
10. при чистом результате роли слоёв атомарно поворачиваются, а counts
    `HnextBuckets` обнуляются для следующей глубины:

```text
Hprev <- old Hcurr
Hcurr <- new unique Hnext
CurrentStates <- published next extents
depth <- depth + 1
```

`next_count == 0` после чистой финализации означает исчерпание достижимого
графа. Локальная пустая очередь этого не означает.

State extent и hash-layer arena имеют отдельные archive leases. Rotation может
переименовать логическую роль буфера, но физически перезаписать его разрешено
только после завершения соответствующего D2H. Если lease не закрыт к моменту,
когда capacity требует этот range, это `ARCHIVE_RECLAIM_LAG`, а не скрытое
ожидание диска.

### 10.1. Conservation equations

Категории имеют фиксированный приоритет и не пересекаются:

```text
generated
  = source_local_duplicates
  + source_emitted

source_emitted_global
  = local_loopback_received
  + remote_received

source_emitted_global
  = owner_input_global

owner_input
  = owner_epoch_duplicates
  + prev_hits
  + curr_hits
  + next_already_accepted_hits
  + owner_committed

owner_committed_global
  = next_count.

HASH_FIRST only:
owner_committed
  = materialize_requested
  = materialize_completed.
```

Для fixed-degree Cayley graph дополнительно:

```text
generated = current_frontier_count * move_count.
```

Счётчики ведутся per rank, per peer, per shard и globally. Их совпадение —
обязательный tripwire, но не доказательство правильности множеств; bounded
fixtures сравниваются по полным state bytes с независимым CPU oracle.

## 11. Архив без торможения GPU

Каждый published слой сохраняет:

```text
canonical live StateStorage records
sorted Hash128 layer
shard/bucket directory
depth counters and fingerprints
ConfigDigest256 and ownership epoch
```

Parent/path metadata в v1 не сохраняются, потому что они не входят в заявленный
результат. Добавление пути — новый output contract, а не безобидный флаг.

Путь данных:

```text
VRAM immutable extent
 -> archive_copy_stream
 -> preallocated pinned slot
 -> independent host I/O thread
 -> preallocated per-rank disk extent.
```

State extent становится immutable сразу после `OwnerCommit` в `DENSE` и после
`MATERIALIZED` в `HASH_FIRST`; его D2H разрешён немедленно и перекрывается с
оставшейся работой глубины. Compact `HASH_PLANE` и directories появляются лишь
на `FinalizeDepth`, поэтому они архивируются после bucket compaction. Общий
`LayerCommit` всё равно публикуется только после наличия обоих planes.

GPU range можно переиспользовать после завершения D2H, не после disk write:
pinned slot становится владельцем копии. Диск не создаёт backpressure. Если
archive worker временно не имеет свободного pinned slot, pending descriptor
остаётся только в его заранее выделенной очереди, а GPU продолжает jobs. Если
GPU allocator дошёл до range с незакрытым archive lease либо descriptor/pinned
queue исчерпала capacity, rank поднимает fatal. Он не ждёт диск, не уменьшает
batch и не отключает архивирование. Disk extent full и write/flush error также
fatal.

`FrontierPublish(d)` и `LayerCommit(d)` — разные события. Первое делает слой
доступным следующей GPU-глубине после completed cut и регистрации всех archive
obligations; оно не ждёт их D2H или disk completion. Второе делает слой durable
в архиве позже. Файл состоит из checksummed chunks и
`LayerCommit`. Частично записанный слой без валидного commit не выдаётся как
durable archive layer. Успешный `RunCommit` появляется только после flush всех
rank files и общего manifest. Уже committed старые слои остаются
диагностическим результатом после capacity stop, но сам run имеет статус
`INCOMPLETE`.

On-disk encoding — field-wise little-endian, не `memcpy` host struct. Каждый
rank file начинается с `FileHeader`:

```text
magic, archive_schema, header_bytes
ConfigDigest256, ownership_epoch
rank, world_size
state_profile_id, state_logical_bytes, state_storage_bytes
hash_algorithm_id, hash_bits=128
checksum_algorithm_id=BLAKE3_256
preallocated_extent_bytes
```

Каждый 4096-byte-aligned chunk имеет 64-byte header:

```text
chunk_sequence, kind, flags, depth
record_begin, record_count, payload_bytes
payload_digest_low128
```

Полный BLAKE3-256 payload digest и список chunk sequences входят в
`LayerCommit`; короткое поле header служит ранним corruption check. Chunk kinds:

```text
STATE_PLANE
HASH_PLANE
SHARD_DIRECTORY
BUCKET_DIRECTORY
DEPTH_COUNTERS
LAYER_COMMIT
RUN_COMMIT
```

State plane содержит только live states, без дыр `StateRing`; hash plane —
отсортированный owner-local hash set. Их порядок независим, но counts обязаны
совпасть. Offline verifier повторно хэширует каждый state и точно сравнивает
полные canonical state sets между seed/rank configurations.

## 12. Failure contract

Fatal conditions как минимум:

```text
STATE_RING_FULL
HASH_LAYER_FULL
NEXT_BUCKET_OVERFLOW
NEXT_BUCKET_DIRECTORY_MISMATCH
OWNER_COMMIT_INVARIANT
TARGET_EXTENT_UNMATERIALIZED
ROUTE_SLOT_OVERFLOW
RECEIVE_SLOT_OVERFLOW
MATERIALIZE_SLOT_OVERFLOW
SCRATCH_PLAN_MISMATCH
ARCHIVE_PINNED_FULL
ARCHIVE_RECLAIM_LAG
DISK_EXTENT_FULL
DISK_IO_ERROR
CUDA_ERROR
NCCL_ERROR
NCCL_PROGRESS_TIMEOUT
CONFIG_MISMATCH
CONSERVATION_MISMATCH
UNSUPPORTED_SM_OR_BACKEND
```

Первый local fatal запрещает создание новых semantic jobs. Все ranks входят в
одинаковую fixed-order fatal collective, прекращают запуск следующих epochs и
доводят уже поставленные операции только до безопасной точки владения
буферами. Частичная глубина не получает `LayerCommit`.

Fatal collective не запускается вне очереди поверх активного NCCL epoch.
Sequencer сначала закрывает или аварийно завершает текущий номер, затем выдаёт
всем ranks один и тот же `FATAL_EPOCH`; иначе сама обработка ошибки могла бы
создать несовпадающий NCCL order и зависание.

Если сам NCCL epoch перестал прогрессировать, ждать следующую collective уже
нельзя. Out-of-band control watchdog видит неизменный epoch heartbeat,
вызывает `ncclCommAbort` на всех workers и завершает запуск как
`NCCL_PROGRESS_TIMEOUT`. Это аварийный stop, не retry; текущая глубина остаётся
uncommitted.

В v1 нет retry, миграции owner shard, уменьшения batch, смены backend или
перехода на CPU. Ошибка видна как точный код, depth, rank, shard, job, requested
и available capacity.

## 13. Preflight

До depth 0 все ranks обязаны подтвердить:

1. одинаковый полный `ConfigDigest256`;
2. один уникальный GPU UUID на rank и правильный `LOCAL_RANK`;
3. `world_size == 2^owner_bits` и rank map является перестановкой;
4. одинаковые start/generator/inverse manifests;
5. `StateTraits` canonicalization и inverse tests;
6. CPU/GPU fixed vectors для каждого move и hash seed;
7. поддержку выбранного backend фактическим SM/cubin;
8. checked arithmetic всех count/byte high-water;
9. успешное выделение всей VRAM и заявленного untouched reserve;
10. успешное выделение всех pinned slots и disk extents;
11. NCCL bootstrap и малый ordered P2P smoke между всеми ranks;
12. нулевые counters, свободные slots и пустой архивный queue;
13. напечатанную collision bound для `expected_max_unique_states`.

Любая ошибка одного rank не позволяет ни одному rank войти в BFS.

### 13.1. Инициализация глубины 0

Все ranks независимо вычисляют один и тот же `H(A)` и owner prefix. Только
назначенный owner резервирует один state extent, записывает canonical `A` и
единственный `Hcurr` record. `Hprev` пуст. Global reductions обязаны получить
`current_count=1` и ровно одного владельца; затем слой 0 проходит обычный
archive enqueue и `FrontierPublish(0)`. Parent jobs могут начаться сразу после
регистрации archive lease; D2H и disk `LayerCommit(0)` завершаются асинхронно.

## 14. Calibration и выбор профиля

Перед большим production run выполняется отдельный bounded calibration с теми
же state/action/hash/owner правилами. Она не меняет будущий run, а сохраняет по
глубинам:

```text
frontier states
generated children
source-local duplicates
prev hits, curr hits
owner-epoch duplicates
next-already-accepted hits
next survivors
state/hash bytes before and after routing
per-rank and per-shard max/mean
per-bucket max/mean occupancy and accepted-range read amplification
time of GenerateHash, sort, exchange, owner commit, materialize,
bucket compaction and archive D2H
StateRing and HnextBuckets high-water
```

Выбор делается по end-to-end времени и capacity:

- высокий survivor ratio и дорогая rematerialization склоняют к `DENSE`;
- низкий survivor ratio и дорогой state traffic склоняют к `HASH_FIRST`;
- `LocalPreDedup ON/OFF` сравниваются, потому что sort нужен в любом случае, но
  compaction может не окупиться на свободно растущем графе;
- `BMMA_BUCKET` сравнивается только с тем же профилем и теми же выходными
  множествами.

Автоматического переключения на пике или спаде фронтира нет. Для другого режима
запускается другой production run.

### 14.1. Что уже подтверждает выбор, а что ещё нет

- [REF-011](experiments/REF-011-wire-record-strategies.md) показал, что deferred
  metadata/state transfer проигрывает на ранних слоях, где принимаются все
  кандидаты. Поэтому `HASH_FIRST` не объявлен универсальным.
- [REF-015](experiments/REF-015-cub-sort-unique-visited.md) показал цену полной
  sort/unique, когда у 32-bit dense rank уже есть дешёвый bitmap. Здесь sort
  оправдывается только тем, что один и тот же порядок используется для local
  unique, owner routing и последующего merge.
- [REF-016](experiments/REF-016-cayley-s8-successor-locality.md) показал сильную
  зависимость локальных дублей от parent-major/generator-major порядка. Поэтому
  v1 фиксирует parent-major logical layout и измеряет его физическое сохранение.
- [REF-017](experiments/REF-017-fused-gpu-cayley-s9.md) подтвердил полезность
  fused generation на одном небольшом GPU-resident обходе, но не является
  доказательством application-size или multi-GPU скорости этой архитектуры.

## 15. Верификация

### 15.1. Семантика

- независимый CPU successor oracle, не использующий GPU generator table code;
- exhaustive tiny Cayley groups с полным сравнением state sets на каждой
  глубине;
- self-loop, duplicate generator, same-parent alias и cross-parent convergence;
- одинаковый ребёнок в разных owner epochs: первый commit остаётся, все поздние
  arrivals отклоняются, а финализация не меняет count;
- non-bipartite Cayley fixture для обязательной проверки `F_d`;
- намеренно не inverse-closed manifest должен падать в preflight;
- forced Hash128 collision fixture должен демонстрировать принятую границу, а
  не называться exact BFS.

### 15.2. Эквивалентность реализаций

Для каждого bounded графа обязаны совпасть полные состояния слоёв:

```text
DENSE == HASH_FIRST
LocalPreDedup OFF == ON
CUB_SORT_MERGE == BMMA_BUCKET
CUDA_ZOBRIST == TC_HASH
1 rank == 2 ranks == 4 ranks
несколько hash seeds после сравнения по full state bytes.
```

Counts/fingerprints используются как быстрые regression checks, но не заменяют
точное set comparison.

### 15.3. Runtime и hardware ladder

1. CPU protocol simulator с переставленными ready events и capacity faults;
2. one-GPU CUDA tests и все Compute Sanitizer режимы;
3. настоящий `2 ranks / 2 T4` запуск на Kaggle, а не два процесса на одной GPU;
4. Nsight Systems timeline с одновременными generate/sort/comm/owner/D2H;
5. H100/B200 multi-GPU, затем multi-node;
6. restart не тестируется как recovery: v1 обязан корректно обозначить run
   incomplete.

T4 и B200 имеют отдельные compiled policies. Результат T4 не доказывает
эффективность Blackwell kernel, а одно-GPU прохождение не подтверждает NCCL
ordering.

### 15.4. Performance acceptance

На каждой глубине сохраняются:

```text
semantic counts and all conservation categories
per-stage GPU time and end-to-end depth time
critical rank/shard, max/mean imbalance
sort input/output, routed records and bytes per peer
owner old/duplicate/survivor counts
next-already-accepted hits and per-bucket occupancy
StateRing, HnextBuckets, scratch and pinned high-water
comm epochs, average payload and zero-payload participation
archive D2H and disk throughput
fatal/capacity status
```

Tensor backend принимается только по полному depth time и capacity при той же
семантике. Изолированный высокий BMMA/GEMM throughput недостаточен.

## 16. Реализационные границы

```text
crates/mgbfs-core/        StateTraits, config, manifests, CPU oracle
crates/mgbfs-runtime/     scheduler, slots, StateRing, finalization, archive
crates/mgbfs-cli/         native rank worker and manifest CLI

cuda/generate/            CUDA_ZOBRIST, TC_HASH, MATRIX_TC_HASH
cuda/route/               radix sort, local unique, owner segmentation
cuda/owner/               merge/set-difference, BMMA, bucket commit
cuda/materialize/         HASH_FIRST request apply/response
cuda/finalize/            bucket scan/compact, invariant check, hash directories

python/cayleypy_mgbfs/    graph/config adapter and launcher only
tests/cpu_oracle/
tests/cuda/
tests/multigpu/
```

Первый vertical slice:

```text
один permutation StateTraits
CUDA_ZOBRIST
DENSE
LocalPreDedup ON и OFF
CUB_SORT_MERGE
1 GPU
lossless archive
полное сравнение с CPU oracle
```

Второй slice добавляет настоящий 2xT4 owner exchange. Третий — `HASH_FIRST` и
materialization. Tensor backends подключаются только после стабильного CUB
эталона; иначе невозможно отличить новую оптимизацию от ошибки семантики.

## 17. Реестр решений

### 17.1. Confirmed

- strict layer-setting BFS;
- только inverse-closed Cayley graphs в v1;
- Hash128 identity с управляемым seed и честной probabilistic гарантией;
- owner/shard/bucket из prefix bits без дополнительной перестановки;
- один source radix sort, затем owner merge;
- обязательный owner-side дедуп против `Hprev`, `Hcurr` и уже committed
  `Hnext`, после которого state финален;
- фиксированный flat prefix-bucket accepted store без late dedup;
- крупные последовательные reads/writes и scan-derived offsets;
- один static `StateRing`, потоковое освобождение parents;
- два immutable frontier profiles;
- optional local pre-dedup;
- CUB baseline и экспериментальный BMMA bucket backend;
- единственный semantic barrier между глубинами;
- обязательный lossless disk archive через preallocated pinned ring;
- fail-fast при любой нехватке capacity.

### 17.2. Proposed, требуется измерение

- конкретные `bucket_bits`, `next_bucket_capacity_records`,
  `exchange_trigger_records`, `owner_job_candidate_records` и
  `owner_job_bucket_descriptors` выбирает calibration;
- `DENSE` либо `HASH_FIRST` выбирается до production run по end-to-end depth
  time и byte high-water;
- `BMMA_BUCKET`, `TC_HASH` и `MATRIX_TC_HASH` остаются экспериментальными до
  равенства выходов и выигрыша у baseline на целой глубине.

### 17.3. Open за границей v1

- injective state encoding / collision-resolving exact identity;
- bidirectional BFS;
- target/path/parent outputs;
- directed generators с rolling window длиннее двух старых слоёв;
- динамическая смена профиля;
- recovery, elastic ranks и owner migration;
- non-power-of-two world size в native v1;
- algebraic/coset ownership вместо hash ownership;
- 2D generator-sharded decomposition.

### 17.4. Rejected

- hash-only frontier без full states;
- материализация всего `frontier * move_count` одним массивом;
- random per-candidate append/allocator;
- giant pointer-linked bucket/page structure;
- dirty owner runs с отложенным cross-epoch dedup на финализации;
- GEMM, объявленный сортировкой;
- Tensor Core local compare, объявленный глобальным dedup;
- пустая локальная очередь, объявленная завершённым слоем;
- успешный count/fingerprint, объявленный доказательством равенства множеств;
- fake multi-rank test на одной видимой GPU;
- молчаливый fallback после overflow или I/O lag.

## 18. Definition of architecture-ready

Кодирование production runtime можно начинать, когда одновременно готовы:

1. versioned `RunConfig`, `ConfigDigest256` и все ABI static assertions;
2. конкретный первый `StateTraits` с независимыми CPU vectors;
3. byte-exact memory planner и fatal high-water checks;
4. StateRing/extent simulator с wrap, плотными final extents и reclamation
   leases;
5. exchange-epoch simulator с одинаковым NCCL order на всех ranks;
6. CUB owner oracle с несколькими epochs, committed `Hnext` lookup и bucket
   scan/compact;
7. archive chunk/commit schema и injected short-write tests;
8. conservation counters в первом же vertical slice;
9. отдельный calibration CLI;
10. зафиксированный 1GPU -> real 2xT4 verification gate.

До этих десяти пунктов документ является архитектурой, а не заявлением о
готовой или быстрой реализации.

## 19. NVIDIA contract references

- CUTLASS — GEMM и связанные tiled/data-movement primitives, но не алгоритм
  radix sort: <https://docs.nvidia.com/cutlass/latest/overview.html>
- PTX single-bit MMA определяет `xor.popc` и `and.popc` semantics:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- NCCL grouped calls требуют одинакового issue order между ranks:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html>
