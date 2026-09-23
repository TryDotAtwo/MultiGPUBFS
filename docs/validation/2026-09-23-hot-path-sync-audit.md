# CPU synchronization audit: MultiGPUBFS

Передача основному агенту по просьбе Ивана. Дата: 2026-09-23.

## Вывод

В runtime остаются существенные CPU-зависимости внутри обработки батчей:
чтение GPU-счётчиков, построение заданий на CPU, ожидание CUDA-stream,
последовательное управление шардами и блокирующие коллективные согласования.
Удаление одного ожидания в cuCollections не делает весь BFS асинхронным.

Это статический аудит исходников на HEAD `33eb506` с существующими рабочими
изменениями. Он подтверждает наличие зависимостей, но не измеряет их стоимость,
не доказывает ошибочность результатов BFS и не гарантирует отсутствие других
ошибок. Номера строк относятся к осмотренному состоянию файлов.

## Подтверждённые места

Все пути ниже относительно корня MultiGPUBFS.

### 1. Скрытая синхронизация в загрузке данных

`crates/mgbfs-runtime/src/distributed_native.rs:156–179`:

- `Buffer::put()` вызывает `cudaMemcpyAsync`, затем безусловный
  `cudaStreamSynchronize` (строка 169).
- `read()` использует синхронный `cudaMemcpy`; `one()` вызывает `read()`.
- Эти методы используются в горячем пути для counts, controls, extents и jobs.

Аналогичная схема есть в `crates/mgbfs-runtime/src/native.rs:52–58`.
Поиск только явных ожиданий в вызывающем коде пропускает эти зависимости.
Удалять ожидание из `put()` без решения lifetime host-буфера нельзя.

### 2. Library owner: несколько CPU round trips на шард

`crates/mgbfs-runtime/src/distributed_native.rs`, `commit_library_batch()`:

- 1267: ожидание stream после построения директории.
- 1269–1272: чтение fatal и директории бакетов на CPU.
- 1275: CPU-цикл по шардам.
- 1315: compare возвращает host-visible число survivors.
- 1342: blocking snapshot после резервирования.
- 1404: blocking snapshot после материализации.
- 1417–1428: CPU публикует extent в host-список.

Скрытые ожидания подтверждены в:

- `experiments/library_owner/cuco_owner.cuh:218–224`: D2H двух счётчиков,
  `stream_.synchronize()`, проверка capacity на CPU.
- `experiments/library_owner/control_transfer.cpp:67–75`: D2H control,
  extent, ring и optional count, затем `cudaStreamSynchronize`.

Следствие: несколько последовательных host/GPU зависимостей на каждый
непустой шард. Размер эффекта требует профилирования.

### 3. Native owner также зависит от CPU

`crates/mgbfs-runtime/src/distributed_native.rs`:

- 1481–1498: stream drain, D2H directory, CPU `split()`, затем блокирующий
  `jobs_gpu.put()`.
- 1642–1656: в DENSE ожидание конца owner batch, чтение ring и descriptors,
  формирование следующего списка extents на CPU.
- 1542–1548 и 1617–1634: HASH_FIRST дополнительно читает control, survivor
  count и extent внутри обработки отдельных spans.

DENSE уже объединяет часть результатов, но граница batch остаётся синхронной.

### 4. Маршрутизация и транспорт

`crates/mgbfs-runtime/src/distributed_native.rs`:

- 2015–2018: при local pre-dedup счётчик routed читается через host wait.
- 2046–2049: безусловное ожидание упаковки и D2H `owner_counts`.
- 2104: загрузка размера сообщения через блокирующий `put()`.
- 2105–2117: NCCL-обмен размерами, ожидание communication stream, чтение
  размера приёма на CPU; только затем ставится пересылка payload.

Это host dependency на peer round. Само использование NCCL не устраняет её.
Пересылка максимальных буферов с padding не должна становиться неявным
исправлением: её расход трафика и памяти нужно явно оценить.

### 5. Освобождение родителей

`crates/mgbfs-runtime/src/distributed_native.rs`:

- 2164–2174: retirement kernel, stream drain, чтение ring на CPU.
- 2179: дополнительный безусловный stream drain перед owner processing.
- 2244–2256: похожая последовательность для HASH_FIRST с последующим
  коллективным согласованием fatal.

Здесь критичны lifetime родителей и завершение archive/transport readers.
Заменять ожидания нужно вместе с протоколом событий и владения.

### 6. Коллективные согласования внутри глубины

`all_max()` в `crates/mgbfs-runtime/src/distributed_native.rs:1228–1239`:
blocking upload → NCCL all-reduce → stream drain → D2H результата.

Вызовы есть внутри батчей и rounds, в частности на строках 1944, 1981,
2221, 2256 и 2261. HASH_FIRST добавляет вызовы на 1713, 1737, 1755 и 1830.

Это не буквальный MPI barrier, но блокирующие межранговые зависимости
в control flow. Утверждение «согласование только на FinalizeDepth» текущему
исполнению не соответствует. Нельзя удалить их без сохранения одинакового
NCCL issue order, zero-payload участия, termination и fatal propagation.

### 7. HASH_FIRST materialization

`materialize_hash_first()` в
`crates/mgbfs-runtime/src/distributed_native.rs:1661–1830` содержит
последовательные ожидания и host reads для fatal, размеров обмена и публикации.
Оптимизация только owner compare оставит эту цепочку зависимостей.

### 8. Однокарточный native path

`crates/mgbfs-runtime/src/native.rs`:

- 752: host `cudaEventSynchronize` перед продолжением обработки.
- 770–771: retirement drain и чтение ring.
- 864–869: ожидание и чтение control/extent внутри owner processing.

Одна GPU и наличие producer stream сами по себе не доказывают непрерывное
асинхронное исполнение.

## Что не считать дефектом автоматически

- Синхронизации при инициализации, Drop, snapshot и окончательном завершении.
- Ожидания на семантической границе слоя, например
  `dense_device.rs:437` перед проверкой итогов и swap.
- GPU-side `cudaStreamWaitEvent`, защищающие настоящие зависимости.
- Ожидания archive events в отдельном фоновом worker.
- Синхронизации, включаемые только `trace_route`; для замеров trace должен
  быть выключен.
- `macro_native.rs:745/816`: это границы produce/settle, требующие отдельного
  анализа семантики; в этом аудите они не объявляются лишними.

## Почему это получилось

По коду сохраняется первоначальная синхронная интеграция: host-sized ABI
(`SurvivorsV1.rows`), директории и extent-списки на CPU, host-проверки
резервирования и размеры NCCL payload, известные на CPU перед launch.
`library_native.rs:357–359` прямо называет этот путь первым integration
backend и не заявляет overlap.

Документ `docs/plans/device-driven-library-owner.md` признаёт эти ограничения,
но документ не является реализованным конвейером. Причины приоритизации
работ самим агентом этот аудит не устанавливает.

## Что требуется от основного агента

1. Добавить эти зависимости в completion ledger с привязкой к текущему коду.
2. Реализовывать сквозной путь owner → transport → retirement. Удаление
   отдельного wait или смену API на Async не выдавать за готовый результат.
3. Перенести необходимые counts, offsets, capacity decisions и descriptors
   на GPU; фиксировать lifetime буферов и событий для каждого потребителя.
4. Сохранить fail-fast, fatal propagation, NCCL ordering и correctness при
   пустых батчах, разных размерах фронтира и capacity failures.
5. Проверить полный owner DAG на capture без host readback. Отдельно снять
   Nsight Systems timeline реального BFS: D2H counts, host waits, launch gaps,
   collective stalls и степень overlap. Capture сам по себе не заменяет timeline.
6. После изменения выполнить full-state проверки профилей и рангов, четыре
   Compute Sanitizer gate и согласованные замеры времени/VRAM с повторами.

Не обещать отсутствие всех ошибок по этому статическому разбору. Он задаёт
конкретные подтверждённые точки, которые нужно устранить или обосновать.
