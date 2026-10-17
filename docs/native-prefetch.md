# Предзагрузка native-сборок без GPU

Установка пакета не требует GPU. Предзагрузка тоже не опрашивает карты, не загружает CUDA-библиотеки и не запускает дочерние процессы или BFS. На login/SSH-узле достаточно Linux x86_64, Python3.10+, доступа к GitHub и места в постоянном кэше.

## Установка

```bash
python -m pip install --upgrade "multigpubfs[cayleypy] @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261017-v24/multigpubfs-0.2.0.dev24-py3-none-any.whl#sha256=693917bbede1fe2f4808481f32335c3df0d4bda77e7f0eafe09e48537e36980d"
```

Если CayleyPy/PyTorch уже установлены, можно использовать `multigpubfs @` вместо extra. Команда одна и та же для SSH и вычислительного окружения; в Kaggle установку выполняют через `%pip`. После обновления нужен новый Python-процесс.

## Выбранные или все сборки

Выберите постоянный каталог, который виден и login-узлу, и вычислительным заданиям. Если домашняя директория общая, подходит стандартный `$HOME/.cache/multigpubfs/native`. Иначе укажите доступный общий диск через `--cache` и используйте тот же `MGBFS_NATIVE_CACHE` в заданиях.

```bash
export MGBFS_NATIVE_CACHE="$HOME/.cache/multigpubfs/native"

# Только посмотреть оставшийся объём. Ничего не скачивает и не записывает.
python -m multigpubfs prefetch --all --dry-run

# Только выбранные SM: например SM100 и SM120.
python -m multigpubfs prefetch --sm 100,120

# Или все штатные SM и универсальная сборка.
python -m multigpubfs prefetch --all
```

`--sm` можно повторять; `--sm 100 --sm 120` эквивалентно списку выше. Поддерживаются 75,80,86,89,90,100,103,120 и `universal`. `--sm` и `--all` взаимоисключающие. Для машин с разными поколениями GPU внутри одного задания добавьте `universal` либо используйте `--all`.

| Набор, чистый кэш | Объём архивов |
|---|---:|
| Общий runtime + SM100 и SM120 | 327,579,748 байт |
| Общий runtime + восемь отдельных SM без universal | 405,482,922 байт |
| `--all`: общий runtime + восемь SM + universal | 514,463,349 байт |

Это размеры архивов компонентов; зависимости pip и сетевые накладные расходы считаются отдельно. Общий runtime скачивается один раз для этого набора компонентов. Новая архитектура добавляет только отсутствующий компонент. `--dry-run` проверяет хеши имеющихся файлов и сообщает консервативный максимум оставшихся загрузок; частично скачанный архив при повторном обращении может продолжиться через HTTP Range.

`--all` относится к штатным вычислительным сборкам. Изолированные Static/legacy-контроли для глубокой диагностики не входят в него и могут потребовать отдельного сетевого доступа при соответствующей проверке. Не называйте предзагрузку доказательством работы GPU: статус `PREFETCHED` подтверждает файлы и контрольные суммы; `PLAN_ONLY` означает только предварительный расчёт.

## Вычислительные задания без скачиваний

В каждом задании установите тот же путь к кэшу и запретите загрузки:

```bash
export MGBFS_NATIVE_CACHE="$HOME/.cache/multigpubfs/native"
export MGBFS_NATIVE_OFFLINE=1
python my_calculation.py
```

Библиотека определит выделенные заданию GPU и выберет подходящие файлы из кэша. Переустановка при смене карты не требуется. Старые сборки сохраняются рядом. Несколько заданий используют общий runtime и свои подходящие библиотеки; блокировки и атомарная публикация предотвращают повторное скачивание одного компонента конкурентными процессами. GPU visibility остаётся под управлением планировщика.

Если нужной сборки нет, `MGBFS_NATIVE_OFFLINE=1` выдаст явную ошибку вместо скрытого скачивания. Если кэш временный, очищается после задания или не виден другой ноде, переиспользования не будет. Один и тот же кэш можно использовать из нескольких Python-окружений. Неизменившиеся компоненты переиспользуются по SHA256 и после обновления Python-пакета.

## Python-only доступ

На узле без GPU:

```python
from multigpubfs import prefetch_native
report = prefetch_native(sm=[100, 120], cache="/path/to/shared/cache")
# Или: prefetch_native(all_architectures=True, cache="/path/to/shared/cache")
print(report["status"], report["downloaded_bytes"])
```

В другом процессе с выделенными GPU и уже установленными зависимостями:

```python
import os
os.environ["MGBFS_NATIVE_CACHE"] = "/path/to/shared/cache"
os.environ["MGBFS_NATIVE_OFFLINE"] = "1"

import multigpubfs, cayleypy
from multigpubfs.generic_session import persistent_graph_workers
multigpubfs.enable_native()
with persistent_graph_workers():
    for n in (10, 11):
        graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.coxeter(n), device="cpu")
        result = graph.bfs()
        print(n, result.bfs_completed, result.layer_sizes)
```

`prefetch_native` не меняет глобальные переменные окружения. Путь из `cache=` нужно передать вычислительным заданиям через `MGBFS_NATIVE_CACHE`. Ответ содержит готовый словарь `environment`. Для автоматизации используйте `python -m multigpubfs prefetch --all --json`: stdout содержит один JSON, сообщения прогресса идут в stderr. Дедлайн предзагрузки по умолчанию600 секунд; `--timeout` позволяет задать до1800. Это сетевой этап, а не ограничение размера или времени самого BFS.

[Проверка выпуска](native-prefetch-acceptance-2026-10-15.md): настоящие архивы загружены на машине без GPU; затем реальные параллельные и persistent-запуски на2×T4 работали из подготовленного кэша без новых загрузок. Выполнение на других архитектурах этим тестом не подтверждается.
