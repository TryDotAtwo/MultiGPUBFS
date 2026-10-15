# Обновить пакет и начать расчёт на сервере

Команду нужно выполнять внутри задачи, которой уже выделены GPU. Она обновляет
только MultiGPUBFS, сохраняет существующие зависимости Torch/CayleyPy и запускает
диагностику в новом Python-процессе:

```sh
python -m pip install --upgrade --force-reinstall --no-deps "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261015-v16/multigpubfs-0.2.0.dev16-py3-none-linux_x86_64.whl#sha256=65941655242e7a48fa3f27989992fe5f8d5d898c716d481ce0c92cdd9d4a5a6b" && python -m multigpubfs.diagnostics
```

Это обновление установленного окружения. Для чистой установки с CayleyPy есть
[обычная команда pip с extra](install-native-package.md).

Штатная библиотека уже содержит готовые SM75/SM86/SM120 ядра и shared CUDA runtime.
Компилятор на сервере, ручные пути библиотек и выбор диагностического варианта не
нужны. GPU видны так, как их выделил планировщик; CUDA_VISIBLE_DEVICES менять не нужно.

При исправном окружении обычный doctor завершится PASS, включая `gpu-N-bfs`,
а при нескольких GPU — `nccl-exchange` и `multi-gpu-bfs`. Это реальные маленькие
BFS с проверкой точных слоёв. FAIL/TIMEOUT означает неуспешную проверку; полный
отчёт и ZIP сохраняются в напечатанном каталоге. Ничего автоматически не отправляется.

Если сервис принимает только Python-код, отправьте такой файл. Обновление и
диагностика выполняются отдельными процессами, поэтому старые импорты не мешают:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade",
                "--force-reinstall", "--no-deps", "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261015-v16/multigpubfs-0.2.0.dev16-py3-none-linux_x86_64.whl#sha256=65941655242e7a48fa3f27989992fe5f8d5d898c716d481ce0c92cdd9d4a5a6b"], check=True)
subprocess.run([sys.executable, "-m", "multigpubfs.diagnostics"], check=True)
```

После успешной проверки запускайте расчёт из нового Python-процесса:

```python
from time import perf_counter
import multigpubfs
from cayleypy import CayleyGraph, PermutationGroups
from multigpubfs.generic_session import persistent_graph_workers

multigpubfs.enable_native()
with persistent_graph_workers():
    for n in (5, 11, 12):
        graph = CayleyGraph(PermutationGroups.coxeter(n), device="cpu")
        started = perf_counter()
        result = graph.bfs()
        if not result.bfs_completed:
            print(n, "INCOMPLETE", result.completed_depth, result.output)
            break
        print(n, result.num_vertices(), result.diameter(),
              round(perf_counter() - started, 3), result.output)
```

`device="cpu"` относится к описанию графа. После явного `enable_native()` сам
BFS выполняется нашим native GPU-движком. По умолчанию он использует выделенные
видимые карты и автоподбор; общий persistent-контекст сохраняет workers между
графами. Точные размеры всех слоёв доступны в `result.layer_sizes` и отчёте;
состояния сохраняются компактными выборками, не всеми слоями целиком.

`PermutationGroups.coxeter(n)` можно заменить поддерживаемым конструктором
CayleyPy. Для перебора LRX/Rapaport M2 есть [пример пар](../examples/cayleypy_pair_sweep.ipynb).
Возможность закончить конкретный граф зависит от доступной памяти и объёма работы.

При разборе оставшейся ошибки используйте `python -m multigpubfs.diagnostics --deep`:
до 30 минут тестов, проверки штатного BFS раньше старых сборок и один ZIP.
`production_bfs_acceptance` отдельно показывает результат штатного запуска;
FAIL в эксперименте старой сборки не превращает успешный штатный BFS в неуспешный
расчёт. Общий статус deep сохраняет сведения обо всех ошибках.

[Измеренные проверки и границы покрытия](sm120-production-acceptance-2026-10-15.md).
На T4 проверена эта библиотека с её SM75-кодом; запуск именно на серверных SM120
нужно подтвердить свежим doctor на том сервере.
