"""Exact nearest paths and bounded labeled walks on one or more local CUDA GPUs."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import operator
from pathlib import Path
import subprocess
from struct import Struct
from tempfile import TemporaryDirectory
from time import perf_counter

from .graph_definition import GraphDefinition, I64_MAX, _flatten, _normalize_graph
from .path_gpu import GpuLayers, Layer, SearchStopped as _Stopped, inverse_action
from .path_lineage import DiskLineage, DiskLinks, _Blocks
from .path_network import DistributedLayers, connect
from .specialized import match_lrx


@dataclass(frozen=True)
class PathSearchResult:
    graph: GraphDefinition
    output: Path
    report: dict

    @property
    def status(self):
        return self.report["status"]

    def iter_paths(self):
        """Yield records with target_index, length and forward generator_ids."""
        if "path_file" not in self.report:
            return
        import pyarrow.parquet as pq

        with pq.ParquetFile(self.output / self.report["path_file"]) as reader:
            for batch in reader.iter_batches(batch_size=1024):
                yield from batch.to_pylist()

    def verify_paths(self, cayley_graph):
        """Replay saved words with CayleyPy and return the number checked."""
        if not callable(getattr(cayley_graph, "apply_path", None)):
            raise TypeError("PATH_VERIFICATION_REQUIRES_CAYLEYPY_GRAPH")
        if _normalize_graph(cayley_graph, self.graph.start).digest() != self.graph.digest():
            raise ValueError("PATH_VERIFICATION_GRAPH_MISMATCH")
        targets = self.report["targets"]
        count = 0
        for path in self.iter_paths():
            actual = _flatten(cayley_graph.apply_path(self.graph.start, path["generator_ids"]))
            if actual != list(targets[path["target_index"]]):
                raise RuntimeError(f"PATH_VERIFICATION_TARGET_MISMATCH at saved row {count}")
            count += 1
        return count


def _cuda_devices(torch, devices):
    if not torch.version.cuda or getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        raise RuntimeError("PATH_SEARCH_REQUIRES_NVIDIA_CUDA")
    count = torch.cuda.device_count()
    if devices is None:
        devices = range(count)
    elif isinstance(devices, int):
        devices = (devices,)
    elif isinstance(devices, str):
        try:
            devices = tuple(int(item) for item in devices.split(","))
        except ValueError:
            raise ValueError("PATH_CUDA_DEVICES") from None
    try:
        ids = tuple(devices)
    except TypeError:
        raise ValueError("PATH_CUDA_DEVICES") from None
    if not ids or any(type(rank) is not int or not 0 <= rank < count for rank in ids):
        raise ValueError("PATH_CUDA_DEVICES")
    if len(set(ids)) != len(ids):
        raise ValueError("PATH_CUDA_DEVICES")
    return tuple(torch.device("cuda", rank) for rank in ids)


def _nonnegative(value, error, maximum=None):
    if isinstance(value, bool):
        raise ValueError(error)
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(error) from None
    if value < 0 or (maximum is not None and value > maximum):
        raise ValueError(error)
    return value


def _deadline(max_seconds, started):
    if max_seconds is None:
        return None
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise ValueError("PATH_MAX_SECONDS")
    try:
        valid = math.isfinite(max_seconds) and max_seconds >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("PATH_MAX_SECONDS")
    return started + max_seconds if max_seconds else None


def _graph_and_targets(graph, start, targets):
    graph = _normalize_graph(graph, start)
    try:
        states = list(dict.fromkeys(tuple(_flatten(target)) for target in targets))
    except TypeError:
        raise ValueError("PATH_TARGETS_REQUIRED") from None
    if not states:
        raise ValueError("PATH_TARGETS_REQUIRED")
    if any(len(state) != graph.state_elements for state in states):
        raise ValueError("PATH_TARGET_SHAPE")
    return graph, states


def _active_path_targets(graph, targets, direction):
    """Keep report IDs while selecting roots in the exact reverse domain."""
    action = graph.action
    if direction == "bidirectional" and action["kind"] == "matrix":
        modulus = action["generators"][0]["modulo"]
        if modulus:
            # An admitted start and all forward successors are canonical.
            return [(i, state) for i, state in enumerate(targets)
                    if all(0 <= value < modulus for value in state)]
    return list(enumerate(targets))


def _empty_native_path_receipt(stop_reason=None):
    return dict(status="UNREACHABLE" if stop_reason is None else "INCOMPLETE",
                stop_reason=stop_reason, d_min=None, target_index=None, generator_ids=[])


def _walks(history, rank, row, check, *, root_counts=None):
    """Walk predecessors iteratively, yielding root coordinates and edge labels."""
    stack = []
    edges = []
    depth = len(history) - 1
    while True:
        check()
        if not depth:
            if root_counts is not None and not (0 <= rank < len(root_counts) and 0 <= row < root_counts[rank]):
                raise RuntimeError("NATIVE_WINDOW_ROOT_BOUNDS")
            yield rank, row, tuple(edges)
        else:
            links = history[depth].links
            if root_counts is not None and not 0 <= rank < len(links):
                raise RuntimeError("NATIVE_WINDOW_PARENT_RANK")
            parents = links[rank]
            begin, end = parents.bounds(row)
            if begin != end:
                stack.append([depth, begin, end, parents])
                edges.append(None)
        while stack and stack[-1][1] == stack[-1][2]:
            stack.pop()
            edges.pop()
        if not stack:
            return
        parent_depth, index, _, parents = stack[-1]
        stack[-1][1] += 1
        rank, row, edges[-1] = parents.parent(index)
        depth = parent_depth - 1


def _nearest(gpu, roots, goals, direction, keep_links, store):
    histories = [[roots], [goals]]
    visited = [roots.states, goals.states]
    depths = [0, 0]
    while True:
        gpu.checkpoint()
        left, right = histories[0][-1], histories[1][-1]
        match = gpu.match_nearest(left, right)
        if match is not None:
            return sum(depths), (match, histories) if keep_links else None
        if not left.size or not right.size:
            return None, None
        side = 0
        # Every completed expansion increases the covered distance by one.
        if direction == "bidirectional" and (right.size, depths[1]) < (left.size, depths[0]):
            side = 1
        previous = histories[side][-1]
        lineage = (store, f"n{side}", depths[side] + 1) if keep_links else None
        layer, visited[side] = gpu.advance(
            previous, visited[side], reverse=bool(side), lineage=lineage,
        )
        depths[side] += 1
        if keep_links:
            # A prior layer is needed only for its lineage after expansion.
            histories[side][-1] = Layer([], previous.links)
            histories[side].append(layer)
        else:
            histories[side][-1] = layer


def _witness(gpu, match, histories, goal_ids):
    rank, left_at, right_at = match
    _, _, prefix = next(_walks(histories[0], rank, left_at, gpu.check))
    goal_rank, goal_at, suffix = next(_walks(histories[1], rank, right_at, gpu.check))
    yield goal_ids[goal_rank][goal_at], prefix[::-1] + suffix


def _joined_paths(gpu, matches, forward, reverse, goal_ids):
    for rank, left_at, right_at in matches:
        for _, _, prefix in _walks(forward, rank, left_at, gpu.check):
            forward_labels = prefix[::-1]
            for goal_rank, goal_at, suffix in _walks(reverse, rank, right_at, gpu.check):
                yield goal_ids[goal_rank][goal_at], forward_labels + suffix


def _window(gpu, roots, goals, goal_ids, direction, distance, limit, add, progress, store):
    forward = [roots]
    reverse = [goals]
    for length in range(distance, limit + 1):
        gpu.checkpoint()
        prefix_depth = length if direction == "forward" else length // 2
        suffix_depth = length - prefix_depth
        while len(forward) <= prefix_depth:
            layer, _ = gpu.advance(forward[-1], lineage=(store, "w0", len(forward)))
            forward[-1] = Layer([], forward[-1].links)
            forward.append(layer)
        while len(reverse) <= suffix_depth:
            layer, _ = gpu.advance(reverse[-1], reverse=True, lineage=(store, "w1", len(reverse)))
            reverse[-1] = Layer([], reverse[-1].links)
            reverse.append(layer)
        for matches in gpu.match_batches(forward[-1], reverse[-1]):
            gpu.write_paths(_joined_paths(gpu, matches, forward, reverse, goal_ids), add)
        progress["completed_through_length"] = length


def _file_hash(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


class _ParquetPaths:
    def __init__(self, output, max_paths, max_bytes, check, arrow, pq):
        self.output = output
        self.max_paths = max_paths
        self.max_bytes = max_bytes
        self.check = check
        self.arrow = arrow
        self.schema = arrow.schema([
            ("target_index", arrow.int64()),
            ("length", arrow.int64()),
            ("generator_ids", arrow.list_(arrow.int32())),
        ])
        self.staged = output / ".paths.parquet.tmp"
        self.writer = pq.ParquetWriter(
            self.staged, self.schema, compression="zstd", write_statistics=False,
        )
        self.staged_bytes = self.staged.stat().st_size
        self.pending = []
        self.pending_bytes = 0
        self.counts = Counter()
        self.total = 0
        self.row_groups = 0

    def add(self, target, generators):
        self.check()
        if self.max_paths is not None and self.total >= self.max_paths:
            raise _Stopped("PATH_LIMIT")
        # Admission allows for uncompressed row data and per-row-group metadata.
        # The actual completed file is checked again before it becomes visible.
        size = 32 + 8 * len(generators)
        if self.max_bytes is not None:
            estimated = self.staged_bytes + 768 + 384 * self.row_groups
            if estimated + self.pending_bytes + size > self.max_bytes:
                raise _Stopped("OUTPUT_LIMIT")
        self.pending.append({
            "target_index": target,
            "length": len(generators),
            "generator_ids": generators,
        })
        self.pending_bytes += size
        self.counts[len(generators)] += 1
        self.total += 1
        if len(self.pending) >= 256 or self.pending_bytes >= 1 << 20:
            self.flush()

    def flush(self):
        if self.pending:
            self.writer.write_table(self.arrow.Table.from_pylist(self.pending, schema=self.schema))
            self.pending.clear()
            self.pending_bytes = 0
            self.row_groups += 1
            if self.max_bytes is not None:
                self.staged_bytes = self.staged.stat().st_size

    def finish(self):
        self.flush()
        self.writer.close()
        size = self.staged.stat().st_size
        if self.max_bytes is not None and size > self.max_bytes:
            self.staged.unlink()
            return None
        final = self.output / "paths.parquet"
        self.staged.replace(final)
        return {"path_file": final.name, "path_bytes": size}

    def abort(self):
        self.writer.close()
        self.staged.unlink(missing_ok=True)


def _empty_output_dir(output):
    output.mkdir(parents=True, exist_ok=True)
    if next(output.iterdir(), None) is not None:
        raise FileExistsError(f"PATH_OUTPUT_NOT_EMPTY: {output}")


def _reserve_output(output, graph):
    _empty_output_dir(output)
    with (output / "graph.json").open("x", encoding="utf-8") as stream:
        stream.write(graph.to_json() + "\n")


def _run(gpu, graph, targets, direction, goal_mode, extra, add, report, store):
    gpu.checkpoint()
    roots, _ = gpu.seed([graph.start], [0])
    ids, selected = zip(*targets) if targets else ((), ())
    goals, goal_ids = gpu.seed(selected, ids)
    distance, witness = _nearest(gpu, roots, goals, direction, goal_mode == "nearest", store)
    report["d_min"] = distance
    if distance is None:
        report["status"] = "UNREACHABLE"
    elif goal_mode == "nearest":
        report["max_length"] = distance
        gpu.write_paths(_witness(gpu, *witness, goal_ids), add)
        report["completed_through_length"] = distance
    else:
        if distance + extra > I64_MAX:
            raise _Stopped("LENGTH_RANGE")
        report["max_length"] = distance + extra
        _window(gpu, roots, goals, goal_ids, direction, distance, distance + extra, add, report, store)


def _run_native_nearest(native, graph, targets, output, deadline, check, add, report, *, bidirectional):
    executable, environment, device, *options = native
    native_targets = [state for _, state in targets]
    check()
    with TemporaryDirectory(prefix=".pq-", dir=output) as directory:
        query = Path(directory) / "query.json"
        request = {
            "graph": graph.to_dict(),
            "targets": native_targets,
            "device": device,
            "max_seconds": None if deadline is None else max(0.0, deadline - perf_counter()),
        }
        with query.open("x", encoding="utf-8") as stream:
            json.dump(request, stream)
        check()
        result = None
        if isinstance(device, tuple):
            from math import ceil
            from .launch import run_graph

            result = run_graph(
                graph, Path(directory) / "ranks", devices=device, executable=executable,
                backend="generic", autotune=False, shards=1, _native_env=environment,
                max_seconds=None if deadline is None else max(1, ceil(deadline - perf_counter())),
                _path_targets=native_targets, _path_bidirectional=bidirectional,
                memory_preset=report["preset"],
            )
        elif options and options[0]:
            from .generic_session import resident_nearest

            result = resident_nearest(executable, environment, device, query, deadline)
        if result is None:
            child = subprocess.run(
                [executable, "graph-path-nearest", str(query)], env=environment,
                capture_output=True, text=True,
            )
            if child.returncode:
                raise RuntimeError("NATIVE_PATH_SEARCH_FAILED: " + child.stderr[-4000:])
            try:
                result = json.loads(child.stdout)
            except ValueError as error:
                raise RuntimeError("NATIVE_PATH_RESULT_INVALID") from error
    _record_native_nearest(result, graph, targets, check, add, report)


def _validate_native_nearest_receipt(result, target_count, generator_count):
    try:
        status = result["status"]
        distance = result["d_min"]
        reason = result["stop_reason"]
        target = result["target_index"]
        labels = result["generator_ids"]
        checked_depth = result.get("search_confirmed_through_depth")
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeError("NATIVE_PATH_RESULT_INVALID") from error
    valid_distance = distance is None or type(distance) is int and 0 <= distance <= I64_MAX
    if not valid_distance or not isinstance(labels, list):
        raise RuntimeError("NATIVE_PATH_RESULT_INVALID")
    if checked_depth is not None and (type(checked_depth) is not int or not 0 <= checked_depth <= I64_MAX
                                      or distance is not None and checked_depth != distance):
        raise RuntimeError("NATIVE_PATH_RESULT_INVALID")
    if status == "COMPLETE":
        if (reason is not None or distance is None or type(target) is not int or not 0 <= target < target_count
                or len(labels) != distance
                or any(type(label) is not int or not 0 <= label < generator_count for label in labels)):
            raise RuntimeError("NATIVE_PATH_RESULT_INVALID")
    elif (target is not None or labels
          or status == "UNREACHABLE" and (distance is not None or reason is not None)
          or status == "INCOMPLETE" and reason not in ("TIME_LIMIT", "CANCELLED", "CUDA_MEMORY", "LINEAGE_DISK_FULL", "DEPTH_LIMIT")
          or status not in ("UNREACHABLE", "INCOMPLETE")):
        raise RuntimeError("NATIVE_PATH_RESULT_INVALID")
    return status, distance, reason, target, labels, checked_depth


def _record_native_nearest(result, graph, targets, check, add, report):
    status, distance, reason, target, labels, checked_depth = _validate_native_nearest_receipt(
        result, len(targets), graph.generator_count)
    report.update(status=status, stop_reason=reason, d_min=distance, max_length=distance)
    if "search_confirmed_through_depth" in result:
        report["search_confirmed_through_depth"] = checked_depth
    workspace = result.get("native_workspace", result.get("workspace"))
    if workspace is not None:
        report["native_workspace"] = workspace
    if result.get("native_storage") == "SEGMENTED_BATCH_HASH":
        report["native_storage"] = result["native_storage"]
        if report["goal_mode"] == "nearest":
            report["lineage_storage"] = "native_segmented_parents"
        for field in ("native_exchange", "native_state_bytes", "native_search_depths"):
            if field in result:
                report[field] = result[field]
    if status == "COMPLETE":
        # Validate the single native witness against the exact submitted graph
        # before adding its row to the artifact.
        endpoint = graph.start
        for label in labels:
            if add is not None:
                check()
            endpoint = graph.successor(endpoint, label)
        target, state = targets[target]
        if tuple(endpoint) != state:
            raise RuntimeError("NATIVE_PATH_WITNESS_MISMATCH")
        if add is None:
            report["nearest_witness"] = {"target_index": target, "generator_ids": labels}
        else:
            add(target, labels)
            report["completed_through_length"] = distance


def _agree_window_receipts(replies, *, bidirectional=None):
    if not replies:
        raise RuntimeError("NATIVE_WINDOW_CONSENSUS_MISMATCH")
    if bidirectional is None:
        bidirectional = "side_layer_rows" in replies[0]
    fields = ("status", "stop_reason", "graph_completed_through_depth")
    if bidirectional:
        fields += ("first_cut_length", "last_cut_length")
    first = replies[0]
    mode_fields = ("native_storage", "native_exchange", "native_state_bytes")
    fields += tuple(key for key in mode_fields if any(key in reply for reply in replies))

    def nonnegative(value):
        return type(value) is int and value >= 0

    def shape(reply):
        if bidirectional != ("side_layer_rows" in reply):
            raise ValueError
        if any(not nonnegative(reply[key]) for key in ("lineage_peak_bytes", "lineage_csr_bytes")):
            raise ValueError
        layers = reply["side_layer_rows"] if bidirectional else (reply["layer_rows"],)
        if len(layers) != (2 if bidirectional else 1):
            raise ValueError
        if any(not nonnegative(count) for side in layers for count in side):
            raise ValueError
        return tuple(map(len, layers))

    try:
        sizes = shape(first)
        status, reason = first["status"], first["stop_reason"]
        if not (status == "COMPLETE" and reason is None
                or status == "INCOMPLETE" and isinstance(reason, str) and reason):
            raise ValueError
        if any(any(reply[key] != first[key] for key in fields) or shape(reply) != sizes
               for reply in replies[1:]):
            raise ValueError
        confirmed = first["graph_completed_through_depth"]
        if confirmed is not None and not nonnegative(confirmed):
            raise ValueError
        if bidirectional:
            earliest, last = first["first_cut_length"], first["last_cut_length"]
            if not nonnegative(earliest):
                raise ValueError
            if last is not None and (
                not nonnegative(last) or confirmed is None or not earliest <= last <= confirmed
                or sizes[0] <= last // 2 or sizes[1] <= (last + 1) // 2
            ):
                raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("NATIVE_WINDOW_CONSENSUS_MISMATCH") from error
    return {"ranks": replies, **{key: first[key] for key in fields}}


_TERMINAL = Struct("<QQ")


def _native_terminal_rows(path, cache):
    rows, remainder = divmod(cache.size(path), _TERMINAL.size)
    if remainder:
        raise ValueError("NATIVE_WINDOW_TERMINAL_SIZE")
    for row in range(rows):
        yield cache.unpack(path, row, _TERMINAL)


def _read_native_forward_window(root, receipt, distance, limit, target_ids, check, add, report, flush, cache):
    rank_receipts = receipt["ranks"]
    history = [Layer([], [])]
    rank_dirs = [root / f"rank-{rank}" for rank in range(len(rank_receipts))]
    sizes = [item["layer_rows"] for item in rank_receipts]
    roots = [rank_sizes[0] if rank_sizes else 0 for rank_sizes in sizes]
    confirmed = receipt["graph_completed_through_depth"]
    report["lineage_peak_bytes"] = sum(item["lineage_peak_bytes"] for item in rank_receipts)
    report["lineage_committed_csr_bytes"] = sum(item["lineage_csr_bytes"] for item in rank_receipts)
    report["native_process_peak_rss_bytes_by_rank"] = [
        item.get("native_process_peak_rss_bytes") for item in rank_receipts
    ]
    report["status"] = receipt["status"]
    report["stop_reason"] = receipt["stop_reason"]
    if confirmed is None:
        return
    for depth in range(min(confirmed, limit) + 1):
        check()
        present = depth < len(sizes[0])
        if depth and present:
            links = [DiskLinks.from_native_owner(folder / f"{depth}.csr", cache, rank_sizes[depth])
                     for folder, rank_sizes in zip(rank_dirs, sizes, strict=True)]
            history.append(Layer([], links))
        if depth < distance:
            continue
        if present:
            for rank, folder in enumerate(rank_dirs):
                for target_id, row in _native_terminal_rows(folder / f"{depth}.targets", cache):
                    if target_id >= len(target_ids) or row >= sizes[rank][depth]:
                        raise RuntimeError("NATIVE_WINDOW_TERMINAL_BOUNDS")
                    for _, _, labels in _walks(history, rank, row, check, root_counts=roots):
                        add(target_ids[target_id], labels[::-1])
        if flush is not None:
            flush()
        report["completed_through_length"] = depth


def _read_native_bidirectional_window(root, receipt, distance, limit, target_ids, check, add, report, flush, cache):
    rank_receipts = receipt["ranks"]
    rank_dirs = [root / f"rank-{rank}" for rank in range(len(rank_receipts))]
    sizes = [item["side_layer_rows"] for item in rank_receipts]
    roots = [[rank_sizes[side][0] if rank_sizes[side] else 0 for rank_sizes in sizes] for side in range(2)]
    histories = [[Layer([], [])], [Layer([], [])]]
    confirmed = receipt["graph_completed_through_depth"]
    report["status"], report["stop_reason"] = receipt["status"], receipt["stop_reason"]
    report["lineage_peak_bytes"] = sum(item["lineage_peak_bytes"] for item in rank_receipts)
    report["lineage_committed_csr_bytes"] = sum(item["lineage_csr_bytes"] for item in rank_receipts)
    report["native_process_peak_rss_bytes_by_rank"] = [
        item.get("native_process_peak_rss_bytes") for item in rank_receipts
    ]
    if confirmed is None:
        return
    if receipt["first_cut_length"] != distance:
        raise RuntimeError("NATIVE_WINDOW_CUT_IDENTITY")
    for length in range(distance, min(limit, confirmed) + 1):
        check()
        if receipt["last_cut_length"] is not None and length <= receipt["last_cut_length"]:
            depths = length // 2, (length + 1) // 2
            for side, name in enumerate(("forward", "reverse")):
                history = histories[side]
                while len(history) <= depths[side]:
                    depth = len(history)
                    links = [DiskLinks.from_native_owner(folder / name / f"{depth}.csr", cache,
                                                         rank_sizes[side][depth])
                             for folder, rank_sizes in zip(rank_dirs, sizes, strict=True)]
                    history.append(Layer([], links))
            for rank, folder in enumerate(rank_dirs):
                for left, right in _native_terminal_rows(folder / f"{length}.matches", cache):
                    if left >= sizes[rank][0][depths[0]] or right >= sizes[rank][1][depths[1]]:
                        raise RuntimeError("NATIVE_WINDOW_JOIN_BOUNDS")
                    for _, _, prefix in _walks(histories[0], rank, left, check, root_counts=roots[0]):
                        for owner, row, suffix in _walks(histories[1], rank, right, check, root_counts=roots[1]):
                            target, root_row = cache.unpack(
                                rank_dirs[owner] / "reverse" / "0.targets", row, _TERMINAL)
                            if target >= len(target_ids) or root_row != row:
                                raise RuntimeError("NATIVE_WINDOW_ROOT_BOUNDS")
                            add(target_ids[target], prefix[::-1] + suffix)
        if flush is not None:
            flush()
        report["completed_through_length"] = length


def _read_native_window(root, receipt, distance, limit, target_ids, check, add, report, *,
                        flush=None, owner_files=None):
    reader = (_read_native_bidirectional_window if report["direction"] == "bidirectional"
              else _read_native_forward_window)
    if "native_workspace" in receipt:
        report["native_window_workspace"] = receipt["native_workspace"]
    for key in ("native_storage", "native_exchange", "native_state_bytes"):
        if key in receipt:
            report[key.replace("native_", "native_window_", 1)] = receipt[key]
    if receipt.get("native_storage") == "SEGMENTED_LAYER_BATCH_HASH":
        layer_key = "side_layer_rows" if report["direction"] == "bidirectional" else "layer_rows"
        report["native_window_layer_rows_by_rank"] = [item[layer_key] for item in receipt["ranks"]]
    if any("lineage_io" in item for item in receipt["ranks"]):
        report["native_window_lineage_io_by_rank"] = [item.get("lineage_io") for item in receipt["ranks"]]
    available = receipt["graph_completed_through_depth"]
    report["window_graph_completed_through_length"] = receipt.get("native_graph_completed_through_depth", available)
    report["window_lineage_available_through_length"] = available
    cache = _Blocks(report["lineage_cache_bytes"], open_files=8, owner_files=owner_files)
    try:
        reader(root, receipt, distance, limit, target_ids, check, add, report, flush, cache)
    finally:
        report["lineage_cache_peak_bytes"] = cache.peak
        cache.close()
        if owner_files is not None:
            report.update(lineage_read_mode="rank_owned", lineage_remote_reads=owner_files.requests,
                          lineage_remote_read_bytes=owner_files.bytes_read)


def _run_native_window(native, graph, targets, output, deadline, check, add, report, lineage_dir, flush):
    from math import ceil
    from .launch import _window_shards, run_graph

    executable, environment, devices, *_ = native
    bidirectional = report["direction"] == "bidirectional"
    target_ids = tuple(index for index, _ in targets)
    native_targets = [state for _, state in targets]
    directory = output if lineage_dir is None else Path(lineage_dir)
    directory.mkdir(parents=True, exist_ok=True)

    def launch(root, config=None):
        check()
        return run_graph(
            graph, root, devices=devices, executable=executable, backend="generic",
            autotune=False, shards=_window_shards(environment), _native_env=environment, _path_targets=native_targets,
            _path_window=config, _path_bidirectional=bidirectional,
            memory_preset=report["preset"],
            max_seconds=None if deadline is None else max(1, ceil(deadline - perf_counter())),
        )

    with TemporaryDirectory(prefix=".pw-", dir=directory) as temporary:
        root = Path(temporary)
        nearest = launch(root / "nearest")
        _record_native_nearest(nearest, graph, targets, check, None, report)
        distance = report["d_min"]
        if report["status"] != "COMPLETE":
            return
        limit = distance + report["k"]
        if limit > 0xffff_ffff:
            raise ValueError("NATIVE_WINDOW_DEPTH_RANGE")
        report["max_length"] = limit
        budget = report["max_lineage_bytes"]
        if budget is None:
            budget = (1 << 64) - 1
        report["lineage_rank_budget_bytes"] = budget // len(devices)
        config = {"max_length": limit, "disk_budget": budget // len(devices)}
        if bidirectional:
            config["min_length"] = distance
        result = launch(root / "window", config)
        _read_native_window(root / "window", result, distance, limit, target_ids, check, add, report, flush=flush)


def _path_options(
    *, direction="forward", goal_mode="nearest", extra_steps=0, preset="balanced",
    max_paths=None, max_output_bytes=None, max_lineage_bytes=None, lineage_cache_bytes=8 << 20,
):
    extra = _nonnegative(extra_steps, "PATH_EXTRA_STEPS", I64_MAX)
    if max_paths is not None:
        max_paths = _nonnegative(max_paths, "PATH_MAX_PATHS")
    if max_output_bytes is not None:
        max_output_bytes = _nonnegative(max_output_bytes, "PATH_MAX_OUTPUT_BYTES")
    if max_lineage_bytes is not None:
        max_lineage_bytes = _nonnegative(max_lineage_bytes, "PATH_MAX_LINEAGE_BYTES")
    lineage_cache_bytes = _nonnegative(lineage_cache_bytes, "PATH_LINEAGE_CACHE_BYTES")
    if direction not in ("forward", "bidirectional"):
        raise ValueError("PATH_DIRECTION")
    if goal_mode not in ("nearest", "window"):
        raise ValueError("PATH_GOAL_MODE")
    if goal_mode == "nearest" and extra:
        raise ValueError("PATH_EXTRA_STEPS_REQUIRE_WINDOW")
    if preset == "max_capacity":
        preset = "maxmemory"
    if preset not in ("balanced", "maxmemory"):
        raise ValueError("PATH_PRESET")
    return extra, preset, max_paths, max_output_bytes, max_lineage_bytes, lineage_cache_bytes


def solve(
    graph, *, start=None, targets, direction="forward", goal_mode="nearest", extra_steps=0,
    preset="balanced", devices=None, output, max_seconds=None, max_paths=None,
    max_output_bytes=None, distributed=None, lineage_dir=None, max_lineage_bytes=None,
    lineage_cache_bytes=8 << 20, native=None, _native_receipt=None, _started=None, _native_window_data=None,
    _native_window_files=None, _output_reserved=False,
):
    """Search exact states with NVIDIA CUDA and write a streamed path artifact.

    Targets are an iterable of states. Duplicate states share the ID of their
    first occurrence in the deduplicated list in the report. For window
    searches, each distinct generator sequence ending at a target is included
    once if its length is at most the minimum distance plus extra_steps.
    With explicit distributed rank options, every rank participates and only
    rank zero returns the result and writes the output.
    Lineage is kept in temporary indexed files under lineage_dir or output.
    The lineage byte limit includes temporary sort runs and final indices.
    """
    started = perf_counter() if _started is None else _started
    deadline = _deadline(max_seconds, started)
    extra, preset, max_paths, max_output_bytes, max_lineage_bytes, lineage_cache_bytes = _path_options(
        direction=direction, goal_mode=goal_mode, extra_steps=extra_steps, preset=preset,
        max_paths=max_paths, max_output_bytes=max_output_bytes, max_lineage_bytes=max_lineage_bytes,
        lineage_cache_bytes=lineage_cache_bytes,
    )
    native_path = native is not None or _native_receipt is not None or _native_window_data is not None
    native_window = (native is not None or _native_window_data is not None) and goal_mode == "window"
    if native_path and (distributed is not None or goal_mode != "nearest" and not native_window):
        raise ValueError("NATIVE_PATH_REQUIRES_NEAREST")
    graph, states = _graph_and_targets(graph, start, targets)
    action = graph.action
    reverse = inverse_action(graph) if direction == "bidirectional" else None
    active_targets = _active_path_targets(graph, states, direction)
    try:
        import pyarrow as arrow
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("PATH_SEARCH_REQUIRES_PYARROW") from error
    torch = None
    if not native_path:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PATH_SEARCH_REQUIRES_CUDA_TORCH") from error
        devices = _cuda_devices(torch, devices)
    elif native is not None:
        ordinals = native[2] if isinstance(native[2], tuple) else (native[2],)
        devices = tuple(f"cuda:{ordinal}" for ordinal in ordinals)

    def check():
        if deadline is not None and perf_counter() >= deadline:
            raise _Stopped("TIME_LIMIT")

    output = Path(output).resolve()
    graph_name = graph.to_dict()["name"]
    lrx = match_lrx(graph)
    report = {
        "schema": 1,
        "mode": "paths",
        "goal_mode": goal_mode,
        "direction": direction,
        "preset": preset,
        "backend": ("native_cuda_nccl" if _native_receipt is not None or _native_window_data is not None or native is not None and isinstance(native[2], tuple) else
                    "native_cuda" if native is not None else
                    "torch_cuda" if distributed is None else "torch_cuda_nccl"),
        "devices": [str(device) for device in devices],
        "graph_family": "LRX" if lrx else (graph_name or action["kind"]),
        "graph_sha256": graph.digest(),
        "generators_sha256": sha256(json.dumps(action, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "start_state": graph.start,
        "targets": states,
        "k": extra,
        "d_min": None,
        "max_length": None,
        "completed_through_length": None,
        "status": "COMPLETE",
        "stop_reason": None,
        "max_seconds": max_seconds,
        "max_paths": max_paths,
        "max_output_bytes": max_output_bytes,
        "lineage_storage": "native_predecessor_scan" if native_path and not native_window else "disk_csr",
        "max_lineage_bytes": max_lineage_bytes,
        "lineage_cache_bytes": lineage_cache_bytes,
        "lineage_cache_peak_bytes": 0,
        "lineage_committed_csr_bytes": 0,
        "lineage_peak_bytes": 0,
    }
    if native_path and preset == "maxmemory":
        report["capacity_slowdown_status"] = "UNVERIFIED"
    if action["kind"] == "permutation":
        report["n"] = graph.state_elements
    if lrx:
        report["r"] = lrx["r"]
    def prepare_writer():
        if not _output_reserved:
            _reserve_output(output, graph)
        return _ParquetPaths(output, max_paths, max_output_bytes, check, arrow, pq)

    def prepare_lineage():
        return DiskLineage(lineage_dir if lineage_dir is not None else output, max_lineage_bytes, lineage_cache_bytes)

    context = nullcontext() if distributed is None else connect(torch, devices, distributed, check)
    inference = torch.no_grad() if torch is not None else nullcontext()
    resource_errors = (_Stopped, torch.OutOfMemoryError) if torch is not None else (_Stopped,)
    with context as network, inference:
        if network is None:
            writer = prepare_writer()
        else:
            signature = {key: value for key, value in report.items() if key != "devices"}
            network.verify_identity(sha256(json.dumps(signature, sort_keys=True).encode()).digest())
            writer = network.step(prepare_writer if network.is_root else None, deadline=False)
            report["world_size"] = network.world
            report["process_group"] = "persistent" if distributed.get("persistent", False) else "per_query"
        batch_records = 65536 if preset == "balanced" else 8192
        add = writer.add if writer is not None else None
        store = None
        try:
            try:
                if _native_receipt is not None:
                    _record_native_nearest(_native_receipt, graph, active_targets, check, add, report)
                elif _native_window_data is not None:
                    nearest, receipt, root, window_stop_reason = _native_window_data
                    _record_native_nearest(nearest, graph, active_targets, check, None, report)
                    distance = report["d_min"]
                    if distance is not None:
                        report["max_length"] = distance + extra
                    if receipt is not None:
                        target_ids = tuple(index for index, _ in active_targets)
                        _read_native_window(root, receipt, distance, distance + extra, target_ids,
                                            check, add, report, flush=writer.flush,
                                            owner_files=_native_window_files)
                    elif window_stop_reason is not None:
                        report.update(status="INCOMPLETE", stop_reason=window_stop_reason)
                elif not active_targets:
                    check()
                    report["status"] = "UNREACHABLE"
                elif native_window:
                    _run_native_window(native, graph, active_targets, output, deadline, check, add,
                                       report, lineage_dir, writer.flush)
                elif native is not None:
                    _run_native_nearest(native, graph, active_targets, output, deadline, check, add, report,
                                        bidirectional=direction == "bidirectional")
                else:
                    if network is None:
                        check()
                        store = prepare_lineage()
                        gpu = GpuLayers(torch, devices, graph, reverse, batch_records, check)
                    else:
                        store = network.step(prepare_lineage if network.is_root else None)
                        gpu = network.step(lambda: DistributedLayers(network, graph, reverse, batch_records))
                    _run(gpu, graph, active_targets, direction, goal_mode, extra, add, report, store)
            except resource_errors as error:
                reason = str(error) if isinstance(error, _Stopped) else "CUDA_MEMORY"
                report.update(status="INCOMPLETE", stop_reason=reason)
            finally:
                if store is not None:
                    report["lineage_peak_bytes"] = store.peak_bytes
                    report["lineage_committed_csr_bytes"] = store.csr_bytes
                    report["lineage_cache_peak_bytes"] = store.cache.peak
                    store.close()
            if network is None:
                artifact = writer.finish()
            else:
                artifact = network.step(writer.finish if writer is not None else None, deadline=False)
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
    if writer is None:
        return None
    if artifact is not None:
        artifact["path_sha256"] = _file_hash(output / artifact["path_file"])
        report.update(artifact)
    else:
        report.update(status="INCOMPLETE", stop_reason="OUTPUT_LIMIT", completed_through_length=None)
        writer.counts.clear()
        writer.total = 0
    report["counts"] = [{"length": length, "count": count} for length, count in sorted(writer.counts.items())]
    report["path_count"] = writer.total
    report["elapsed_seconds"] = perf_counter() - started
    # JSON round-trip keeps the immediate result identical to the saved report.
    report = json.loads(json.dumps(report))
    staged = output / ".report.json.tmp"
    staged.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    staged.replace(output / "report.json")
    return PathSearchResult(graph, output, report)
