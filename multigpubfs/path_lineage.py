"""Bounded external sort and indexed scratch storage for exact path DAGs."""

from collections import OrderedDict
from contextlib import ExitStack
import errno
from heapq import merge
from pathlib import Path
from struct import Struct
from tempfile import TemporaryDirectory

from .path_gpu import SearchStopped


_RAW = Struct("<QQQQ")
_EDGE = Struct("<QII")
_OFFSET = Struct("<Q")
_OFFSET_PAIR = Struct("<QQ")
_SEAL = Struct("<8sQ")
_BLOCK = 64 << 10
_FAN_IN = 16
_RUN_ROWS = 8192


def _disk_call(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except OSError as error:
        if error.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)) or getattr(error, "winerror", None) == 112:
            raise SearchStopped("LINEAGE_DISK_FULL") from error
        raise


class _Blocks:
    def __init__(self, maximum, *, open_files=0, owner_files=None):
        self.maximum = maximum
        self.block_size = min(_BLOCK, 1 << (max(1, maximum).bit_length() - 1))
        self.max_blocks = maximum // self.block_size
        self.used = 0
        self.peak = 0
        self.cached = OrderedDict()
        self.open_files = open_files
        self.streams = OrderedDict()
        self.owner_files = owner_files

    def close(self):
        self.cached.clear()
        self.used = 0
        while self.streams:
            self.streams.popitem()[1].close()

    def _read(self, path, offset, size):
        if self.owner_files is not None and self.owner_files.is_remote(path):
            return self.owner_files.read(path, offset, size)
        if not self.open_files:
            with path.open("rb", buffering=0) as stream:
                stream.seek(offset)
                return stream.read(size)
        stream = self.streams.pop(path, None)
        if stream is None:
            if len(self.streams) == self.open_files:
                self.streams.popitem(last=False)[1].close()
            stream = path.open("rb", buffering=0)
        self.streams[path] = stream
        stream.seek(offset)
        return stream.read(size)

    def size(self, path):
        if self.owner_files is not None and self.owner_files.is_remote(path):
            return self.owner_files.size(path)
        return path.stat().st_size

    def unpack(self, path, index, codec):
        offset = index * codec.size
        if self.maximum < codec.size or self.block_size % codec.size:
            return codec.unpack(self._read(path, offset, codec.size))
        return self._unpack_cached(path, offset, codec)

    def offset_pair(self, path, index):
        offset = index * _OFFSET.size
        if self.block_size >= _OFFSET_PAIR.size:
            if offset % self.block_size <= self.block_size - _OFFSET_PAIR.size:
                return self._unpack_cached(path, offset, _OFFSET_PAIR)
        return self.unpack(path, index, _OFFSET)[0], self.unpack(path, index + 1, _OFFSET)[0]

    def _unpack_cached(self, path, offset, codec):
        first = offset - offset % self.block_size
        key = path, first
        block = self.cached.get(key)
        if block is None:
            # Even short EOF blocks retain a key, a path, and a cache entry.
            # Limit their count as well as the cached payload.
            while len(self.cached) >= self.max_blocks:
                self.used -= len(self.cached.popitem(last=False)[1])
            block = self._read(path, first, self.block_size)
            self.used += len(block)
            self.peak = max(self.peak, self.used)
            self.cached[key] = block
        else:
            self.cached.move_to_end(key)
        return codec.unpack_from(block, offset - first)


class DiskLinks:
    def __init__(self, offsets, edges, cache, rows, edge_count, *, require_predecessors=False):
        self.offsets, self.edges, self.cache, self.rows = offsets, edges, cache, rows
        self.edge_count = edge_count
        self.require_predecessors = require_predecessors

    @classmethod
    def from_native_owner(cls, committed, cache, rows):
        directory = Path(committed)
        offsets, edges = directory / "offsets", directory / "edges"
        if rows < 0 or cache.size(offsets) != (rows + 1) * _OFFSET.size:
            raise ValueError("PATH_LINEAGE_NATIVE_SIZE")
        first, = cache.unpack(offsets, 0, _OFFSET)
        terminal, = cache.unpack(offsets, rows, _OFFSET)
        if first or terminal < rows or cache.size(edges) != terminal * _EDGE.size:
            raise ValueError("PATH_LINEAGE_NATIVE_SIZE")
        return cls(offsets, edges, cache, rows, terminal, require_predecessors=True)

    def bounds(self, child):
        if not 0 <= child < self.rows:
            raise ValueError("PATH_LINEAGE_CHILD")
        start, end = self.cache.offset_pair(self.offsets, child)
        if not start <= end <= self.edge_count or (self.require_predecessors and start == end):
            raise ValueError("PATH_LINEAGE_EDGE_BOUNDS")
        return start, end

    def parent(self, index):
        row, rank, label = self.cache.unpack(self.edges, index, _EDGE)
        return rank, row, label


class DiskLineage:
    def __init__(self, parent, limit, cache_bytes):
        parent = Path(parent)
        _disk_call(parent.mkdir, parents=True, exist_ok=True)
        self.directory = _disk_call(TemporaryDirectory, prefix=".pl-", dir=parent)
        self.root = Path(self.directory.name)
        self.limit = limit
        self.cache = _Blocks(cache_bytes)
        self.bytes = self.peak_bytes = self.csr_bytes = 0

    def layer(self, side, depth, owners):
        return _DiskLayer(self, side, depth, owners)

    def close(self):
        self.cache.cached.clear()
        self.directory.cleanup()

    def write(self, stream, data):
        if self.limit is not None and self.bytes + len(data) > self.limit:
            raise SearchStopped("LINEAGE_LIMIT")
        remaining = memoryview(data)
        while remaining:
            written = _disk_call(stream.write, remaining)
            if not written:
                raise OSError("PATH_LINEAGE_SHORT_WRITE")
            self.bytes += written
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            remaining = remaining[written:]

    def discard(self, path):
        size = path.stat().st_size
        path.unlink()
        self.bytes -= size


def _records(stream):
    while data := stream.read(_BLOCK):
        yield from _RAW.iter_unpack(data)


class _DiskLayer:
    def __init__(self, store, side, depth, owners):
        self.store = store
        self.directory = store.root / f"{side}-{depth}"
        _disk_call(self.directory.mkdir)
        self.runs = [0] * owners
        self.links = None

    def _run(self, owner, generation, index):
        return self.directory / f"{owner}-{generation}-{index}.run"

    def add(self, owner, rows):
        for begin in range(0, len(rows), _RUN_ROWS):
            part = rows[begin:begin + _RUN_ROWS]
            values = part[part[:, 0].argsort(stable=True)].tolist()
            path = self._run(owner, 0, self.runs[owner])
            block_rows = _BLOCK // _RAW.size
            with _disk_call(path.open, "xb", buffering=0) as stream:
                for offset in range(0, len(values), block_rows):
                    block = values[offset:offset + block_rows]
                    self.store.write(stream, b"".join(_RAW.pack(*record) for record in block))
            self.runs[owner] += 1

    def _merge(self, paths, output):
        with ExitStack() as stack:
            inputs = [stack.enter_context(path.open("rb", buffering=_BLOCK)) for path in paths]
            writer = stack.enter_context(_disk_call(output.open, "xb", buffering=0))
            buffer = bytearray()
            for row in merge(*map(_records, inputs), key=lambda row: row[0]):
                buffer += _RAW.pack(*row)
                if len(buffer) >= _BLOCK:
                    self.store.write(writer, buffer)
                    buffer.clear()
                    yield
            if buffer:
                self.store.write(writer, buffer)
        for path in paths:
            self.store.discard(path)
        yield

    def _to_csr(self, source, index_path, edge_path, rows):
        offsets = bytearray(_OFFSET.pack(0))
        edges = bytearray()
        next_child = count = 0
        with ExitStack() as stack:
            records = () if source is None else _records(stack.enter_context(source.open("rb", buffering=_BLOCK)))
            index = stack.enter_context(_disk_call(index_path.open, "xb", buffering=0))
            payload = stack.enter_context(_disk_call(edge_path.open, "xb", buffering=0))
            for child, rank, parent, label in records:
                if child >= rows or child < next_child - 1 or child > next_child:
                    raise ValueError("PATH_LINEAGE_CHILD_ORDER")
                if child == next_child:
                    if child:
                        offsets += _OFFSET.pack(count)
                    next_child += 1
                edges += _EDGE.pack(parent, rank, label)
                count += 1
                if len(offsets) >= _BLOCK or len(edges) >= _BLOCK:
                    self.store.write(index, offsets)
                    self.store.write(payload, edges)
                    offsets.clear()
                    edges.clear()
                    yield
            if next_child != rows:
                raise ValueError("PATH_LINEAGE_MISSING_CHILD")
            if rows:
                offsets += _OFFSET.pack(count)
            self.store.write(index, offsets)
            self.store.write(payload, edges)
        if source is not None:
            self.store.discard(source)
        yield

    def seal(self, sizes):
        links = []
        csr_bytes = 0
        for owner, (count, rows) in enumerate(zip(self.runs, sizes, strict=True)):
            if not rows:
                if count:
                    raise ValueError("PATH_LINEAGE_UNEXPECTED_CHILD")
                links.append(None)
                continue
            generation = 0
            while count > 1:
                next_count = 0
                for first in range(0, count, _FAN_IN):
                    inputs = [self._run(owner, generation, i) for i in range(first, min(count, first + _FAN_IN))]
                    output = self._run(owner, generation + 1, next_count)
                    if len(inputs) == 1:
                        _disk_call(inputs[0].replace, output)
                    else:
                        yield from self._merge(inputs, output)
                    next_count += 1
                generation += 1
                count = next_count
            source = self._run(owner, generation, 0) if count else None
            index_path = self.directory / f"{owner}.offsets"
            edge_path = self.directory / f"{owner}.edges"
            yield from self._to_csr(source, index_path, edge_path, rows)
            edge_bytes = edge_path.stat().st_size
            csr_bytes += index_path.stat().st_size + edge_bytes
            links.append(DiskLinks(index_path, edge_path, self.store.cache, rows, edge_bytes // _EDGE.size))
        # Expose the layer only after all owner files were closed successfully.
        staged = self.directory / ".seal.tmp"
        with _disk_call(staged.open, "xb", buffering=0) as stream:
            self.store.write(stream, _SEAL.pack(b"MGBPLN01", len(sizes)))
            self.store.write(stream, b"".join(_OFFSET.pack(count) for count in sizes))
        _disk_call(staged.replace, self.directory / "seal")
        self.store.csr_bytes += csr_bytes
        self.links = links
