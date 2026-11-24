"""Read committed native window files at their owner ranks."""

from hashlib import sha256
from itertools import count
from os import fstat
from struct import Struct
from time import perf_counter

from .path_gpu import SearchStopped
from .path_lineage import _BLOCK


_REPLY = Struct("<Q32s")
_READ_SECONDS = 15
_DRAIN_SECONDS = 5
WINDOW_DRAIN_SECONDS = _READ_SECONDS + _DRAIN_SECONDS


def _csr_files(side, depth):
    return f"{side}{depth}.csr/offsets", f"{side}{depth}.csr/edges"


def _committed_files(receipt):
    if "side_layer_rows" in receipt:
        last = receipt["last_cut_length"]
        if last is not None:
            yield "reverse/0.targets"
            for depth in range(1, last // 2 + 1):
                yield from _csr_files("forward/", depth)
            for depth in range(1, (last + 1) // 2 + 1):
                yield from _csr_files("reverse/", depth)
            for length in range(receipt["first_cut_length"], last + 1):
                yield f"{length}.matches"
        return
    for depth in range(len(receipt["layer_rows"])):
        yield f"{depth}.targets"
        if depth:
            yield from _csr_files("", depth)


def serve_window_files(store, prefix, root, rank, receipt):
    """Keep one rank's files readable until the coordinator closes this query."""
    folder = root / f"rank-{rank}"
    allowed = set(_committed_files(receipt["ranks"][rank]))
    peer_key = f"{prefix}{rank}/"
    for index in count():
        request = store.await_value(f"{peer_key}request/{index}", stop_key=prefix + "done")
        if request is None:
            return
        name, offset, size = request
        if (name not in allowed or type(offset) is not int or offset < 0
                or type(size) is not int or not 0 <= size <= _BLOCK):
            raise RuntimeError("NATIVE_WINDOW_READ_REQUEST")
        with (folder / name).open("rb", buffering=0) as file:
            file_size = fstat(file.fileno()).st_size
            if offset > file_size:
                raise RuntimeError("NATIVE_WINDOW_READ_OFFSET")
            file.seek(offset)
            expected = min(size, file_size - offset)
            data = file.read(expected)
        if len(data) != expected:
            raise OSError("NATIVE_WINDOW_SOURCE_TRUNCATED")
        store.put_block(peer_key + "reply", index, _REPLY.pack(file_size, sha256(data).digest()) + data,
                        deadline=perf_counter() + _READ_SECONDS)


class WindowOwnerFiles:
    """A single outstanding block and a bounded lifetime for each remote read."""

    def __init__(self, store, prefix, root, world, deadline):
        self.store, self.prefix, self.root, self.deadline = store, prefix, root, deadline
        self.next_read = [0] * world
        self.requests = 0
        self.bytes_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.store.put(self.prefix + "done", True, deadline=perf_counter() + _DRAIN_SECONDS)

    def is_remote(self, path):
        return path.relative_to(self.root).parts[0] != "rank-0"

    def size(self, path):
        return self._fetch(path, 0, 0)[0]

    def read(self, path, offset, size):
        return self._fetch(path, offset, size)[1]

    def _fetch(self, path, offset, size):
        folder, *parts = path.relative_to(self.root).parts
        rank = int(folder.removeprefix("rank-"))
        if not 0 < rank < len(self.next_read):
            raise ValueError("NATIVE_WINDOW_READ_RANK")
        index = self.next_read[rank]
        peer_key = f"{self.prefix}{rank}/"
        request = f"{peer_key}request/{index}"
        reply = peer_key + "reply"
        deadline = perf_counter() + _READ_SECONDS
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        try:
            self.store.put(request, ("/".join(parts), offset, size), deadline=deadline)
            block = self.store.get_block(reply, index, deadline=deadline)
            if len(block) < _REPLY.size:
                raise RuntimeError("NATIVE_WINDOW_READ_REPLY")
            file_size, digest = _REPLY.unpack_from(block)
            data = block[_REPLY.size:]
            if offset > file_size or len(data) != min(size, file_size - offset) or sha256(data).digest() != digest:
                raise RuntimeError("NATIVE_WINDOW_READ_REPLY")
            self.store.acknowledge_block(reply, index, command_key=request, deadline=deadline)
        except TimeoutError as error:
            if self.deadline is not None and deadline == self.deadline:
                raise SearchStopped("TIME_LIMIT") from error
            raise RuntimeError(f"NATIVE_WINDOW_OWNER_UNAVAILABLE: rank {rank}") from error
        self.next_read[rank] += 1
        self.requests += 1
        self.bytes_read += len(data)
        return file_size, data
