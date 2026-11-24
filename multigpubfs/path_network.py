"""NCCL routing for exact path searches with one CUDA owner per process."""

from contextlib import closing, contextmanager, nullcontext
from datetime import timedelta
import math
from threading import Lock

from .path_gpu import (
    DeviceAction, GpuLayers, Layer, SearchStopped, match_indices, owners_for_rows,
)


# A control tensor makes local resource and I/O stops collective before the
# next payload exchange. Transport failures themselves abort the process group.
_REASONS = ("", "TIME_LIMIT", "PATH_LIMIT", "OUTPUT_LIMIT", "LENGTH_RANGE", "CUDA_MEMORY",
            "LINEAGE_LIMIT", "LINEAGE_DISK_FULL")
_UNEXPECTED = len(_REASONS)
_group_lock = Lock()
_owned_group = None


def _close_owned_group():
    global _owned_group
    group, _owned_group = _owned_group, None
    if group is not None and group[0].is_initialized():
        group[0].destroy_process_group()


def close_path_group():
    """Release this process's persistent path communicator before GPU handoff."""
    with _group_lock:
        _close_owned_group()


@contextmanager
def connect(torch, devices, options, check, *, backend="nccl"):
    global _owned_group
    dist = torch.distributed
    rank, world = options["rank"], options["world_size"]
    address = options["init_method"]
    timeout = options.get("timeout_seconds", 120)
    persistent = options.get("persistent", False)
    if (type(rank) is not int or type(world) is not int or world < 2 or not 0 <= rank < world
            or len(devices) != 1 or not isinstance(address, str) or not address.startswith("tcp://")
            or not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(timeout) or timeout <= 0 or type(persistent) is not bool):
        raise ValueError("PATH_DISTRIBUTED_OPTIONS")
    if not dist.is_available() or (backend == "nccl" and not dist.is_nccl_available()):
        raise RuntimeError("PATH_DISTRIBUTED_REQUIRES_TORCH_NCCL")
    device = devices[0]
    identity = dist, backend, address, world, rank, device, timeout
    context = torch.cuda.device(device) if backend == "nccl" else nullcontext()
    if not _group_lock.acquire(blocking=False):
        raise RuntimeError("PATH_DISTRIBUTED_GROUP_IN_USE")
    try:
        with context:
            if _owned_group is not None:
                if not persistent or _owned_group != identity or not dist.is_initialized():
                    raise RuntimeError("PATH_DISTRIBUTED_PERSISTENT_GROUP_MISMATCH")
            else:
                if dist.is_initialized():
                    raise RuntimeError("PATH_DISTRIBUTED_DEFAULT_GROUP_ALREADY_ACTIVE")
                dist.init_process_group(
                    backend, init_method=address, rank=rank, world_size=world, timeout=timedelta(seconds=timeout),
                )
                if persistent:
                    _owned_group = identity
            try:
                yield PathNetwork(torch, device, rank, world, check)
            except BaseException:
                if persistent:
                    _close_owned_group()
                raise
            finally:
                if not persistent:
                    dist.destroy_process_group()
    finally:
        _group_lock.release()


class PathNetwork:
    def __init__(self, torch, device, rank, world, check):
        self.torch = torch
        self.dist = torch.distributed
        self.device = device
        self.rank = rank
        self.world = world
        self.is_root = rank == 0
        self.check = check
        self.control = torch.empty((2,), dtype=torch.int64, device=device)

    def step(self, work=None, *, deadline=True):
        """Check all ranks' local work before any rank advances."""
        error = value = None
        try:
            if deadline and self.is_root:
                self.check()
            if work is not None:
                value = work()
            code = 0
        except SearchStopped as fault:
            code = _REASONS.index(str(fault)) if str(fault) in _REASONS else _UNEXPECTED
            error = fault
        except self.torch.OutOfMemoryError as fault:
            code = _REASONS.index("CUDA_MEMORY")
            error = fault
        except Exception as fault:
            code = _UNEXPECTED
            error = fault
        self.control[0] = code
        self.control[1] = self.rank + 1 if code == _UNEXPECTED else 0
        self.dist.all_reduce(self.control, self.dist.ReduceOp.MAX)
        code, failed = self.control.tolist()
        if code == _UNEXPECTED:
            raise RuntimeError(f"PATH_DISTRIBUTED_RANK_FAILURE: rank {failed - 1}") from error
        if code:
            raise SearchStopped(_REASONS[code]) from error
        return value

    def reduce(self, value, operation):
        self.control[0] = value
        self.dist.all_reduce(self.control[:1], operation)
        return self.control[0].item()

    def verify_identity(self, identity):
        value = self.step(lambda: self.torch.tensor(
            [int.from_bytes(identity[i:i + 8], "little", signed=True) for i in range(0, len(identity), 8)],
            dtype=self.torch.int64, device=self.device,
        ), deadline=False)
        expected = self.step(value.clone, deadline=False)
        self.dist.broadcast(expected, 0)
        valid = self.reduce(int(self.torch.equal(value, expected)), self.dist.ReduceOp.MIN)
        if not valid:
            raise ValueError("PATH_DISTRIBUTED_QUERY_MISMATCH")

    def exchange(self, records, counts):
        """Exchange variable numbers of fixed-width rows; counts are per owner."""
        receive_counts = self.step(lambda: self.torch.empty_like(counts))
        self.dist.all_to_all_single(receive_counts, counts)

        def allocate():
            splits = receive_counts.tolist()
            receive = records.new_empty((sum(splits), records.shape[1]))
            return receive, splits, counts.tolist()

        receive, output_splits, input_splits = self.step(allocate)
        self.dist.all_to_all_single(receive, records, output_splits, input_splits)
        return receive, output_splits

    def metadata_batches(self, rows):
        """Deliver one bounded batch of owner metadata to rank zero at a time."""
        batch = max(1, 8192 // self.world)
        rounds = self.reduce((len(rows) + batch - 1) // batch, self.dist.ReduceOp.MAX)
        for begin in range(0, rounds * batch, batch):
            def prepare():
                transfer = rows[begin:begin + batch].to(self.device)
                counts = transfer.new_zeros((self.world,))
                counts[0] = len(transfer)
                return transfer, counts

            records, counts = self.step(prepare)
            candidates, splits = self.exchange(records, counts)
            yield self.step(lambda: candidates.cpu().split(splits) if self.is_root else ())

    def rank_sizes(self, local):
        send = self.step(lambda: self.control.new_full((self.world,), local))
        take = self.step(lambda: self.torch.empty_like(send))
        self.dist.all_to_all_single(take, send)
        return take.tolist()

    def root_tasks(self, tasks):
        done = object()
        while True:
            def execute():
                if self.is_root:
                    for _ in range(32):
                        if next(tasks, done) is done:
                            return 1
                return 0

            finished = self.step(execute)
            if self.reduce(finished, self.dist.ReduceOp.MAX):
                return

    def write_paths(self, rows, add):
        rows = iter(rows)
        while True:
            def emit():
                if self.is_root:
                    for _ in range(256):
                        row = next(rows, None)
                        if row is None:
                            return 1
                        add(*row)
                return 0

            finished = self.step(emit)
            if self.reduce(finished, self.dist.ReduceOp.MAX):
                return


class DistributedLayers(GpuLayers):
    def __init__(self, network, graph, reverse, batch_records):
        self.network = network
        self.torch = network.torch
        self.width = graph.state_elements
        self.action = DeviceAction(self.torch, graph.action, network.device)
        self.reverse_action = DeviceAction(self.torch, reverse, network.device) if reverse is not None else None
        self.batch_parents = max(1, batch_records // (network.world * self.action.degree))
        self.check = network.check

    def checkpoint(self):
        self.network.step()

    def write_paths(self, rows, add):
        self.network.write_paths(rows, add)

    def seed(self, states, labels):
        net = self.network
        torch = self.torch

        def prepare():
            roots = torch.tensor(states, dtype=torch.int64, device=net.device).reshape(-1, self.width)
            ids = torch.tensor(labels, dtype=torch.int64, device=net.device)
            owners = owners_for_rows(torch, roots, net.world)
            root_labels = [ids[owners == rank].cpu().tolist() for rank in range(net.world)] if net.is_root else []
            return Layer([roots[owners == net.rank]], [None] * net.world, len(states)), root_labels

        return net.step(prepare)

    def advance(self, layer, visited=None, *, reverse=False, lineage=None):
        net = self.network
        torch = self.torch
        rows = layer.states[0]
        table = visited[0] if visited is not None else rows.new_empty((0, self.width))
        committed = len(table)
        action = self.reverse_action if reverse else self.action
        rounds = net.reduce((len(rows) + self.batch_parents - 1) // self.batch_parents, net.dist.ReduceOp.MAX)
        disk = None
        if lineage is not None:
            store, side, depth = lineage
            disk = net.step(lambda: store.layer(side, depth, net.world) if net.is_root else None)
        for begin in range(0, self.batch_parents * rounds, self.batch_parents):
            def expand():
                parents = rows[begin:begin + self.batch_parents]
                candidates = action.expand(parents) if len(parents) else rows.new_empty((0, self.width))
                owners = owners_for_rows(torch, candidates, net.world)
                order = owners.argsort(stable=True)
                counts = torch.bincount(owners, minlength=net.world)
                origins = torch.stack((
                    torch.full_like(order, net.rank),
                    order.div(action.degree, rounding_mode="floor") + begin,
                    order.remainder(action.degree),
                ), dim=1)
                return torch.cat((candidates[order], origins), dim=1), counts

            records, counts = net.step(expand)
            arrived, _ = net.exchange(records, counts)

            def accept():
                if not len(arrived):
                    return table, arrived.new_empty((0, 4)) if lineage is not None else None
                origin = arrived[:, self.width:]
                merged, ids, fresh = self.merge(table, arrived[:, :self.width])
                if lineage is None:
                    return merged, None
                if visited is not None:
                    origin, ids = origin[fresh], ids[fresh]
                return merged, torch.cat(((ids - committed)[:, None], origin), dim=1)

            table, origins = net.step(accept)
            if lineage is not None:
                for owner_blocks in net.metadata_batches(origins):
                    def store_blocks():
                        if net.is_root:
                            for owner, block in enumerate(owner_blocks):
                                disk.add(owner, block)

                    net.step(store_blocks)
        shard = net.step(lambda: table[committed:].clone() if committed else table)
        records = []
        if lineage is None:
            global_size = net.reduce(len(shard), net.dist.ReduceOp.SUM)
        else:
            sizes = net.rank_sizes(len(shard))
            global_size = sum(sizes)
            if net.is_root:
                with closing(disk.seal(sizes)) as tasks:
                    net.root_tasks(tasks)
                records = disk.links
            else:
                net.root_tasks(None)
        return Layer([shard], records, global_size), [table]

    def match_batches(self, left, right):
        net = self.network
        local = net.step(lambda: match_indices(self.torch, left.states[0], right.states[0]))
        for parts in net.metadata_batches(local):
            yield [(rank, left_at, right_at) for rank, part in enumerate(parts)
                   for left_at, right_at in part.tolist()]

    def match_nearest(self, left, right):
        net = self.network
        matches = net.step(lambda: match_indices(self.torch, left.states[0], right.states[0]))
        winner = net.reduce(net.rank if len(matches) else net.world, net.dist.ReduceOp.MIN)
        if winner == net.world:
            return None

        def prepare():
            if net.rank == winner:
                return matches[0].contiguous()
            return matches.new_empty((2,))

        selected = net.step(prepare)
        net.dist.broadcast(selected, winner)
        return winner, *selected.tolist()
