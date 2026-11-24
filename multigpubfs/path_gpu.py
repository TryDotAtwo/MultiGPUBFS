"""Exact CUDA state ownership, expansion and layer joins for path search."""

from contextlib import closing
from dataclasses import dataclass


@dataclass
class Layer:
    states: list
    links: list
    global_size: int | None = None

    @property
    def size(self):
        return self.global_size if self.global_size is not None else sum(len(shard) for shard in self.states)


class SearchStopped(Exception):
    pass


def owners_for_rows(torch, rows, ranks):
    mixed = torch.zeros(len(rows), dtype=torch.int64, device=rows.device)
    if ranks == 1:
        return mixed
    for i in range(rows.shape[1]):
        mixed ^= rows[:, i] + (mixed << 6) + (mixed >> 2) - 7046029254386353131
    return mixed.remainder(ranks)


def match_indices(torch, source, target):
    if not len(source) or not len(target):
        return source.new_empty((0, 2))
    unique, inverse = torch.unique(torch.cat((source, target)), dim=0, return_inverse=True)
    right_row = torch.full((len(unique),), -1, dtype=torch.int64, device=source.device)
    right_row[inverse[len(source):]] = torch.arange(len(target), device=source.device)
    indices = right_row[inverse[:len(source)]]
    selected = (indices >= 0).nonzero().flatten()
    return torch.stack((selected, indices[selected]), dim=1)


def _inverse_prime_power(matrix, n, ring, prime):
    rows = [
        [v % ring for v in matrix[i * n:(i + 1) * n]] + [int(i == j) for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col] % prime), None)
        if pivot is None:
            raise ValueError("PATH_MATRIX_REVERSE_UNSUPPORTED")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = pow(rows[col][col], -1, ring)
        rows[col] = [v * scale % ring for v in rows[col]]
        for r, row in enumerate(rows):
            if r != col:
                scale = row[col]
                rows[r] = [(v - scale * pivot_v) % ring for v, pivot_v in zip(row, rows[col])]
    return [v for row in rows for v in row[n:]]


def _reverse_modular(matrix, n, modulus):
    if not (modulus & (modulus - 1)):
        ring = modulus or (1 << 64)
        inverse = _inverse_prime_power(matrix, n, ring, 2)
        return [v - ring if not modulus and v >= (1 << 63) else v for v in inverse]

    remaining, combined, prime = modulus, 1, 2
    inverse = [0] * len(matrix)
    while remaining > 1:
        if prime * prime > remaining:
            prime = remaining
        if remaining % prime:
            prime += 1
            continue
        power = 1
        while remaining % prime == 0:
            remaining //= prime
            power *= prime
        residues = _inverse_prime_power(matrix, n, power, prime)
        scale = pow(combined, -1, power)
        inverse = [x + combined * ((v - x) * scale % power) for x, v in zip(inverse, residues)]
        combined *= power
    return [v - modulus if v > modulus // 2 else v for v in inverse]


def inverse_action(graph):
    action = graph.action
    if action["kind"] == "permutation":
        inverses = []
        for forward in action["generators"]:
            inverse = [0] * len(forward)
            for i, source in enumerate(forward):
                inverse[source] = i
            inverses.append(inverse)
        return dict(action, generators=inverses)
    n = action["rows"]
    identity = [int(i == j) for i in range(n) for j in range(n)]
    modulus = action["generators"][0]["modulo"]

    def compose(left, right):
        right = right["matrix"]
        if modulus:
            right = [value % modulus for value in right]
        return graph._multiply(left["matrix"], right, n, n, modulus)

    if any(g["modulo"] != modulus for g in action["generators"]):
        raise ValueError("PATH_MATRIX_REVERSE_UNSUPPORTED")
    if modulus and any(v < 0 or v >= modulus for v in graph.start):
        raise ValueError("PATH_MATRIX_REVERSE_UNSUPPORTED")
    binary_ring = not (modulus & (modulus - 1))

    def overflow_free(matrix):
        return all(
            (modulus - 1) * sum(map(abs, matrix[i:i + n])) <= (1 << 63) - 1
            for i in range(0, len(matrix), n)
        )

    inverses = []
    for forward in action["generators"]:
        if not binary_ring and not overflow_free(forward["matrix"]):
            raise ValueError("PATH_MATRIX_REVERSE_UNSUPPORTED")
        inverse = dict(forward, matrix=_reverse_modular(forward["matrix"], n, modulus))
        if (
            (not binary_ring and not overflow_free(inverse["matrix"]))
            or compose(forward, inverse) != identity
            or compose(inverse, forward) != identity
        ):
            raise ValueError("PATH_MATRIX_REVERSE_UNSUPPORTED")
        inverses.append(inverse)
    return dict(action, generators=inverses)


class DeviceAction:
    def __init__(self, torch, action, device):
        self.torch = torch
        self.kind = action["kind"]
        self.degree = len(action["generators"])
        if self.kind == "permutation":
            self.permutations = torch.tensor(action["generators"], dtype=torch.int64, device=device)
        else:
            self.rows = action["rows"]
            self.cols = action["cols"]
            self.matrices = [
                (torch.tensor(g["matrix"], dtype=torch.int64, device=device).view(self.rows, self.rows), g["modulo"])
                for g in action["generators"]
            ]

    def expand(self, parents):
        width = parents.shape[1]
        if self.kind == "permutation":
            return parents[:, self.permutations].reshape(-1, width)
        torch = self.torch
        values = parents.reshape(-1, self.rows, self.cols)
        children = torch.empty((len(parents), self.degree, width), device=parents.device, dtype=torch.int64)
        for index, (matrix, modulus) in enumerate(self.matrices):
            child = torch.zeros_like(values)
            # Signed int64 tensor arithmetic wraps just like the graph ABI.
            for term in range(self.rows):
                child.add_(matrix[:, term][None, :, None] * values[:, term, :][:, None, :])
            if modulus:
                child.remainder_(modulus)
            children[:, index] = child.flatten(1)
        return children.flatten(0, 1)


class GpuLayers:
    def __init__(self, torch, devices, graph, reverse, batch_records, check):
        self.torch = torch
        self.devices = devices
        self.width = graph.state_elements
        self.actions = [DeviceAction(torch, graph.action, device) for device in devices]
        self.reverse_actions = (
            [DeviceAction(torch, reverse, device) for device in devices] if reverse is not None else None
        )
        self.batch_parents = max(1, batch_records // self.actions[0].degree)
        self.check = check

    def owners(self, rows):
        return owners_for_rows(self.torch, rows, len(self.devices))

    def checkpoint(self):
        self.check()

    def write_paths(self, rows, add):
        for row in rows:
            add(*row)

    def seed(self, states, labels):
        torch = self.torch
        roots = torch.tensor(states, dtype=torch.int64, device=self.devices[0]).reshape(-1, self.width)
        ids = torch.tensor(labels, dtype=torch.int64, device=self.devices[0])
        owners = self.owners(roots)
        ordered = [(owners == rank).nonzero().flatten() for rank in range(len(self.devices))]
        return (
            Layer([roots[at].to(device)
                   for at, device in zip(ordered, self.devices)], [None for _ in self.devices]),
            [ids[at].cpu().tolist() for at in ordered],
        )

    def merge(self, history, incoming):
        """Return append-only rows, stable IDs for every candidate, and first arrivals."""
        torch = self.torch
        base = len(history)
        total = base + len(incoming)
        rows = torch.cat((history, incoming))
        unique, inverse = torch.unique(rows, dim=0, return_inverse=True)
        positions = torch.arange(total, dtype=torch.int64, device=rows.device)
        first = torch.full((len(unique),), total, dtype=torch.int64, device=rows.device)
        first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
        groups = inverse[base:]
        fresh = (first[groups] == positions[base:]).nonzero().flatten()
        stable = torch.empty_like(first)
        stable[inverse[:base]] = positions[:base]
        stable[groups[fresh]] = positions[base:base + len(fresh)]
        if len(fresh):
            history = torch.cat((history, incoming[fresh]))
        return history, stable[groups], fresh

    def advance(self, layer, visited=None, *, reverse=False, lineage=None):
        torch = self.torch
        bases = (visited if visited is not None else
                 [rows.new_empty((0, self.width)) for rows in layer.states])
        committed = [len(rows) for rows in bases]
        tables = list(bases)
        disk = None
        if lineage is not None:
            store, side, depth = lineage
            disk = store.layer(side, depth, len(self.devices))
        actions = self.reverse_actions if reverse else self.actions
        for source, (rows, action) in enumerate(zip(layer.states, actions)):
            for begin in range(0, len(rows), self.batch_parents):
                self.check()
                candidates = action.expand(rows[begin:begin + self.batch_parents])
                owners = self.owners(candidates)
                for rank, device in enumerate(self.devices):
                    take = (owners == rank).nonzero().flatten()
                    if not len(take):
                        continue
                    incoming = candidates[take].to(device)
                    tables[rank], ids, fresh = self.merge(tables[rank], incoming)
                    if disk is None:
                        continue
                    original = take.to(device)
                    if visited is not None:
                        original, ids = original[fresh], ids[fresh]
                    records = torch.stack((
                        ids - committed[rank],
                        torch.full_like(ids, source),
                        original.div(action.degree, rounding_mode="floor") + begin,
                        original.remainder(action.degree),
                    ), dim=1)
                    for chunk in records.split(8192):
                        self.check()
                        disk.add(rank, chunk.cpu())
        self.check()
        states = [table[rows:].clone() if rows else table for table, rows in zip(tables, committed)]
        if disk is not None:
            with closing(disk.seal([len(shard) for shard in states])) as tasks:
                for _ in tasks:
                    self.check()
        return Layer(states, disk.links if disk is not None else []), tables

    def match_batches(self, left, right):
        for rank, (source, target) in enumerate(zip(left.states, right.states)):
            for chunk in match_indices(self.torch, source, target).split(8192):
                self.check()
                if len(chunk):
                    yield [(rank, left_at, right_at) for left_at, right_at in chunk.cpu().tolist()]

    def match_nearest(self, left, right):
        for rank, (source, target) in enumerate(zip(left.states, right.states)):
            self.check()
            indices = match_indices(self.torch, source, target)
            if len(indices):
                return rank, *indices[0].tolist()
        return None
