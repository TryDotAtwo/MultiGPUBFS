"""Relay native segmented host-agreement records between rank-local disks."""

import base64
import json
from pathlib import Path
import re

from .network_control import _deadline_remaining


_PHASE = re.compile(r"[A-Za-z0-9-]+")


class AgreementRelay:
    def __init__(self, store, label, bootstrap, rank, world):
        self.store = store
        self.label = label
        self.root = Path(str(bootstrap) + ".segmented-host")
        self.rank = rank
        self.world = world
        self.pending = set()
        self.finished = set()

    def _verify(self, item, phase, rank):
        expected = {"schema": 1, "phase": phase, "rank": rank, "world": self.world}
        if not isinstance(item, dict) or any(item.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"SEGMENTED_NETWORK_RECORD_IDENTITY phase={phase} rank={rank}")

    def step(self, phase=None, *, deadline=None):
        wait = {} if deadline is None else {"deadline": deadline}
        if deadline is not None:
            _deadline_remaining(deadline)
        suffix = f"-rank-{self.rank}.json"
        if phase is not None and _PHASE.fullmatch(phase) is None:
            raise RuntimeError("SEGMENTED_NETWORK_RECORD_PHASE")
        paths = self.root.glob(f"*{suffix}") if phase is None else (self.root / f"{phase}{suffix}",)
        for path in paths:
            name = path.name[:-len(suffix)]
            if name in self.pending or name in self.finished or not path.is_file():
                continue
            if _PHASE.fullmatch(name) is None:
                raise RuntimeError("SEGMENTED_NETWORK_RECORD_PHASE")
            record = json.loads(path.read_text(encoding="utf-8"))
            self._verify(record, name, self.rank)
            self.store.put(f"{self.label}/agreement/{name}/{self.rank}", record, **wait)
            self.pending.add(name)
        for phase in tuple(self.pending):
            records = self.store.try_get(f"{self.label}/agreement/{phase}/", ranks=self.world, **wait)
            if records is None:
                continue
            for rank, record in enumerate(records):
                if deadline is not None:
                    _deadline_remaining(deadline)
                self._verify(record, phase, rank)
                if rank != self.rank:
                    path = self.root / f"{phase}-rank-{rank}.json"
                    temporary = path.with_suffix(".relay-tmp")
                    temporary.write_text(json.dumps(record), encoding="utf-8")
                    temporary.replace(path)
            self.pending.remove(phase)
            self.finished.add(phase)

    def bootstrap(self, path, observed_phase, delivered, *, deadline=None):
        """Relay both host agreement and the bootstrap ID for a live rank."""
        bounded = {} if deadline is None else {"deadline": deadline}
        if observed_phase.startswith("segmented-host-"):
            self.step(observed_phase.removeprefix("segmented-host-"), **bounded)
        if delivered:
            return True
        key = f"{self.label}/nccl-id"
        if self.rank == 0:
            if not path.is_file():
                return False
            self.store.put(key, base64.b64encode(path.read_bytes()).decode(), **bounded)
            return True
        value = self.store.try_get(key, **bounded)
        if value is None:
            return False
        temporary = path.with_suffix(".relay-tmp")
        temporary.write_bytes(base64.b64decode(value, validate=True))
        temporary.replace(path)
        return True
