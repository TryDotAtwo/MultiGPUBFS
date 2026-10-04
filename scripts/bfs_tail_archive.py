"""SSD tail archive. Input chunks contain packed, globally completed layers.

This module owns no CUDA state and performs no CUDA synchronization. A caller
must supply all ranks' chunks after the existing global layer completion vote.
"""
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
try:
    from .state_layout import state_layout
except ImportError:
    from state_layout import state_layout

GB = 1_000_000_000


def packed_width(n, alphabet):
    return state_layout(n, alphabet)['bytes_per_state']


def pack_state(symbols):
    if any(type(x) is not int or x < 0 for x in symbols):
        raise ValueError("nonnegative integer symbols required")
    layout = state_layout(len(symbols), max(symbols, default=0) + 1)
    return sum(x << (layout['bits_per_symbol'] * i) for i, x in enumerate(symbols)).to_bytes(layout['bytes_per_state'], "little")


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if os.name == 'posix':
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


class TailArchive:
    """Single writer; retention thresholds count raw packed bytes, decimal GB.

    An existing directory is rejected to prevent overwriting another run.
    Failed writes never advance the completed depth. Restart recovery is via
    the last manifest; this class deliberately does not resume a search.
    """
    def __init__(self, root, *, n, r, start, actions, program_commit,
                 launch_config, sample_interval_seconds,
                 complete_bytes=10 * GB, incomplete_bytes=GB, retained_layers=None):
        retained_layers = launch_config.get('retained_layers') if retained_layers is None else retained_layers
        if retained_layers is not None and (type(retained_layers) is not int or retained_layers <= 0):
            raise ValueError('positive fixed layer count required')
        self.retained_layers = retained_layers
        self.compact_policy = launch_config.get('retention_policy') == 'last_complete_small_1000'
        self.width = packed_width(n, max(start, default=0) + 1)
        if len(start) != n or not program_commit or set(actions) != {"L", "R", "X"}:
            raise ValueError("exact graph/configuration required")
        pack_state(start)
        if sample_interval_seconds <= 0 or not math.isfinite(sample_interval_seconds):
            raise ValueError("positive monitor interval required")
        if complete_bytes <= 0 or incomplete_bytes <= 0:
            raise ValueError("positive retention budgets required")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.tail = self.root / "tail"
        self.tail.mkdir()
        self.complete_bytes, self.incomplete_bytes = complete_bytes, incomplete_bytes
        self.retained = []
        self.closed = False
        self.manifest = dict(schema=1, status="INCOMPLETE", last_completed_layer=-1,
            stop_reason="running", graph=dict(n=n, r=r, start=list(start), actions=actions),
            packing=dict(**state_layout(n,max(start,default=0)+1),
                         byte_order="little", symbol_order="symbol i at bits bits_per_symbol*i", padding="zero"),
            program_commit=program_commit, launch_config=launch_config,
            vram_sampling_interval_seconds=sample_interval_seconds,
            byte_unit="decimal GB", layers=[], files=[])
        if retained_layers is not None:
            self.manifest['retention'] = dict(policy='fixed_whole_layers', layers=retained_layers,
                                             applies_to=['COMPLETE','INCOMPLETE'])
        if self.compact_policy:
            self.manifest['retention'] = dict(policy='last_complete_small_1000',
                complete='last whole layer plus whole layers with at most 1000 states',
                incomplete_max_states=1000)
        atomic_json(self.root / "manifest.json", self.manifest)

    def completed_layer(self, depth, count, chunks, seconds, vram_peak_bytes):
        """chunks: iterable of bytes; no paths, hashes or duplicate metrics."""
        if self.closed:
            raise ValueError("archive working tail already released")
        if depth != len(self.manifest["layers"]) or type(count) is not int or count < 0:
            raise ValueError("noncontiguous depth or invalid count")
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("invalid layer duration")
        if not vram_peak_bytes or any(v is not None and (type(v) is not int or v < 0) for v in vram_peak_bytes.values()):
            raise ValueError("observed peak required for each GPU")
        path = self.tail / f"layer-{depth:06d}.bin"
        temp = path.with_suffix(".tmp")
        digest, size = hashlib.sha256(), 0
        try:
            with temp.open("wb") as stream:
                for chunk in chunks:
                    if len(chunk) % self.width:
                        raise ValueError("chunk splits a packed state")
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                if size != count * self.width:
                    raise ValueError("layer count mismatch")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            if os.name == 'posix':
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        entry = dict(depth=depth, states=count, full_layer=True,
                     bytes=size, sha256=digest.hexdigest(), path=path.relative_to(self.root).as_posix())
        self.retained.append(entry)
        self.manifest["layers"].append(dict(depth=depth, states=count, seconds=seconds,
                                          vram_peak_bytes=dict(vram_peak_bytes)))
        self.manifest["last_completed_layer"] = depth
        # Keep at least three entire layers AND enough bytes, until graph start.
        if self.compact_policy:
            obsolete = [x for x in self.retained[:-1] if x['states'] > 1000]
            self.retained = [x for x in self.retained if x not in obsolete]
            for old in obsolete:
                (self.root / old['path']).unlink()
        while (not self.compact_policy and (len(self.retained) > self.retained_layers if self.retained_layers is not None
               else len(self.retained) > 3 and sum(x["bytes"] for x in self.retained[1:]) >= self.complete_bytes)):
            old = self.retained.pop(0)
            (self.root / old["path"]).unlink()
        self.snapshot()

    def release_working_tail(self):
        """Final-only cleanup: committed snapshots retain their own links/copies.

        During search keep the entire COMPLETE-capable tail. After the traversal
        stops, only the sealed final snapshot is needed for publication/retry.
        """
        current = json.loads((self.root/'manifest.json').read_text())
        for entry in current['files']:
            payload = (self.root/entry['path']).resolve()
            if not payload.is_relative_to(self.root.resolve()) or not payload.exists():
                raise ValueError('final snapshot payload missing before tail release')
        for entry in self.retained:
            payload = (self.root/entry['path']).resolve()
            if payload.parent != self.tail.resolve():
                raise ValueError('working tail release escapes tail directory')
            payload.unlink(missing_ok=True)
        self.closed = True
        self.retained = []

    def snapshot(self, complete=False, reason="running"):
        """Publish local snapshot, limiting INCOMPLETE to whole packed records.

        COMPLETE keeps whole layers; INCOMPLETE may take the suffix of the
        oldest selected layer. Payload files are immutable for this generation.
        """
        if self.closed:
            raise ValueError("archive working tail already released")
        generation = self.root / f"snapshot-{len(self.manifest['layers']):06d}-{uuid.uuid4().hex}"
        # Repeated calls (e.g. final stop reason) reuse verified immutable files.
        generation.mkdir(exist_ok=True)
        selected, remaining = [], self.incomplete_bytes // self.width * self.width
        if self.compact_policy:
            remaining = 1000 * self.width
        full_tail = complete or (self.retained_layers is not None and not self.compact_policy)
        source_entries = self.retained if full_tail else reversed(self.retained)
        for entry in source_entries:
            take = entry["bytes"] if full_tail else min(entry["bytes"], remaining)
            if not full_tail and remaining == 0:
                break
            target = generation / Path(entry["path"]).name
            offset = entry["bytes"] - take
            digest = hashlib.sha256()
            if take == entry["bytes"]:
                os.link(self.root / entry["path"], target)
                checksum = entry["sha256"]
            else:
                with (self.root / entry["path"]).open("rb") as source, target.open("wb") as dest:
                    source.seek(offset)
                    left = take
                    while left:
                        chunk = source.read(min(left, 4 * 1024 * 1024))
                        if not chunk:
                            raise ValueError("tail truncated")
                        dest.write(chunk)
                        digest.update(chunk)
                        left -= len(chunk)
                    dest.flush()
                    os.fsync(dest.fileno())
                checksum = digest.hexdigest()
            selected.append(dict(entry, path=target.relative_to(self.root).as_posix(),
                                 states=take // self.width, bytes=take,
                                 full_layer=take == entry["bytes"],
                                 first_state_ordinal=offset // self.width,
                                 sha256=checksum))
            remaining -= take
        manifest = dict(self.manifest, status="COMPLETE" if complete else "INCOMPLETE",
                        stop_reason=reason, files=sorted(selected, key=lambda x: x["depth"]))
        atomic_json(generation / "manifest.json", manifest)
        atomic_json(self.root / "manifest.json", manifest)
        # Old snapshot copies are removed only after the new manifest is durable.
        for old in self.root.glob("snapshot-*"):
            if old != generation and old.is_dir():
                shutil.rmtree(old)
        return generation / "manifest.json"


def publish_snapshot(manifest_path, upload_file):
    """Run in a background worker; callback returns only after remote receipt.

    Caller pins this generation against local cleanup until this returns.
    The manifest is uploaded last; failed payload upload cannot publish it.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.parent
    for entry in manifest["files"]:
        path = root / entry["path"]
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("snapshot checksum mismatch")
        upload_file(path, entry["path"])
    upload_file(manifest_path, "manifest.json")
