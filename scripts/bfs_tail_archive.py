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

GB = 1_000_000_000


def packed_width(n, alphabet):
    if not 1 <= n <= 32 or not 1 <= alphabet <= 16:
        raise ValueError("four-bit packing requires n<=32 and alphabet<=16")
    return 8 if n <= 16 else 16


def pack_state(symbols):
    width = packed_width(len(symbols), max(symbols, default=0) + 1)
    if any(type(x) is not int or x < 0 or x > 15 for x in symbols):
        raise ValueError("symbol outside [0,15]")
    return sum(x << (4 * i) for i, x in enumerate(symbols)).to_bytes(width, "little")


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class TailArchive:
    """Single writer; retention thresholds count raw packed bytes, decimal GB.

    An existing directory is rejected to prevent overwriting another run.
    Failed writes never advance the completed depth. Restart recovery is via
    the last manifest; this class deliberately does not resume a search.
    """
    def __init__(self, root, *, n, r, start, actions, program_commit,
                 launch_config, sample_interval_seconds,
                 complete_bytes=10 * GB, incomplete_bytes=GB):
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
        self.manifest = dict(schema=1, status="INCOMPLETE", last_completed_layer=-1,
            stop_reason="running", graph=dict(n=n, r=r, start=list(start), actions=actions),
            packing=dict(bytes_per_state=self.width, bits_per_symbol=4,
                         byte_order="little", symbol_order="symbol i at bits 4*i", padding="zero"),
            program_commit=program_commit, launch_config=launch_config,
            vram_sampling_interval_seconds=sample_interval_seconds,
            byte_unit="decimal GB", layers=[], files=[])
        atomic_json(self.root / "manifest.json", self.manifest)

    def completed_layer(self, depth, count, chunks, seconds, vram_peak_bytes):
        """chunks: iterable of bytes; no paths, hashes or duplicate metrics."""
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
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        entry = dict(depth=depth, states=count, full_layer=True,
                     bytes=size, sha256=digest.hexdigest(), path=str(path.relative_to(self.root)))
        self.retained.append(entry)
        self.manifest["layers"].append(dict(depth=depth, states=count, seconds=seconds,
                                          vram_peak_bytes=dict(vram_peak_bytes)))
        self.manifest["last_completed_layer"] = depth
        # Keep at least three entire layers AND enough bytes, until graph start.
        while len(self.retained) > 3 and sum(x["bytes"] for x in self.retained[1:]) >= self.complete_bytes:
            old = self.retained.pop(0)
            (self.root / old["path"]).unlink()
        self.snapshot()

    def snapshot(self, complete=False, reason="running"):
        """Publish local snapshot, limiting INCOMPLETE to whole packed records.

        COMPLETE keeps whole layers; INCOMPLETE may take the suffix of the
        oldest selected layer. Payload files are immutable for this generation.
        """
        generation = self.root / f"snapshot-{len(self.manifest['layers']):06d}-{uuid.uuid4().hex}"
        # Repeated calls (e.g. final stop reason) reuse verified immutable files.
        generation.mkdir(exist_ok=True)
        selected, remaining = [], self.incomplete_bytes // self.width * self.width
        source_entries = self.retained if complete else reversed(self.retained)
        for entry in source_entries:
            take = entry["bytes"] if complete else min(entry["bytes"], remaining)
            if not complete and remaining == 0:
                break
            target = generation / Path(entry["path"]).name
            offset = entry["bytes"] - take
            digest = hashlib.sha256()
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
            selected.append(dict(entry, path=str(target.relative_to(self.root)),
                                 states=take // self.width, bytes=take,
                                 full_layer=take == entry["bytes"],
                                 first_state_ordinal=offset // self.width,
                                 sha256=digest.hexdigest()))
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
