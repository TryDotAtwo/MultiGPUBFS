"""RAM-only intermediate samples; SSD contains selected final payloads only."""
import hashlib
import math
import os
from pathlib import Path
try:
    from .bfs_tail_archive import TailArchive
except ImportError:
    from bfs_tail_archive import TailArchive


class SelectedTailArchive(TailArchive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.small = {}
        self.last_sample = b''
        self.terminal_parts = None
        self.materialized = False
        self.manifest['state_export'] = dict(
            policy='bounded_prefix_and_terminal_replacement',
            intermediate_storage='RAM', per_rank_prefix_limit=1000,
            incomplete_max_states=1000, per_state_hash_plane=False)

    def selected_layer(self, depth, count, parts, seconds, peaks):
        if self.materialized or depth != len(self.manifest['layers']):
            raise ValueError('selected layer phase')
        if (type(count) is not int or count < 0 or not math.isfinite(seconds)
                or seconds < 0 or not peaks or any(v is not None and
                    (type(v) is not int or v < 0) for v in peaks.values())):
            raise ValueError('selected layer metadata')
        if any(not isinstance(p,bytes) or len(p)%self.width or len(p)>1000*self.width for p in parts):
            raise ValueError('selected rank sample shape')
        # Each rank supplies its first <=1000 states. Taking the global prefix
        # in rank order therefore never creates holes in state ordinals.
        payload = b''.join(parts)[:1000*self.width]
        if len(payload) != min(count, 1000)*self.width:
            raise ValueError('selected prefix shape')
        if count <= 1000:
            self.small[depth] = payload
        self.last_sample = payload
        self.manifest['layers'].append(dict(depth=depth, states=count,
            seconds=seconds, vram_peak_bytes=dict(peaks)))
        self.manifest['last_completed_layer'] = depth

    def terminal(self, depth, parts):
        if depth != self.manifest['last_completed_layer'] or self.materialized:
            raise ValueError('terminal layer phase')
        self.terminal_parts = parts

    def snapshot(self, complete=False, reason='running'):
        if not self.materialized:
            depth = self.manifest['last_completed_layer']
            selected = dict(self.small) if complete else {}
            if depth >= 0:
                selected[depth] = self.last_sample
            if complete and depth >= 0 and self.manifest['layers'][depth]['states'] > 1000:
                if self.terminal_parts is None:
                    raise ValueError('complete terminal payload missing')
                selected[depth] = self.terminal_parts
            created = []
            retained_before = len(self.retained)
            try:
                for at, payload in sorted(selected.items()):
                    target = self.tail/f'layer-{at:06d}.bin'
                    digest = hashlib.sha256(); size = 0
                    with target.open('xb') as out:
                        created.append(target)
                        sources = payload if isinstance(payload, list) else [payload]
                        for source in sources:
                            if isinstance(source, bytes):
                                out.write(source); digest.update(source); size += len(source)
                            else:
                                with Path(source).open('rb') as stream:
                                    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                                        out.write(chunk); digest.update(chunk); size += len(chunk)
                        out.flush(); os.fsync(out.fileno())
                    original = self.manifest['layers'][at]['states']
                    if size % self.width or size > original*self.width or (
                            complete and at == depth and size != original*self.width):
                        raise ValueError('selected final payload shape')
                    self.retained.append(dict(depth=at, states=size//self.width,
                        full_layer=size == original*self.width, bytes=size,
                        first_state_ordinal=0, sha256=digest.hexdigest(),
                        path=target.relative_to(self.root).as_posix()))
            except Exception:
                for target in created:
                    target.unlink(missing_ok=True)
                del self.retained[retained_before:]
                raise
            self.materialized = True
        return super().snapshot(complete, reason)
