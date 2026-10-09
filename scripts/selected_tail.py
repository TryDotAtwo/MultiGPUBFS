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
        self.partial_parts = None
        self.partial_metadata = None
        self.manifest['state_export'] = dict(
            policy='whole_small_and_terminal_replacement',
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
        if not (count>1000 and not payload) and len(payload) != min(count, 1000)*self.width:
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

    def partial_terminal(self,depth,parts,metadata):
        if depth!=self.manifest['last_completed_layer'] or self.materialized:
            raise ValueError('partial terminal phase')
        sizes=[len(x) if isinstance(x,bytes) else Path(x[0]).stat().st_size for x in parts]
        if any(n%self.width for n in sizes) or sum(sizes)>1000*self.width:
            raise ValueError('partial sample global bound')
        if len(metadata)!=len(parts) or any(int(m['saved'])*self.width!=size for m,size in zip(metadata,sizes)):
            raise ValueError('partial sample completion metadata')
        if any(int(m.get('unprocessed',m['saved'])) < int(m['saved']) or int(m['saved'])<0
               or int(m.get('processed',0))<0 for m in metadata):
            raise ValueError('partial sample cursor bounds')
        self.partial_parts=parts;self.partial_metadata=metadata

    def snapshot(self, complete=False, reason='running'):
        if not self.materialized:
            depth = self.manifest['last_completed_layer']
            # A terminal descriptor is a whole layer only after its total shape
            # matches the globally completed layer. Never relabel a partial
            # failed export as a complete snapshot.
            if depth>=0 and self.terminal_parts is not None:
                sources=self.terminal_parts if isinstance(self.terminal_parts,list) else [self.terminal_parts]
                sizes=[len(x) if isinstance(x,bytes) else Path(x[0] if isinstance(x,tuple) else x).stat().st_size for x in sources]
                if any(size%self.width for size in sizes) or sum(sizes)!=self.manifest['layers'][depth]['states']*self.width:
                    self.terminal_parts=None
                    if complete:raise ValueError('selected final payload shape')
            self.whole_incomplete = False
            selected = dict(self.small) if complete else {}
            if complete:
                if depth>=0:selected[depth]=self.last_sample
                if depth>=0 and (self.terminal_parts is not None or self.manifest['layers'][depth]['states']>1000):
                    if self.terminal_parts is None:raise ValueError('complete terminal payload missing')
                    selected[depth]=self.terminal_parts
            else:
                self.manifest['state_export'].update(incomplete_max_states=1000,
                    incomplete_policy='unprocessed_current_sample',sample_is_partial=True,
                    sample_status=('verified_cpu_cursor_suffix' if self.partial_metadata and all(
                        m.get('completion_source')=='CPU_EXACT_CURSOR' for m in self.partial_metadata)
                        else 'verified_gpu_completion_suffix') if self.partial_parts is not None else 'unavailable',
                    sample_rank_metadata=self.partial_metadata or [])
                self.manifest['retention']['incomplete_max_states']=1000
                if depth>=0 and self.partial_parts is not None:selected[depth]=self.partial_parts
            created, moved = [], []
            retained_before = len(self.retained)
            try:
                for at, payload in sorted(selected.items()):
                    original=self.manifest['layers'][at]['states']
                    sources=payload if isinstance(payload,list) else [payload]
                    planned=[];ordinal=0
                    for rank,source in enumerate(sources):
                        if isinstance(source,bytes):
                            size=len(source);digest=hashlib.sha256(source).hexdigest()
                        elif isinstance(source,tuple):
                            path,count,digest=source;path=Path(path);size=path.stat().st_size
                            if path.is_symlink() or size!=count*self.width or len(digest)!=64:
                                raise ValueError('selected terminal descriptor shape')
                            source=(path,count,digest)
                        else:
                            path=Path(source);size=path.stat().st_size
                            with path.open('rb') as stream:
                                digest=hashlib.file_digest(stream,'sha256').hexdigest()
                            source=(path,size//self.width,digest)
                        if size%self.width:raise ValueError('selected final payload shape')
                        planned.append((rank,source,size,digest,ordinal));ordinal+=size//self.width
                    if ordinal>original or (complete and at==depth and ordinal!=original):
                        raise ValueError('selected final payload shape')
                    for rank,source,size,digest,ordinal in planned:
                        suffix=f'-rank-{rank:04d}' if isinstance(payload,list) else ''
                        target=self.tail/f'layer-{at:06d}{suffix}.bin'
                        if target.exists():raise FileExistsError(target)
                        if isinstance(source,bytes):
                            with target.open('xb') as out:
                                created.append(target);out.write(source)
                                out.flush();os.fsync(out.fileno())
                        else:
                            path,_,_=source
                            os.rename(path,target);moved.append((path,target))
                            with target.open('r+b') as stream:os.fsync(stream.fileno())
                        self.retained.append(dict(depth=at,states=size//self.width,
                            full_layer=complete and size==original*self.width,layer_complete=complete,
                            sample_of_unprocessed_current=not complete,
                            bytes=size,first_state_ordinal=ordinal,sha256=digest,
                            path=target.relative_to(self.root).as_posix()))
            except Exception:
                for target in created:target.unlink(missing_ok=True)
                for source,target in reversed(moved):os.rename(target,source)
                del self.retained[retained_before:]
                raise
            self.materialized = True
        return super().snapshot(complete, reason)
