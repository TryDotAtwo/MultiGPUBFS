"""Exact packed-word BFS for small orbits; no CUDA construction or estimates."""
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from .bfs_tail_archive import TailArchive, atomic_json
    from .selected_tail import SelectedTailArchive
    from .state_layout import orbit_layout, state_layout
except ImportError:
    from bfs_tail_archive import TailArchive, atomic_json
    from selected_tail import SelectedTailArchive
    from state_layout import orbit_layout, state_layout


def transitions(word, n, bits):
    symbol = (1 << bits) - 1
    mask = (1 << (n * bits)) - 1
    difference = (word ^ (word >> bits)) & symbol
    return ((word >> bits) | ((word & symbol) << ((n - 1) * bits)),
            ((word << bits) & mask) | (word >> ((n - 1) * bits)),
            word ^ difference ^ (difference << bits))


def exact_search(n, r, seed_hex, on_layer, *, deadline=None, cancelled=None):
    """Commit a source layer only after its complete next frontier was generated.

    A bijective seeded key transform changes dedup table keys between repetitions.
    Packed canonical words remain the exported states. Cancellation discards the
    unfinished expansion and leaves earlier committed layers intact.
    """
    orbit_layout(n, r)
    layout = state_layout(n, n-r+1)
    if len(seed_hex) != 32 or any(c not in '0123456789abcdef' for c in seed_hex):
        raise ValueError('CPU hash seed must be 128-bit lowercase hex')
    bits = layout['bits_per_symbol']
    start = list(range(n-r+1)) + [n-r]*(r-1)
    word = sum(symbol << (i*bits) for i, symbol in enumerate(start))
    key_bits = max(64, n*bits)
    key_mask = (1 << key_bits)-1
    seed = int(seed_hex, 16)
    odd = ((seed ^ (seed >> 64)) | 1) & key_mask
    xor = (seed ^ (seed << (key_bits//2))) & key_mask
    def encode(value):
        return ((value ^ xor)*odd) & key_mask
    frontier = [word]
    visited = {encode(word)}
    depth = 0
    started = time.perf_counter()
    while frontier:
        tick = time.perf_counter()
        begin = time.time()
        future = []
        for index, parent in enumerate(frontier):
            if index % 1024 == 0:
                reason = cancelled() if cancelled else None
                if reason or (deadline is not None and time.monotonic() >= deadline):
                    return dict(complete=False, reason=reason or 'CPU search deadline',
                                search_seconds=time.perf_counter()-started,
                                partial_depth=depth, partial_count=len(frontier),
                                partial_processed=index, partial_unprocessed=len(frontier)-index,
                                partial_words=frontier[index:index+1000], partial_prefix=frontier[:1000])
            for child in transitions(parent, n, bits):
                key = encode(child)
                if key not in visited:
                    visited.add(key)
                    future.append(child)
        on_layer(depth, frontier, time.perf_counter()-tick, begin, time.time())
        frontier = future
        depth += 1
    expected = math.factorial(n)//math.factorial(r)
    if len(visited) != expected:
        raise ValueError('CPU exact orbit size differs from enumerated graph')
    return dict(complete=True, reason='graph exhausted', states=len(visited),
                search_seconds=time.perf_counter()-started)


def eligible(config):
    orbit_layout(config['n'],config['r'])
    limit = config.get('small_graph_cpu_max_states', 0)
    if type(limit) is not int or limit < 0:
        raise ValueError('invalid small-graph CPU routing threshold')
    return limit > 0 and math.factorial(config['n'])//math.factorial(config['r']) <= limit


def run(config, source, root, runtime_env, *, cancelled=None, deadline=None,
        program_commit=None, publisher_api=None):
    n, r, world = config['n'], config['r'], config.get('world', 2)
    if world not in (1,2,4,8):
        raise ValueError('graph/topology')
    orbit = orbit_layout(n, r)
    layout = state_layout(n, n-r+1)
    order = math.factorial(n)//math.factorial(r)
    if not eligible(config):
        raise ValueError('CPU orbit exceeds routing threshold')
    source, root = Path(source).resolve(), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    try:
        from .resident_session import active_session
    except ImportError:
        from resident_session import active_session
    session = active_session()
    metadata = session.metadata if session else {}
    identity = str(source)+'commit'
    commit = program_commit or metadata.get(identity)
    if commit is None:
        commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        metadata[identity] = commit
    seed = config.get('env',{}).get('MGBFS_HASH_SEED_HEX','000000000000000000000000013527dc')
    saved = dict(config, execution_backend='CPU_EXACT_PACKED',
                 orbit_layout=orbit, python_version=sys.version,
                 python_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 dedup_key_transform='bijective seeded affine packed-integer keys',
                 runtime_paths=runtime_env,
                 automatic_phase_seconds=dict(admission=0.0,graph_calibration=0.0))
    selected = config.get('retention_policy') == 'last_complete_small_1000'
    archive = (SelectedTailArchive if selected else TailArchive)(root/'saved',n=n,r=r,
        start=list(range(n-r+1))+[n-r]*(r-1),actions=dict(L='cyclic left rotation',
        R='cyclic right rotation',X='swap positions 0 and 1'),program_commit=commit,
        launch_config=saved,sample_interval_seconds=.05,snapshot_each_layer=False)
    archive.manifest['execution_backend'] = 'CPU_EXACT_PACKED'
    archive.manifest['vram_observation'] = dict(source='separate nvidia-smi monitor',
        timestamp='host receipt time',window='CPU whole-layer expansion',
        caveat='short CPU layers can have no sample; CPU search allocates no VRAM')
    archive.manifest['layer_time_scope'] = 'whole exact CPU layer expansion'
    samples, lock = session.memory_monitor() if session else ([], threading.Lock())
    # One bounded preallocated payload arena; no state files during traversal.
    width = layout['bytes_per_state']
    arena = bytearray(order*width)
    layers = []
    offset = 0
    def committed(depth, words, seconds, begin, end):
        nonlocal offset
        length = len(words)*width
        if offset+length > len(arena):
            raise ValueError('CPU payload arena exceeded exact orbit bound')
        start = offset
        for word in words:
            arena[offset:offset+width] = word.to_bytes(width,'little')
            offset += width
        with lock:
            peaks = {str(gpu):max((used for at,index,used in samples
                if index==str(gpu) and begin<=at<=end),default=None) for gpu in range(world)}
        layers.append((depth,len(words),start,length,seconds,peaks))
    cpu_deadline = time.monotonic()+max(0, config.get('timeout_seconds',300))
    if deadline is not None:
        cpu_deadline = min(cpu_deadline,time.monotonic()+max(0,deadline-time.time()))
    outcome = exact_search(n,r,seed,committed,deadline=cpu_deadline,cancelled=cancelled)
    archive_started = time.perf_counter()
    for depth,count,start,length,seconds,peaks in layers:
        if selected:
            archive.selected_layer(depth,count,[bytes(arena[start:start+min(count,1000)*width])],seconds,peaks)
        else:
            archive.completed_layer(depth,count,[memoryview(arena)[start:start+length]],seconds,peaks)
    if selected and outcome['complete'] and layers and layers[-1][1] > 1000:
        depth,count,start,length,_,_=layers[-1]
        archive.terminal(depth,[bytes(arena[start:start+length])])
    if selected and not outcome['complete']:
        depth=outcome['partial_depth'];count=outcome['partial_count']
        # This frontier was generated completely; its expansion was interrupted.
        sample=b''.join(word.to_bytes(width,'little') for word in outcome['partial_words'])
        archive.selected_layer(depth,count,[b''.join(word.to_bytes(width,'little') for word in outcome['partial_prefix'])],0.0,
            {str(gpu):None for gpu in range(world)})
        archive.manifest['layers'][-1]['expansion_complete']=False
        archive.partial_terminal(depth,[sample],[dict(saved=len(outcome['partial_words']),
            processed=outcome['partial_processed'],unprocessed=outcome['partial_unprocessed'],
            scope='unprocessed_current_suffix',completion_source='CPU_EXACT_CURSOR')])
    final = archive.snapshot(outcome['complete'],outcome['reason'])
    archive.release_working_tail()
    result = dict(status='COMPLETE' if outcome['complete'] else 'INCOMPLETE',
        execution_backend='CPU_EXACT_PACKED',hash_seed_hex=seed,
        search_complete_seconds=outcome['search_seconds'] if outcome['complete'] else None,
        search_prefix_seconds=None if outcome['complete'] else outcome['search_seconds'],
        archive_seconds=time.perf_counter()-archive_started,layer_sizes=[x[1] for x in layers],
        arena_capacity_bytes=len(arena),arena_used_bytes=offset,cuda_allocated_bytes=0,
        program_commit=commit)
    (root/'result').mkdir()
    atomic_json(root/'result/cpu.json',result)
    if config.get('repo_id'):
        try:
            from .tail_upload import Publisher
        except ImportError:
            from tail_upload import Publisher
        token = None
        if publisher_api is None:
            from huggingface_hub import get_token
            token = get_token()
        publisher = Publisher(root/'upload-pins',config['repo_id'],config['run_id'],token,
            api=publisher_api,storage_format=config.get('live_upload_format','packed'),
            prefix=config.get('live_upload_prefix'))
        publisher.enqueue(final)
        publisher.finish()
    atomic_json(root/'run-summary.json',dict(status=result['status'],reason=outcome['reason'],
        manifest=str(final),execution_backend=result['execution_backend'],program_commit=commit,
        publication_status='COMPLETE' if config.get('repo_id') else 'NOT_REQUESTED'))
    return final
