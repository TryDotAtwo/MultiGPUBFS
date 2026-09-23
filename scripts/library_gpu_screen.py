"""Contracts for library BFS screening, not final Pareto acceptance."""
import math
import json
import subprocess
import sys
from pathlib import Path
from distributed_gpu_bench import run_group, stats


def validate_result(row, expected_states, pool_bytes, *, owner='CUDF_RELATIONAL', expected=None):
    """Reject incomplete/search-only/fallback runs before interpreting timings."""
    if row.get('status') != 'COMPLETE':
        raise ValueError('SCREEN_INCOMPLETE')
    counts = row.get('layer_sizes')
    if (not isinstance(counts, list) or not counts
            or any(type(x) is not int or x < 0 for x in counts)
            or sum(counts) != expected_states):
        raise ValueError('SCREEN_COUNTS')
    durable = row.get('durable_run_commit_seconds')
    if type(durable) not in (int, float) or not math.isfinite(durable) or durable < 0:
        raise ValueError('SCREEN_DURABLE')
    ranks = row.get('rank_results')
    if not isinstance(ranks, list) or not ranks or any(not isinstance(rank, dict) for rank in ranks):
        raise ValueError('SCREEN_RANKS')
    if expected is not None and 'world_size' in expected:
        world = expected['world_size']
        ids = [rank.get('rank') for rank in ranks]
        if (len(ranks) != world or any(type(rank) is not int for rank in ids)
                or sorted(ids) != list(range(world))):
            raise ValueError('SCREEN_RANKS')
    for rank in ranks:
        native = owner in ('CUB_SORT_MERGE', 'BMMA_BUCKET')
        if native and (pool_bytes != 0 or not rank.get('backend', '').startswith('native_')):
            raise ValueError('SCREEN_NATIVE_CONTRACT')
        if (rank.get('owner_backend') != owner
                or rank.get('archive_enabled') is not True
                or rank.get('warmup_completed') is not True
                or rank.get('library_pool_reserved_bytes', 0 if native else None) != pool_bytes):
            raise ValueError('SCREEN_LIBRARY_CONTRACT')
        for field, value in (expected or {}).items():
            if type(rank.get(field)) is not type(value) or rank[field] != value:
                raise ValueError('SCREEN_CONFIGURATION: ' + field)


def run_case(cli, output, archive_root, group, expected_states, world, batch,
             capacity, ring, pool_bytes, profile, pre_dedup, inherited, *, owner='CUDF_RELATIONAL',
             native_example=None, nsys=None):
    """One screening run; optional diagnostic trace is never benchmark evidence."""
    if world not in (1, 2) or profile not in ('DENSE', 'HASH_FIRST'):
        raise ValueError('SCREEN_CONFIG')
    native = owner in ('CUB_SORT_MERGE', 'BMMA_BUCKET')
    if owner not in ('CUDF_RELATIONAL', 'CUCO_INDEXED', 'CUCO_RANK', 'CUB_SORT_MERGE', 'BMMA_BUCKET'):
        raise ValueError('SCREEN_OWNER')
    if native != (native_example is not None) or (pool_bytes != 0 if native else pool_bytes <= 0):
        raise ValueError('SCREEN_BACKEND_CONFIG')
    if pre_dedup not in ('ON', 'OFF') or min(batch, capacity, ring) <= 0:
        raise ValueError('SCREEN_CONFIG')
    output, archive_root = Path(output), Path(archive_root)
    output.mkdir(parents=True, exist_ok=False)
    archive_root.mkdir(parents=True, exist_ok=False)
    env = dict(inherited, CUDA_VISIBLE_DEVICES=','.join(map(str, range(world))),
        MGBFS_BENCH_WORLD_SIZE=str(world), MGBFS_RANK_MAP=','.join(map(str, range(world))),
        MGBFS_OWNER_BACKEND=owner, MGBFS_LIBRARY_POOL_BYTES=str(pool_bytes),
        MGBFS_PROFILE=profile, MGBFS_PRE_DEDUP=pre_dedup,
        MGBFS_HASH_FIRST_GENERATION='SCALAR', MGBFS_BENCH_CAPACITY=str(capacity),
        MGBFS_FUTURE_CAPACITY=str(ring), MGBFS_CAPACITY_MODE='max_per_rank',
        MGBFS_STATE_CODEC='matrix_u8', MGBFS_ARCHIVE_CODEC='matrix_u8',
        MGBFS_BENCH_WARMUP='1', MGBFS_BENCH_SKIP_ARCHIVE='0', MGBFS_ARCHIVE_STREAM='0')
    executable = [str(native_example)] if native else [str(cli), 'bench', '--reference']
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc-per-node={world}', '--no-python', *executable,
        group, str(batch), str(archive_root/'bootstrap'), str(archive_root/'archive'),
        '{RANK_OUT}']
    if nsys is not None:
        command = [str(nsys), 'profile', '--trace=cuda,nvtx,osrt',
            '--sample=none', '--cpuctxsw=none', '--force-overwrite=false',
            '--output=' + str(output/'timeline'), *command]
    report = dict(status='INCOMPLETE', scope='single screening sample; not Pareto acceptance',
                  memory_scope='50ms nvidia-smi sampled full device consumption; not exact peak',
                  configuration=env, archive_verification=[])
    report['profiled'] = nsys is not None
    if nsys is not None:
        report['scope'] = 'diagnostic trace including startup/warmup/archive; not performance evidence'
    # Do not serialize inherited credentials or unrelated environment settings.
    report['configuration'] = {k: v for k, v in env.items()
                               if k.startswith('MGBFS_') and k in {
                                   'MGBFS_OWNER_BACKEND', 'MGBFS_LIBRARY_POOL_BYTES',
                                   'MGBFS_PROFILE', 'MGBFS_PRE_DEDUP', 'MGBFS_BENCH_CAPACITY',
                                   'MGBFS_FUTURE_CAPACITY', 'MGBFS_RANK_MAP'}}
    try:
        row = run_group(command, output, 'measure', env, timeout=1800)
        row['profiled'] = nsys is not None
        # Keep the standalone measurement artifact as unambiguous as the
        # enclosing screen summary; downstream consumers may read either.
        (output/'measure.json').write_text(json.dumps(row, indent=2))
        report['measurement'] = row
        validate_result(row, expected_states, pool_bytes, owner=owner,
            expected=dict(group=group, batch=batch, world_size=world,
                frontier_profile=profile, pre_dedup=pre_dedup,
                declared_capacity_records=capacity, declared_state_ring_records=ring,
                capacity_mode='MaxPerRank', hash_first_generation='SCALAR', generation_variant=1))
        if nsys is not None:
            trace = output/'timeline.nsys-rep'
            if not trace.is_file() or trace.stat().st_size == 0:
                raise ValueError('SCREEN_TRACE_MISSING')
            report['trace'] = str(trace)
        for rank in range(world):
            archive = archive_root/f'archive-rank-{rank}.mgbfsar1'
            verified = subprocess.run([str(cli), 'verify', str(archive)], env=env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=True, timeout=1800)
            (output/f'verify-rank-{rank}.log').write_text(verified.stdout)
            result = json.loads(verified.stdout)
            if result.get('status') != 'VERIFIED':
                raise ValueError('SCREEN_ARCHIVE_VERIFY')
            report['archive_verification'].append(result)
        if nsys is None:
            report['statistics'] = stats([row])
        report['status'] = 'COMPLETE'
        return report
    except Exception as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        (output/'screen-summary.json').write_text(json.dumps(report, indent=2))
