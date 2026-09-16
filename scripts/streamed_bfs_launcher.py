"""Platform-independent launch plan for native BFS with bounded HF consumers.

No provisioning or billing control is performed here. The caller must arrange
an independent rental deadline before launching any paid workload.
"""
import re
import sys


def make_plan(config, source, root):
    positive = ('world', 'n', 'batch', 'capacity', 'archive_rows', 'archive_slots',
                'upload_slots', 'max_slot_bytes', 'rows_per_shard', 'shards',
                'buckets', 'bucket_capacity', 'scratch_bytes', 'reserve_bytes',
                'timeout_seconds', 'job_buckets', 'library_pool_bytes')
    if any(type(config.get(k)) is not int or config[k] <= 0 for k in positive):
        raise ValueError('STREAM_CONFIG_POSITIVE_INTEGER')
    world = config['world']
    if world not in (1, 2, 4, 8) or not 2 <= config['n'] <= 20:
        raise ValueError('STREAM_CONFIG_TOPOLOGY')
    for key in ('shards', 'buckets'):
        value = config[key]
        if value & (value - 1) or value < world:
            raise ValueError('STREAM_CONFIG_GEOMETRY')
    if config['buckets'] < config['shards']:
        raise ValueError('STREAM_CONFIG_GEOMETRY')
    if (config['job_buckets'] > config['buckets']//world or
            any(config[k] >= 2**32 for k in ('batch', 'capacity', 'archive_rows',
                'archive_slots', 'shards', 'buckets', 'bucket_capacity', 'job_buckets'))):
        raise ValueError('STREAM_CONFIG_CAPACITY')
    for key, allowed in [('profile', ('DENSE', 'HASH_FIRST')),
                         ('owner', ('CUB_SORT_MERGE', 'CUCO_INDEXED')),
                         ('pre_dedup', ('ON', 'OFF'))]:
        if config.get(key) not in allowed:
            raise ValueError('STREAM_CONFIG_' + key)
    run_id = config.get('run_id', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,90}', run_id):
        raise ValueError('STREAM_CONFIG_RUN_ID')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', config.get('repo_id', '')):
        raise ValueError('STREAM_CONFIG_REPO')
    group = f"s{config['n']}"
    env = dict(MGBFS_BENCH_WORLD_SIZE=str(world), MGBFS_CAPACITY_MODE='max_per_rank',
               MGBFS_ARCHIVE_STREAM='1', MGBFS_BENCH_SKIP_ARCHIVE='0',
               MGBFS_ARCHIVE_CODEC='permutation_u8', MGBFS_STATE_CODEC='permutation_u8',
               MGBFS_PROFILE=config['profile'], MGBFS_OWNER_BACKEND=config['owner'],
               MGBFS_PRE_DEDUP=config['pre_dedup'], MGBFS_BENCH_WARMUP='0',
               MGBFS_TRACE_DEPTHS='1', MGBFS_RANK_MAP=','.join(map(str, range(world))))
    for name, key in [('BENCH_CAPACITY', 'capacity'), ('FUTURE_CAPACITY', 'capacity'),
                      ('ARCHIVE_ROWS', 'archive_rows'), ('ARCHIVE_SLOTS', 'archive_slots'),
                      ('SHARDS', 'shards'), ('BUCKETS', 'buckets'),
                      ('BUCKET_CAPACITY', 'bucket_capacity'), ('JOB_BUCKETS', 'job_buckets'),
                      ('LIBRARY_POOL_BYTES', 'library_pool_bytes')]:
        env['MGBFS_' + name] = str(config[key])
    prefix = root / 'archive'
    fifos = [str(root / f'archive-rank-{r}.mgbfsar1') for r in range(world)]
    branches = [f'staging-{run_id}-rank-{r}' for r in range(world)]
    consumers, commits = [], []
    for rank in range(world):
        staging = root / f'rank-{rank}-slots'
        commits.append(str(staging / f'rank-{rank:05d}-stream-commit.json'))
        consumers.append([
            sys.executable, str(source / 'scripts/stream_hf_archive.py'),
            '--run-id', run_id, '--group-id', group, '--rank', str(rank),
            '--input', fifos[rank], '--staging-dir', str(staging),
            '--repo-id', config['repo_id'], '--branch', branches[rank],
            '--rows-per-shard', str(config['rows_per_shard']),
            '--slot-count', str(config['upload_slots']),
            '--max-slot-bytes', str(config['max_slot_bytes'])])
    return dict(env=env, fifos=fifos, branches=branches, consumers=consumers,
                commits=commits,
                upload_bytes=world*config['upload_slots']*config['max_slot_bytes'],
                search=['torchrun', '--standalone', f'--nproc-per-node={world}',
                        '--no-python', str(source / 'target/release/mgbfs'),
                        'bench', '--reference', group, str(config['batch']),
                        str(root / 'bootstrap'), str(prefix), '{RANK_OUT}'])
