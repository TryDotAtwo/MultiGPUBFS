"""Platform-independent launch plan for native BFS with bounded HF consumers.

No provisioning or billing control is performed here. The caller must arrange
an independent rental deadline before launching any paid workload.
"""
import re
import sys
import json
import subprocess
import time
from contextlib import ExitStack


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
            '--repo-id', config['repo_id'], '--branch', branches[rank], '--create-branch',
            '--rows-per-shard', str(config['rows_per_shard']),
            '--slot-count', str(config['upload_slots']),
            '--max-slot-bytes', str(config['max_slot_bytes'])])
    return dict(env=env, fifos=fifos, branches=branches, consumers=consumers,
                commits=commits,
                promotion=[sys.executable, str(source / 'scripts/promote_hf_stream.py'),
                           '--repo-id', config['repo_id'], '--world-size', str(world),
                           '--reference', str(config.get('reference',
                               source / 'data/reference/lrx13-layers.json')), *commits],
                upload_bytes=world*config['upload_slots']*config['max_slot_bytes'],
                search=['torchrun', '--standalone', f'--nproc-per-node={world}',
                        '--no-python', str(source / 'target/release/mgbfs'),
                        'bench', '--reference', group, str(config['batch']),
                        str(root / 'bootstrap'), str(prefix), '{RANK_OUT}'])


def execute_plan(plan, logs, env, timeout_seconds):
    """One deadline for search, draining consumers and atomic publication.

    Caller preflights resources and creates FIFOs. A timeout stops processes,
    not rental billing. Failed staging branches are retained for diagnosis.
    """
    from distributed_gpu_bench import run_group, stop_group
    deadline = time.monotonic() + timeout_seconds

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('STREAM_RUN_DEADLINE')
        return value

    children = []
    with ExitStack() as files:
        try:
            for rank, command in enumerate(plan['consumers']):
                remaining()
                output = files.enter_context((logs / f'consumer-{rank}.log').open('wb'))
                children.append(subprocess.Popen(command, env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True))
            result = run_group(plan['search'], logs, 'search', env,
                               timeout=remaining(), required_processes=children)
            if result['status'] != 'COMPLETE':
                raise RuntimeError('SEARCH_' + result['status'])
            while True:
                codes = [child.poll() for child in children]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError(f'CONSUMER_FAILED {codes}')
                if all(code == 0 for code in codes):
                    break
                time.sleep(min(0.1, remaining()))
            remaining()
            output = files.enter_context((logs / 'promotion.log').open('wb'))
            publication = subprocess.Popen(plan['promotion'], env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            children.append(publication)
            try:
                code = publication.wait(timeout=remaining())
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('STREAM_PUBLICATION_DEADLINE') from error
            output.flush()
            if code != 0:
                raise RuntimeError(f'PROMOTION_FAILED {code}')
            receipt = json.loads((logs / 'promotion.log').read_text(encoding='utf-8').splitlines()[-1])
            if receipt.get('status') != 'COMPLETE' or not receipt.get('commit_url'):
                raise RuntimeError('PROMOTION_RECEIPT')
            result['publication'] = receipt
            (logs / 'stream-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            return result
        finally:
            for child in children:
                stop_group(child)
