"""Startup-only capacity admission from native warmed allocation queries.

All probes are isolated native subprocesses; no sampling enters the BFS loop.
Query failure is a startup error, never a graph resource-pruning observation.
"""
import json
import os
import signal
import subprocess
from pathlib import Path


def select_capacity(query, upper, *, minimum=32768):
    """Find largest admitted capacity. Exact queries correct affine prediction."""
    minimum = min(minimum, upper)
    cache = {}

    def probe(rows):
        if rows not in cache:
            records = query(rows)
            if not records:
                raise ValueError('missing native memory query records')
            for record in records:
                for key in ('required_bytes', 'reserve_bytes', 'free_after_nccl_warmup_bytes'):
                    if type(record.get(key)) is not int or record[key] < 0:
                        raise ValueError('invalid native memory query')
            cache[rows] = records
        return cache[rows]

    def fits(rows):
        return all(x['required_bytes'] + x['reserve_bytes'] <=
                   x['free_after_nccl_warmup_bytes'] for x in probe(rows))

    if not fits(minimum):
        raise ValueError('minimum fast batch does not fit warmed VRAM')
    if upper == minimum:
        return minimum, cache
    second = min(upper, minimum * 2)
    first_records, second_records = probe(minimum), probe(second)
    if len(first_records) != len(second_records):
        raise ValueError('memory query rank count changed')
    prediction = upper
    for first, other in zip(first_records, second_records):
        slope = (other['required_bytes'] - first['required_bytes']) / (second - minimum)
        if slope <= 0:
            raise ValueError('non-increasing native allocation plan')
        free = min(first['free_after_nccl_warmup_bytes'], other['free_after_nccl_warmup_bytes'])
        room = free - max(first['reserve_bytes'], other['reserve_bytes']) - first['required_bytes']
        prediction = min(prediction, minimum + int(room / slope))
    candidate = max(minimum, min(upper, prediction))
    low, high = minimum, upper + 1
    if fits(candidate):
        low = candidate
        if candidate == upper:
            return low, cache
        # Plans are almost affine: test the immediately adjacent row first.
        if not fits(candidate + 1):
            return low, cache
        low = candidate + 1
    else:
        high = candidate
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low, cache


def native_query(config, source, root, runtime_env, *, timeout=90):
    """Stop native construction after actual NCCL warmup, before large allocs."""
    source, root = Path(source).resolve(), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    world = config['world']
    cli = (source / config.get('binary_path', 'target/release/mgbfs')).resolve()
    if not cli.is_relative_to(source):
        raise ValueError('query binary must belong to checkout')
    env = dict(os.environ, **runtime_env)
    env.update(config['env'])
    env.update(MGBFS_MEMORY_QUERY='1', MGBFS_BENCH_SKIP_ARCHIVE='1',
        MGBFS_ARCHIVE_STREAM='0', MGBFS_BENCH_WARMUP='0',
        MGBFS_BENCH_WORLD_SIZE=str(world), MGBFS_RANK_MAP=','.join(map(str, range(world))),
        MGBFS_STATE_CODEC='permutation_u8', MGBFS_ARCHIVE_CODEC='permutation_u8')
    command = ['torchrun', '--standalone', f'--nproc-per-node={world}', '--no-python',
        str(cli), 'bench', '--reference', f"lrx{config['n']}r{config['r']}",
        str(config['batch']), str(root/'bootstrap'), str(root/'archive'), str(root/'result')]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    (root/'query.log').write_text(output, encoding='utf-8')
    prefix = 'MGBFS_MEMORY_QUERY '
    records = [json.loads(line.split(prefix, 1)[1]) for line in output.splitlines()
               if prefix in line]
    if (len(records) != world or 'MEMORY_QUERY_DONE' not in output or
            sorted(x.get('rank', -1) for x in records) != list(range(world))):
        raise RuntimeError('native memory query failed; inspect '+str(root/'query.log'))
    return sorted(records, key=lambda x: x['rank'])
