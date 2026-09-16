"""Physical H200 oracle/sanitizer gate. Does not rent or stop billing.

Pass the release library_multi_gpu test executable produced by cargo --no-run.
The separate eight-process CLI and full LRX13 gates remain mandatory.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time

TEST = 'library_eight_rank_layers_and_archives_match_oracle'
TOOLS = ('plain', 'memcheck', 'racecheck', 'initcheck', 'synccheck')


def validate_inventory(text):
    rows = list(csv.reader(io.StringIO(text.strip()), skipinitialspace=True))
    if len(rows) != 8 or any(len(row) != 5 for row in rows):
        raise ValueError('EIGHT_H200_REQUIRED')
    indices, uuids = set(), set()
    devices = []
    for index, name, uuid, total, free in rows:
        if ('H200' not in name.split() or not uuid.startswith('GPU-') or
                index in indices or uuid in uuids):
            raise ValueError('EIGHT_DISTINCT_H200_REQUIRED')
        index_number, total_number, free_number = int(index), int(total), int(free)
        if index_number < 0 or not 0 <= free_number <= total_number or total_number <= 0:
            raise ValueError('GPU_INVENTORY_VALUES')
        indices.add(index); uuids.add(uuid)
        devices.append(dict(index=index_number, name=name, uuid=uuid,
                            total_mib=total_number, free_mib=free_number))
    return devices


def validate_log(text, tool):
    results = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    if results != [('1', '0', '0')]:
        raise ValueError('ORACLE_DID_NOT_RUN_EXACTLY_ONCE')
    if tool == 'plain':
        return
    if tool not in TOOLS:
        raise ValueError('UNKNOWN_SANITIZER')
    errors = re.findall(r'ERROR SUMMARY: (\d+) errors', text)
    races = re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)', text)
    if any(int(n) for n in errors) or any(any(int(n) for n in row) for row in races):
        raise ValueError('SANITIZER_FINDINGS')
    if not errors and not (tool == 'racecheck' and races):
        raise ValueError('SANITIZER_SUMMARY_MISSING')


def run_command(command, log_path, env, timeout):
    from distributed_gpu_bench import stop_group
    if timeout <= 0:
        raise TimeoutError('EIGHT_GPU_GATE_DEADLINE')
    with log_path.open('wb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('EIGHT_GPU_GATE_DEADLINE') from error
            if code != 0:
                raise RuntimeError(f'GATE_COMMAND_FAILED exit={code} log={log_path}')
        finally:
            stop_group(child)
    return log_path.read_text(encoding='utf-8', errors='replace')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not args.executable.is_file():
        parser.error('positive timeout and existing executable required')
    args.logs.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.timeout_seconds
    env = {k:v for k,v in os.environ.items() if not k.startswith('MGBFS_')}
    report = dict(status='INCOMPLETE', scope='eight device threads; not torchrun CLI',
                  executable=str(args.executable.resolve()),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  stages=[])
    try:
        inventory = run_command(['nvidia-smi',
            '--query-gpu=index,name,uuid,memory.total,memory.free', '--format=csv,noheader,nounits'],
            args.logs/'inventory.log', env, deadline-time.monotonic())
        devices = validate_inventory(inventory)
        report['devices'] = devices
        env['CUDA_VISIBLE_DEVICES'] = ','.join(device['uuid'] for device in devices)
        run_command(['nvidia-smi', 'topo', '-m'], args.logs/'topology.log', env,
                    deadline-time.monotonic())
        run_command(['compute-sanitizer', '--version'], args.logs/'sanitizer-version.log', env,
                    deadline-time.monotonic())
        for tool in TOOLS:
            command = [str(args.executable.resolve()), TEST, '--ignored', '--exact',
                       '--nocapture', '--test-threads=1']
            if tool != 'plain':
                command = ['compute-sanitizer', '--tool', tool, '--error-exitcode', '97', *command]
            started = time.monotonic()
            text = run_command(command, args.logs/f'{tool}.log', env, deadline-started)
            validate_log(text, tool)
            report['stages'].append(dict(tool=tool, status='PASS', seconds=time.monotonic()-started))
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (args.logs/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
