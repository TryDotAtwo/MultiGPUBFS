"""Bounded build and two-rank oracle on the isolated rental."""
import json
import os
import subprocess
from pathlib import Path

root = Path('/root/tail-src')
build = Path('/root/tail-build')
env = dict(os.environ, **json.loads((build/'runtime-env.json').read_text()))
nccl = '/venv/main/lib/python3.12/site-packages/nvidia/nccl'
env.update(CPATH=nccl+'/include', LIBRARY_PATH=nccl+'/lib',
           LD_LIBRARY_PATH=nccl+'/lib:'+env['LD_LIBRARY_PATH'])
out = Path('/root/tail-gate')
out.mkdir(exist_ok=True)
commands = [
    ('cli-build', ['cargo','build','--locked','--release','-p','mgbfs-cli','--features','library-owner']),
    ('multiset-oracle', ['cargo','test','--locked','--release','-p','mgbfs-runtime',
        '--features','cuda,library-owner','--test','lrx_multiset_gpu',
        'lrx_multiset_two_rank_cuco_full_state_oracle','--','--ignored','--nocapture'])]
for label, command in commands:
    print('STAGE '+label, flush=True)
    with (out/(label+'.log')).open('w') as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise SystemExit(result.returncode)
print('TWO_RANK_ORACLE_PASS', flush=True)
