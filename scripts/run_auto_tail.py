"""One GPU-host command: automatic pair sweep, HF publication and readback.

The external platform must own the lease deadline. No provisioning here.
Capacity sizing is conservative, not a proof of maximum hardware capacity.
"""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from bfs_tail_archive import atomic_json
from run_tail_bfs import run
from sweep_tail_bfs import automatic_pairs, execute


def device_budget(inventory):
    if len(inventory) not in (1, 2, 4, 8) or any(x['free_bytes'] <= 0 for x in inventory):
        raise ValueError('unsupported or unavailable GPU topology')
    free = min(x['free_bytes'] for x in inventory)
    # Reserve space for CUDA/NCCL and leave headroom above the conservative
    # per-row allowance. Native allocation failures remain explicit evidence.
    usable = int(free * .65)
    if usable < 512 << 20:
        raise ValueError('insufficient free device memory')
    pool = max(64 << 20, usable // 3)
    rows = min((2**32-1)//2, (usable-pool)//2048)
    if rows < 256:
        raise ValueError('insufficient free device memory')
    return dict(policy='conservative-free-vram-v1', free_bytes=free,
                usable_bytes=usable, library_pool_bytes=pool,
                max_rows_per_rank=rows, bytes_per_row_allowance=2048,
                maximum_hardware_capacity_proven=False)


def pair_config(base, n, r):
    cfg = json.loads(json.dumps(base))
    order = math.factorial(n)//math.factorial(r)
    capacity = max(256, min(order, base['resource_plan']['max_rows_per_rank']))
    pool = min(base['resource_plan']['library_pool_bytes'],
               max(64 << 20, capacity*512))
    cfg['env'].update(MGBFS_BENCH_CAPACITY=str(capacity),
        MGBFS_FUTURE_CAPACITY=str(capacity*2), MGBFS_BUCKET_CAPACITY=str(capacity),
        MGBFS_LIBRARY_POOL_BYTES=str(pool))
    cfg['batch'] = min(32768, capacity)
    return cfg


def verify_hf(root, repo, api, token):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_url
    import requests
    from verify_tail_hf import verify_payload
    ledger = json.loads((root/'sweep.json').read_text())
    revision = api.repo_info(repo, repo_type='dataset').sha
    run_id = ledger['configuration']['base']['run_id']
    work = []
    for key, record in ledger['cases'].items():
        if not record.get('attempted'): continue
        local = json.loads((root/key/'saved/manifest.json').read_text())
        prefix = 'tail-runs/'+run_id+'-'+key+'/'
        with requests.get(hf_hub_url(repo, prefix+'manifest.json',
                repo_type='dataset', revision=revision),
                headers={'Authorization':'Bearer '+token}, timeout=60) as response:
            response.raise_for_status()
            if response.json() != local: raise ValueError('HF manifest differs')
        work.extend((prefix, entry) for entry in local['files'])
    def check(item):
        prefix, entry = item
        with requests.get(hf_hub_url(repo, prefix+entry['path'],
                repo_type='dataset', revision=revision), stream=True,
                headers={'Authorization':'Bearer '+token}, timeout=(30,120)) as response:
            return verify_payload(response, entry)
    with ThreadPoolExecutor(max_workers=4) as workers:
        size = sum(workers.map(check, work))
    # Read the complete ledger too: exclusions are part of the output contract.
    with requests.get(hf_hub_url(repo, 'tail-sweeps/'+root.name+'/sweep.json',
            repo_type='dataset', revision=revision),
            headers={'Authorization':'Bearer '+token}, timeout=60) as response:
        response.raise_for_status()
        if response.json() != ledger: raise ValueError('HF sweep ledger differs')
    return dict(revision=revision, files=len(work), bytes=size,
                manifests_verified=sum(x.get('attempted',False) for x in ledger['cases'].values()),
                all_checksums_verified=True, sweep_ledger_verified=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--runtime-env',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    args = p.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix-time.time()<300:
        p.error('finite deadline with at least 300 seconds remaining required')
    if os.name!='posix': p.error('GPU execution requires Linux')
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi, get_token
    from publish_tail_batch import publish
    token = get_token()
    if not token: p.error('HF write credential required')
    api = HfApi(token=token)
    args.root.mkdir(parents=True,exist_ok=True)
    config_path=args.root/'automatic-config.json'
    if config_path.exists():
        base=json.loads(config_path.read_text())
    else:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        inventory=[]
        for line in lines:
            index,name,free=line.split(',')
            inventory.append(dict(index=int(index),name=name.strip(),free_bytes=int(free)*1024**2))
        base=dict(world=len(inventory),run_id=args.root.name,timeout_seconds=120,
            gpu_inventory=inventory,resource_plan=device_budget(inventory),env=dict(
                MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_PRE_DEDUP='ON',
                MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BUCKETS='16',MGBFS_SHARDS='8',
                MGBFS_JOB_BUCKETS='2',MGBFS_ARCHIVE_ROWS='8192',MGBFS_ARCHIVE_SLOTS='2048',
                NCCL_CUMEM_ENABLE='0'))
        atomic_json(config_path,base)
    runtime=json.loads(args.runtime_env.read_text())
    def adaptive(config,source,case,env):
        return run(pair_config(config,config['n'],config['r']),source,case,env)
    # Each layer still writes its bounded intermediate snapshot immediately.
    # The final GPU-host bulk publisher avoids per-layer HF commit quota storms.
    remaining=args.deadline_unix-time.time()
    publication_reserve=min(1200,remaining*.25)
    ledger=execute(base,args.source,args.root,runtime,automatic_pairs(),
                   remaining-publication_reserve,adaptive)
    receipt=publish(args.root,args.repo_id,api,args.deadline_unix)
    verified=verify_hf(args.root,args.repo_id,api,token)
    report=dict(status='VERIFIED',pending=ledger['pending'],publication=receipt,
        verification=verified,resource_plan=base['resource_plan'],
        attempted=sum(x.get('attempted',False) for x in ledger['cases'].values()),
        complete=sum(x['status']=='COMPLETE' for x in ledger['cases'].values()),
        pruned=sum('pruned_by' in x for x in ledger['cases'].values()))
    atomic_json(args.root/'automatic-report.json',report)
    api.upload_file(path_or_fileobj=str(args.root/'automatic-report.json'),
        repo_id=args.repo_id,repo_type='dataset',
        path_in_repo='tail-sweeps/'+args.root.name+'/automatic-report.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
