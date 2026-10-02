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
import threading
from pathlib import Path

from bfs_tail_archive import atomic_json
from run_tail_bfs import run
from sweep_tail_bfs import automatic_pairs, execute


class SweepPublisher:
    """Coalesce only queued ledgers; completed case payloads stay immutable."""
    def __init__(self, root, repo, api, deadline, publish):
        self.root,self.repo,self.api,self.deadline,self.publish=root,repo,api,deadline,publish
        self.condition=threading.Condition()
        self.pending=None;self.closed=False;self.error=None;self.receipt=None;self.generations=0
        self.thread=threading.Thread(target=self._work,daemon=True);self.thread.start()

    def enqueue(self, ledger):
        frozen=json.loads(json.dumps(ledger))
        frozen['pending']=[pair for pair in frozen['configuration']['grid']
            if f'n{pair[0]}-m{pair[1]}' not in frozen['cases']]
        with self.condition:
            if self.error:raise RuntimeError('background sweep publication failed') from self.error
            if self.closed:raise RuntimeError('publisher already closed')
            self.pending=frozen;self.condition.notify()

    def _work(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.closed)
                    if self.pending is None:return
                    ledger=self.pending;self.pending=None
                self.receipt=self.publish(self.root,self.repo,self.api,self.deadline,ledger=ledger)
                self.generations+=1
        except BaseException as error:
            with self.condition:self.error=error;self.condition.notify_all()

    def finish(self):
        with self.condition:self.closed=True;self.condition.notify()
        self.thread.join()
        if self.error:raise RuntimeError('background sweep publication failed') from self.error
        return self.receipt


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
    # Use the existing batch-sized native scratch instead of emitting four
    # small archive frames per full batch. Pinned bytes still share the same
    # bounded host budget; no additional device allocation is introduced.
    rows=cfg['batch']
    host=base.get('host_available_bytes',8<<30)
    host_slots=(host//4)//(base.get('world',2)*(n+16)*rows)
    target_slots=(order+rows-1)//rows+2
    slots=min(8192,host_slots,max(64,target_slots))
    if slots<64:raise ValueError('insufficient host archive credit memory')
    cfg['env'].update(MGBFS_ARCHIVE_ROWS=str(rows),MGBFS_ARCHIVE_SLOTS=str(slots))
    return cfg


def tune_pair(base, source, case, runtime_env, *, query=None):
    """Native warmed admission, before creating any case archive or running BFS."""
    from vram_autotune import native_query, select_capacity
    n,r=base['n'],base['r']
    upper=max(256,min(math.factorial(n)//math.factorial(r),(2**31-1)//3))
    def configuration(rows):
        draft=json.loads(json.dumps(base))
        # The numeric pool is a compatible parser placeholder. Native autosize
        # replaces it with the queried worst-case bound before admission.
        draft['resource_plan'].update(max_rows_per_rank=rows,
            library_pool_bytes=max(64<<20,((rows*512+255)//256)*256))
        draft['env']['MGBFS_LIBRARY_POOL_AUTOSIZE']='1'
        return pair_config(draft,n,r)
    def probe(rows):
        cfg=configuration(rows)
        if query is not None:return query(cfg)
        return native_query(cfg,source,case.parent/(case.name+f'-query-{rows}'),runtime_env)
    rows,probes=select_capacity(probe,upper)
    cfg=configuration(rows)
    pools=[x.get('library_pool_bytes') for x in probes[rows]]
    if all(type(x) is int and x>0 for x in pools):
        # The native autosizer remains enabled; persist the actual bound as well
        # so selection/configuration metadata does not retain the old placeholder.
        cfg['env']['MGBFS_LIBRARY_POOL_BYTES']=str(max(pools))
    cfg['resource_plan']=dict(policy='native-warmed-admission-v2',
        selected_rows_per_rank=rows, max_rows_per_rank=rows,
        library_pool_bytes=int(cfg['env']['MGBFS_LIBRARY_POOL_BYTES']),
        probes=[dict(rows_per_rank=k,ranks=v) for k,v in sorted(probes.items())],
        pool_policy='native-cuco-extent-cub-query-worst-shard-history-plus-fragmentation-slack',
        maximum_hardware_capacity_proven=False)
    return cfg


def verify_hf(root, repo, api, token):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_url
    import requests
    from verify_tail_hf import verify_payload
    ledger = json.loads((root/'sweep.json').read_text())
    revision = api.repo_info(repo, repo_type='dataset').sha
    run_id = ledger['configuration']['base']['run_id']
    session=requests.Session()
    session.headers['Authorization']='Bearer '+token
    locals_=threading.local()
    pace_lock=threading.Lock()
    next_request=0.
    def get(client,url,**kwargs):
        nonlocal next_request
        for attempt in range(4):
            # Resolve URLs may involve two Hub requests. Keep the readback
            # below the observed 5000-resolves/300s budget, with spare capacity.
            with pace_lock:
                delay=max(0,next_request-time.monotonic())
                next_request=max(next_request,time.monotonic())+1/7
            if delay:time.sleep(delay)
            response=client.get(url,**kwargs)
            if response.status_code!=429:return response
            wait=min(300,max(1,int(response.headers.get('Retry-After','30'))))
            response.close()
            if attempt==3:raise RuntimeError('HF readback rate limit persists')
            time.sleep(wait)
        raise RuntimeError('HF readback unavailable')
    work = []
    for key, record in ledger['cases'].items():
        if not record.get('attempted'): continue
        from tail_parquet import publication_root
        saved=publication_root(root/key/'saved',
            ledger['configuration']['base'].get('archive_format','packed'))
        local = json.loads((saved/'manifest.json').read_text())
        prefix = 'tail-runs/'+run_id+'-'+key+'/'
        with get(session,hf_hub_url(repo, prefix+'manifest.json',
                repo_type='dataset', revision=revision),
                timeout=60) as response:
            response.raise_for_status()
            if response.json() != local: raise ValueError('HF manifest differs')
        work.extend((prefix, entry) for entry in local['files'])
    def check(item):
        prefix, entry = item
        if not hasattr(locals_,'session'):
            locals_.session=requests.Session()
            locals_.session.headers['Authorization']='Bearer '+token
        with get(locals_.session,hf_hub_url(repo, prefix+entry['path'],
                repo_type='dataset', revision=revision), stream=True,
                timeout=(30,120)) as response:
            return verify_payload(response, entry)
    with ThreadPoolExecutor(max_workers=8) as workers:
        size = sum(workers.map(check, work))
    # Read the complete ledger too: exclusions are part of the output contract.
    with get(session,hf_hub_url(repo, 'tail-sweeps/'+root.name+'/sweep.json',
            repo_type='dataset', revision=revision),
            timeout=60) as response:
        response.raise_for_status()
        if response.json() != ledger: raise ValueError('HF sweep ledger differs')
    return dict(revision=revision, files=len(work), bytes=size,
                manifests_verified=sum(x.get('attempted',False) for x in ledger['cases'].values()),
                all_checksums_verified=True, sweep_ledger_verified=True)


def main(cancelled=None):
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
        from streamed_bfs_launcher import available_host_bytes
        base=dict(world=len(inventory),run_id=args.root.name,timeout_seconds=120,
            archive_format='parquet',
            native_capacity_probe=True,
            host_available_bytes=available_host_bytes(),
            gpu_inventory=inventory,resource_plan=device_budget(inventory),env=dict(
                MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_PRE_DEDUP='ON',
                MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BUCKETS='16',MGBFS_SHARDS='8',
                MGBFS_JOB_BUCKETS='2',MGBFS_ARCHIVE_ROWS='8192',MGBFS_ARCHIVE_SLOTS='2048',
                NCCL_CUMEM_ENABLE='0'))
        atomic_json(config_path,base)
    runtime=json.loads(args.runtime_env.read_text())
    publisher=SweepPublisher(args.root,args.repo_id,api,args.deadline_unix,publish)
    last_published=0
    def progress(ledger):
        nonlocal last_published
        if publisher.error:raise RuntimeError('background publication failed') from publisher.error
        count=sum(x.get('attempted',False) for x in ledger['cases'].values())
        if count-last_published>=20:
            publisher.enqueue(ledger);last_published=count
    def adaptive(config,source,case,env):
        cfg=(tune_pair(config,source,case,env) if config.get('native_capacity_probe')
             else pair_config(config,config['n'],config['r']))
        return run(cfg,source,case,env,cancelled=cancelled)
    # Each layer writes its bounded intermediate snapshot immediately.
    # Completed cohorts publish in the background without per-layer quota storms.
    remaining=args.deadline_unix-time.time()
    publication_reserve=min(1200,remaining*.25)
    ledger=None
    try:
        ledger=execute(base,args.source,args.root,runtime,automatic_pairs(),
                       remaining-publication_reserve,adaptive,on_progress=progress,should_stop=cancelled)
        publisher.enqueue(ledger)
        receipt=publisher.finish()
        verified=verify_hf(args.root,args.repo_id,api,token)
    except BaseException as error:
        if not publisher.closed:
            try:publisher.finish()
            except BaseException:pass
        if ledger is None:
            path=args.root/'sweep.json'
            ledger=json.loads(path.read_text()) if path.exists() else dict(cases={})
            ledger['pending']=[list(pair) for pair in automatic_pairs()
                if f'n{pair[0]}-m{pair[1]}' not in ledger['cases']]
        atomic_json(args.root/'automatic-report.json',dict(status='INCOMPLETE',
            pending=ledger['pending'],stop_reason=ledger.get('global_stop_reason'),
            publication_status='FAILED',failure=str(error),local_snapshots_retained=True,
            background_publication_generations=publisher.generations))
        raise
    report=dict(status='VERIFIED',pending=ledger['pending'],publication=receipt,
        sweep_status='INCOMPLETE' if ledger['pending'] else 'COMPLETE',
        stop_reason=ledger.get('global_stop_reason'),
        verification=verified,resource_plan=base['resource_plan'],
        attempted=sum(x.get('attempted',False) for x in ledger['cases'].values()),
        complete=sum(x['status']=='COMPLETE' for x in ledger['cases'].values()),
        pruned=sum('pruned_by' in x for x in ledger['cases'].values()),
        background_publication_generations=publisher.generations)
    atomic_json(args.root/'automatic-report.json',report)
    api.upload_file(path_or_fileobj=str(args.root/'automatic-report.json'),
        repo_id=args.repo_id,repo_type='dataset',
        path_in_repo='tail-sweeps/'+args.root.name+'/automatic-report.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    from run_tail_bfs import cancellation_signals
    with cancellation_signals() as cancelled:main(cancelled)
