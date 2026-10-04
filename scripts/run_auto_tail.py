"""One GPU-host command: automatic pair sweep, HF publication and readback.

The external platform must own the lease deadline. No provisioning here.
Capacity sizing is conservative, not a proof of maximum hardware capacity.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import threading
import shutil
from pathlib import Path

from bfs_tail_archive import atomic_json,TailArchive
from run_tail_bfs import run
from sweep_tail_bfs import automatic_pairs, execute
from paired_tail import publication_records, run_pair


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
                with self.condition:self.condition.notify_all()
        except BaseException as error:
            with self.condition:self.error=error;self.condition.notify_all()

    def wait_for_capacity(self, ledger, *, deadline, cancelled=None,
                          max_pending_bytes=10_000_000_000):
        """Case-boundary backpressure; never adds a GPU/batch synchronization.

        Count only final snapshot files still present on this host. The worker
        unlinks verified closed groups; producer metadata remains immutable.
        A whole case can overshoot the target before the next case is delayed.
        """
        if max_pending_bytes <= 0 or not math.isfinite(deadline):
            raise ValueError('positive pending-state bound and finite deadline required')
        while True:
            with self.condition:
                if self.error:
                    raise RuntimeError('background sweep publication failed') from self.error
            pending = 0
            for key, record in publication_records(ledger).items():
                if not record.get('attempted', True):
                    continue
                saved = Path(self.root)/key/'saved'
                manifest = json.loads((saved/'manifest.json').read_text())
                for entry in manifest['files']:
                    path = (saved/entry['path']).resolve()
                    if not path.is_relative_to(saved.resolve()):
                        raise ValueError('pending-state file escapes saved root')
                    try:
                        pending += path.stat().st_size
                    except FileNotFoundError:
                        # A durable release receipt precedes the worker's unlink.
                        # Payload planning/readback independently validates that receipt.
                        pass
            if pending <= max_pending_bytes:
                return True
            if time.time() >= deadline or (cancelled and cancelled()):
                return False
            with self.condition:
                self.condition.wait(timeout=min(1, max(.01, deadline-time.time())))

    def finish(self):
        with self.condition:self.closed=True;self.condition.notify()
        self.thread.join()
        if self.error:raise RuntimeError('background sweep publication failed') from self.error
        return self.receipt


class EndUploadDiskAdmission:
    """Reserve physical SSD space for the next tail and final Parquet conversion.

    Only newly completed cases are read; there is no per-layer disk polling.
    This is a conservative storage estimate, not a pair/state-count limit.
    """
    def __init__(self, root):
        self.root=Path(root);self.seen=set();self.retained=0;self.largest_layer=0
        self.fixed_layers=None;self.repetitions=1

    def observe(self, ledger):
        self.repetitions=2 if ledger.get('configuration',{}).get('base',{}).get('two_seeds') else 1
        self.fixed_layers=ledger.get('configuration',{}).get('base',{}).get('retained_layers')
        for key,record in publication_records(ledger).items():
            if key in self.seen or not record.get('attempted',False):continue
            manifest=json.loads((self.root/key/'saved/manifest.json').read_text())
            self.retained+=sum(entry['bytes'] for entry in manifest['files']
                if (self.root/key/'saved'/entry['path']).exists())
            width=manifest['packing']['bytes_per_state']
            self.largest_layer=max(self.largest_layer,
                max((layer['states']*width for layer in manifest['layers']),default=0))
            self.seen.add(key)

    def required_free_bytes(self):
        # Parquet row metadata and a full next-case working tail/spools coexist
        # with retained packed data until verified publication can release it.
        tail=(self.fixed_layers*self.largest_layer if self.fixed_layers is not None
              else max(10_000_000_000,3*self.largest_layer))
        return (2*self.retained+self.repetitions*tail
                +2*self.largest_layer+1_000_000_000+(64<<20))

    def stop_reason(self):
        free=shutil.disk_usage(self.root).free;required=self.required_free_bytes()
        if free<required:
            return f'SSD admission stopped compute for final upload: free={free} required={required}'
        return None


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


def automatic_reserve(runtime, environ=None):
    """Startup-only reserve; preserve explicit settings and native floor."""
    environ=os.environ if environ is None else environ
    value=runtime.get('MGBFS_VRAM_RESERVE_BYTES',environ.get(
        'MGBFS_VRAM_RESERVE_BYTES',str(256<<20)))
    try:bytes_=int(value)
    except (TypeError,ValueError):raise ValueError('invalid automatic VRAM reserve')
    if not (64<<20)<=bytes_<=2**64-1:
        raise ValueError('invalid automatic VRAM reserve')
    return str(bytes_)


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


def tune_pair(base, source, case, runtime_env, *, query=None, deadline=None, cancelled=None):
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
        if (deadline is not None and deadline<=time.time()) or (cancelled and cancelled()):
            raise TimeoutError('startup capacity probe deadline or cancellation')
        cfg=configuration(rows)
        if query is not None:return query(cfg)
        timeout=90 if deadline is None else min(90,max(.1,deadline-time.time()))
        return native_query(cfg,source,case.parent/(case.name+f'-query-{rows}'),runtime_env,timeout=timeout)
    from resident_session import active_session
    session = active_session() if query is None else None
    # r changes the orbit bound/start word, not native byte-state geometry.
    # Keep admission evidence per n and launch policy inside this rank session.
    profile_key = json.dumps(dict(n=n, world=base['world'], env=base['env'],
        margin=base.get('startup_memory_margin_bytes',64<<20),runtime=runtime_env),sort_keys=True)
    cached = session.capacity_profiles.get(profile_key) if session else None
    reused = (cached is not None and cached[0] <= upper
              and (cached[0] < cached[2] or upper <= cached[2]))
    if reused:
        rows,probes=cached[:2]
    else:
        rows,probes=select_capacity(probe,upper)
        if session is not None and (cached is None or upper > cached[2]):
            session.capacity_profiles[profile_key]=(rows,probes,upper)
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
    cfg['resource_plan']['resident_admission_reused'] = reused
    return cfg


def startup_failure_snapshot(config, source, case, reason):
    """Publish honest empty INCOMPLETE metadata when no BFS layer was started."""
    source,case=Path(source).resolve(),Path(case)
    binary=(source/config.get('binary_path','target/release/mgbfs')).resolve()
    if not binary.is_relative_to(source):raise ValueError('binary outside source checkout')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    cfg=json.loads(json.dumps(config))
    cfg.update(search_started=False,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        probe_configurations=[json.loads(p.read_text()) for p in sorted(
            case.parent.glob(case.name+'-query-*/query-config.json'))])
    case.mkdir(parents=True,exist_ok=False)
    n,r=config['n'],config['r']
    archive=TailArchive(case/'saved',n=n,r=r,start=list(range(n-r+1))+[n-r]*(r-1),
        actions=dict(L='cyclic left rotation',R='cyclic right rotation',X='swap positions 0 and 1'),
        program_commit=commit,launch_config=cfg,sample_interval_seconds=.05)
    return archive.snapshot(False,reason)


def run_adaptive(config, source, case, runtime, *, deadline, cancelled=None):
    """Admission and optional matched calibration happen before the real BFS."""
    adaptive_started=time.monotonic()
    try:
        cfg=(tune_pair(config,source,case,runtime,deadline=deadline,cancelled=cancelled)
             if config.get('native_capacity_probe') else pair_config(config,config['n'],config['r']))
    except Exception as error:
        return startup_failure_snapshot(config,source,case,'startup capacity probe failed: '+str(error))
    admission_finished=time.monotonic()
    cfg['automatic_phase_seconds']=dict(admission=admission_finished-adaptive_started,graph_calibration=0.0)
    explicit_graph=runtime.get('MGBFS_CUDA_GRAPH_BATCHES',os.environ.get('MGBFS_CUDA_GRAPH_BATCHES'))
    if explicit_graph is not None:
        cfg['env'].setdefault('MGBFS_CUDA_GRAPH_BATCHES',explicit_graph)
    if (config.get('graph_calibration', True)
            and cfg['env'].get('MGBFS_TRANSPORT_BACKEND') == 'NCCL_LSA'
            and 'MGBFS_CUDA_GRAPH_BATCHES' not in cfg['env']):
        from calibrate_graph_profile import calibrate
        remaining=deadline-time.time()
        from bfs_tail_archive import packed_width
        width = packed_width(cfg['n'], cfg['n']-cfg['r']+1)
        comparison_bytes = min(512<<20, cfg.get('host_available_bytes',8<<30)//16)
        order = math.factorial(cfg['n'])//math.factorial(cfg['r'])
        # Large admitted buffers and prefix verification took several minutes
        # in the measured n15/r4 gate. A 90-second ceiling cannot finish its six
        # repeats; preserve the 10% fraction but permit a bounded 600s prefix gate.
        ceiling = 600 if order*width > comparison_bytes else 90
        allowance = min(ceiling,max(0,remaining*.1))
        calibration_deadline=time.time()+allowance
        calibration_started=time.monotonic()
        decision=calibrate(cfg,source,case.parent/(case.name+'-graph-calibration'),runtime,
            deadline=calibration_deadline,cancelled=cancelled)
        cfg['automatic_phase_seconds']['graph_calibration']=time.monotonic()-calibration_started
        decision['startup_budget_seconds']=allowance
        cfg['graph_profile_selection']=decision
        cfg['env']['MGBFS_CUDA_GRAPH_BATCHES']=str(decision['graph_batches'])
    remaining=deadline-time.time()
    if remaining<=0 or (cancelled and cancelled()):
        return startup_failure_snapshot(cfg,source,case,'automatic run deadline or cancellation before BFS')
    # A pair has no independent time limit: only the externally authorized
    # compute deadline/cancellation can stop the traversal and archive drain.
    cfg['timeout_seconds']=remaining
    return run(cfg,source,case,runtime,cancelled=cancelled)


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
    from publish_tail_batch import publication_cases
    publications = publication_cases(root, ledger)
    seen_payloads = {}
    for key, record in publication_records(ledger).items():
        if not record.get('attempted'): continue
        saved, manifest_path = publications[key]
        local = json.loads(manifest_path.read_text())
        prefix = 'tail-runs/'+run_id+'-'+key+'/'
        with get(session,hf_hub_url(repo, prefix+'manifest.json',
                repo_type='dataset', revision=revision),
                timeout=60) as response:
            response.raise_for_status()
            if response.json() != local: raise ValueError('HF manifest differs')
        for entry in local['files']:
            remote = entry.get('repo_path', prefix+entry['path'])
            signature = (entry['bytes'], entry['sha256'])
            if remote in seen_payloads:
                if seen_payloads[remote] != signature:
                    raise ValueError('conflicting shared HF payload')
                continue
            seen_payloads[remote] = signature
            work.append((remote, entry))
    def check(item):
        remote, entry = item
        if not hasattr(locals_,'session'):
            locals_.session=requests.Session()
            locals_.session.headers['Authorization']='Bearer '+token
        with get(locals_.session,hf_hub_url(repo, remote,
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
                manifests_verified=sum(x.get('attempted',False) for x in publication_records(ledger).values()),
                all_checksums_verified=True, sweep_ledger_verified=True)


def main(cancelled=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--two-seeds',action='store_true')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--runtime-env',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--upload-mode',choices=('search','graph','end','background'),default=None,
        help='search: live Parquet snapshots; graph: after each graph; end (default): SSD then upload/resume cycles; background: completed cohorts in parallel')
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--retained-layers',type=int,default=None,
        help='retain only this many final whole completed layers, for COMPLETE and INCOMPLETE; no byte target')
    p.add_argument('--retention-policy',choices=('last_complete_small_1000',),default=None)
    args = p.parse_args()
    if args.retained_layers is not None and args.retained_layers <= 0:
        p.error('--retained-layers must be positive')
    if args.retention_policy and args.retained_layers is not None:
        p.error('choose a retention policy or fixed layers')
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
        runtime=json.loads(args.runtime_env.read_text())
        base=dict(world=len(inventory),run_id=args.root.name,
            upload_mode=args.upload_mode or 'end',
            archive_format='parquet_cohort',cohort_group_size=1 if args.upload_mode=='graph' else 20,cohort_group_bytes=2_000_000_000,
            native_capacity_probe=True,
            host_available_bytes=available_host_bytes(),
            gpu_inventory=inventory,resource_plan=device_budget(inventory),env=dict(
                MGBFS_VRAM_RESERVE_BYTES=automatic_reserve(runtime),
                MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_PRE_DEDUP='ON',
                MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BUCKETS='16',MGBFS_SHARDS='8',
                MGBFS_JOB_BUCKETS='2',MGBFS_ARCHIVE_ROWS='8192',MGBFS_ARCHIVE_SLOTS='2048',
                NCCL_CUMEM_ENABLE='0'))
        from automatic_transport import select_transport
        runtime=json.loads(args.runtime_env.read_text())
        base=select_transport(base,args.source,args.root/'transport-gate',runtime,
            deadline=min(args.deadline_unix-120,time.time()+60),cancelled=cancelled)
        atomic_json(config_path,base)
    if args.retained_layers is not None:
        if base.get('retained_layers') not in (None,args.retained_layers):
            p.error('resume retained layer count differs from saved configuration')
        if 'retained_layers' not in base and (args.root/'sweep.json').exists():
            p.error('changing retention requires a new run root')
        base['retained_layers']=args.retained_layers
        atomic_json(config_path,base)
    if args.retention_policy:
        if (args.root/'sweep.json').exists() and base.get('retention_policy') != args.retention_policy:
            p.error('changing retention requires a new run root')
        base['retention_policy']=args.retention_policy
        atomic_json(config_path,base)
    mode=args.upload_mode or base.get('upload_mode','end')
    if base.get('upload_mode',mode)!=mode:
        p.error('resume upload mode differs from saved configuration; use a new run root')
    if args.two_seeds:
        if mode != 'end': p.error('two-seed validation currently requires end upload mode')
        if (args.root/'sweep.json').exists() and not base.get('two_seeds'): p.error('two-seed mode requires a new run root')
        base['two_seeds']=True
        atomic_json(config_path,base)
    runtime=json.loads(args.runtime_env.read_text())
    disk_admission=EndUploadDiskAdmission(args.root)
    previous_ledger=args.root/'sweep.json'
    if disk_admission and previous_ledger.exists():
        disk_admission.observe(json.loads(previous_ledger.read_text()))
    def compute_stop():
        return ((cancelled() if cancelled else None)
                or (disk_admission.stop_reason() if disk_admission else None))
    def publish_and_release(root, repo, api, deadline, *, ledger):
        from release_tail_cohorts import release
        receipt = publish(root, repo, api, deadline, ledger=ledger)
        receipt['local_release'] = release(root, ledger, repo, api, token, deadline=deadline)
        return receipt
    publisher=SweepPublisher(args.root,args.repo_id,api,args.deadline_unix,publish_and_release)
    last_published=0
    def progress(ledger):
        nonlocal last_published,publisher
        if publisher.error:raise RuntimeError('background publication failed') from publisher.error
        disk_admission.observe(ledger)
        if mode=='end':return
        if mode=='graph':
            publisher.enqueue(ledger);publisher.finish()
            publisher=SweepPublisher(args.root,args.repo_id,api,
                args.deadline_unix,publish_and_release)
            return
        from tail_cohort import case_groups
        sealed_count = sum(len(members) for members, sealed in case_groups(args.root, ledger) if sealed)
        if sealed_count > last_published:
            publisher.enqueue(ledger);last_published=sealed_count
        if not publisher.wait_for_capacity(ledger,
                deadline=args.deadline_unix-publication_reserve,cancelled=cancelled):
            ledger['global_stop_reason']='publication backlog at compute deadline or cancellation'
    def adaptive(config,source,case,env):
        if mode=='search':
            config=dict(config,repo_id=args.repo_id,live_upload_format='parquet',
                live_upload_prefix='tail-live/'+config['run_id'])
        def one(cfg,src,destination,runtime):
            return run_adaptive(cfg,src,destination,runtime,
                deadline=args.deadline_unix-publication_reserve,cancelled=cancelled)
        if base.get('two_seeds'):
            return run_pair(config,source,case,env,one,startup_failure_snapshot)
        return one(config,source,case,env)
    # Each layer still writes an intermediate SSD snapshot. End mode uploads
    # at a physical-storage boundary and resumes the same immutable ledger.
    ledger=None;upload_cycles=[]
    try:
        from resident_session import resident
        import uuid
        with resident(args.root/('resident-'+uuid.uuid4().hex)):
            while True:
                remaining=args.deadline_unix-time.time()
                publication_reserve=min(1200,remaining*.25)
                compute_started=time.time()
                ledger=execute(base,args.source,args.root,runtime,automatic_pairs(),
                    max(.001,remaining-publication_reserve),adaptive,
                    on_progress=progress,should_stop=compute_stop)
                compute_finished=time.time()
                from tail_cohort import seal_current_cohort
                seal_current_cohort(args.root,ledger)
                atomic_json(args.root/'sweep.json',ledger)
                publisher.enqueue(ledger)
                receipt=publisher.finish()
                upload_cycles.append(dict(compute_started_at=compute_started,
                    compute_finished_at=compute_finished,upload_finished_at=time.time(),
                    stop_reason=ledger.get('global_stop_reason'),receipt=receipt))
                atomic_json(args.root/'upload-cycles.json',upload_cycles)
                reason=ledger.get('global_stop_reason','')
                if (not ledger['pending']
                        or not (reason.startswith('SSD admission stopped compute')
                                or reason.startswith('SSD full'))
                        or (cancelled and cancelled())
                        or args.deadline_unix-time.time()<300):
                    break
                disk_admission=EndUploadDiskAdmission(args.root)
                disk_admission.observe(ledger)
                if disk_admission.stop_reason():
                    ledger['global_stop_reason']='SSD admission cannot resume after verified upload'
                    atomic_json(args.root/'sweep.json',ledger)
                    break
                publisher=SweepPublisher(args.root,args.repo_id,api,
                    args.deadline_unix,publish_and_release)
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
            upload_mode=mode,background_publication_generations=publisher.generations))
        raise
    report=dict(status='VERIFIED',pending=ledger['pending'],publication=receipt,
        upload_cycles=upload_cycles,
        sweep_status='INCOMPLETE' if ledger['pending'] else 'COMPLETE',
        stop_reason=ledger.get('global_stop_reason'),
        verification=verified,resource_plan=base['resource_plan'],
        attempted=sum(x.get('attempted',False) for x in ledger['cases'].values()),
        complete=sum(x['status']=='COMPLETE' for x in ledger['cases'].values()),
        pruned=sum('pruned_by' in x for x in ledger['cases'].values()),
        upload_mode=mode,background_publication_generations=publisher.generations)
    atomic_json(args.root/'automatic-report.json',report)
    api.upload_file(path_or_fileobj=str(args.root/'automatic-report.json'),
        repo_id=args.repo_id,repo_type='dataset',
        path_in_repo='tail-sweeps/'+args.root.name+'/automatic-report.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    from run_tail_bfs import cancellation_signals
    with cancellation_signals() as cancelled:main(cancelled)
