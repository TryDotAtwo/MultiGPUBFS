"""Automatic supported (n,r) sweep; legacy m arguments map to native r."""
import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
from bfs_tail_archive import atomic_json
from run_tail_bfs import run


def pairs(n_min=2,n_max=128,m_min=1,m_max=None):
    if not 2<=n_min<=n_max<=128 or m_min<1 or (m_max is not None and m_max<m_min):
        raise ValueError('invalid finite sweep bounds')
    return [(n,m) for n in range(n_min,n_max+1)
            for m in range(m_min,min(n,m_max if m_max is not None else n)+1)]


def unsupported_reason(n,r):
    if not 2<=n<=128 or not 1<=r<=n:return 'outside supported n/r domain'
    return None


def automatic_pairs(n_min=2,n_max=128,r_min=1,r_max=None):
    # Ascend n within every independent fixed-r branch.
    return pairs(n_min,n_max,r_min,r_max)


def resource_stop(record):
    if record.get('status')!='INCOMPLETE' or not record.get('attempted',False):return False
    if record.get('replicas') and not all(resource_stop(replica) for replica in record['replicas']):return False
    reason=record.get('reason','').lower()
    if 'no space left on device' in reason:return False
    if record.get('resource_classification') in ('cuda_allocation_failure','confirmed_shard_capacity'):return True
    # cuda/state_commit.cu uses sticky code 16 for layer/request capacity.
    # Other rank-depth codes and remote cancellation alone are not evidence.
    if re.search(r'\blibrary_rank_depth_fatal_(?:16_(?:0|16)|0_16)\b',reason):return True
    # Source-confirmed search-ring row/descriptor/layer capacity, not a
    # generic ring failure (protocol/transport codes must remain eligible).
    if re.search(r'\bgroup_state_ring_retire_fatal_(?:11|12|16)\b',reason):return True
    return any(marker in reason for marker in (
        'cuda_error_out_of_memory','cudaerrormemoryallocation',
        'cuda out of memory','cuda error: out of memory',
        'search capacity exceeded','search capacity exhausted',
        'gpu capacity exceeded','gpu capacity exhausted'))


def allocation_failure(case,source):
    """Status 2 alone is ambiguous: confirm its source is a CUDA allocation."""
    log=case/'native.log'
    if not log.exists():return False
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-65536))
        tail=stream.read().decode('utf-8',errors='replace')
    source=source.resolve()
    for file,line in re.findall(r'MGBFS_NATIVE_STATUS status=2 file=(\S+) line=(\d+)',tail):
        path=(source/file).resolve()
        if not path.is_relative_to(source) or not path.is_file():continue
        lines=path.read_text(encoding='utf-8').splitlines()
        index=int(line)-1
        if 0<=index<len(lines) and any(name in lines[index] for name in (
                'cudaMalloc(', 'cudaMallocAsync(')):
            return True
    return False


def execute(base,source,root,runtime,grid,deadline_seconds,runner=run, *, on_progress=None, should_stop=None):
    if deadline_seconds<=0:
        raise ValueError('positive sweep deadline required')
    root.mkdir(parents=True,exist_ok=True)
    ledger_path=root/'sweep.json'
    fingerprint=dict(base=base,grid=grid,pruning_policy='fixed-r-resource-stop-v3')
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        if ledger['configuration']!=json.loads(json.dumps(fingerprint)):
            raise ValueError('resume configuration differs')
    else:
        ledger=dict(configuration=fingerprint,cases={})
    # A storage/upload cycle resumes eligible pairs; its previous stop reason
    # is historical and must not label the new pass.
    ledger.pop('global_stop_reason',None)
    # Classify structural exclusions even when the compute deadline expires.
    for n,m in grid:
        reason=unsupported_reason(n,m)
        key=f'n{n}-m{m}'
        if reason and key not in ledger['cases']:
            ledger['cases'][key]=dict(status='INCOMPLETE',last_completed_layer=-1,
                reason=reason,attempted=False,n=n,m=m)
    atomic_json(ledger_path,ledger)
    deadline=time.monotonic()+deadline_seconds
    previous_runner_finished = None
    prune_checkpoint_count = 0
    blocked={}
    for record in ledger['cases'].values():
        if resource_stop(record):
            r=record['m']
            if r not in blocked or record['n']<blocked[r]['n']:blocked[r]=record
    for n,m in grid:
        stop = should_stop() if should_stop else None
        if stop:
            ledger['global_stop_reason']=stop
            break
        key=f'n{n}-m{m}'
        if key in ledger['cases']:
            continue  # Never overwrite/retry an existing run implicitly.
        if m in blocked and n>blocked[m]['n'] and unsupported_reason(n,m) is None:
            stop=blocked[m]
            ledger['cases'][key]=dict(status='INCOMPLETE',last_completed_layer=-1,
                reason=f"skipped larger n at fixed r={m} after resource stop at n={stop['n']}: {stop['reason']}",
                attempted=False,n=n,m=m,pruned_by=dict(n=stop['n'],r=m),
                pruning_is_heuristic=True)
            # These decisions are reproducible from the already durable failing
            # graph. Batch checkpoints instead of fsyncing the growing ledger
            # for every skipped pair; attempted graphs still commit immediately.
            prune_checkpoint_count += 1
            if prune_checkpoint_count >= 64:
                atomic_json(ledger_path,ledger)
                prune_checkpoint_count = 0
            continue
        remaining=deadline-time.monotonic()
        if remaining<=0:
            ledger['global_stop_reason']='compute deadline exhausted'
            break
        config=copy.deepcopy(base)
        case_key=ledger.get('retry_case_keys',{}).get(key,key)
        if case_key in ('', '.', '..') or '/' in case_key or '\\' in case_key:
            raise ValueError('unsafe retry case key')
        config.update(n=n,r=m,run_id=base.get('run_id','sweep')+'-'+case_key,
                      timeout_seconds=remaining)
        unsupported=unsupported_reason(n,m)
        if unsupported:
            record=dict(status='INCOMPLETE',last_completed_layer=-1,reason=unsupported,
                        attempted=False,n=n,m=m)
        else:
            case=root/case_key
            runner_started = time.monotonic()
            runner_started_unix = time.time()
            transition_before = (None if previous_runner_finished is None
                                 else runner_started-previous_runner_finished)
            manifest = {}
            try:
                manifest_path=runner(config,source,case,runtime)
                manifest=json.loads(manifest_path.read_text())
                record=dict(status=manifest['status'],last_completed_layer=manifest['last_completed_layer'],
                            reason=manifest['stop_reason'],attempted=True,n=n,m=m,manifest=str(manifest_path))
            except Exception as error:
                path=case/'saved/manifest.json'
                manifest=json.loads(path.read_text()) if path.exists() else {}
                record=dict(status='INCOMPLETE',last_completed_layer=manifest.get('last_completed_layer',-1),
                            reason=manifest.get('stop_reason',str(error)),attempted=True,n=n,m=m)
            previous_runner_finished = time.monotonic()
            searches = []
            for rank in range(config.get('world',2)):
                report_path = case/'result'/f'rank-{rank}.json'
                if report_path.exists():
                    try:
                        native = json.loads(report_path.read_text())
                    except (OSError,ValueError):
                        continue  # A killed rank may leave a partial report.
                    seconds = native.get('search_complete_seconds')
                    if seconds is None:seconds=native.get('search_prefix_seconds')
                    if type(seconds) in (int,float) and math.isfinite(seconds) and seconds >= 0:
                        searches.append(seconds)
            record['timing'] = dict(runner_started_at_unix=runner_started_unix,
                runner_wall_seconds=previous_runner_finished-runner_started,
                transition_before_seconds=transition_before,
                completed_layer_seconds=sum(layer['seconds'] for layer in manifest.get('layers',[])),
                production_search_seconds=max(searches) if len(searches)==config.get('world',2) else None,
                automatic_phase_seconds=manifest.get('launch_config',{}).get('automatic_phase_seconds'),
                scope='runner includes admission/calibration/startup/archive cleanup; transition includes ledger/progress/backpressure; search requires every rank report')
        if 'comparison' in manifest:
            record['comparison']=manifest['comparison']
        if manifest.get('replicas'):
            record['replicas']=manifest['replicas']
            record['comparison']=manifest['comparison']
            for replica in record['replicas']:
                if replica['status']=='INCOMPLETE' and allocation_failure(root/replica['key'],source):
                    replica['resource_classification']='cuda_allocation_failure'
        if record['status']=='INCOMPLETE' and allocation_failure(root/case_key,source):
            record['resource_classification']='cuda_allocation_failure'
        if case_key != key:
            record['publication_key']=case_key
        ledger['cases'][key]=record
        if resource_stop(record):
            blocked[m]=record
            # Record future exclusions now, even if the global deadline is near.
            for next_n,next_r in grid:
                next_key=f'n{next_n}-m{next_r}'
                if next_r==m and next_n>n and next_key not in ledger['cases'] and unsupported_reason(next_n,next_r) is None:
                    ledger['cases'][next_key]=dict(status='INCOMPLETE',last_completed_layer=-1,
                        reason=f"skipped larger n at fixed r={m} after resource stop at n={n}: {record['reason']}",
                        attempted=False,n=next_n,m=next_r,pruned_by=dict(n=n,r=m),
                        pruning_is_heuristic=True)
        atomic_json(ledger_path,ledger)
        if on_progress is not None:on_progress(ledger)
        if 'no space left on device' in record.get('reason','').lower():
            ledger['global_stop_reason']='SSD full; pending pairs remain eligible on restored storage'
            break
    ledger['pending']=[list(pair) for pair in grid if f'n{pair[0]}-m{pair[1]}' not in ledger['cases']]
    atomic_json(ledger_path,ledger)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--runtime-env',type=Path)
    p.add_argument('--source',type=Path)
    p.add_argument('--root',type=Path)
    p.add_argument('--n-min',type=int,default=2)
    p.add_argument('--n-max',type=int,default=128)
    p.add_argument('--r-min','--m-min',dest='m_min',type=int,default=1)
    p.add_argument('--r-max','--m-max',dest='m_max',type=int)
    p.add_argument('--deadline-seconds',type=float)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    grid=automatic_pairs(args.n_min,args.n_max,args.m_min,args.m_max)
    if args.plan_only:
        print(json.dumps(grid))
        return
    missing=[name for name in ('config','runtime_env','source','root','deadline_seconds')
             if getattr(args,name) is None]
    if missing:p.error('GPU execution requires '+', '.join('--'+name.replace('_','-') for name in missing))
    ledger=execute(json.loads(args.config.read_text()),args.source,args.root,
        json.loads(args.runtime_env.read_text()),grid,args.deadline_seconds)
    print(json.dumps(ledger,indent=2))


if __name__=='__main__':
    main()
