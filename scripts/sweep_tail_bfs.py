"""Explicit finite (n,m) sweep; m is the repeated-symbol count r in native CLI."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
from bfs_tail_archive import atomic_json
from run_tail_bfs import run


def pairs(n_min,n_max,m_min=1,m_max=None):
    if not 2<=n_min<=n_max<=32 or m_min<1 or (m_max is not None and m_max<m_min):
        raise ValueError('invalid finite sweep bounds')
    return [(n,m) for n in range(n_min,n_max+1)
            for m in range(m_min,min(n,m_max if m_max is not None else n)+1)]


def execute(base,source,root,runtime,grid,deadline_seconds,runner=run):
    if deadline_seconds<=0:
        raise ValueError('positive sweep deadline required')
    root.mkdir(parents=True,exist_ok=True)
    ledger_path=root/'sweep.json'
    fingerprint=dict(base=base,grid=grid)
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        if ledger['configuration']!=json.loads(json.dumps(fingerprint)):
            raise ValueError('resume configuration differs')
    else:
        ledger=dict(configuration=fingerprint,cases={})
    deadline=time.monotonic()+deadline_seconds
    for n,m in grid:
        key=f'n{n}-m{m}'
        if key in ledger['cases']:
            continue  # Never overwrite/retry an existing run implicitly.
        remaining=deadline-time.monotonic()
        if remaining<=0:
            break
        config=copy.deepcopy(base)
        config.update(n=n,r=m,run_id=base.get('run_id','sweep')+'-'+key,
                      timeout_seconds=min(base.get('timeout_seconds',300),remaining))
        unsupported='alphabet exceeds four bits' if n-m+1>16 else (
            'native orbit count exceeds u64' if math.factorial(n)//math.factorial(m)>=2**64 else None)
        if unsupported:
            record=dict(status='INCOMPLETE',last_completed_layer=-1,reason=unsupported,
                        attempted=False,n=n,m=m)
        else:
            case=root/key
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
        ledger['cases'][key]=record
        atomic_json(ledger_path,ledger)
    ledger['pending']=[list(pair) for pair in grid if f'n{pair[0]}-m{pair[1]}' not in ledger['cases']]
    atomic_json(ledger_path,ledger)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--runtime-env',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--n-min',type=int,required=True)
    p.add_argument('--n-max',type=int,required=True)
    p.add_argument('--m-min',type=int,default=1)
    p.add_argument('--m-max',type=int)
    p.add_argument('--deadline-seconds',type=float,required=True)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    grid=pairs(args.n_min,args.n_max,args.m_min,args.m_max)
    if args.plan_only:
        print(json.dumps(grid))
        return
    ledger=execute(json.loads(args.config.read_text()),args.source,args.root,
        json.loads(args.runtime_env.read_text()),grid,args.deadline_seconds)
    print(json.dumps(ledger,indent=2))


if __name__=='__main__':
    main()
