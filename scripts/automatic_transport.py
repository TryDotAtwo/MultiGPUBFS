"""One startup LSA capability gate using a complete independent CPU oracle.

Capability is not a throughput claim. Production profile timings are separate.
An explicit transport is authoritative and never replaced by this gate.
"""
import copy
import os
import time
from pathlib import Path
from bfs_tail_archive import atomic_json


def select_transport(base, source, root, runtime, *, deadline, runner=None, verifier=None,
                     cancelled=None):
    if runner is None:
        from run_tail_bfs import run
        runner = run
    if verifier is None:
        from verify_tail_oracle import verify
        verifier = verify
    result = copy.deepcopy(base)
    explicit = result['env'].get('MGBFS_TRANSPORT_BACKEND',runtime.get(
        'MGBFS_TRANSPORT_BACKEND',os.environ.get('MGBFS_TRANSPORT_BACKEND')))
    if explicit is not None:
        result['env']['MGBFS_TRANSPORT_BACKEND'] = explicit
        return result
    root = Path(root)
    root.mkdir(parents=True,exist_ok=False)
    decision = dict(selected='HOST_SIZED_NCCL',status='NOT_TESTED',
                    scope='startup capability only; no throughput claim')
    if cancelled and cancelled():
        decision['reason'] = 'startup gate cancelled'
    elif base['world'] == 1:
        decision['reason'] = 'single rank needs no peer LSA transport'
    elif deadline-time.time() < 30:
        decision['reason'] = 'insufficient startup gate time'
    else:
        cfg=copy.deepcopy(base)
        cfg.pop('repo_id',None)
        cfg.update(n=4,r=1,batch=2,timeout_seconds=min(30,deadline-time.time()))
        cfg['env'].update(MGBFS_TRANSPORT_BACKEND='NCCL_LSA',
            NCCL_CUMEM_ENABLE='1',MGBFS_CUDA_GRAPH_BATCHES='0',
            MGBFS_BENCH_CAPACITY='256',MGBFS_FUTURE_CAPACITY='512',
            MGBFS_BUCKET_CAPACITY='256',MGBFS_BUCKETS='8',MGBFS_SHARDS='4',
            MGBFS_JOB_BUCKETS='2',MGBFS_LIBRARY_POOL_BYTES=str(64<<20),
            MGBFS_ARCHIVE_ROWS='2',MGBFS_ARCHIVE_SLOTS='64')
        if cfg['env'].get('MGBFS_OWNER_BACKEND') == 'SHARD_AB':
            cfg['env'].pop('MGBFS_LIBRARY_POOL_BYTES',None)
            cfg['env'].pop('MGBFS_LIBRARY_POOL_AUTOSIZE',None)
        try:
            runner(cfg,source,root/'lsa-probe',runtime,cancelled=cancelled)
            oracle=verifier(root/'lsa-probe/saved')
            if oracle.get('status') != 'VERIFIED_FULL_STATE_LAYERS' or oracle.get('states') != 24:
                raise ValueError('LSA startup CPU oracle not verified')
            decision.update(selected='NCCL_LSA',status='VERIFIED',oracle=oracle)
            result['env']['NCCL_CUMEM_ENABLE']='1'
        except Exception as error:
            decision.update(status='FAILED',error_type=type(error).__name__,reason=str(error))
    if base['world']>1 and base['env'].get('MGBFS_OWNER_BACKEND')=='SHARD_AB' and decision['selected']!='NCCL_LSA':
        atomic_json(root/'decision.json',decision)
        raise RuntimeError('SHARD_AB_DEVICE_TRANSPORT_REQUIRED: '+decision.get('reason','startup gate failed'))
    result['env']['MGBFS_TRANSPORT_BACKEND']=decision['selected']
    result['transport_selection']=decision
    atomic_json(root/'decision.json',decision)
    return result
