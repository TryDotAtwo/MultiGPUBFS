"""Fail-closed hardware admission, then compact two-seed automatic sweep."""
import argparse,csv,fcntl,json,math,os,subprocess,sys,time
from pathlib import Path
from bfs_tail_archive import atomic_json

def inventory(text,expected):
    rows=list(csv.reader(text.strip().splitlines(),skipinitialspace=True))
    if len(rows)!=expected: raise ValueError('EXPECTED_GPU_COUNT')
    seen=set(); out=[]
    for row in rows:
        if len(row)!=6: raise ValueError('GPU_INVENTORY_FORMAT')
        index,name,uuid,total,free,cap=row
        if 'B200' not in name.split() or cap.strip()!='10.0': raise ValueError('B200_SM100_REQUIRED')
        if uuid in seen or not uuid.startswith('GPU-'): raise ValueError('DISTINCT_PHYSICAL_GPUS_REQUIRED')
        seen.add(uuid)
        total,free=int(total),int(free)
        if total<=0 or not 0<free<=total: raise ValueError('GPU_MEMORY_VALUES')
        out.append(dict(index=int(index),name=name,uuid=uuid,total_mib=total,free_mib=free))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--expected-gpus',type=int,choices=(1,2,8),default=8)
    p.add_argument('--token-file',type=Path)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix-time.time()<600: p.error('at least 600 seconds to external work deadline required')
    if a.token_file: os.environ['HF_TOKEN']=a.token_file.read_text().strip()
    os.environ['HF_HUB_DISABLE_XET']='1'
    a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'production.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path('/opt/mgbfs/b200-src'); build=Path('/opt/mgbfs/b200-build')
    runtime=json.loads((build/'runtime-env.json').read_text())
    env=os.environ.copy();env.update(runtime)
    report=dict(status='FAILED',scope='B200 hardware startup admission; no throughput claim',started_at=time.time())
    gate=a.root/('startup-'+str(time.time_ns()));gate.mkdir()
    try:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap','--format=csv,noheader,nounits'],text=True)
        report['devices']=inventory(text,a.expected_gpus)
        env['CUDA_VISIBLE_DEVICES']=','.join(x['uuid'] for x in report['devices'])
        runtime['CUDA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES']
        report['topology']=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
        from huggingface_hub import HfApi,get_token
        token=get_token()
        if not token: raise ValueError('HF_WRITE_CREDENTIAL_REQUIRED')
        api=HfApi(token=token)
        marker=json.dumps(dict(kind='b300-startup',run=a.root.name)).encode()
        remote='evidence/'+a.root.name+'/'+gate.name+'/access-probe.json'
        receipt=api.upload_file(path_or_fileobj=marker,path_in_repo=remote,repo_id=a.repo_id,repo_type='dataset',commit_message='B200 startup access probe')
        from huggingface_hub import hf_hub_download
        probe=hf_hub_download(a.repo_id,remote,repo_type='dataset',revision=receipt.oid,token=token,local_dir=gate/'readback')
        if Path(probe).read_bytes()!=marker: raise ValueError('HF_READBACK_MISMATCH')
        report['hf_revision']=receipt.oid
        primitive=build/'native-build/mgbfs-batch-graph-window-test'
        output=subprocess.check_output([str(primitive)],env=env,text=True,stderr=subprocess.STDOUT,timeout=min(120,a.deadline_unix-time.time()-300))
        (gate/'graph32.log').write_text(output)
        if 'BATCH_GRAPH_32_MULTISTREAM_UPDATE_EXTERNAL_ARCHIVE_PASS' not in output: raise ValueError('GRAPH32_PRIMITIVE_NOT_VERIFIED')
        from run_tail_bfs import run
        from verify_tail_oracle import verify
        cfg=json.loads((source/'configs/tail-smoke.json').read_text()) if (source/'configs/tail-smoke.json').exists() else dict(env={})
        cfg.update(n=4,r=1,world=a.expected_gpus,batch=2,timeout_seconds=120,run_id='startup-oracle',retention_policy='last_complete_small_1000')
        cfg['env'].update(MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_PRE_DEDUP='ON',MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_TRANSPORT_BACKEND='HOST_SIZED_NCCL',MGBFS_CUDA_GRAPH_BATCHES='0',MGBFS_BENCH_CAPACITY='256',MGBFS_FUTURE_CAPACITY='512',MGBFS_BUCKET_CAPACITY='256',MGBFS_BUCKETS='8',MGBFS_SHARDS='4',MGBFS_JOB_BUCKETS='2',MGBFS_LIBRARY_POOL_BYTES=str(64<<20),MGBFS_ARCHIVE_ROWS='2',MGBFS_ARCHIVE_SLOTS='64',NCCL_CUMEM_ENABLE='0')
        run(cfg,source,gate/'oracle',runtime)
        oracle=verify(gate/'oracle/saved')
        if oracle.get('status')!='VERIFIED_FULL_STATE_LAYERS' or oracle.get('states')!=24: raise ValueError('FULL_STATE_CPU_ORACLE_NOT_VERIFIED')
        report.update(status='VERIFIED_STARTUP',oracle=oracle)
    except Exception as error:
        report['error_type']=type(error).__name__
        raise
    finally:
        report['finished_at']=time.time();atomic_json(gate/'admission.json',report)
    if a.preflight_only: return
    command=[sys.executable,str(source/'scripts/run_auto_tail.py'),'--source',str(source),'--runtime-env',str(build/'runtime-env.json'),'--root',str(a.root),'--repo-id',a.repo_id,'--deadline-unix',str(a.deadline_unix),'--two-seeds','--retention-policy','last_complete_small_1000','--upload-mode','end']
    # Keep lock across exec, preventing concurrent resumes into this root.
    os.set_inheritable(lock.fileno(),True)
    os.execvpe(command[0],command,env)

if __name__=='__main__': main()
