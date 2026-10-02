"""Physical 10 GB SSD -> background HF -> streamed readback acceptance gate.

Payloads are synthetic; this validates storage/publication, never BFS reachability.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from bfs_tail_archive import TailArchive,atomic_json
from tail_upload import Publisher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    args=p.parse_args()
    if shutil.disk_usage(args.root.parent).free<25_000_000_000:
        p.error('25 GB free SSD required')
    os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi,get_token,hf_hub_url
    from verify_tail_hf import verify_payload
    import requests
    api=HfApi(token=get_token())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    archive=TailArchive(args.root,n=16,r=1,start=list(range(16)),
        actions=dict(L='rotate left',R='rotate right',X='swap first two'),
        program_commit=commit,launch_config=dict(synthetic_storage_stress=True,
        storage_bytes=13_000_000_000,deadline_unix=args.deadline_unix),sample_interval_seconds=.05)
    latest=Publisher(args.root/'pins-live',args.repo_id,args.root.name+'-live',get_token())
    report=dict(kind='synthetic physical storage/publication test, not graph completion',
                program_commit=commit,intermediate=[])
    for depth in range(5):
        if time.time()>args.deadline_unix:raise TimeoutError('storage deadline')
        chunk=bytes([depth+1])*(4<<20)
        def blocks():
            left=2_600_000_000
            while left:
                take=min(left,len(chunk));yield chunk[:take];left-=take
        archive.completed_layer(depth,325_000_000,blocks(),0,{'synthetic_storage':None})
        m=json.loads((args.root/'manifest.json').read_text())
        assert sum(x['bytes'] for x in m['files'])==1_000_000_000
        path=args.root/Path(m['files'][0]['path']).parent/'manifest.json'
        latest.enqueue(path)
        report['intermediate'].append(dict(depth=depth,bytes=1_000_000_000,
            layers_recorded=len(m['layers']),full_layer=m['files'][0]['full_layer']))
    latest.finish()
    def publish_check(path,label):
        if time.time()>args.deadline_unix:raise TimeoutError('publication deadline')
        m=json.loads(path.read_text());run_id=args.root.name+'-'+label
        worker=Publisher(args.root/('pins-'+label),args.repo_id,run_id,get_token())
        worker.enqueue(path);worker.finish()
        revision=api.repo_info(args.repo_id,repo_type='dataset').sha
        prefix='tail-runs/'+run_id+'/'
        session=requests.Session();session.headers['Authorization']='Bearer '+get_token()
        with session.get(hf_hub_url(args.repo_id,prefix+'manifest.json',repo_type='dataset',revision=revision),timeout=60) as response:
            response.raise_for_status();assert response.json()==m
        size=0
        for entry in m['files']:
            if time.time()>args.deadline_unix:raise TimeoutError('verification deadline')
            with session.get(hf_hub_url(args.repo_id,prefix+entry['path'],repo_type='dataset',revision=revision),stream=True,timeout=(30,120)) as response:
                size+=verify_payload(response,entry)
        return dict(revision=revision,run_id=run_id,bytes_verified=size,
            files=len(m['files']),all_checksums_verified=True,
            depths=[x['depth'] for x in m['files']],full_layers=[x['full_layer'] for x in m['files']])
    complete=archive.snapshot(True,'synthetic storage acceptance, not graph exhaustion')
    m=json.loads(complete.read_text())
    assert [x['depth'] for x in m['files']]==[1,2,3,4]
    assert sum(x['bytes'] for x in m['files'])==10_400_000_000
    assert all(x['full_layer'] for x in m['files'])
    report['complete']=publish_check(complete,'complete')
    stopped=archive.snapshot(False,'synthetic interruption')
    m=json.loads(stopped.read_text())
    assert sum(x['bytes'] for x in m['files'])==1_000_000_000
    assert m['files'][0]['first_state_ordinal']==200_000_000
    report['incomplete']=publish_check(stopped,'stopped')
    report['status']='VERIFIED';atomic_json(args.root/'storage-hf-report.json',report)
    api.upload_file(path_or_fileobj=str(args.root/'storage-hf-report.json'),repo_id=args.repo_id,
        repo_type='dataset',path_in_repo='evidence/'+args.root.name+'/storage-hf-report.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
