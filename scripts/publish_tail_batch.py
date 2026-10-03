"""Publish a staged finite sweep from its GPU host in two ordered HF commits."""
import argparse
import hashlib
import json
import math
import time
import tempfile
from pathlib import Path, PurePosixPath


def publication_cases(root, ledger):
    format_ = ledger['configuration']['base'].get('archive_format', 'packed')
    if format_ == 'parquet_cohort':
        try:
            from .tail_cohort import publication_cases as cohorts
        except ImportError:
            from tail_cohort import publication_cases as cohorts
        return cohorts(root, ledger)
    try:
        from .tail_parquet import publication_root
    except ImportError:
        from tail_parquet import publication_root
    result = {}
    for key, record in ledger['cases'].items():
        if key in ('', '.', '..') or '/' in key or '\\' in key:
            raise ValueError('unsafe case key')
        if record.get('attempted', True):
            base = publication_root(Path(root)/key/'saved', format_)
            result[key] = (base, base/'manifest.json')
    return result


def plan(root, max_bytes=25_000_000_000, *, ledger=None):
    root=Path(root).resolve()
    if ledger is None:ledger=json.loads((root/'sweep.json').read_text())
    run_id=ledger['configuration']['base']['run_id']
    if not isinstance(run_id,str) or '/' in run_id or '\\' in run_id or run_id in ('','.','..'):
        raise ValueError('unsafe sweep run ID')
    payloads,manifests,total=[],[],0
    cases = publication_cases(root, ledger)
    seen = {}
    for key,record in ledger['cases'].items():
        if '/' in key or '\\' in key or key in ('','.','..'):
            raise ValueError('unsafe case key')
        if not record.get('attempted',True):continue
        base, manifest_path = cases[key]
        manifest=json.loads(manifest_path.read_text())
        prefix='tail-runs/'+run_id+'-'+key+'/'
        for entry in manifest['files']:
            relative=PurePosixPath(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or '\\' in entry['path']:
                raise ValueError('unsafe payload path')
            path=(base/entry['path']).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError('payload escapes case root')
            remote = entry.get('repo_path', prefix+entry['path'])
            repo_path = PurePosixPath(remote)
            if repo_path.is_absolute() or '..' in repo_path.parts or '\\' in remote:
                raise ValueError('unsafe repository payload path')
            signature = (entry['bytes'], entry['sha256'])
            if remote in seen:
                if seen[remote] != signature:
                    raise ValueError('conflicting shared payload')
                continue
            seen[remote] = signature
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4<<20),b''):digest.update(chunk)
            if path.stat().st_size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
                raise ValueError('staged payload checksum mismatch')
            total+=entry['bytes']
            if max_bytes is not None and total>max_bytes:raise ValueError('batch byte bound exceeded')
            payloads.append((path,remote))
        manifests.append((manifest_path,prefix+'manifest.json'))
    return payloads,manifests,total


def publish(root,repo_id,api,deadline_unix=None, *, ledger=None, max_batch_bytes=25_000_000_000):
    from huggingface_hub import CommitOperationAdd
    try:
        from .tail_upload import retry_upload
    except ImportError:
        from tail_upload import retry_upload
    # Freeze the ledger before planning payloads. The producer may add cases
    # while the background worker uploads this exact completed-case cohort.
    if ledger is None:ledger=json.loads((Path(root)/'sweep.json').read_text())
    if max_batch_bytes<=0:raise ValueError('positive publication batch bound required')
    payloads,manifests,total=plan(root,max_bytes=None,ledger=ledger)
    def commit(items,message):
        if deadline_unix is not None and time.time()>=deadline_unix:
            raise TimeoutError('HF publication deadline; staged inputs retained')
        return retry_upload(lambda:api.create_commit(repo_id=repo_id,repo_type='dataset',
            operations=[CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(path))
                        for path,remote in items],commit_message=message))
    batch=[];batch_bytes=0;payload_commits=0
    for item in payloads:
        size=item[0].stat().st_size
        if size>max_batch_bytes:raise ValueError('one payload exceeds publication batch bound')
        if batch and (batch_bytes+size>max_batch_bytes or len(batch)>=10000):
            commit(batch,'Publish checked BFS sweep payload batch')
            payload_commits+=1;batch=[];batch_bytes=0
        batch.append(item);batch_bytes+=size
    if batch:
        commit(batch,'Publish checked BFS sweep payload batch');payload_commits+=1
    with tempfile.TemporaryDirectory(prefix='hf-ledger-',dir=root) as staging:
        frozen=Path(staging)/'sweep.json'
        frozen.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
        metadata=[(frozen,'tail-sweeps/'+Path(root).name+'/sweep.json')]
        receipt=commit(manifests+metadata,'Publish BFS sweep manifests after payloads')
    return dict(cases=len(manifests),files=len(payloads),bytes=total,receipt=str(receipt),
                payload_commits=payload_commits,max_batch_bytes=max_batch_bytes)


def main():
    from huggingface_hub import HfApi,get_token
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-root',type=Path,required=True)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    args=parser.parse_args()
    if not math.isfinite(args.deadline_unix):parser.error('finite publication deadline required')
    api=HfApi(token=get_token())
    while True:
        if time.time()>=args.deadline_unix:
            raise TimeoutError('HF publication deadline; GPU-host inputs retained')
        try:
            # A tiny probe avoids pre-uploading a large batch while commits are
            # explicitly blocked by the repository quota.
            api.upload_file(path_or_fileobj=b'{"publication_ready":true}',repo_id=args.repo_id,
                repo_type='dataset',path_in_repo='tail-sweeps/'+args.sweep_root.name+'/ready.json')
            break
        except Exception as error:
            if getattr(getattr(error,'response',None),'status_code',None)!=429:raise
            time.sleep(min(60,max(0,args.deadline_unix-time.time())))
    print(json.dumps(publish(args.sweep_root,args.repo_id,api,args.deadline_unix)))


if __name__=='__main__':main()
