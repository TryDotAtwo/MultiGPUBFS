"""Upload a data-only SSH producer's stream; keep the HF token local."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path,PurePosixPath
from bfs_tail_archive import atomic_json


def validate_header(header,repo_id,run_id,max_bytes):
    path=header.get('path_in_repo','')
    if (header.get('repo_id')!=repo_id or header.get('repo_type')!='dataset'
        or not isinstance(path,str) or not path.startswith('tail-runs/'+run_id+'/')
        or '..' in PurePosixPath(path).parts or '\\' in path
        or any(ord(char)<32 for char in path)
        or type(header.get('bytes')) is not int or not 0<=header['bytes']<=max_bytes
        or not isinstance(header.get('sha256'),str)
        or not re.fullmatch('[0-9a-f]{64}',header.get('sha256',''))):
        raise ValueError('relay request outside authorized scope')


def read_exact(stream,size):
    chunks=[]
    while size:
        block=stream.read(size)
        if not block:raise EOFError('truncated upload stream')
        chunks.append(block);size-=len(block)
    return b''.join(chunks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--token-dpapi',type=Path)
    parser.add_argument('--max-bytes',type=int,default=25_000_000_000)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',args.run_id):parser.error('safe run ID required')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('SSH producer command required')
    args.root.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    from huggingface_hub import HfApi,get_token
    from publish_tail_run import decrypt_windows_token
    token=decrypt_windows_token(args.token_dpapi) if args.token_dpapi else get_token()
    if not token:raise ValueError('local HF credential required')
    api=HfApi(token=token)
    transferred=0;receipts=[];result=None
    with (args.root/'producer.stderr').open('wb') as errors:
        # Never put the credential in the producer's environment or arguments.
        env={k:v for k,v in os.environ.items() if k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN')}
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,env=env)
        try:
            while line:=process.stdout.readline():
                if line.startswith(b'HFRESULT '):result=json.loads(line[9:]);break
                if not line.startswith(b'HFRPC1 '):continue # SSH login banner only.
                size=int(line[7:]);
                if not 0<size<=8192:raise ValueError('relay header size')
                header=json.loads(read_exact(process.stdout,size))
                validate_header(header,args.repo_id,args.run_id,args.max_bytes-transferred)
                path=args.root/'payload.tmp'
                digest=hashlib.sha256();left=header['bytes']
                with path.open('wb') as output:
                    while left:
                        block=read_exact(process.stdout,min(left,1<<20));output.write(block)
                        digest.update(block);left-=len(block)
                if digest.hexdigest()!=header['sha256']:raise ValueError('relay checksum')
                transferred+=header['bytes'];started=time.time()
                receipt=api.upload_file(path_or_fileobj=str(path),repo_id=args.repo_id,
                    repo_type='dataset',path_in_repo=header['path_in_repo'])
                receipts.append(dict(header,local_upload_begin_unix=started,
                    local_upload_end_unix=time.time(),receipt=str(receipt)))
                atomic_json(args.root/'receipts.json',receipts)
                process.stdin.write(json.dumps(dict(status='UPLOADED',receipt=str(receipt))).encode()+b'\n')
                process.stdin.flush();path.unlink()
            code=process.wait(timeout=30)
            if code or result is None:raise RuntimeError('producer incomplete')
            atomic_json(args.root/'result.json',dict(status='UPLOADED',producer=result,
                files_uploaded=len(receipts),bytes_transferred=transferred))
            print(json.dumps(dict(status='UPLOADED',files_uploaded=len(receipts),bytes=transferred)))
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=30)


if __name__=='__main__':main()
