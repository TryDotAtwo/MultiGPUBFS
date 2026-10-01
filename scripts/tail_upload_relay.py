"""Data-only stdout/stdin upload adapter; no HF credential on the GPU host."""
import hashlib
import json
import sys
import time
from pathlib import Path


class StdIOUploadRelay:
    def upload_file(self, *, path_or_fileobj, repo_id, repo_type, path_in_repo):
        path=Path(path_or_fileobj)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
        header=json.dumps(dict(repo_id=repo_id,repo_type=repo_type,path_in_repo=path_in_repo,
            bytes=path.stat().st_size,sha256=digest.hexdigest(),sent_unix=time.time())).encode()
        output=sys.stdout.buffer
        output.write(b'HFRPC1 '+str(len(header)).encode()+b'\n')
        output.write(header)
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):output.write(chunk)
        output.flush()
        acknowledgement=json.loads(sys.stdin.buffer.readline())
        if acknowledgement.get('status')!='UPLOADED':
            raise RuntimeError('local upload relay failed')
        return acknowledgement['receipt']
