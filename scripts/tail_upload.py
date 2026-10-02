"""Single background HF publisher with pinned inputs and bounded pending bytes."""
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path


def retry_upload(operation):
    """Retry idempotent same-path publication after bounded transport failures."""
    for attempt in range(4):
        try:
            return operation()
        except Exception as error:
            code = getattr(getattr(error, 'response', None), 'status_code', None)
            transient = type(error).__name__ in {
                'RemoteProtocolError', 'ReadTimeout', 'ConnectTimeout', 'ConnectError',
                'ReadError', 'ConnectionError', 'Timeout', 'ChunkedEncodingError', 'SSLError'}
            transient = transient or code == 429 or (isinstance(code, int) and code >= 500)
            if not transient or attempt == 3:
                raise
            time.sleep(2 ** attempt)


class Publisher:
    def __init__(self, root, repo_id, run_id, token=None, max_pending_bytes=25_000_000_000, *, api=None, coalesce_pending=True):
        if api is None:
            from huggingface_hub import HfApi
            if not token:
                raise ValueError('HF write token required')
            api = HfApi(token=token)
        self.api = api
        self.repo_id, self.prefix = repo_id, 'tail-runs/' + run_id
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        self.queue, self.error = queue.Queue(), None
        self.pending, self.limit, self.lock = 0, max_pending_bytes, threading.Lock()
        self.coalesce_pending, self.closed = coalesce_pending, False
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def enqueue(self, manifest_path):
        if self.closed:
            raise RuntimeError('publisher already finishing')
        if self.error:
            raise RuntimeError('HF worker failed') from self.error
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        size = sum(f['bytes'] for f in manifest['files'])
        if self.coalesce_pending:
            # Every snapshot is self-contained and includes all completed-layer
            # statistics. Replace only queued inputs; the in-flight item remains
            # pinned until its manifest commit succeeds.
            while True:
                try:
                    older = self.queue.get_nowait()
                except queue.Empty:
                    break
                if older is None:
                    self.queue.put(None)
                    self.queue.task_done()
                    raise RuntimeError('publisher already finishing')
                try:
                    shutil.rmtree(older[0])
                finally:
                    with self.lock:
                        self.pending -= older[2]
                    self.queue.task_done()
        with self.lock:
            if self.pending + size > self.limit:
                raise RuntimeError('HF pending byte bound exceeded')
            self.pending += size
        pinned = self.root / uuid.uuid4().hex
        pinned.mkdir()
        try:
            for entry in manifest['files']:
                source = manifest_path.parent.parent / entry['path']
                dest = pinned / entry['path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, dest)
            (pinned / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            self.queue.put((pinned, manifest, size))
        except BaseException:
            with self.lock:
                self.pending -= size
            shutil.rmtree(pinned)
            raise

    def _work(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                pinned, manifest, size = item
                if self.error:
                    continue
                payloads = []
                for entry in manifest['files']:
                    path = pinned / entry['path']
                    sha = hashlib.sha256()
                    with path.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                            sha.update(chunk)
                    if path.stat().st_size != entry['bytes'] or sha.hexdigest() != entry['sha256']:
                        raise ValueError('pinned snapshot checksum')
                    payloads.append((path, self.prefix+'/'+entry['path']))
                if payloads and callable(getattr(self.api, 'create_commit', None)):
                    from huggingface_hub import CommitOperationAdd
                    retry_upload(lambda: self.api.create_commit(repo_id=self.repo_id, repo_type='dataset',
                        operations=[CommitOperationAdd(path_in_repo=remote,
                            path_or_fileobj=str(path)) for path, remote in payloads],
                        commit_message='Upload checked BFS snapshot payloads'))
                else:
                    for path, remote in payloads:
                        retry_upload(lambda: self.api.upload_file(path_or_fileobj=str(path), repo_id=self.repo_id,
                            repo_type='dataset', path_in_repo=remote))
                receipt = retry_upload(lambda: self.api.upload_file(path_or_fileobj=str(pinned/'manifest.json'),
                    repo_id=self.repo_id, repo_type='dataset', path_in_repo=self.prefix+'/manifest.json'))
                (self.root/'receipt.json').write_text(json.dumps(dict(
                    commit_url=str(receipt), status=manifest['status'],
                    last_completed_layer=manifest['last_completed_layer'])))
                shutil.rmtree(pinned)
            except BaseException as error:
                self.error = error
            finally:
                if item is not None:
                    with self.lock:
                        self.pending -= item[2]
                self.queue.task_done()

    def finish(self):
        self.closed = True
        self.queue.put(None)
        self.queue.join()
        self.thread.join()
        if self.error:
            raise RuntimeError('HF publication failed; pinned inputs retained') from self.error
