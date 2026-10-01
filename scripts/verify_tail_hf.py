"""Verify HF tail payloads by streaming on the GPU host, without saving states."""
import argparse
import hashlib
import json
from pathlib import PurePosixPath


def verify_payload(response, entry):
    response.raise_for_status()
    size, digest = 0, hashlib.sha256()
    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
        size += len(chunk)
        if size > entry['bytes']:
            raise ValueError('remote payload exceeds declared size')
        digest.update(chunk)
    if size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise ValueError('remote payload checksum or size mismatch')
    return size


def main():
    from huggingface_hub import HfApi, get_token, hf_hub_url
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if '/' in args.run_id or args.run_id in ('', '.', '..'):
        parser.error('invalid run ID')
    token = get_token()
    revision = HfApi(token=token).repo_info(args.repo_id, repo_type='dataset').sha
    session = requests.Session()
    if token:
        session.headers['Authorization'] = 'Bearer ' + token
    prefix = 'tail-runs/' + args.run_id + '/'
    def url(path):
        return hf_hub_url(args.repo_id, prefix + path, repo_type='dataset', revision=revision)
    with session.get(url('manifest.json'), timeout=(30, 120)) as response:
        response.raise_for_status()
        manifest = response.json()
    total = 0
    for entry in manifest['files']:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in entry['path']:
            raise ValueError('unsafe manifest path')
        with session.get(url(entry['path']), stream=True, timeout=(30, 120)) as response:
            total += verify_payload(response, entry)
    print(json.dumps(dict(repo_id=args.repo_id, run_id=args.run_id, revision=revision,
                         files=len(manifest['files']), bytes=total,
                         status=manifest['status'], checksums_verified=True)))


if __name__ == '__main__':
    main()
