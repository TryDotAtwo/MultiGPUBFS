"""Publish an already saved tail after local HF login; manifest is uploaded last."""
import argparse
import json
import re
from pathlib import Path
from huggingface_hub import get_token
from tail_upload import Publisher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_root',type=Path)
    parser.add_argument('--repo-id',default='TryDotAtwo/multigpubfs-bfs-results')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',args.run_id):
        parser.error('safe fresh run ID required')
    token=get_token()
    if not token:
        parser.error('HF write token missing; authenticate locally, never put token in command arguments')
    manifest=json.loads((args.saved_root/'manifest.json').read_text())
    if manifest['files']:
        path=args.saved_root/Path(manifest['files'][0]['path']).parent/'manifest.json'
    else:
        path=args.saved_root/'manifest.json'
    worker=Publisher(args.saved_root/('publish-pins-'+args.run_id),args.repo_id,args.run_id,token)
    worker.enqueue(path)
    worker.finish()
    print((worker.root/'receipt.json').read_text())


if __name__=='__main__':
    main()
