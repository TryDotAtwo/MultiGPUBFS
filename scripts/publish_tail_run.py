"""Publish an already saved tail after local HF login; manifest is uploaded last."""
import argparse
import json
import re
from pathlib import Path
from huggingface_hub import get_token
from tail_upload import Publisher


def decrypt_windows_token(path):
    import ctypes
    import os
    from ctypes import wintypes
    if os.name != 'nt':
        raise ValueError('DPAPI credential requires the original Windows user')
    class Blob(ctypes.Structure):
        _fields_=[('cbData',wintypes.DWORD),('pbData',ctypes.POINTER(ctypes.c_char))]
    raw=path.read_bytes()
    buffer=ctypes.create_string_buffer(raw)
    source=Blob(len(raw),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_char)))
    result=Blob()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source),None,None,
            None,None,0,ctypes.byref(result)):
        raise ValueError('cannot decrypt local HF credential')
    try:
        return ctypes.string_at(result.pbData,result.cbData).decode()
    finally:
        ctypes.windll.kernel32.LocalFree(result.pbData)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_root',type=Path)
    parser.add_argument('--repo-id',default='TryDotAtwo/multigpubfs-bfs-results')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--token-dpapi',type=Path,help='encrypted Windows credential file')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}',args.run_id):
        parser.error('safe fresh run ID required')
    token=decrypt_windows_token(args.token_dpapi) if args.token_dpapi else get_token()
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
