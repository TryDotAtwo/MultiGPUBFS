"""Remote BFS producer for a local SSH HF uploader; reads no credential."""
import argparse
import json
import sys
from pathlib import Path
from run_tail_bfs import run
from tail_upload_relay import StdIOUploadRelay


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('config','source','root','runtime-env'):
        parser.add_argument('--'+option,type=Path,required=True)
    args=parser.parse_args()
    # Start-up import is outside the native measured search window.
    import numpy
    config=json.loads(args.config.read_text())
    config['upload_transport']='SSH data-only relay; HF credential on local client'
    manifest=run(config,args.source,args.root,json.loads(args.runtime_env.read_text()),
                 publisher_api=StdIOUploadRelay())
    print('HFRESULT '+json.dumps(dict(manifest=str(manifest))),flush=True)
