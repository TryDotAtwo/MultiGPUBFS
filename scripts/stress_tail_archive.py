"""Physical SSD retention stress, synthetic payloads, never BFS evidence."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from bfs_tail_archive import TailArchive,atomic_json


def stress(root):
    layer_bytes=2_600_000_000
    if shutil.disk_usage(root.parent).free < 18_000_000_000:
        raise RuntimeError('at least 18 decimal GB free required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    archive=TailArchive(root,n=16,r=1,start=list(range(16)),
        actions={'L':'rotate left','R':'rotate right','X':'swap first two'},
        program_commit=commit,launch_config={'synthetic_storage_stress':True},
        sample_interval_seconds=.05)
    chunk=bytes(range(256))*16384
    def chunks():
        left=layer_bytes
        while left:
            take=min(left,len(chunk));yield chunk[:take];left-=take
    timings=[]
    for depth in range(5):
        at=time.monotonic()
        archive.completed_layer(depth,layer_bytes//8,chunks(),0,{'storage_test':None})
        elapsed=time.monotonic()-at
        timings.append(elapsed)
        intermediate=json.loads((root/'manifest.json').read_text())
        assert intermediate['status']=='INCOMPLETE'
        assert sum(e['bytes'] for e in intermediate['files'])==1_000_000_000
        print(json.dumps({'depth':depth,'write_and_snapshot_seconds':elapsed}),flush=True)
    path=archive.snapshot(True,'synthetic storage stress complete, not graph exhaustion')
    manifest=json.loads(path.read_text())
    assert [e['depth'] for e in manifest['files']]==[1,2,3,4]
    assert all(e['full_layer'] for e in manifest['files'])
    assert sum(e['bytes'] for e in manifest['files'])==10_400_000_000
    assert not (root/'tail/layer-000000.bin').exists()
    for entry in manifest['files']:
        digest=hashlib.sha256()
        with (root/entry['path']).open('rb') as stream:
            for data in iter(lambda:stream.read(4<<20),b''):digest.update(data)
        assert digest.hexdigest()==entry['sha256']
    incomplete=archive.snapshot(False,'synthetic stopped')
    short=json.loads(incomplete.read_text())
    assert sum(e['bytes'] for e in short['files'])==1_000_000_000
    assert short['files'][0]['first_state_ordinal']==200_000_000
    digest=hashlib.sha256()
    with (root/short['files'][0]['path']).open('rb') as stream:
        for data in iter(lambda:stream.read(4<<20),b''):digest.update(data)
    assert digest.hexdigest()==short['files'][0]['sha256']
    result=dict(kind='synthetic SSD stress, not GPU/BFS result',program_commit=commit,
        bytes_written=5*layer_bytes,complete_snapshot_bytes=10_400_000_000,
        retained_depths=[1,2,3,4],complete_checksums_verified=4,
        incomplete_snapshot_bytes=1_000_000_000,write_seconds=timings)
    result['incomplete_checksum_verified']=True
    atomic_json(root/'stress-result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    stress(parser.parse_args().root)
