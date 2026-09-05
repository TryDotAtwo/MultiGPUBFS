"""CPU-only replay of one immutable S11 shard; no CUDA build or publication."""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix='mgbfs-codec-', dir='/tmp'))
source = '14ec25e35cfce7c1f2019158203cca61215fa561'
revision = '2f77dce99de88f2d8eac20859305da5051d6187e'
repo = 'TryDotAtwo/multigpubfs-bfs-results'
path = 'states/s11-native-2xt4-20260905-142232-rank-00000-part-00000000.parquet'
from huggingface_hub import HfApi, hf_hub_download
info = HfApi().get_paths_info(repo, [path], repo_type='dataset', revision=revision)
if len(info) != 1 or not 0 < info[0].size <= 128 * 1024**2:
    raise RuntimeError('SHARD_CAPACITY')
local = hf_hub_download(repo, path, repo_type='dataset', revision=revision,
                        cache_dir=str(root / 'hf'))
script = root / 'bench.py'
urllib.request.urlretrieve(
    f'https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/{source}/scripts/bench_archive_codec.py', script)
output = Path('/kaggle/working/codec-profile.json')
subprocess.run([sys.executable, str(script), local, str(output)], check=True, timeout=900)
Path('/kaggle/working/codec-input.json').write_text(json.dumps(
    dict(source=source, revision=revision, repo=repo, path=path, bytes=info[0].size)), encoding='utf-8')
print('CODEC_REPLAY_COMPLETE', flush=True)
