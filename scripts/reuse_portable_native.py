"""Extract only checksummed native binaries from an immutable GitHub Release."""
import hashlib, json, sys, urllib.request, zipfile
from pathlib import Path
tag=sys.argv[1];base='https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/'+tag+'/'
receipt=json.loads(urllib.request.urlopen(base+'manifest.json',timeout=120).read())
work=Path('/tmp/mgbfs-portable');(work/'target/release').mkdir(parents=True);(work/'native-build').mkdir()
wheel=work/'accepted.whl';urllib.request.urlretrieve(base+receipt['wheel'],wheel)
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==receipt['sha256']
with zipfile.ZipFile(wheel) as z:
 names={name[name.index('multigpubfs/_native/'):]:name for name in z.namelist() if 'multigpubfs/_native/' in name and not name.endswith('/')}
 manifest=json.loads(z.read(names['multigpubfs/_native/manifest.json']))
 for name,target in [('bin/mgbfs',work/'target/release/mgbfs'),('lib/libmgbfs_cuda.so',work/'native-build/libmgbfs_cuda.so')]:
  data=z.read(names['multigpubfs/_native/'+name]);assert hashlib.sha256(data).hexdigest()==manifest['sha256'][name];target.write_bytes(data)
(work/'target/release/mgbfs').chmod(0o755)
(work/'native-manifest.json').write_text(json.dumps(manifest))
