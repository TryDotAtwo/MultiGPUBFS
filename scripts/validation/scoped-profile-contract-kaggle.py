"""Bounded CPU-only GREEN gate against immutable production replay instrumentation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import urllib.request

out = Path('/kaggle/working/scoped-profile-green')
out.mkdir(parents=True, exist_ok=False)
work = Path('/tmp/mgbfs-scoped-green')
work.mkdir(exist_ok=False)
sources = [
    ('820423350fef37ad77b3812a9d3f4baebf473231', 'scripts/replay_lsa_cancel_candidate.py', 'replay.py'),
    ('5017c90e8cd2ea7b83d4a7bf23c8a3c728e9a64f', 'scripts/validation/test_replay_scoped_profile.py', 'test.py'),
]
records = []
for ref, path, name in sources:
    with urllib.request.urlopen('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/' + ref + '/' + path, timeout=30) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise RuntimeError('SOURCE_CAPACITY')
    (work / name).write_bytes(data)
    records.append(dict(ref=ref, path=path, sha256=hashlib.sha256(data).hexdigest()))
result = subprocess.run([sys.executable, str(work / 'test.py'), str(work / 'replay.py')], capture_output=True, text=True, timeout=30)
text = result.stdout + result.stderr
(out / 'test.log').write_text(text)
expected = result.returncode == 0 and 'Ran 4 tests' in text and 'OK' in text and 'FAILED' not in text
summary = dict(status='SCOPED_COMMAND_GREEN' if expected else 'GREEN_NOT_AUTHENTICATED', sources=records, returncode=result.returncode, scope='instrumentation command only; not GPU proof')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
if not expected:
    raise RuntimeError('GREEN_NOT_AUTHENTICATED')
