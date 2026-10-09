"""Bounded CPU-only RED gate against immutable production replay instrumentation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import urllib.request

out = Path('/kaggle/working/scoped-profile-red')
out.mkdir(parents=True, exist_ok=False)
work = Path('/tmp/mgbfs-scoped-red')
work.mkdir(exist_ok=False)
sources = [
    ('70f4f5bb6e955f4133bbbe3a0aaaca8882a75310', 'scripts/replay_lsa_cancel_candidate.py', 'replay.py'),
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
expected = result.returncode == 1 and 'UNKNOWN_PROCESS_INSTRUMENTATION' in text and 'test_search_capture_has_explicit_profiler_boundary' in text and 'Ran 4 tests' in text
summary = dict(status='EXPECTED_RED' if expected else 'RED_NOT_AUTHENTICATED', sources=records, returncode=result.returncode, scope='instrumentation command only; not GPU proof')
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
if not expected:
    raise RuntimeError('RED_NOT_AUTHENTICATED')
