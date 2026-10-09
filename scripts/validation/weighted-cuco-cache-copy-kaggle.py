"""Immutable cloud-only copy of v44's compiled weighted-CUCO closure.
Private ingress supplies basename, signed URL and retained SHA256 in RAM.
No compilation or local artifact download.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

inputs = globals().pop("_MGBFS_COMPILED_COPY_INPUTS")
assert len(inputs) == 2
root = Path("/kaggle/working")
records = []
for name, url, expected in inputs:
    if Path(name).name != name or not name.startswith("compiled-mgbfs_") or not name.endswith(".tar.gz"):
        raise RuntimeError("CACHE_COPY_BASENAME")
    destination = root / name
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected:
        destination.unlink()
        raise RuntimeError("CACHE_COPY_CHECKSUM")
    records.append(dict(name=name, bytes=destination.stat().st_size, sha256=expected))
receipt = dict(status="COMPLETE", producer="trydotatwo/multigpubfs-current-host-full-regression",
    producer_version=44, runtime="6f35c433ebd3269da45c8569fba1424f4e20a1d6",
    scope="compiled cache retention only; not weighted CUCO GPU acceptance", files=records)
(root / "cache-copy.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
