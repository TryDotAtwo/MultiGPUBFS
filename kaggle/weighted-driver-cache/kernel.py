"""Cloud-only immutable cache copy. Private signed ingress is injected in RAM."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

OUTPUT = Path("/kaggle/working")
EXPECTED = {'compiled-mgbfs_cuda-d9aab1e1d579dd1164ca9bcf4fec3880f9d026afb6fcb9ef5eca0d116333e1c6.tar.gz': '005870777a5fa968c7f78470d2ae1a8953c8a56f22207808fe87b406ac59e98e', 'compiled-mgbfs_library_owner-d20be0f060383a6a0d4e0a5c640794afc7ceeb0b30ff2da58b1fae9ee277da86.tar.gz': 'e81c569cf3299116c023bcf255d099193f441fea0bb6d560498a428855519e3b'}
summary = dict(status="INCOMPLETE", source_kernel="trydotatwo/multigpubfs-current-host-full-regression",
               source_version=25, runtime="c9d5bbafcffaad105ddf76a9bde57a802d50b9bb", files={})
manifest = OUTPUT / "cache-copy.json"
manifest.write_text(json.dumps(summary, indent=2))
try:
    inputs = globals().pop("CACHE_COPY_INPUTS")
    if sorted(name for name, _ in inputs) != sorted(EXPECTED):
        raise RuntimeError("CACHE_COPY_INPUT_IDENTITY")
    for name, url in inputs:
        if Path(name).name != name:
            raise RuntimeError("CACHE_COPY_BASENAME")
        target = OUTPUT / name
        temporary = OUTPUT / (name + ".partial")
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 64 * 1024 * 1024:
                    raise RuntimeError("CACHE_COPY_SIZE_LIMIT")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if digest.hexdigest() != EXPECTED[name]:
            raise RuntimeError("CACHE_COPY_DIGEST")
        if target.exists():
            raise RuntimeError("CACHE_COPY_OVERWRITE")
        temporary.rename(target)
        summary["files"][name] = dict(bytes=size, sha256=digest.hexdigest())
        print("CACHE_COPY_VERIFIED", name, size, digest.hexdigest(), flush=True)
    summary["status"] = "COMPLETE"
    manifest.write_text(json.dumps(summary, indent=2))
except Exception as error:
    summary["error_type"] = type(error).__name__
    manifest.write_text(json.dumps(summary, indent=2))
    raise RuntimeError("CLOUD_CACHE_COPY_FAILED") from None
