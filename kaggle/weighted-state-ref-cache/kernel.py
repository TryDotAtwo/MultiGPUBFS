"""Cloud-only immutable cache copy. Private signed ingress is injected in RAM."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

OUTPUT = Path("/kaggle/working")
EXPECTED = {'compiled-mgbfs_cuda-fa8968be9f8e3e85bd69f7eec44bcb7227a6fd063319b2817559f5d26a75898b.tar.gz': '921b9ce8612dd0e9c9fcfde4f5ee6b81d60d8dbdb18f2d3ba79f383b7ff1b2d6', 'compiled-mgbfs_library_owner-35e5b1e04c13d889552c7969a5b49c67aa77c746fac8b8fe29302daea5d068dd.tar.gz': '3fc4623f832241f10ec2a6be49e17fc595a9b7d760c9202d31cfa84fed0a041c'}
summary = dict(status="INCOMPLETE", source_kernel="trydotatwo/multigpubfs-current-host-full-regression",
               source_version=21, runtime="15e2ed480ff1061e3059dc2aa3c7773515634c35", files={})
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
