"""Cloud-only immutable cache copy. Private signed ingress is injected in RAM."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

OUTPUT = Path("/kaggle/working")
EXPECTED = {
    "compiled-mgbfs_cuda-12a2271977627d2260457dbd2e15aefa4178ba683ddf2fbcd214cd4fa2cddc6e.tar.gz":
        "3341adda5f75876c2a94d03dfe53a3819e44a3056a444339417b25a1bf833df4",
}
summary = dict(status="INCOMPLETE", source_kernel="trydotatwo/multigpubfs-current-host-full-regression",
               source_version=14, runtime="ba65ea410b951427e3e8b5cd53b4dcf5466e79e8", files={})
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
