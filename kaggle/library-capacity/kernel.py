"""Full 1/2-T4 regression for the N-peer runtime, all sanitizers, no speed panel."""
import hashlib
import os
import urllib.request

RUNNER_COMMIT = "007da906164504c56d1bd5d7ddde14ca63b8e7a8"
RUNNER_SHA256 = "54dd5a23dd63565adf9ba8b804b800b69e308b80f743b69e6e48dc535c81f632"

if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
           + RUNNER_COMMIT + "/kaggle/library-owner/kernel.py")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("PINNED_RUNNER_DIGEST_MISMATCH")
    namespace = {"__name__": "capacity_runner", "__file__": url}
    exec(compile(payload, url, "exec"), namespace)
    namespace.update(SOURCE_COMMIT="fce7de137d2ffb121854759efcea330b7c25a803",
                     SCREEN_POOL_BYTES=96 << 20, SCREEN_ARCHIVE_SLOTS=256,
                     SCREEN_REPEATS=5, SCREEN_WORLDS=(1, 2),
                     CUCO_PREVIOUS_COMMIT=None, PROFILE_SCREEN=False,
                     FULL_BFS_GATE=True, LOAD_SCREEN=False,
                     NATIVE_COMPARISON=True,
                     SANITIZER_TOOLS=("memcheck", "racecheck", "initcheck", "synccheck"))
    # Both pinned reference executables read these before allocation. The
    # runner preserves MGBFS_* environment for both native and library cases.
    os.environ["MGBFS_SHARDS"] = "4"
    os.environ["MGBFS_BUCKETS"] = "256"
    print("N_PEER_REGRESSION: physical 1/2 T4 correctness, all sanitizers; "
          "eight-device test remains an explicit separate hardware gate", flush=True)
    namespace["main"]()
