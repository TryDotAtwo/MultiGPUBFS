"""Fixed-192-MiB cuco/native diagnostic timelines; not a timing benchmark."""
import hashlib
import urllib.request

RUNNER_COMMIT = "b829df97f4c0ff676dffcb8068309c9892709773"
RUNNER_SHA256 = "b90a7fbdac3386479a8f35e26753aea860578b14fd7c1bcd4868b1f6a928d98b"

if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
           + RUNNER_COMMIT + "/kaggle/library-owner/kernel.py")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("PINNED_RUNNER_DIGEST_MISMATCH")
    namespace = {"__name__": "capacity_runner", "__file__": url}
    exec(compile(payload, url, "exec"), namespace)
    namespace.update(SOURCE_COMMIT=RUNNER_COMMIT,
                     SCREEN_POOL_BYTES=192 << 20, SCREEN_ARCHIVE_SLOTS=256,
                     SCREEN_REPEATS=1, SCREEN_WORLDS=(1, 2),
                     CUCO_PREVIOUS_COMMIT=None, PROFILE_SCREEN=True,
                     NATIVE_COMPARISON=True, SANITIZER_TOOLS=())
    # Full plain correctness remains enabled. Sanitizers at this configuration
    # are not claimed; prior unchanged runtime evidence is recorded separately.
    print("NSYS_DIAGNOSTIC pool=192MiB archive_slots=256 worlds=1,2; "
          "cuco and preserved native; no performance or sanitizer claim", flush=True)
    namespace["main"]()
