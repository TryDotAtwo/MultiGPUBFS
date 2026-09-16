"""Independent fixed-192-MiB pool screen; no fallback and no native rerun."""
import hashlib
import urllib.request

RUNNER_COMMIT = "3e90b740b23f51cd6bf95a2ab19a7509ccb40e1d"
RUNNER_SHA256 = "a6d51dbbd6269741a6844cd9c14b2d699d32fd9cf3d88e9f8096c35de4e71a8c"

if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
           + RUNNER_COMMIT + "/kaggle/library-owner/kernel.py")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("PINNED_RUNNER_DIGEST_MISMATCH")
    namespace = {"__name__": "capacity_runner", "__file__": url}
    exec(compile(payload, url, "exec"), namespace)
    namespace.update(SCREEN_POOL_BYTES=192 << 20, SCREEN_ARCHIVE_SLOTS=256,
                     SCREEN_REPEATS=5, SCREEN_WORLDS=(1, 2),
                     NATIVE_COMPARISON=False, SANITIZER_TOOLS=())
    # Full plain correctness remains enabled. Sanitizers at this configuration
    # are not claimed; prior unchanged runtime evidence is recorded separately.
    print("CAPACITY_SCREEN pool=192MiB archive_slots=256 repeats=5 worlds=1,2; "
          "plain correctness only; no same-run sanitizer claim", flush=True)
    namespace["main"]()
