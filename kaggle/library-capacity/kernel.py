"""Independent fixed-192-MiB pool screen; no fallback and no native rerun."""
import hashlib
import urllib.request

RUNNER_COMMIT = "552b32c73ebe9a42da7df4a7ceca4967dd366b1d"
RUNNER_SHA256 = "2d247cfed28b00babdae5712082feda9b6f12f5cfb633fa8f3ce6a63e9a1b5ea"

if __name__ == "__main__":
    url = ("https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/"
           + RUNNER_COMMIT + "/kaggle/library-owner/kernel.py")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("PINNED_RUNNER_DIGEST_MISMATCH")
    namespace = {"__name__": "capacity_runner", "__file__": url}
    exec(compile(payload, url, "exec"), namespace)
    namespace.update(SOURCE_COMMIT="05efe4cff5f315e5dbdd2fb7c2435ec4d2638e96",
                     SCREEN_POOL_BYTES=192 << 20, SCREEN_ARCHIVE_SLOTS=256,
                     SCREEN_REPEATS=5, SCREEN_WORLDS=(1, 2),
                     CUCO_PREVIOUS_COMMIT="5ea41ff0a9861c974f979a30c4562f92a168e0b8",
                     NATIVE_COMPARISON=False, SANITIZER_TOOLS=())
    # Full plain correctness remains enabled. Sanitizers at this configuration
    # are not claimed; prior unchanged runtime evidence is recorded separately.
    print("CAPACITY_SCREEN pool=192MiB archive_slots=256 repeats=5 worlds=1,2; "
          "plain correctness only; no same-run sanitizer claim", flush=True)
    namespace["main"]()
