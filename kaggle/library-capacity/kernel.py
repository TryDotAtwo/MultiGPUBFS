"""RED contract: already-drained owner completion must not synchronize again."""
import hashlib
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
    namespace.update(SOURCE_COMMIT="11e2a274db5bff5619a56522957f062095e2f11c",
                     SCREEN_POOL_BYTES=96 << 20, SCREEN_ARCHIVE_SLOTS=256,
                     SCREEN_REPEATS=1, SCREEN_WORLDS=(1, 2),
                     CUCO_PREVIOUS_COMMIT=None, PROFILE_SCREEN=False,
                     FULL_BFS_GATE=False, LOAD_SCREEN=False,
                     NATIVE_COMPARISON=False,
                     SANITIZER_TOOLS=())
    print("DRAINED_COMPLETION_RED: expected runtime assertion, not benchmark", flush=True)
    namespace["main"]()
