# dev34 automatic verified recovery

After `multigpubfs doctor --recover --devices 0,1 --total-timeout 1800`, a successful native and CayleyPy persistent check installs a node/package/environment-bound recipe in the user cache. Ordinary `multigpubfs.enable_native()` uses the matching recipe automatically. No profile path is required. Failed checks never install a recipe.

67 focused tests passed. A complete source diagnostic passed on 2x RTX 3060 in 209.77 seconds, including measured UT(4,47) prefix, NCCL, mode checks and two real CayleyPy API checks. S3/S4/S3 layers and terminal states matched independent CPU oracles; worker PIDs were reused. The automatic check ran in a new process.

This is not acceptance of another server, all possible faults, or full UT(4,47). Native binaries are unchanged. The published wheel was then installed from GitHub and passed the full diagnostic in 207.63 seconds. A separate ordinary API process automatically loaded the saved recipe and completed S3/S4/S3 with the same two workers, autotuning enabled and max_seconds=None. 75 release CI tests passed. See recovery-dev34-installed.json. The repository-wide Rust formatting check remains failing on unchanged files; it is not reported as passed.

See docs/auto-recovery.md for cache identity, explicit settings, opt-out and limitations.
