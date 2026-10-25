# dev34 automatic verified recovery

After `multigpubfs doctor --recover --devices 0,1 --total-timeout 1800`, a successful native and CayleyPy persistent check installs a node/package/environment-bound recipe in the user cache. Ordinary `multigpubfs.enable_native()` uses the matching recipe automatically. No profile path is required. Failed checks never install a recipe.

67 focused tests passed. A complete source diagnostic passed on 2x RTX 3060 in 209.77 seconds, including measured UT(4,47) prefix, NCCL, mode checks and two real CayleyPy API checks. S3/S4/S3 layers and terminal states matched independent CPU oracles; worker PIDs were reused. The automatic check ran in a new process.

This is not acceptance of another server, all possible faults, or full UT(4,47). Native binaries are unchanged. Installed wheel acceptance will be recorded separately.

See docs/auto-recovery.md for cache identity, explicit settings, opt-out and limitations.
