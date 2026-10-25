# Automatic verified recovery profiles (dev34)

Run one parent inside the allocated GPU job:

```bash
python -m multigpubfs doctor --recover --devices 0,1,2,3,4,5,6,7 --total-timeout 1800
```

After native acceptance, doctor exercises the actual CayleyPy API on three small graphs in one persistent session. It repeats this in a fresh process using ordinary `enable_native()` and automatic cache selection. Only successful checks publish a profile in `~/.cache/multigpubfs/recovery` (or `$XDG_CACHE_HOME/multigpubfs/recovery`). Set `MGBFS_RECOVERY_CACHE` to the same persistent directory in diagnostic and production jobs when needed.

Then use the ordinary API, without a profile filename or hand-written NCCL settings:

```python
import multigpubfs
from cayleypy import CayleyGraph, PermutationGroups
multigpubfs.enable_native()
with multigpubfs.persistent_graph_workers():
    for n in (10, 11, 12):
        result = CayleyGraph(PermutationGroups.coxeter(n), device="cpu").bfs()
        print(n, result.bfs_completed, result.completed_depth)
```

Activation prints the selected verified profile. Production remains automatically tuned and has no artificial time limit. The 30 minute budget belongs to diagnostics only. Nothing starts an unbounded production graph as a side effect of doctor.

Identity binds node/boot, GPU/driver evidence, Python, native/package code and relevant loader/transport environment. Other nodes and changed environments do not inherit a stale workaround: a warning explains that ordinary activation continues and doctor should be rerun. Explicit GPU selection is never silently replaced. Invalid/corrupt proofs are ignored with a warning. Profiles are immutable bundles with an atomic pointer, supporting concurrent writers and multiple nodes in shared storage.

Call `enable_native()` before creating resident GPU workers. An already running worker cannot change transport; activation rejects that case. `disable_native()` restores only environment values installed by the automatic profile and preserves subsequent user changes. Set `MGBFS_RECOVERY_AUTO=0` or use `enable_native(recovery=False)` to opt out. Explicit `enable_profile()` remains available.

Diagnostics contain `api-proof.json` and `auto-registration.json` plus the existing archive. A successful finite test does not prove absence of all driver faults, certify arbitrary large graphs, or promise peak transport speed. Failed production BFS is never silently restarted.
