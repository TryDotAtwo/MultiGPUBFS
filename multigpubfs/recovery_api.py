"""Real public CayleyPy API acceptance, isolated by the diagnostic supervisor."""
import json
import os
import shutil
from pathlib import Path

def check(config,root):
    import multigpubfs
    from cayleypy import CayleyGraph,PermutationGroups
    from .graph_definition import from_cayleypy
    from .deep_diagnostics import oracle
    from .recovery_diagnostics import enable_profile
    from . import recovery_registry
    devices=config['devices']
    if config.get('auto'):
        os.environ['MGBFS_RECOVERY_CACHE']=config['cache']
        recovery_registry.install(config['profile'],config['api_proof'])
        multigpubfs.enable_native(multigpubfs.NativeOptions(output_root=root/'results'))
        if recovery_registry._applied is None:raise RuntimeError('RECOVERY_AUTO_NOT_APPLIED')
    else:enable_profile(config['profile'],output_root=root/'results')
    results=[];pids=[]
    try:
        with multigpubfs.persistent_graph_workers() as pool:
            try:
                for n in (3,4,3):
                    graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
                    layers,terminal=oracle(from_cayleypy(graph))
                    result=graph.bfs()
                    if not result.bfs_completed or result.layer_sizes!=layers or result.report.get('devices')!=devices:raise RuntimeError('CAYLEYPY_ORACLE_OR_DEVICE_MISMATCH')
                    saved=json.loads((result.output/'states.json').read_text())
                    if sorted(tuple(row) for row in saved['current'])!=terminal:raise RuntimeError('CAYLEYPY_TERMINAL_MISMATCH')
                    current=[w.process.pid for w in pool.workers]
                    if len(current)!=len(devices):raise RuntimeError('CAYLEYPY_WORKER_COUNT')
                    pids.append(current);results.append({'n':n,'layers':layers,'devices':devices,'worker_pids':current})
            finally:
                if pool.root.exists():shutil.copytree(pool.root,root/'session-evidence',dirs_exist_ok=True,ignore=shutil.ignore_patterns('job-*','*.bin','states*'))
        if not all(v==pids[0] for v in pids):raise RuntimeError('CAYLEYPY_PERSISTENT_NOT_REUSED')
        return {'devices':devices,'api_verified':True,'persistent_reuse':True,'automatic_profile':bool(config.get('auto')),'graphs':results}
    finally:multigpubfs.disable_native()
