"""GPU-host startup calibration; packed states never leave the host.

Whole traversals or a matched prefix stopped at a native completed-layer
boundary. Both modes require every measured layer to remain fully archived.
"""
import copy
import hashlib
import json
import math
import subprocess
import shutil
import time
from pathlib import Path
from bfs_tail_archive import atomic_json
from graph_profile_selection import select_graph_profile


def full_state_fingerprint(manifest_path, *, calibration_limit=None, max_bytes=512<<20):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'COMPLETE' and not (
            calibration_limit is not None and manifest['status']=='INCOMPLETE'
            and manifest['stop_reason']=='calibration layer limit'
            and len(manifest['layers'])==calibration_limit
            and manifest['last_completed_layer']==calibration_limit-1):
        raise ValueError('calibration traversal incomplete')
    width = manifest['packing']['bytes_per_state']
    if sum(layer['states'] for layer in manifest['layers'])*width>max_bytes:
        raise ValueError('calibration comparison host memory bound exceeded')
    archive_root = (manifest_path.parent.parent if manifest_path.parent.name.startswith('snapshot-')
                    else manifest_path.parent)
    entries = {entry['depth']: entry for entry in manifest['files']}
    if len(entries) != len(manifest['files']):
        raise ValueError('duplicate archived depth')
    result = []
    for layer in manifest['layers']:
        entry = entries.get(layer['depth'])
        if not entry or entry['states'] != layer['states'] or not entry['full_layer']:
            raise ValueError('calibration archive omits complete layers')
        data = (archive_root / entry['path']).read_bytes()
        if (len(data) != layer['states'] * width or len(data) != entry['bytes']
                or hashlib.sha256(data).hexdigest() != entry['sha256']):
            raise ValueError('calibration archive checksum or count differs')
        states = sorted(data[i:i+width] for i in range(0, len(data), width))
        if any(a == b for a, b in zip(states, states[1:])):
            raise ValueError('duplicate calibration states')
        digest = hashlib.sha256()
        for state in states:
            digest.update(state)
        result.append((layer['depth'], layer['states'], digest.hexdigest()))
    return result


def calibrate(config, source, root, runtime, *, deadline, cancelled=None,
              runner=None, identity=None):
    """Return an auditable decision; calibration errors retain direct launches."""
    if runner is None:
        from run_tail_bfs import run
        runner = run
    source, root = Path(source), Path(root)
    root.mkdir(parents=True, exist_ok=False)
    decision = dict(graph_batches=0, policy='matched-complete-archive-v1',
                    status='NOT_CALIBRATED', samples=[])
    order = math.factorial(config['n']) // math.factorial(config['r'])
    width = 8 if config['n'] <= 16 else 16
    if config['env'].get('MGBFS_TRANSPORT_BACKEND') != 'NCCL_LSA':
        decision['reason'] = 'full-window Graph requires NCCL_LSA'
    elif order < config['world'] * config.get('batch',1) * 32:
        decision['reason'] = 'state count cannot supply a full Graph window per rank'
    elif deadline - time.time() < 30:
        decision['reason'] = 'insufficient calibration time'
    else:
        try:
            measurement_config=copy.deepcopy(config)
            max_bytes=min(512<<20,config.get('host_available_bytes',8<<30)//16)
            limit=34 if order*width>max_bytes else None
            if limit is not None:
                measurement_config['env']['MGBFS_CALIBRATION_LAYERS']=str(limit)
            decision['calibration_layers']=limit
            if identity is None:
                normalized = copy.deepcopy(measurement_config)
                normalized.pop('run_id', None)
                normalized.pop('timeout_seconds', None)
                normalized['env'].pop('MGBFS_CUDA_GRAPH_BATCHES', None)
                binary = source/config.get('binary_path', 'target/release/mgbfs')
                details = dict(configuration=normalized, runtime=runtime,
                    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version',
                        '--format=csv,noheader'],text=True).splitlines())
                identity = hashlib.sha256(json.dumps(details,sort_keys=True).encode()).hexdigest()
                decision['identity_details'] = details
            reference = None
            samples = []
            # Alternate order within each pair to limit thermal/order bias.
            modes = [0,32,32,0,0,32]
            for index, mode in enumerate(modes):
                if cancelled and cancelled():
                    raise RuntimeError('calibration cancelled')
                remaining = deadline-time.time()
                if remaining < 5:
                    raise TimeoutError('calibration deadline')
                cfg = copy.deepcopy(measurement_config)
                cfg.pop('repo_id', None)
                cfg['timeout_seconds'] = min(config.get('timeout_seconds',120),remaining)
                cfg['env']['MGBFS_CUDA_GRAPH_BATCHES'] = str(mode)
                case = root/f'run-{index}'
                manifest = runner(cfg,source,case,runtime,cancelled=cancelled)
                fingerprint = full_state_fingerprint(manifest,calibration_limit=limit,max_bytes=max_bytes)
                if reference is None:
                    reference = fingerprint
                elif fingerprint != reference:
                    raise ValueError('Graph calibration full-state parity differs')
                ranks = [json.loads((case/'result'/f'rank-{rank}.json').read_text())
                         for rank in range(config['world'])]
                from run_tail_bfs import native_completion
                complete,_=native_completion(ranks,limit,len(fingerprint))
                archived=json.loads(Path(manifest).read_text())
                samples.append(dict(configuration_identity=identity,pair=index//2,
                    graph_batches=mode,status='COMPLETE' if complete else 'PREFIX_COMPLETE',full_state_parity=True,
                    layers=archived['layers'],archive_status=archived['status'],
                    last_completed_layer=archived['last_completed_layer'],
                    vram_sampling_interval_seconds=archived['vram_sampling_interval_seconds'],
                    program_commit=archived['program_commit'],
                    binary_sha256=archived['launch_config'].get('binary_sha256'),
                    calibration_inputs_retained=True,
                    search_seconds=max(x.get('search_complete_seconds',x.get('search_prefix_seconds')) for x in ranks),
                    full_windows_per_rank=[(x.get('batch_graph') or {}).get('full_windows',0)
                                           for x in ranks]))
                decision['samples'] = samples
                atomic_json(root/'decision.json', decision)
                # Calibration inputs are temporary. Preserve native reports and
                # the decision, while preventing six state archives per case
                # from exhausting the GPU host's SSD during an automatic sweep.
                saved=case/'saved'
                if saved.is_symlink() or saved.resolve()!=case.resolve()/'saved' or case.resolve().parent!=root.resolve():
                    raise ValueError('calibration cleanup path outside owned case')
                shutil.rmtree(saved)
                samples[-1]['calibration_inputs_retained']=False
                atomic_json(root/'decision.json',decision)
            decision.update(select_graph_profile(samples), status='CALIBRATED')
        except Exception as error:
            decision.update(status='NOT_CALIBRATED',graph_batches=0,
                            reason=str(error),error_type=type(error).__name__)
    atomic_json(root/'decision.json',decision)
    return decision
