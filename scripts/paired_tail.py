"""Two independent hash-seeded searches; compare deterministic layer tables."""
import copy
import json
import time
from pathlib import Path
try:
    from .bfs_tail_archive import atomic_json
except ImportError:
    from bfs_tail_archive import atomic_json

SEEDS = ['000000000000000000000000013527dc', '6a09e667f3bcc909bb67ae8584caa73b']

def publication_records(ledger):
    records = dict(ledger['cases'])
    for record in ledger['cases'].values():
        for replica in record.get('replicas', []):
            key = replica['key']
            if key in records: raise ValueError('duplicate replica identity')
            records[key] = replica
    return records

def compare_tables(first, second, seeds_verified):
    fields = ('graph', 'packing', 'status', 'last_completed_layer')
    differences = [key for key in fields if first.get(key) != second.get(key)]
    a = [(x['depth'], x['states']) for x in first.get('layers', [])]
    b = [(x['depth'], x['states']) for x in second.get('layers', [])]
    if a != b: differences.append('layer_state_counts')
    if not seeds_verified: differences.append('native_seeds_unverified')
    if not a or not b: differences.append('empty_completed_layer_table')
    return dict(matched=not differences, differences=differences,
        scope='Completed-layer counts and status only; timings/VRAM excluded. Equal counts do not prove identical state sets.',
        complete_graph_verified=not differences and first.get('status')=='COMPLETE')

def run_pair(config, source, case, runtime, runner, failure_snapshot):
    case = Path(case)
    paths = [case, case.parent/(case.name+'-rep2')]
    manifests = []; timings = []; seeds_ok = True
    for index, path in enumerate(paths):
        cfg = copy.deepcopy(config)
        cfg['run_id'] = config['run_id'] + ('' if index==0 else '-rep2')
        cfg.setdefault('env', {})['MGBFS_HASH_SEED_HEX'] = SEEDS[index]
        cfg['repetition'] = index+1
        at = time.monotonic()
        try: manifest_path = runner(cfg, source, path, runtime)
        except Exception as error:
            manifest_path = path/'saved/manifest.json'
            if not manifest_path.exists():
                manifest_path = failure_snapshot(cfg, source, path, str(error))
        manifest = json.loads(Path(manifest_path).read_text())
        native = []
        for rank in range(config['world']):
            try: native.append(json.loads((path/'result'/f'rank-{rank}.json').read_text()))
            except (OSError, ValueError): pass
        verified = len(native)==config['world'] and all(x.get('hash_seed_hex')==SEEDS[index] for x in native)
        seeds_ok = seeds_ok and verified
        searches = [x.get('search_complete_seconds', x.get('search_prefix_seconds')) for x in native]
        timings.append(dict(repetition=index+1, seed_hex=SEEDS[index], native_seed_verified=verified,
            runner_wall_seconds=time.monotonic()-at,
            production_search_seconds=max(searches) if len(searches)==config['world'] and all(type(x) in (int,float) for x in searches) else None))
        manifest['repetition'] = index+1
        manifest['hash_seed_hex'] = SEEDS[index]
        manifests.append(manifest)
        if index == 0:
            try:
                from .sweep_tail_bfs import resource_stop, allocation_failure
            except ImportError:
                from sweep_tail_bfs import resource_stop, allocation_failure
            first = dict(status=manifest['status'], attempted=True,
                         reason=manifest.get('stop_reason', ''))
            if allocation_failure(path, source):
                first['resource_classification'] = 'cuda_allocation_failure'
            if resource_stop(first):
                comparison = dict(matched=None, differences=[],
                    status='NOT_COMPARED', second_run_skipped=True,
                    skip_reason='first_run_resource_exhausted',
                    complete_graph_verified=False, runs=timings,
                    scope='Only one run: confirmed resource exhaustion; no comparison performed.')
                manifest['comparison'] = comparison
                manifest['replicas'] = []
                atomic_json(path/'saved/manifest.json', manifest)
                atomic_json(case/'comparison.json', comparison)
                return path/'saved/manifest.json'
    comparison = compare_tables(*manifests, seeds_ok)
    comparison['runs'] = timings
    replica = dict(key=paths[1].name, n=config['n'], m=config['r'], attempted=True,
        status=manifests[1]['status'], last_completed_layer=manifests[1]['last_completed_layer'],
        reason=manifests[1]['stop_reason'], repetition=2, timing=timings[1])
    manifests[0]['replicas'] = [replica]
    for path, manifest in zip(paths, manifests):
        manifest['comparison'] = comparison
        atomic_json(path/'saved/manifest.json', manifest)
    atomic_json(case/'comparison.json', comparison)
    return case/'saved/manifest.json'
