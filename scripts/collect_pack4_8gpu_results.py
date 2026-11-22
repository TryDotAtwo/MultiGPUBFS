#!/usr/bin/env python3
"""Assemble the measured packed-bitmap evidence; absent optional tests stay unverified."""
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

repo = Path('/workspace/mgbfs-pack4')
ev = Path('/workspace/eight-gpu-evidence')
results = Path('/workspace/mgbfs-pack4-evidence')
output = repo / 'docs/acceptance/2026-11-22-pack4-8gpu-results.json'

def digest(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def read_json(path):
    return json.loads(Path(path).read_text())

def marker(log, token):
    p = ev / log
    if not p.is_file():
        return {'status': 'NOT_RUN_OR_NO_LOG', 'log': log}
    data = p.read_text(errors='replace')
    return {'status': 'MARKER_OBSERVED' if token in data else 'NOT_ESTABLISHED',
            'log': log, 'expected_marker': token, 'log_sha256': digest(p)}

oracle = read_json(ev/'mahonian-window-oracle.json')
benchmarks = []
expected = [
    results/'benchmark-s12-max_capacity-8gpu-30623/summary.json',
    results/'benchmark-s13-max_capacity-8gpu-2031694/summary.json',
]
expected += [Path(x) for x in sorted(glob.glob(str(results/'benchmark-s13-max_capacity-8gpu-shards8-*/summary.json')))]
expected += [Path(x) for x in sorted(glob.glob(str(results/'benchmark-nvml-ab-s12-max_capacity-8gpu-*/summary.json')))]
for summary in expected:
    if not summary.exists():
        continue
    for item in read_json(summary):
        record = dict(item)
        case = oracle.get(str(record['n']))
        run = summary.parent / ('run-%s-%s' % (record['run'], '1' if record['codec']=='pack4-v1' else '0'))
        # Some test helpers label the codec differently; derive from the actual record when needed.
        if not run.exists():
            possibilities = list(summary.parent.glob('run-%s-*' % record['run']))
            run = possibilities[0] if len(possibilities)==1 else run
        if (run/'launch.json').exists():
            launch = read_json(run/'launch.json')
            record['launch_receipt'] = {k:launch.get(k) for k in ('devices','graph_digest','plan','reserve_policy','profile_status')}
            record['rank_plan_limits'] = [{k:rank.get(k) for k in ('batch','capacity','device_bytes','queue_capacity','shards')} for rank in launch.get('rank_plans',[])]
        if (run/'report.json').exists():
            report = read_json(run/'report.json')
            rank_keys = ('rank','device','admitted_batch','queue_batch_ceiling','batch_recovery','batch_recoveries','prefetched_chunks','actual_queue_capacity','owner_lanes','storage_allocated_bytes','storage_peak_requested_bytes','segment_index_allocated_bytes','segment_state_allocated_bytes','segment_counter_allocated_bytes','bitmap_bytes','queues_allocated_bytes','owner_spill_allocated_bytes','extra_send_bytes','reserve_bytes','device_free_bytes','device_total_bytes','peak_live_rows','rows_per_bank','configured_segment_rows','segment_count','capacity_limit_scope','capacity_limit_per_rank','segment_strategy','resident_codec','bitmap_owner_page')
            record['runtime_metrics_by_rank'] = [{k:rank.get(k) for k in rank_keys} for rank in report.get('segment_memory_by_rank',[])]
            record['layer_sizes'] = report.get('layer_sizes')
            record['layer_seconds'] = report.get('layer_seconds')
            seconds = record['layer_seconds']
            sizes = record['layer_sizes']
            if seconds is None or sizes is None or len(seconds) != len(sizes)-1:
                raise ValueError('layer time indexing not established: '+str(run))
            if case and record.get('window_depths') == case['depths']:
                lo, hi = case['depths']
                if not 1 <= lo <= hi <= len(seconds):
                    raise ValueError('invalid timed window: '+str(run))
                corrected = sum(seconds[lo-1:hi])
                record['original_summary_window_seconds'] = record['window_seconds']
                record['window_seconds'] = corrected
                record['window_time_indexing'] = 'layer_seconds[d-1] times creation of depth d'
                record['window_new_states'] = case['window_new_states']
                record['window_generated_edges'] = case['window_generated_edges']
                record['window_new_states_per_second'] = case['window_new_states']/corrected
                record['window_generated_edges_per_second'] = case['window_generated_edges']/corrected
        record['summary_source'] = str(summary.relative_to(results))
        benchmarks.append(record)

capacity_path = ev/'s14-s16-mahonian-calculation.jsonl'
capacity = [json.loads(s) for s in capacity_path.read_text().splitlines() if s.strip()]
files = [
    'tests/bitmap_pack4_timing.py',
    'tests/test_bitmap_pack4_timing.py',
    'tests/bitmap_pack4_runtime_gpu.py',
    'tests/bitmap_pack4_cayleypy_gpu.py',
    'tests/bitmap_pack4_benchmark_gpu.py',
    'tests/bitmap_pack4_rank_failure_gpu.py',
    'tests/bitmap_pack4_shards_gpu.py',
    'tests/bitmap_pack4_nvml_ab_gpu.py',
    'tests/segmented_resource_public_gpu.py',
    'scripts/mahonian_capacity_lower_bound.py',
    'scripts/mahonian_cyclic_rank_capacity.py',
    'tests/test_mahonian_cyclic_rank_capacity.py',
    'scripts/collect_pack4_8gpu_results.py',
]
out = {
 'schema': 1,
 'created_utc': datetime.now(timezone.utc).isoformat(),
 'scope': 'experimental source build; eight RTX 3060; not a released wheel or B200 acceptance',
 'source_code_checkpoint': 'e0cf4098f06ca9a01b5c19f02952061a8845227c',
 'original_benchmark_public_checkpoint': 'da7a4e51609b154d05f2f1b03ae947ddae87b1ea',
 'public_correction_checkpoint': '78bf61482d10935707ab4c220d09406a623ea463',
 'cayleypy_source_commit': '2d35a8917ee1d123d6ec8c771bc08002c974a57f',
 'hardware': {'world':8,'gpu':'RTX 3060 12 GiB SM86','driver':'580.178.04','cuda':'12.8','nccl':'2.30.7',
              'topology':(ev/'topology.txt').read_text(),'device_inventory_csv':(ev/'gpus.csv').read_text()},
 'native_sha256': {'executable':digest('/workspace/mgbfs-pack4/target/release/mgbfs'),
                   'cuda_library':digest('/workspace/mgbfs-native-build/libmgbfs_cuda.so')},
 'dependencies_sha256': (ev/'dependency-sha256.txt').read_text(),
 'loaded_module_maps_sha256': digest(ev/'loaded-libraries-during-s12.txt'),
 'measured_scripts_sha256': {f:digest(repo/f) for f in files},
 'small_multiset_4shard_test_script_sha256':digest(ev/'bitmap_pack4_runtime_gpu-baseline.py'),
 's12_s13_benchmark_script_as_run_sha256':digest(ev/'bitmap_pack4_benchmark_gpu-used-s12-s13.py'),
 'baseline_layer_oracle': oracle,
 'correctness': {
   'native_full_multiset_small_graphs_4_shards': marker('pack4-exact-layers.log','PACK4_RUNTIME_EXACT_LAYERS_PASS'),
   'native_full_multiset_small_graphs_8_shards': marker('pack4-exact-layers-shards8.log','PACK4_RUNTIME_EXACT_LAYERS_PASS'),
   'cayleypy_persistent_small_graphs': marker('pack4-cayleypy.log','PACK4_CAYLEYPY_MULTI_GPU_PASS'),
   'cayleypy_tight_cap_autotune': marker('pack4-autotune.log','PUBLIC_RESOURCE_GPU_PASS'),
   'single_rank_failure_and_fresh_pool': marker('pack4-rank-failure.log','PACK4_RANK_FAILURE_PASS'),
   'large_graphs': 'all layer counts checked against Mahonian oracle; full state sets checked only for final and penultimate layers',
   'full_layer_archive': 'not established for the segmented CayleyPy path',
 },
 'measurement_caveats': [
   'same native executable/library across comparisons; device and exact geometry must still be compared per run',
   'NVML maxima are samples with a 0.5-second post-query delay; actual intervals can be longer, short peaks may be missed, and these are not exact allocator high-water',
   'per-rank peak live row maxima need not occur simultaneously',
   '2 GiB reserve in geometry is an admission allowance, not a memory allocation',
   'one S13 observation per codec cannot establish normal variation or production defaults',
   'original S12/S13 benchmark summaries had a one-layer time-window offset; this receipt recalculates from recorded layer_seconds',
 ],
 'benchmarks': benchmarks,
 'mahonian_capacity_arithmetic_only': capacity,
 'cyclic_owner_capacity_arithmetic_only': [json.loads(line) for line in (ev/'cyclic-rank-capacity.jsonl').read_text().splitlines() if line.strip()],
 'cyclic_owner_capacity_physical_peak_receipt_checks': read_json(ev/'cyclic-rank-capacity-receipt-check.json'),
 'cyclic_owner_capacity_cpu_test_log_sha256': digest(ev/'cyclic-rank-capacity-tests.log'),
}
output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print('WROTE',output,'benchmark_records',len(benchmarks),'bytes',output.stat().st_size)
