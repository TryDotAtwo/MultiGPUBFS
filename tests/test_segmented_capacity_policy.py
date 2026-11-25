"""Capacity-first autotuning must distinguish preset defaults from user limits."""
import os
import subprocess
import time
from unittest.mock import patch

import pytest
from multigpubfs import autotune
from multigpubfs.graph_definition import GraphDefinition
from multigpubfs.launch import _apply_memory_preset


def test_expired_startup_budget_does_not_start_optional_admission(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("expired profile started native admission")

    monkeypatch.setattr(autotune, "_admit", unexpected)
    graph = GraphDefinition.permutation([[1, 0]], [0, 1])
    result = autotune.choose_profile(
        graph, [0], None, 60, "mgbfs", {}, pilot_limit=3, pilot_wall_seconds=5,
        startup_deadline=time.monotonic() - 1,
    )
    assert result["status"] == "TUNING_BUDGET_EXHAUSTED_CONSERVATIVE_PROFILE"
    assert result["shards"] is None and not result["measured"]


def measure(tmp_path, *, explicit_lane=None, explicit_queue=None,
            capacity_loss=False, bitmap=False, slow_baseline=False, use_cache=False, free_bytes=100000, very_short=False, include_inventory=True, invalid_queue_receipt=False, cached_bytes=0, pool_bytes=0, total_bytes=200000, reserve_bytes=1000,
            bounded=False, preserve_implicit_bitmap_banks=False, timeout_first=False, trace=None):
    native = tmp_path / 'native'
    native.write_bytes(b'native test executable')
    graph = GraphDefinition.permutation([[1, 0, 2], [0, 2, 1]], [0, 1, 2])
    calls = []
    env = {'MGBFS_GENERIC_HISTORY': 'hash', 'MGBFS_EXPERIMENTAL_SEGMENTED24': '1'}
    backend = 'permutation_bitmap' if bitmap else 'segmented24'
    if bitmap:
        env['MGBFS_EXPERIMENTAL_BITMAP'] = '1'
    if explicit_lane is not None:
        env['MGBFS_SEGMENT_OWNER_LANES'] = str(explicit_lane)
    if explicit_queue is not None:
        env['MGBFS_SEGMENT_QUEUE_CAP'] = str(explicit_queue)
    env = _apply_memory_preset(env, backend, 'max_capacity')
    inventory = [
        {'device': d, 'free_bytes': free_bytes, 'total_bytes': total_bytes,
         'cached_reusable_bytes': cached_bytes, 'pool_reusable_bytes': pool_bytes,
         'compute_major': 8, 'compute_minor': 6, 'sm_count': 28,
         'additional_reserve_bytes': reserve_bytes} for d in [0, 1]] if include_inventory else []

    def admit(graph, devices, capacity, shards, native, env, directory, **kwargs):
        return {'devices': [0, 1], 'inventory': inventory, 'plan': {
            'capacity': 100000, 'batch': 131072, 'queue_capacity': 16384,
            'history_algorithm': 'HASH', 'history_layers': 3, 'owner_lanes': 0}}

    def pilot(graph, target, **kw):
        actual = kw['_native_env']
        lane = int(actual['MGBFS_SEGMENT_OWNER_LANES'])
        queue = int(actual['MGBFS_SEGMENT_QUEUE_CAP'])
        calls.append((lane, queue))
        if trace is not None:
            trace.append((kw['memory_preset'], actual.get('MGBFS_SEGMENT_COMPACT_POS'),
                          actual.get('MGBFS_BITMAP_BANKS'), kw.get('_profile_deadline'), kw['max_seconds']))
        if timeout_first and len(calls)==1:
            raise RuntimeError('AUTOTUNE_PILOT_WALL_DEADLINE; evidence=test')
        budget = (10500 if queue == 2048 else 10000) if capacity_loss == 'partial' else (9000 if capacity_loss and queue > 2048 else 10000)
        speed = 0.1 if (lane, queue) == (4, 16384) else 1.0
        receipts = [{'owner_lanes': lane, 'exact_exchange': False,
                     'actual_queue_capacity': queue, 'storage_budget_bytes': budget,
                     'bytes_per_segment': 1000, 'rows_per_bank': 100,
                     'capacity_limit_per_rank': None, 'capacity_limit_is_hard': False,
                     'capacity_limit_scope': 'one next layer per rank',
                     'segment_rows_per_bank_by_shard': [[100]], 'peak_live_rows': 1} for _ in range(2)]
        if invalid_queue_receipt:
            receipts[1]['actual_queue_capacity'] = queue + 1
        depth = 3 if very_short and queue == 2048 else (4 if slow_baseline and queue == 2048 else (8 if slow_baseline or very_short else 4))
        return {'backend': 'EXPERIMENTAL_PERMUTATION_BITMAP' if bitmap else 'EXPERIMENTAL_SEGMENTED24',
                'segment_memory_by_rank': receipts,
                'layer_sizes': [1] + [20000] * depth,
                'layer_seconds': [speed] * depth,
                'status': 'INCOMPLETE', 'reason': 'PROFILE_LAYER_LIMIT'}

    missing = subprocess.CompletedProcess([], 0 if use_cache else 1, 'mock', '')
    dependency = {'cuda_library_sha256': 'dummy', 'resolved_dependencies': []} if use_cache else None
    policy = {'shards': [1, 4, 8], 'probe_capacity_per_rank': 50000}
    with (
        patch.dict(os.environ, {'MGBFS_PROFILE_CACHE': str(tmp_path / 'profiles')}),
        patch.object(autotune, '_admit', side_effect=admit),
        patch.object(autotune, '_system_info', return_value=missing),
        patch.object(autotune, '_dependency_identity', return_value=dependency),
        patch('multigpubfs.hardware_profiles.startup_policy', return_value=policy),
        patch('multigpubfs.hardware_profiles.small_permutation_workload', return_value=False),
        patch('multigpubfs.launch.run_graph', side_effect=pilot),
    ):
        bounded_options = {'capacity_slowdown_limit': 4.0, 'pilot_limit': 3, 'pilot_wall_seconds': 5} if bounded else {}
        result = autotune.choose_profile(
            graph, [0, 1], None, 90, str(native), env,
            allow_specialized=False, fixed_shards=4,
            preserve_implicit_bitmap_banks=preserve_implicit_bitmap_banks, **bounded_options)
    return result, calls, env


@pytest.mark.parametrize('bitmap', [False, True])
def test_auto_preset_measures_queue_and_lane_without_capacity_loss(tmp_path, bitmap):
    result, calls, env = measure(tmp_path, bitmap=bitmap)
    assert (1, 2048) in calls
    assert (4, 16384) in calls
    assert result['segment_owner_lanes'] == 4
    assert result['segment_queue_cap'] == 16384
    selected = autotune._profile_environment(env, result)
    assert selected['MGBFS_SEGMENT_QUEUE_CAP'] == '16384'
    assert selected['MGBFS_SEGMENT_OWNER_LANES'] == '4'


def test_estimated_capacity_takes_priority_over_faster_candidate(tmp_path):
    result, calls, _ = measure(tmp_path, capacity_loss=True)
    assert any(q > 2048 for _, q in calls)
    assert result['segment_queue_cap'] == 2048
    assert result['pilots'][result['winner_index']]['capacity_score_rows'] == 1000


@pytest.mark.parametrize('lane,queue', [(1, None), (None, 2048), (3, 6000)])
def test_explicit_user_geometry_is_never_overridden(tmp_path, lane, queue):
    result, calls, _ = measure(tmp_path, explicit_lane=lane, explicit_queue=queue)
    if lane is not None:
        assert all(l == lane for l, _ in calls)
        assert result['segment_owner_lanes'] == lane
    if queue is not None:
        assert all(q == queue for _, q in calls)
        assert result['segment_queue_cap'] == queue


def test_cache_distinguishes_implicit_from_explicit_default(tmp_path):
    _, _, implicit = measure(tmp_path)
    _, _, explicit = measure(tmp_path, explicit_lane=1, explicit_queue=2048)
    assert autotune._configuration_digest(implicit) != autotune._configuration_digest(explicit)


def test_capacity_priority_keeps_shorter_slow_prefix_candidate(tmp_path):
    result, calls, _ = measure(tmp_path, capacity_loss=True, slow_baseline=True)
    assert any(q > 2048 for _, q in calls)
    assert result['segment_queue_cap'] == 2048
    assert result['pilots'][result['winner_index']]['capacity_score_rows'] == 1000


def test_no_tuning_preserves_conservative_preset_defaults():
    env = _apply_memory_preset({}, 'segmented24', 'max_capacity')
    assert env['MGBFS_SEGMENT_OWNER_LANES'] == '1'
    assert env['MGBFS_SEGMENT_QUEUE_CAP'] == '2048'


def test_measured_profile_cache_can_be_written_and_read(tmp_path):
    first, first_calls, _ = measure(tmp_path, use_cache=True)
    second, second_calls, _ = measure(tmp_path, use_cache=True)
    assert first_calls and not first['cache_hit']
    assert not second_calls and second['cache_hit']
    assert second['segment_queue_cap'] == first['segment_queue_cap']



def test_partial_segment_memory_is_considered_before_speed(tmp_path):
    result, calls, _ = measure(tmp_path, capacity_loss='partial')
    assert any(q > 2048 for _, q in calls)
    assert result['segment_queue_cap'] == 2048
    assert result['pilots'][result['winner_index']]['capacity_score_rows'] == 1000


def test_capacity_cache_tracks_fresh_free_memory(tmp_path):
    first, _, _ = measure(tmp_path, use_cache=True, free_bytes=100000)
    same, calls, _ = measure(tmp_path, use_cache=True, free_bytes=100000)
    changed, changed_calls, _ = measure(tmp_path, use_cache=True, free_bytes=99000)
    assert same['cache_hit'] and not calls
    assert changed_calls and not changed['cache_hit']



def test_short_high_capacity_probe_cannot_silently_lose_priority(tmp_path):
    result, calls, env = measure(tmp_path, capacity_loss=True, very_short=True)
    assert any(q > 2048 for _, q in calls)
    assert not result['measured']
    assert result['status'] == 'CAPACITY_COMPARISON_INSUFFICIENT_PREFIX'
    selected = autotune._profile_environment(env, result)
    assert selected['MGBFS_SEGMENT_OWNER_LANES'] == '1'
    assert selected['MGBFS_SEGMENT_QUEUE_CAP'] == '2048'





def test_cache_not_used_without_live_capacity_inventory(tmp_path):
    first, _, _ = measure(tmp_path, use_cache=True, include_inventory=False)
    second, calls, _ = measure(tmp_path, use_cache=True, include_inventory=False)
    assert not first['cache_hit'] and not second['cache_hit'] and calls


def test_inconsistent_queue_receipt_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match='QUEUE_RECEIPT_MISMATCH'):
        measure(tmp_path, invalid_queue_receipt=True)


def test_capacity_cache_reuses_equal_effective_inventory_after_backend_switch(tmp_path):
    # Native inventory's free_bytes already includes both reusable pools.
    # Changes in their split alone do not change admission capacity.
    first, first_calls, _ = measure(tmp_path, use_cache=True, cached_bytes=100, pool_bytes=900)
    second, second_calls, _ = measure(tmp_path, use_cache=True, cached_bytes=900, pool_bytes=100)
    assert first_calls and not first['cache_hit']
    assert second['cache_hit'] and not second_calls
    assert second['segment_queue_cap'] == first['segment_queue_cap']

@pytest.mark.parametrize('change', [
    {'free_bytes': 99999}, {'total_bytes': 200001}, {'reserve_bytes': 1001},
])
def test_capacity_cache_rechecks_actual_inventory_dimensions(tmp_path, change):
    first, _, _ = measure(tmp_path, use_cache=True)
    changed, changed_calls, _ = measure(tmp_path, use_cache=True, **change)
    assert not first['cache_hit']
    assert changed_calls and not changed['cache_hit']


@pytest.mark.parametrize('bitmap', [False, True])
def test_capacity_bound_times_a_real_balanced_anchor_and_rejects_slow_capacity(tmp_path, bitmap):
    trace = []
    result, calls, initial_env = measure(tmp_path, bitmap=bitmap, capacity_loss=True, bounded=True,
                                         preserve_implicit_bitmap_banks=not bitmap, trace=trace)
    assert calls == [(4, 16384), (4, 16384), (1, 2048)]
    assert len(trace) == 3 and all(isinstance(probe[3], float) and probe[4] <= 4 for probe in trace)
    assert len({probe[3] for probe in trace}) == 1
    assert trace[0][:3] == ('balanced', None, None)
    assert trace[1][:3] == (('max_capacity', None, '2') if bitmap else ('max_capacity', '1', None))
    bound = result['capacity_slowdown']
    assert bound['status'] == 'MEASURED_MATCHED_PREFIX_BOUND'
    assert bound['timing_scope'] == 'COMPLETED_LAYER_PREFIX'
    assert bound['rejected_over_limit'] == ['pilot-2']
    assert bound['observed_ratio'] == 1.0
    assert result['segment_runtime_preset'] == 'balanced'
    final_env = autotune._profile_environment(initial_env, result)
    assert final_env['MGBFS_MEMORY_PRESET'] == 'balanced'
    assert 'MGBFS_SEGMENT_COMPACT_POS' not in final_env
    assert 'MGBFS_BITMAP_BANKS' not in final_env


def test_automatic_capacity_pilots_keep_implicit_bitmap_banks(tmp_path):
    trace = []
    result, calls, initial_env = measure(tmp_path, bitmap=True, bounded=True,
                                         preserve_implicit_bitmap_banks=True, timeout_first=True, trace=trace)
    assert 2 <= len(calls) <= 3
    assert all(probe[:3] == ('max_capacity', None, '2') for probe in trace)
    assert result['timed_out_trials'] == ['pilot-0']
    assert not result['measured']
    assert result['capacity_slowdown']['status'] == 'UNVERIFIED'
    assert 'observed_ratio' not in result['capacity_slowdown']
    final_env = autotune._profile_environment(initial_env, result)
    assert final_env['MGBFS_MEMORY_PRESET'] == 'max_capacity'
    assert final_env['MGBFS_BITMAP_BANKS'] == '2'


def test_lost_balanced_pilot_does_not_certify_slowdown(tmp_path):
    result, calls, initial_env = measure(tmp_path, bounded=True, timeout_first=True, capacity_loss=True)
    assert len(calls) == 3
    assert result['timed_out_trials'] == ['pilot-0']
    assert result['capacity_slowdown']['status'] == 'UNVERIFIED'
    assert not result['measured']
    assert result['apply_profile_batch'] is True and result['capacity_planning_candidate'] == 'pilot-2'
    final_env = autotune._profile_environment(initial_env, result)
    assert final_env['MGBFS_MEMORY_PRESET'] == 'max_capacity'
    assert final_env['MGBFS_SEGMENT_QUEUE_CAP'] == '2048'


@pytest.mark.parametrize('has_partial', [False, True])
def test_complete_bfs_bound_uses_actual_total_time_not_completed_layer_sum(has_partial):
    common = {'status': 'COMPLETE', 'layer_seconds': [1.0]*5}
    anchor = dict(common, candidate='baseline', segment_runtime_preset='balanced',
                  capacity_score_rows=1000, bfs_seconds=10.)
    candidate = dict(common, candidate='capacity', segment_runtime_preset='max_capacity',
                     capacity_score_rows=2000, bfs_seconds=41.)
    pilots = [anchor, candidate]
    if has_partial:
        pilots.append({'candidate': 'prefix', 'status': 'INCOMPLETE', 'layer_seconds': [1.0]*4,
                       'segment_runtime_preset': 'max_capacity', 'capacity_score_rows': 500})
    _, _, winner, _, evidence = autotune._bounded_segmented_profile(pilots, 4.)
    assert winner == 0
    assert evidence['status'] == 'MEASURED_FULL_BFS_BOUND'
    assert evidence['timing_scope'] == 'FULL_BFS'
    assert evidence['baseline_seconds'] == 10.
    assert evidence['rejected_over_limit'] == ['capacity']


def test_partial_pilot_does_not_replace_faster_full_balanced_anchor():
    complete = {'status': 'COMPLETE', 'layer_seconds': [2.0] * 5}
    full_anchor = dict(complete, candidate='full-anchor', segment_runtime_preset='balanced',
                       capacity_score_rows=1000, bfs_seconds=10.0)
    partial_anchor = dict(candidate='prefix-anchor', segment_runtime_preset='balanced',
                          capacity_score_rows=1000, status='INCOMPLETE', layer_seconds=[1.0] * 4)
    capacity = dict(complete, candidate='capacity', segment_runtime_preset='max_capacity',
                    capacity_score_rows=2000, bfs_seconds=41.0)
    _, _, _, _, evidence = autotune._bounded_segmented_profile([full_anchor, partial_anchor, capacity], 4.0)
    assert evidence['rejected_over_limit'] == ['capacity']


def test_balanced_anchor_and_fallback_drop_only_implicit_packet_transport():
    address = 'MGBFS_SEGMENT_PACKET_CONTROL_ADDR'
    ring = {'MGBFS_SEGMENT_PACKET_RING_SLOTS': '3', 'MGBFS_SEGMENT_PACKET_RX_SLOTS': '3'}
    explicit = {address: '127.0.0.1:12345', 'MGBFS_SEGMENT_EXACT_EXCHANGE': '1', **ring}
    implicit = dict(explicit, _MGBFS_PRESET_IMPLICIT_PACKET='1')
    fallback = {'segment_runtime_preset': 'balanced', 'segment_exact_exchange': True}
    capacity = dict(fallback, segment_runtime_preset='max_capacity')
    packet = [address, *ring]
    assert all(key not in autotune._balanced_segment_environment(implicit) for key in packet)
    fallback_env = autotune._profile_environment(implicit, fallback)
    assert all(key not in fallback_env for key in packet)
    for selected in (autotune._profile_environment(explicit, fallback),
                     autotune._profile_environment(implicit, capacity)):
        assert [selected[key] for key in packet] == [explicit[key] for key in packet]
