"""Memory policies are independent of backend choice and explicitly observable."""
import pytest
from multigpubfs.cayleypy_backend import NativeOptions
from multigpubfs.launch import _apply_memory_preset
from multigpubfs.autotune import _pick_segmented_profile


def test_options_validate_policy():
    assert NativeOptions().memory_preset == 'balanced'
    assert NativeOptions(memory_preset='max_capacity').memory_preset == 'max_capacity'
    for value in ('fastest', '', None, 0):
        with pytest.raises(ValueError, match='INVALID_MEMORY_PRESET'):
            NativeOptions(memory_preset=value)


def test_policy_preserves_explicit_controls_and_source():
    source = {'MGBFS_SEGMENT_OWNER_LANES':'3', 'MGBFS_SEGMENT_QUEUE_CAP':'17', 'MGBFS_SEGMENT_COMPACT_POS':'0'}
    result = _apply_memory_preset(source, 'segmented24', 'max_capacity')
    assert result is not source
    assert result['MGBFS_MEMORY_PRESET'] == 'max_capacity'
    assert result['MGBFS_SEGMENT_OWNER_LANES'] == '3'
    assert result['MGBFS_SEGMENT_QUEUE_CAP'] == '17'
    assert result['MGBFS_SEGMENT_COMPACT_POS'] == '0'
    assert 'MGBFS_MEMORY_PRESET' not in source


def test_policy_defaults_and_bitmap_choice():
    h = _apply_memory_preset({}, 'segmented24', 'max_capacity')
    b = _apply_memory_preset({}, 'permutation_bitmap', 'max_capacity')
    assert h['MGBFS_SEGMENT_OWNER_LANES'] == '1'
    assert h['MGBFS_SEGMENT_QUEUE_CAP'] == '2048'
    assert h['MGBFS_SEGMENT_COMPACT_POS'] == '1'
    assert b['MGBFS_BITMAP_BANKS'] == '2'
    assert 'MGBFS_SEGMENT_COMPACT_POS' not in b
    assert _apply_memory_preset({}, 'segmented24', 'balanced') == {'MGBFS_MEMORY_PRESET':'balanced'}
    with pytest.raises(ValueError, match='MEMORY_PRESET_UNSUPPORTED'):
        _apply_memory_preset({}, 'legacy', 'max_capacity')


def test_measured_selection_uses_capacity_then_speed_on_tie():
    pilots = [{'capacity_score_rows':100}, {'capacity_score_rows':120}, {'capacity_score_rows':120}]
    assert _pick_segmented_profile(pilots, [1., 4., 3.], 'balanced') == 0
    assert _pick_segmented_profile(pilots, [1., 4., 3.], 'max_capacity') == 2
    with pytest.raises(RuntimeError, match='CAPACITY_EVIDENCE'):
        _pick_segmented_profile([{}], [1.], 'max_capacity')
