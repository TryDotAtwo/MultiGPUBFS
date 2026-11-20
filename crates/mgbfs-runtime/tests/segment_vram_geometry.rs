use mgbfs_runtime::segmented_history::{fit_segment_layout, SegmentLayout, LOCAL_ROWS};

#[test]
fn fit_produces_maximal_exact_layout_including_hash_slot_rounding() {
    for bitmap in [false, true] {
        let make = |rows| if bitmap { SegmentLayout::bitmap(15, 1, rows).unwrap() }
                           else { SegmentLayout::new(15, 1, rows, 3).unwrap() };
        for rows in [1, 3, 7, 31, 255, 4096, 10000] {
            let layout = make(rows);
            for budget in [layout.device_bytes.saturating_sub(1), layout.device_bytes, layout.device_bytes + 29] {
                let got = fit_segment_layout(bitmap, 15, 1, 3, 10000, budget).unwrap();
                if let Some(candidate) = got {
                    assert!(candidate.device_bytes <= budget);
                    if candidate.rows_per_bank < 10000 {
                        assert!(make(candidate.rows_per_bank + 1).device_bytes > budget);
                    }
                } else { assert!(make(1).device_bytes > budget); }
            }
        }
    }
}

#[test]
fn eight_shards_are_sized_after_all_owner_lanes_and_allow_a_smaller_tail() {
    let free = 160u64 << 30;
    let reserve = 2u64 << 30;
    let bitmap_bytes = 20_432_412_000u64;
    let lane_bytes = 393_224u64;
    let available = free - reserve - bitmap_bytes - 4 * lane_bytes;
    let preferred = fit_segment_layout(true, 15, 1, 3, LOCAL_ROWS, available / 8).unwrap().unwrap();
    assert!(preferred.device_bytes * 8 <= available);
    let old = fit_segment_layout(true, 15, 1, 3, LOCAL_ROWS, (free-reserve-bitmap_bytes-lane_bytes)/8).unwrap().unwrap();
    assert!(old.device_bytes * 8 > available, "test must reproduce the old admission error");
    let tail_budget = preferred.device_bytes - 100_000;
    let tail = fit_segment_layout(true, 15, 1, 3, preferred.rows_per_bank, tail_budget).unwrap().unwrap();
    assert!(tail.rows_per_bank < preferred.rows_per_bank && tail.device_bytes <= tail_budget);
}

#[test]
fn configured_bounds_and_full_32_bit_cardinality_are_not_truncated() {
    assert!(fit_segment_layout(false, 14, 1, 3, 0, u64::MAX).is_err());
    assert!(fit_segment_layout(false, 14, 1, 3, LOCAL_ROWS + 1, u64::MAX).is_err());
    let full = SegmentLayout::new(14, 1, LOCAL_ROWS, 3).unwrap();
    assert_eq!(fit_segment_layout(false, 14, 1, 3, LOCAL_ROWS, full.device_bytes).unwrap().unwrap().rows_per_bank, LOCAL_ROWS);
}

#[test]
fn compact_positions_halve_storage_without_sacrificing_last_u32_slot() {
    let old = SegmentLayout::new(14, 1, 1000, 3).unwrap();
    let compact = SegmentLayout::new_compact(14, 1, 1000, 3).unwrap();
    assert_eq!(compact.slots, old.slots);
    assert_eq!(compact.position_bytes, old.position_bytes / 2);
    assert_eq!(compact.device_bytes, old.device_bytes - old.position_bytes / 2);
    assert_eq!(compact.position_item_bytes, 4);
    let maximum_32_bit_slot_count = SegmentLayout::new_compact(1, 1, (1u64 << 32) / 6, 3).unwrap();
    assert_eq!(maximum_32_bit_slot_count.slots, 1u64 << 32);
    assert_eq!(maximum_32_bit_slot_count.position_item_bytes, 4);
    let wide = SegmentLayout::new_compact(1, 1, (1u64 << 32) / 6 + 1, 3).unwrap();
    assert!(wide.slots > (1u64 << 32));
    assert_eq!(wide.position_item_bytes, 8);
    assert_eq!(SegmentLayout::new_compact(1, 1, 1, 1).unwrap().position_item_bytes, 0);
}

#[test]
fn bitmap_two_banks_are_admitted_as_a_separate_exact_layout() {
    let two = SegmentLayout::bitmap_banks(14, 1, 1000, 2).unwrap();
    let three = SegmentLayout::bitmap_banks(14, 1, 1000, 3).unwrap();
    assert_eq!(two.state_bytes, 28_000);
    assert_eq!(three.state_bytes, 42_000);
    assert_eq!(two.banks, 2);
    assert_eq!(fit_segment_layout(true, 14, 1, 2, 1000, two.device_bytes).unwrap(), Some(two));
    assert!(SegmentLayout::bitmap_banks(14, 1, 1000, 1).is_err());
}
