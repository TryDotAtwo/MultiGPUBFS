use mgbfs_core::macro_memory::{MacroMemoryInput, MacroMemoryShape};

fn input() -> MacroMemoryInput {
    MacroMemoryInput {
        state_stride: 64,
        parent_batch: 1024,
        macro_count: 44,
        effective_depth: 4,
        layer_capacity: 40_320,
        future_capacity_per_depth: 100_000,
        route_slot_records: 45_056,
    }
}

#[test]
fn shape_accounts_two_k_history_future_depths_and_two_producer_banks() {
    let shape = MacroMemoryShape::derive(input()).unwrap();
    assert_eq!(shape.candidate_records, 45_056);
    assert_eq!(shape.history_layer_slots, 8);
    assert_eq!(shape.history_hash_records, 322_560);
    assert_eq!(shape.future_depth_slots, 4);
    assert_eq!(shape.future_records, 400_000);
    assert_eq!(shape.producer_state_bytes, 2 * 45_056 * 64);
    assert_eq!(shape.producer_hash_bytes, 2 * 45_056 * 16);
    assert_eq!(shape.future_state_bytes, 400_000 * 64);
    assert_eq!(shape.future_hash_ref_bytes, 400_000 * 24);
    assert_eq!(shape.history_hash_bytes, 322_560 * 16);
}

#[test]
fn depth_one_retains_the_existing_three_layer_semantic_window() {
    let mut value = input();
    value.macro_count = 3;
    value.effective_depth = 1;
    value.route_slot_records = 3072;
    let shape = MacroMemoryShape::derive(value).unwrap();
    assert_eq!(shape.history_layer_slots, 2);
    assert_eq!(shape.future_depth_slots, 1);
}

#[test]
fn every_invalid_or_insufficient_shape_fails_preflight() {
    let mut value = input();
    value.route_slot_records -= 1;
    assert_eq!(
        MacroMemoryShape::derive(value).unwrap_err(),
        "MACRO_ROUTE_CAPACITY"
    );
    for field in 0..5 {
        let mut value = input();
        match field {
            0 => value.state_stride = 0,
            1 => value.parent_batch = 0,
            2 => value.macro_count = 0,
            3 => value.effective_depth = 0,
            _ => value.future_capacity_per_depth = 0,
        }
        assert_eq!(
            MacroMemoryShape::derive(value).unwrap_err(),
            "MACRO_MEMORY_SHAPE"
        );
    }
}

#[test]
fn checked_arithmetic_rejects_candidate_history_future_and_byte_overflow() {
    let mut value = input();
    value.parent_batch = u64::MAX;
    assert_eq!(
        MacroMemoryShape::derive(value).unwrap_err(),
        "MACRO_CANDIDATE_OVERFLOW"
    );
    let mut value = input();
    value.effective_depth = u32::MAX;
    assert_eq!(
        MacroMemoryShape::derive(value).unwrap_err(),
        "MACRO_HISTORY_SLOT_OVERFLOW"
    );
    let mut value = input();
    value.future_capacity_per_depth = u64::MAX;
    assert_eq!(
        MacroMemoryShape::derive(value).unwrap_err(),
        "MACRO_FUTURE_OVERFLOW"
    );
}
