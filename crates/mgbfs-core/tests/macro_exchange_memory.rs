use mgbfs_core::{macro_memory::{MacroExchangeInput, MacroExchangeMemoryPlan}, wire::{payload_layout, FrameKind}};

#[test]
fn dense_exchange_plan_counts_every_live_send_and_receive_slot() {
    let plan = MacroExchangeMemoryPlan::derive(MacroExchangeInput {
        world: 2,
        candidate_capacity: 4096,
        state_stride: 16,
        route_slots: 2,
    }).unwrap();
    assert_eq!(plan.send_frame_bytes_per_slot, 198_560);
    assert_eq!(plan.receive_frame_bytes_per_slot, 197_584);
    assert_eq!(plan.owner_count_bytes_per_slot, 8);
    assert_eq!(plan.count_exchange_bytes_per_slot, 8);
    assert_eq!(plan.slot_bytes, 396_160);
    assert_eq!(plan.failure_collective_bytes, 8);
    assert_eq!(plan.requested_device_bytes, 792_328);
}

#[test]
fn dense_exchange_frame_bound_covers_every_small_owner_partition() {
    fn check_partitions(remaining: u32, peers: u32, parts: &mut Vec<u32>, plan: &MacroExchangeMemoryPlan) {
        if peers == 0 {
            if remaining != 0 { return; }
            let sent = parts.iter().filter(|&&n| n != 0).map(|&n|
                256 + payload_layout(FrameKind::MacroDense, n, 32).unwrap().bytes
            ).sum::<u64>();
            assert!(sent <= plan.send_frame_bytes_per_slot, "parts={parts:?}");
            for &n in parts.iter() {
                let received = if n == 0 { 0 } else {
                    256 + payload_layout(FrameKind::MacroDense, n, 32).unwrap().bytes
                };
                assert!(received <= plan.receive_frame_bytes_per_slot);
            }
            return;
        }
        for n in 0..=remaining {
            parts.push(n);
            check_partitions(remaining - n, peers - 1, parts, plan);
            parts.pop();
        }
    }
    for world in [1, 2, 4, 8] {
        let plan = MacroExchangeMemoryPlan::derive(MacroExchangeInput {
            world, candidate_capacity: 9, state_stride: 32, route_slots: 2,
        }).unwrap();
        check_partitions(9, world, &mut Vec::new(), &plan);
    }
}

#[test]
fn dense_exchange_plan_rejects_unsupported_or_unbounded_shapes() {
    let base = MacroExchangeInput { world: 8, candidate_capacity: 16, state_stride: 256, route_slots: 3 };
    assert!(MacroExchangeMemoryPlan::derive(MacroExchangeInput { world: 3, ..base }).is_err());
    assert!(MacroExchangeMemoryPlan::derive(MacroExchangeInput { route_slots: 0, ..base }).is_err());
    assert!(MacroExchangeMemoryPlan::derive(MacroExchangeInput { state_stride: 17, ..base }).is_err());
    assert!(MacroExchangeMemoryPlan::derive(MacroExchangeInput { candidate_capacity: i32::MAX as u32 + 1, ..base }).is_err());
    assert!(MacroExchangeMemoryPlan::derive(MacroExchangeInput { state_stride: u64::MAX - 15, ..base }).is_err());
}
