use mgbfs_runtime::distributed_memory::{native_rank_shared_buffers, SharedBufferShape};
#[test]
fn native_rank_plan_retains_host_sized_receive_planes() {
    let shape = SharedBufferShape {
        state_stride: 16,
        packet_stride: 16,
        batch: 8,
        candidates: 24,
        layer_capacity: 64,
        state_ring_capacity: 128,
        buckets: 16,
        bucket_capacity: 8,
        job_buckets: 4,
        archive_width: 16,
    };
    let ledger = native_rank_shared_buffers(shape, 4).unwrap();
    for (name, bytes) in [
        ("recv_states", 384),
        ("recv_hashes", 384),
        ("recv_count", 4),
    ] {
        let allocation = ledger.allocations.iter().find(|a| a.name == name);
        assert!(
            allocation.is_some(),
            "missing host-sized receive plane: {name}"
        );
        assert_eq!(allocation.unwrap().payload_bytes, bytes);
    }
}

#[test]
fn lsa_native_rank_plan_does_not_duplicate_receive_planes() {
    let shape = SharedBufferShape {
        state_stride: 16,
        packet_stride: 16,
        batch: 8,
        candidates: 24,
        layer_capacity: 64,
        state_ring_capacity: 128,
        buckets: 16,
        bucket_capacity: 8,
        job_buckets: 4,
        archive_width: 16,
    };
    let ledger =
        mgbfs_runtime::distributed_memory::native_rank_shared_buffers_for_transport(shape, 4, true)
            .unwrap();
    for name in ["recv_states", "recv_hashes", "recv_count"] {
        assert!(!ledger.allocations.iter().any(|a| a.name == name));
    }
    for (name, bytes) in [
        ("owner_window", 12),
        ("rank_shard_counts", 16),
        ("rank_shard_offsets", 20),
    ] {
        assert_eq!(
            ledger
                .allocations
                .iter()
                .find(|a| a.name == name)
                .unwrap()
                .payload_bytes,
            bytes
        );
    }
}
