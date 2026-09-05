use mgbfs_runtime::distributed_memory::{shared_buffers, SharedBufferShape};

#[test]
fn allocation_report_preserves_named_payload_and_padding_without_device_addresses() {
    use mgbfs_core::memory::AllocationLedger;
    use mgbfs_runtime::distributed_memory::allocation_report;
    let mut ledger = AllocationLedger::new(4096, 0).unwrap();
    ledger.add("shared.states", 17, 16, 256).unwrap();
    ledger.add("owner.flags", 3, 1, 256).unwrap();
    let report = allocation_report(&ledger);
    assert_eq!(report["payload_bytes"], 275);
    assert_eq!(report["aligned_bytes"], 768);
    assert_eq!(report["planes"][0]["name"], "shared.states");
    assert_eq!(report["planes"][0]["payload_bytes"], 272);
    assert_eq!(report["planes"][0]["reserved_bytes"], 512);
    assert_eq!(report["planes"][1]["payload_bytes"], 3);
    assert_eq!(report["planes"][1]["reserved_bytes"], 256);
    assert!(report["planes"][0].get("offset").is_none());
    assert_eq!(report["planes"].as_array().unwrap().len(), 2);
}

fn shape() -> SharedBufferShape {
    SharedBufferShape {
        state_stride: 32,
        packet_stride: 16,
        batch: 7,
        candidates: 21,
        layer_capacity: 64,
        state_ring_capacity: 128,
        buckets: 8,
        bucket_capacity: 11,
        job_buckets: 2,
        archive_width: 3,
    }
}

#[test]
fn device_admission_preserves_reserve_and_never_wraps_required_bytes() {
    use mgbfs_runtime::distributed_memory::device_admission;
    assert!(device_admission(768, 256, 1024).is_ok());
    assert!(device_admission(769, 256, 1024).is_err());
    assert!(device_admission(u64::MAX, 1, u64::MAX).is_err());
    assert!(device_admission(0, 1025, 1024).is_err());
}

#[test]
fn queried_storage_composition_keeps_alignment_and_rejects_duplicate_planes() {
    use mgbfs_core::{
        memory::AllocationLedger,
        rank_plan::{QueryAllocation, QueryResult},
    };
    use mgbfs_runtime::distributed_memory::append_query;
    let q = QueryResult {
        source: "fixture actual query".into(),
        allocations: vec![QueryAllocation {
            name: "scratch".into(),
            bytes: 257,
            alignment: 256,
        }],
    };
    let mut ledger = AllocationLedger::new(1024, 0).unwrap();
    append_query(&mut ledger, "route", &q).unwrap();
    assert_eq!(ledger.total(), 512);
    assert_eq!(ledger.allocations[0].payload_bytes, 257);
    assert!(append_query(&mut ledger, "route", &q).is_err());
    assert!(append_query(&mut ledger, "", &q).is_err());
    let q = QueryResult {
        source: String::new(),
        ..q
    };
    assert!(append_query(&mut ledger, "owner", &q).is_err());
}

#[test]
fn physical_shared_planes_use_declared_strides_and_abi_sizes() {
    let ledger = shared_buffers(shape()).unwrap();
    let bytes = |name| {
        ledger
            .allocations
            .iter()
            .find(|a| a.name == name)
            .unwrap()
            .payload_bytes
    };
    assert_eq!(ledger.allocations.len(), 29);
    for (name, want) in [
        ("states", 4096),
        ("prev", 1024),
        ("curr", 1024),
        ("children", 336),
        ("packed_states", 336),
        ("recv_states", 336),
        ("accepted", 1408),
        ("jobs_gpu", 576),
        ("counts", 64),
        ("archive_states", 21),
        ("directory", 128),
        ("identity_refs", 168),
    ] {
        assert_eq!(bytes(name), want, "{name}");
    }
    let mut dense = shape();
    dense.packet_stride = 32;
    let dense = shared_buffers(dense).unwrap();
    let sum = |l: &mgbfs_core::memory::AllocationLedger| {
        l.allocations.iter().map(|a| a.payload_bytes).sum::<u64>()
    };
    assert_eq!(sum(&dense) - sum(&ledger), 1008);
    assert_eq!(dense.total() - ledger.total(), 768);
}

#[test]
fn invalid_or_overflowing_storage_is_rejected_before_allocation() {
    for bad in [
        SharedBufferShape {
            candidates: 0,
            ..shape()
        },
        SharedBufferShape {
            state_stride: u64::MAX - 15,
            state_ring_capacity: 128,
            ..shape()
        },
        SharedBufferShape {
            packet_stride: 17,
            ..shape()
        },
        SharedBufferShape {
            buckets: u64::MAX,
            ..shape()
        },
    ] {
        assert!(shared_buffers(bad).is_err());
    }
}
