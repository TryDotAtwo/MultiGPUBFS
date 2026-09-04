use mgbfs_core::{config::FrontierProfile, rank_plan::*};

fn fixture() -> (RankShape, RankQueries) {
    let s = RankShape {
        n: 4,
        moves: 6,
        modulus: 3,
        parents: 2,
        state_records: 32,
        extent_descriptors: 8,
        layer_records: 16,
        shards: 2,
        buckets: 4,
        bucket_records: 8,
        incoming: 12,
        touched_buckets: 2,
        materialize_records: 12,
        generation_lanes: 2,
        route_lanes: 2,
        owner_lanes: 2,
        materialize_lanes: 2,
        archive_slots: 2,
        archive_slot_bytes: 4100,
        profile: FrontierProfile::Dense,
        policy_digest: [7; 32],
    };
    // Literal synthetic query inputs exercise accounting, not CUDA performance.
    let mut q = RankQueries {
        shape: s.clone(),
        device_uuid: "test-device".into(),
        build_digest: [9; 32],
        results: REQUIRED_QUERIES
            .into_iter()
            .map(|k| {
                (
                    k,
                    QueryResult {
                        source: "synthetic accounting fixture".into(),
                        allocations: vec![],
                    },
                )
            })
            .collect(),
    };
    for (kind, name, bytes) in [
        (QueryKind::Generation, "products_s32", 768),
        (QueryKind::Hash, "partials_s32", 768),
        (QueryKind::Route, "cub_radix", 257),
        (QueryKind::Transport, "candidate_receives", 513),
        (QueryKind::ControlPinned, "descriptors", 1),
    ] {
        q.results
            .get_mut(&kind)
            .unwrap()
            .allocations
            .push(QueryAllocation {
                name: name.into(),
                bytes,
                alignment: 256,
            });
    }
    (s, q)
}
fn plan(s: &RankShape, q: &RankQueries) -> mgbfs_core::Result<RankPlan> {
    rank_plan(s, q, "test-device", [9; 32], 1 << 30, 1 << 20, 1 << 20)
}
fn bytes(p: &RankPlan, name: &str) -> u64 {
    p.device
        .iter()
        .find(|a| a.name == name)
        .unwrap()
        .payload_bytes
}

#[test]
fn rank_plan_counts_replicas_intermediates_and_separate_pinned_pool() {
    let (s, q) = fixture();
    let p = plan(&s, &q).unwrap();
    assert_eq!(bytes(&p, "state_ring"), 512);
    assert_eq!(bytes(&p, "accepted_hashes"), 512);
    assert_eq!(bytes(&p, "hash_arena_prev"), 256);
    assert_eq!(bytes(&p, "hash_arena_curr"), 256);
    assert_eq!(bytes(&p, "generation/Generation/products_s32"), 1536);
    assert_eq!(bytes(&p, "generation/Hash/partials_s32"), 1536);
    // Replicate *aligned* lane sizes: 2*align256(257), not align256(2*257).
    assert_eq!(bytes(&p, "route/Route/cub_radix"), 1024);
    assert_eq!(bytes(&p, "rank/Transport/candidate_receives"), 768);
    assert_eq!(p.pinned_bytes, 16640); // 2*align4096(4100) + align256(1)
    assert!(p
        .device
        .windows(2)
        .all(|x| x[0].offset + x[0].reserved_bytes <= x[1].offset));
    assert!(p.device.iter().all(|a| a.offset % 256 == 0));
    let mut hf = s.clone();
    hf.profile = FrontierProfile::HashFirst;
    let mut hq = q.clone();
    hq.shape = hf.clone();
    let hp = plan(&hf, &hq).unwrap();
    // Two-GEMM HASH_FIRST still pays for transient full children.
    assert_eq!(
        bytes(&hp, "generation_states"),
        bytes(&p, "generation_states")
    );
}

#[test]
fn scratch_queries_are_mandatory_and_bound_to_shape_build_and_device() {
    let (s, q) = fixture();
    for k in REQUIRED_QUERIES {
        let mut bad = q.clone();
        bad.results.remove(&k);
        assert!(plan(&s, &bad).unwrap_err().starts_with("MISSING_QUERY"));
    }
    let mut bad = q.clone();
    bad.shape.parents += 1;
    assert_eq!(plan(&s, &bad).unwrap_err(), "QUERY_SHAPE_MISMATCH");
    let mut bad = q.clone();
    bad.build_digest = [0; 32];
    assert_eq!(plan(&s, &bad).unwrap_err(), "QUERY_BUILD_MISMATCH");
    let mut bad = q.clone();
    bad.device_uuid = "other-device".into();
    assert_eq!(plan(&s, &bad).unwrap_err(), "QUERY_DEVICE_MISMATCH");
}

#[test]
fn device_and_pinned_budget_fail_before_any_allocation() {
    let (s, q) = fixture();
    let p = plan(&s, &q).unwrap();
    assert!(rank_plan(
        &s,
        &q,
        "test-device",
        [9; 32],
        p.device_bytes + 1024,
        1024,
        16640
    )
    .is_ok());
    assert!(rank_plan(
        &s,
        &q,
        "test-device",
        [9; 32],
        p.device_bytes + 1023,
        1024,
        16640
    )
    .unwrap_err()
    .starts_with("DEVICE_CAPACITY"));
    assert!(
        rank_plan(&s, &q, "test-device", [9; 32], 1 << 30, 1024, 16639)
            .unwrap_err()
            .starts_with("PINNED_CAPACITY")
    );
}

#[test]
fn owner_scratch_does_not_grow_with_rank_layer() {
    let (s, q) = fixture();
    let p = plan(&s, &q).unwrap();
    let mut larger = s.clone();
    larger.layer_records = 32;
    let mut lq = q.clone();
    lq.shape = larger.clone();
    let lp = plan(&larger, &lq).unwrap();
    let owner = |p: &RankPlan| {
        p.device
            .iter()
            .filter(|a| a.name.starts_with("owner/"))
            .map(|a| a.reserved_bytes)
            .sum::<u64>()
    };
    assert_eq!(owner(&p), owner(&lp));
    assert_eq!(lp.device_bytes - p.device_bytes, 512);
}

#[test]
fn malformed_shapes_and_query_allocations_fail_closed() {
    let (s, q) = fixture();
    for change in 0..5 {
        let mut bad = s.clone();
        match change {
            0 => bad.layer_records = 33,
            1 => bad.shards = 3,
            2 => bad.touched_buckets = 5,
            3 => bad.state_records = u64::MAX,
            _ => bad.generation_lanes = 0,
        }
        let mut bq = q.clone();
        bq.shape = bad.clone();
        assert!(plan(&bad, &bq).is_err());
    }
    for change in 0..4 {
        let mut bad = q.clone();
        let r = bad.results.get_mut(&QueryKind::Hash).unwrap();
        match change {
            0 => r.source.clear(),
            1 => r.allocations.push(r.allocations[0].clone()),
            2 => r.allocations[0].alignment = 3,
            _ => r.allocations[0].bytes = u64::MAX,
        }
        assert!(plan(&s, &bad).is_err());
    }
}

#[test]
fn cluster_capacity_modes_separate_equal_global_memory_from_max_capacity() {
    let equal = cluster_capacity_plan(CapacityMode::EqualGlobal, 11, 2).unwrap();
    assert_eq!(equal.per_rank_records, vec![6, 5]);
    assert_eq!(equal.global_records, 11);
    assert_eq!(equal.rank_records(0).unwrap(), 6);
    assert_eq!(equal.rank_records(1).unwrap(), 5);

    let max = cluster_capacity_plan(CapacityMode::MaxPerRank, 11, 2).unwrap();
    assert_eq!(max.per_rank_records, vec![11, 11]);
    assert_eq!(max.global_records, 22);
}

#[test]
fn cluster_capacity_plan_fails_closed_on_empty_ranks_and_overflow() {
    assert_eq!(
        cluster_capacity_plan(CapacityMode::EqualGlobal, 1, 2).unwrap_err(),
        "RANK_CAPACITY_ZERO"
    );
    assert_eq!(
        cluster_capacity_plan(CapacityMode::MaxPerRank, u64::MAX, 2).unwrap_err(),
        "BYTE_OVERFLOW"
    );
    let plan = cluster_capacity_plan(CapacityMode::EqualGlobal, 4, 2).unwrap();
    assert_eq!(plan.rank_records(2).unwrap_err(), "RANK_OUT_OF_RANGE");
}
