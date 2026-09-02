use mgbfs_core::{config::RunConfigV1, hash::Hash128};

#[test]
fn config_digest_survives_json_roundtrip_but_not_seed_or_rank_changes() {
    let c = RunConfigV1::fixture(5).unwrap();
    c.validate().unwrap();
    let wire = serde_json::to_string_pretty(&c).unwrap();
    let mut copy: RunConfigV1 = serde_json::from_str(&wire).unwrap();
    assert_eq!(c.digest().unwrap(), copy.digest().unwrap());
    copy.seed[0] = 1;
    assert_ne!(c.digest().unwrap(), copy.digest().unwrap());
    copy.seed = c.seed;
    copy.topology.logical_owner_to_rank.swap(0, 1);
    assert_ne!(c.digest().unwrap(), copy.digest().unwrap());
}

#[test]
fn owner_shard_bucket_use_high_bits_and_manual_rank_permutation() {
    let mut t = RunConfigV1::fixture(5).unwrap().topology;
    t.logical_owner_to_rank = vec![1, 0];
    assert_eq!(t.locate(Hash128([0, 0, 0, 0x80000000])).unwrap(), (0, 0, 0));
    assert_eq!(
        t.locate(Hash128([0, 0, 0, 0x7ffe0000])).unwrap(),
        (1, 63, 255)
    );
    t.logical_owner_to_rank = vec![0, 0];
    assert!(t.validate().is_err());
    t.world_size = 3;
    assert!(t.validate().is_err());
}

#[test]
fn preflight_rejects_slot_overflow_unknown_schema_and_zero_capacity() {
    let mut c = RunConfigV1::fixture(5).unwrap();
    c.parent_batch = u64::MAX;
    assert!(c.validate().is_err());
    c.parent_batch = 16384;
    c.capacities.route_slot_records = 1;
    assert!(c.validate().is_err());
    c.capacities.route_slot_records = 6 * 16384;
    c.schema = 2;
    assert!(c.validate().is_err());
    c.schema = 1;
    c.capacities.pinned_archive_slots = 0;
    assert!(c.validate().is_err());
}
#[test]
fn wire_config_pins_generation_hash_and_matrix_schema() {
    let c = RunConfigV1::fixture(5).unwrap();
    let value = serde_json::to_value(c).unwrap();
    assert_eq!(value["generation_backend"], "CUTLASS_U8_SM75_V1");
    assert_eq!(value["hash_backend"], "GEMM_U8_P32X4_V1");
    assert_eq!(value["graph"]["schema"], 1);
    let mut unknown = value;
    unknown["hash_backend"] = serde_json::json!("AUTO");
    assert!(serde_json::from_value::<RunConfigV1>(unknown).is_err());
}
