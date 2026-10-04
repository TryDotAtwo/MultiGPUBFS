use mgbfs_core::{config::RunConfigV1, hash::Hash128};

#[test]
fn preflight_rejects_aggregate_route_bank_byte_overflow() {
    let mut config = RunConfigV1::fixture(5).unwrap();
    // Each bank fits u64 by itself, but the declared three-bank reservation
    // does not. Admission must reject this before any allocation or NCCL call.
    config.capacities.route_slot_records = u64::MAX / 64;
    config.capacities.route_slot_count = 2;
    config.validate().unwrap();
    config.capacities.route_slot_count = 3;
    assert_eq!(config.validate().unwrap_err(), "BYTE_OVERFLOW");
}

#[test]
fn production_epoch_window_is_bounded_and_part_of_run_identity() {
    let config = RunConfigV1::fixture(5).unwrap();
    let original = config.digest().unwrap();
    let mut wire = serde_json::to_value(config).unwrap();
    wire["completion_epoch_window"] = serde_json::json!(3);
    let config: RunConfigV1 = serde_json::from_value(wire.clone()).unwrap();
    config.validate().unwrap();
    assert_ne!(original, config.digest().unwrap());
    for value in [0, 1] {
        wire["completion_epoch_window"] = serde_json::json!(value);
        let config: RunConfigV1 = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(config.validate().unwrap_err(), "CONFIG_EPOCH_WINDOW");
    }
}

#[test]
fn legacy_native_config_keeps_its_frozen_digest_without_a_pool_field() {
    let config: RunConfigV1 =
        serde_json::from_str(include_str!("../../../tests/run-s4-two-rank.json")).unwrap();
    // Independently SHA256'd compact pre-CUCO fixture, in its declared wire order.
    let digest = config
        .digest()
        .unwrap()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect::<String>();
    assert_eq!(
        digest,
        "dd509823ba8d382ee8de7ca80e6aa483d4dfa17cc053a13eba601b4bbb38ea3c"
    );
    assert!(serde_json::to_value(&config)
        .unwrap()
        .get("library_pool_bytes")
        .is_none());
    let mut cuco = config;
    cuco.owner_backend = mgbfs_core::config::RunOwnerBackend::CucoRank;
    cuco.library_pool_bytes = Some(64 << 20);
    let first = cuco.digest().unwrap();
    cuco.library_pool_bytes = Some(96 << 20);
    assert_ne!(first, cuco.digest().unwrap());
}

#[test]
fn production_cuco_requires_explicit_aligned_pool_budget() {
    let mut wire = serde_json::to_value(RunConfigV1::fixture(5).unwrap()).unwrap();
    wire["owner_backend"] = serde_json::json!("CUCO_RANK");
    let decode = |value| serde_json::from_value::<RunConfigV1>(value).unwrap();
    assert_eq!(
        decode(wire.clone()).validate().unwrap_err(),
        "CONFIG_LIBRARY_POOL_REQUIRED"
    );
    wire["library_pool_bytes"] = serde_json::json!(257);
    assert_eq!(
        decode(wire.clone()).validate().unwrap_err(),
        "CONFIG_LIBRARY_POOL_ALIGNMENT"
    );
    wire["library_pool_bytes"] = serde_json::json!(67108864);
    decode(wire.clone()).validate().unwrap();
    wire["owner_backend"] = serde_json::json!("CUB_SORT_MERGE");
    assert_eq!(
        decode(wire).validate().unwrap_err(),
        "CONFIG_UNUSED_LIBRARY_POOL"
    );
}

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
