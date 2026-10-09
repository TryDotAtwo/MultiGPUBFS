use mgbfs_core::config::RunConfigV1;
#[test]
fn production_config_accepts_explicit_host_sized_nccl_transport() {
    let mut value = serde_json::to_value(RunConfigV1::fixture(2).unwrap()).unwrap();
    value["transport_backend"] = serde_json::json!("HOST_SIZED_NCCL");
    let parsed = serde_json::from_value::<RunConfigV1>(value);
    let parsed = parsed.expect("explicit production transport rejected");
    assert_eq!(
        parsed.transport_backend,
        mgbfs_core::config::ReferenceTransport::HostSizedNccl
    );
    parsed.validate().unwrap();
}

#[test]
fn legacy_transport_default_preserves_canonical_config_and_digest() {
    let config = RunConfigV1::fixture(2).unwrap();
    let legacy = serde_json::to_value(&config).unwrap();
    assert!(legacy.get("transport_backend").is_none());
    let parsed: RunConfigV1 = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(
        parsed.transport_backend,
        mgbfs_core::config::ReferenceTransport::Lsa
    );
    assert_eq!(parsed.digest().unwrap(), config.digest().unwrap());
    let mut explicit = legacy;
    explicit["transport_backend"] = serde_json::json!("NCCL_LSA");
    let parsed: RunConfigV1 = serde_json::from_value(explicit).unwrap();
    assert_eq!(parsed.digest().unwrap(), config.digest().unwrap());
    let mut host = config;
    host.transport_backend = mgbfs_core::config::ReferenceTransport::HostSizedNccl;
    assert_ne!(host.digest().unwrap(), parsed.digest().unwrap());
}
#[test]
fn unknown_transport_is_rejected_without_fallback() {
    let mut value = serde_json::to_value(RunConfigV1::fixture(2).unwrap()).unwrap();
    value["transport_backend"] = serde_json::json!("AUTO");
    assert!(serde_json::from_value::<RunConfigV1>(value).is_err());
}

#[test]
fn hash_first_cuco_host_keeps_explicit_pool_and_transport_contract() {
    let mut value = serde_json::to_value(RunConfigV1::fixture(2).unwrap()).unwrap();
    value["transport_backend"] = serde_json::json!("HOST_SIZED_NCCL");
    value["frontier_profile"] = serde_json::json!("HASH_FIRST");
    value["owner_backend"] = serde_json::json!("CUCO_RANK");
    let decode = |v| serde_json::from_value::<RunConfigV1>(v).unwrap();
    assert_eq!(
        decode(value.clone()).validate().unwrap_err(),
        "CONFIG_LIBRARY_POOL_REQUIRED"
    );
    value["library_pool_bytes"] = serde_json::json!(257);
    assert_eq!(
        decode(value.clone()).validate().unwrap_err(),
        "CONFIG_LIBRARY_POOL_ALIGNMENT"
    );
    value["library_pool_bytes"] = serde_json::json!(64 << 20);
    let config = decode(value);
    config.validate().unwrap();
    assert_eq!(
        config.transport_backend,
        mgbfs_core::config::ReferenceTransport::HostSizedNccl
    );
    assert_eq!(
        config.frontier_profile,
        mgbfs_core::config::FrontierProfile::HashFirst
    );
    let wire = serde_json::to_string(&config).unwrap();
    let roundtrip: RunConfigV1 = serde_json::from_str(&wire).unwrap();
    assert_eq!(config.digest().unwrap(), roundtrip.digest().unwrap());
    let mut lsa = roundtrip;
    lsa.transport_backend = mgbfs_core::config::ReferenceTransport::Lsa;
    assert_ne!(config.digest().unwrap(), lsa.digest().unwrap());
}
