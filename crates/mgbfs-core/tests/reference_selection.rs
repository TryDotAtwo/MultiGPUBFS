use mgbfs_core::config::{FrontierProfile, OwnerBackend, ReferenceOwner, ReferenceSelection, ReferenceTransport};

#[test]
fn lsa_transport_is_explicit_and_only_valid_for_dense_rank_owner() {
    let rank = ReferenceSelection::parse("DENSE", "CUCO_RANK", "ON", false, 64, 8).unwrap();
    assert_eq!(rank.transport, ReferenceTransport::HostSizedNccl);
    assert_eq!(rank.with_transport("NCCL_LSA").unwrap().transport, ReferenceTransport::Lsa);
    let cub = ReferenceSelection::parse("DENSE", "CUB_SORT_MERGE", "ON", false, 64, 8).unwrap();
    assert!(cub.with_transport("NCCL_LSA").is_err());
    assert!(rank.with_transport("UNKNOWN").is_err());
}

#[test]
fn cuco_reference_requires_fixed_pool_and_archive_without_native_substitution() {
    for profile in ["DENSE", "HASH_FIRST"] {
        let selected = ReferenceSelection::parse(profile, "CUCO_INDEXED", "ON", false, 64, 8)
            .expect("explicit cuco selection");
        assert!(selected.with_library_pool(None, true).is_err());
        assert!(selected.with_library_pool(Some("67108864"), false).is_err());
        assert!(selected.validate_archive(false).is_err());
        let selected = selected.with_library_pool(Some("67108864"), true).unwrap();
        assert_eq!(selected.library_pool_bytes, Some(64 << 20));
        assert!(selected.validate_archive(true).is_ok());
        assert_ne!(selected.owner, ReferenceOwner::CudfRelational);
        assert_ne!(
            selected.owner,
            ReferenceOwner::Native(OwnerBackend::CubSortMerge)
        );
    }
}

#[test]
fn library_dispatch_requires_compiled_support_and_an_explicit_fixed_pool() {
    let selection =
        ReferenceSelection::parse("DENSE", "CUDF_RELATIONAL", "OFF", false, 64, 8).unwrap();
    assert!(selection
        .with_library_pool(Some("67108864"), false)
        .is_err());
    for invalid in [
        None,
        Some("0"),
        Some("255"),
        Some("257"),
        Some("auto"),
        Some("18446744073709551616"),
    ] {
        assert!(selection.with_library_pool(invalid, true).is_err());
    }
    let selection = selection.with_library_pool(Some("67108864"), true).unwrap();
    assert_eq!(selection.owner, ReferenceOwner::CudfRelational);
    assert_eq!(selection.library_pool_bytes, Some(64 << 20));
}

#[test]
fn native_dispatch_rejects_an_unused_library_pool() {
    let selection =
        ReferenceSelection::parse("DENSE", "CUB_SORT_MERGE", "ON", false, 64, 8).unwrap();
    assert!(selection.with_library_pool(Some("67108864"), true).is_err());
    assert_eq!(
        selection.with_library_pool(None, false).unwrap().owner,
        ReferenceOwner::Native(OwnerBackend::CubSortMerge)
    );
}

#[test]
fn library_measurement_cannot_disable_the_archive_contract() {
    let library =
        ReferenceSelection::parse("DENSE", "CUDF_RELATIONAL", "ON", false, 64, 8).unwrap();
    assert!(library.validate_archive(false).is_err());
    assert!(library.validate_archive(true).is_ok());
    let native = ReferenceSelection::parse("DENSE", "CUB_SORT_MERGE", "ON", false, 64, 8).unwrap();
    assert!(native.validate_archive(false).is_ok());
}

#[test]
fn tensor_generation_is_an_explicit_hash_first_only_reference_choice() {
    let hash_first =
        ReferenceSelection::parse("HASH_FIRST", "CUB_SORT_MERGE", "ON", false, 64, 8).unwrap();
    assert!(
        !hash_first
            .with_hash_first_generation("SCALAR")
            .unwrap()
            .tensor_generation
    );
    assert!(
        hash_first
            .with_hash_first_generation("INT_MMA_SM75")
            .unwrap()
            .tensor_generation
    );
    assert!(hash_first.with_hash_first_generation("auto").is_err());
    let dense = ReferenceSelection::parse("DENSE", "CUB_SORT_MERGE", "ON", false, 64, 8).unwrap();
    assert!(dense.with_hash_first_generation("INT_MMA_SM75").is_err());
}

#[test]
fn reference_backend_selection_is_explicit_and_rejects_unsupported_compact_hash_first() {
    for profile in ["DENSE", "HASH_FIRST"] {
        for owner in ["CUB_SORT_MERGE", "BMMA_BUCKET"] {
            for pre in ["ON", "OFF"] {
                let selected =
                    ReferenceSelection::parse(profile, owner, pre, false, 64, 8).unwrap();
                assert_eq!(
                    selected.profile == FrontierProfile::HashFirst,
                    profile == "HASH_FIRST"
                );
                assert_eq!(
                    selected.owner == ReferenceOwner::Native(OwnerBackend::BmmaBucket),
                    owner == "BMMA_BUCKET"
                );
                assert_eq!(selected.prededup, pre == "ON");
                assert_eq!(
                    selected.materialization_capacity,
                    if profile == "HASH_FIRST" {
                        Some(64)
                    } else {
                        None
                    }
                );
            }
        }
    }
    assert!(ReferenceSelection::parse("HASH_FIRST", "CUB_SORT_MERGE", "ON", true, 64, 8).is_err());
    assert!(ReferenceSelection::parse("DENSE", "CUB_SORT_MERGE", "ON", true, 64, 8).is_ok());
    for (profile, owner, pre, cap, tile) in [
        ("dense", "CUB_SORT_MERGE", "ON", 64, 8),
        ("DENSE", "auto", "ON", 64, 8),
        ("DENSE", "CUB_SORT_MERGE", "auto", 64, 8),
        ("HASH_FIRST", "CUB_SORT_MERGE", "ON", 0, 8),
        ("HASH_FIRST", "CUB_SORT_MERGE", "ON", u32::MAX, 8),
        ("DENSE", "BMMA_BUCKET", "ON", 64, 0),
        ("DENSE", "BMMA_BUCKET", "ON", 64, 257),
    ] {
        assert!(ReferenceSelection::parse(profile, owner, pre, false, cap, tile).is_err());
    }
}
#[test]
fn search_only_explicitly_allows_library_without_archive() {
    for owner in ["CUCO_INDEXED", "CUDF_RELATIONAL", "CUB_SORT_MERGE"] {
        let selection = ReferenceSelection::parse("DENSE", owner, "ON", false, 64, 8).unwrap();
        assert!(selection.validate_archive_contract(false, true).is_ok());
        assert!(selection.validate_archive_contract(true, true).is_err());
    }
}

#[test]
fn rank_owner_is_explicit_dense_only_and_requires_fixed_pool() {
    let selected = ReferenceSelection::parse("DENSE", "CUCO_RANK", "ON", false, 64, 8)
        .expect("rank owner must be selectable without aliasing CUCO_INDEXED");
    assert_eq!(selected.owner, ReferenceOwner::CucoRank);
    assert!(selected.with_library_pool(None, true).is_err());
    assert!(selected.with_library_pool(Some("67108864"), false).is_err());
    assert!(selected.with_library_pool(Some("67108864"), true).is_ok());
    assert!(ReferenceSelection::parse("HASH_FIRST", "CUCO_RANK", "ON", false, 64, 8).is_err());
}
