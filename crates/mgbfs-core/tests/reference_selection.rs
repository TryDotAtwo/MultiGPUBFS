use mgbfs_core::config::{FrontierProfile, OwnerBackend, ReferenceSelection};

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
                    selected.owner == OwnerBackend::BmmaBucket,
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
