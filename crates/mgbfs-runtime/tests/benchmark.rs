use mgbfs_core::config::{FrontierProfile, OwnerBackend, ReferenceOwner};
use mgbfs_runtime::benchmark::{library_backend_label, run_phases, Phase};

#[test]
fn reference_record_labels_rank_owner_without_native_or_indexed_alias() {
    assert_eq!(library_backend_label(ReferenceOwner::CucoRank, FrontierProfile::Dense),
        Some("library_nccl_dense_cuco_rank_v1"));
    assert_eq!(library_backend_label(ReferenceOwner::CucoIndexed, FrontierProfile::Dense),
        Some("library_nccl_dense_cuco_v1"));
    assert_eq!(library_backend_label(ReferenceOwner::CucoIndexed, FrontierProfile::HashFirst),
        Some("library_nccl_hash_first_cuco_v1"));
    assert_eq!(library_backend_label(ReferenceOwner::Native(OwnerBackend::CubSortMerge),
        FrontierProfile::Dense), None);
}

#[test]
fn full_warmup_precedes_measurement_and_failure_prevents_measurement() {
    let mut phases = Vec::new();
    run_phases(true, |phase| {
        phases.push(phase);
        Ok(())
    })
    .unwrap();
    assert_eq!(phases, [Phase::Warmup, Phase::Measure]);
    phases.clear();
    let result = run_phases(true, |phase| {
        phases.push(phase);
        Err("warmup failed".into())
    });
    assert_eq!(result.unwrap_err(), "warmup failed");
    assert_eq!(phases, [Phase::Warmup]);
    phases.clear();
    run_phases(false, |phase| {
        phases.push(phase);
        Ok(())
    })
    .unwrap();
    assert_eq!(phases, [Phase::Measure]);
}
