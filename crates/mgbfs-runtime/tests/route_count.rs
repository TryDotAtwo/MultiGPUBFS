use mgbfs_runtime::route_count::routed_count;

#[test]
fn eight_owner_counts_include_late_peers_and_reject_late_overflow() {
    use mgbfs_runtime::route_count::packed_count;
    assert_eq!(packed_count(36, [1, 2, 3, 4, 5, 6, 7, 8]).unwrap(), 36);
    assert_eq!(packed_count(9, [0, 0, 0, 0, 0, 0, 0, 9]).unwrap(), 9);
    assert_eq!(packed_count(0, [0; 8]).unwrap(), 0);
    assert!(packed_count(35, [1, 2, 3, 4, 5, 6, 7, 8]).is_err());
    assert!(packed_count(u32::MAX, [1, 0, 0, 0, 0, 0, 0, u32::MAX]).is_err());
    assert!(packed_count(0, []).is_err());
    assert!(packed_count(3, [1, 1, 1]).is_err());
}

#[test]
fn no_prededup_preserves_count_without_host_readback() {
    for n in [0, 1, 65_536, 1_048_576] {
        assert_eq!(
            routed_count(false, n, || panic!("unnecessary GPU synchronization")).unwrap(),
            n
        );
    }
}

#[test]
fn prededup_reads_compacted_count_once_and_checks_bound() {
    let mut reads = 0;
    assert_eq!(
        routed_count(true, 100, || {
            reads += 1;
            Ok(17)
        })
        .unwrap(),
        17
    );
    assert_eq!(reads, 1);
    assert_eq!(
        routed_count(true, 100, || Ok(101)).unwrap_err(),
        "ROUTE_COUNT_BOUND"
    );
    assert_eq!(routed_count(true, 0, || Ok(0)).unwrap(), 0);
    assert_eq!(
        routed_count(true, 100, || Err("CUDA_FAILURE".into())).unwrap_err(),
        "CUDA_FAILURE"
    );
}
#[test]
fn packed_owner_counts_reject_corrupt_totals_and_device_fatal() {
    use mgbfs_runtime::route_count::packed_count;
    assert_eq!(packed_count(7, [4, 3]).unwrap(), 7);
    assert_eq!(packed_count(7, [0, 3]).unwrap(), 3);
    assert_eq!(packed_count(0, [0, 0]).unwrap(), 0);
    assert!(packed_count(7, [4, 4]).is_err());
    assert!(packed_count(u32::MAX, [u32::MAX, 0]).is_err());
    assert!(packed_count(u32::MAX, [1, u32::MAX]).is_err());
}
