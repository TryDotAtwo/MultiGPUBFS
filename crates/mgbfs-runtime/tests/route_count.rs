use mgbfs_runtime::route_count::routed_count;

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
