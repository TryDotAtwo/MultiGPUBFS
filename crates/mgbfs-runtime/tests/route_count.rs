use mgbfs_runtime::route_count::routed_count;

#[test]
fn mapped_ranges_preserve_packed_offsets_with_empty_late_owners() {
    use mgbfs_runtime::route_count::packed_rank_ranges;
    let ranges =
        packed_rank_ranges(15, &[2, 0, 3, 0, 1, 4, 0, 5], &[7, 0, 5, 2, 6, 1, 4, 3]).unwrap();
    assert_eq!(
        ranges,
        [
            (2, 0),
            (6, 4),
            (5, 0),
            (10, 5),
            (10, 0),
            (2, 3),
            (5, 1),
            (0, 2)
        ]
    );
    assert_eq!(
        packed_rank_ranges(7, &[4, 3], &[1, 0]).unwrap()[..2],
        [(4, 3), (0, 4)]
    );
    assert_eq!(packed_rank_ranges(7, &[7], &[0]).unwrap()[0], (0, 7));
    assert!(packed_rank_ranges(14, &[2, 0, 3, 0, 1, 4, 0, 5], &[7, 0, 5, 2, 6, 1, 4, 3]).is_err());
    assert!(packed_rank_ranges(7, &[4, 3], &[0, 0]).is_err());
    assert!(packed_rank_ranges(7, &[4, 3], &[0, 2]).is_err());
    assert!(packed_rank_ranges(7, &[4, 3], &[0]).is_err());
    assert!(packed_rank_ranges(7, &[], &[]).is_err());
}

#[test]
fn peer_rounds_match_and_cover_every_directed_pair_once() {
    use mgbfs_runtime::route_count::exchange_peer;
    // Not rank+round: send and receive use the same peer in each NCCL group.
    assert_eq!(
        (0..8)
            .map(|r| exchange_peer(8, 5, r).unwrap())
            .collect::<Vec<_>>(),
        vec![5, 4, 7, 6, 1, 0, 3, 2]
    );
    for world in [1, 2, 4, 8] {
        for rank in 0..world {
            let mut seen = vec![false; world as usize];
            for round in 0..world {
                let peer = exchange_peer(world, rank, round).unwrap();
                assert!(!seen[peer as usize]);
                seen[peer as usize] = true;
                assert_eq!(exchange_peer(world, peer, round).unwrap(), rank);
                assert_eq!(peer == rank, round == 0);
            }
            assert!(seen.into_iter().all(|x| x));
        }
    }
    for (w, r, phase) in [(0, 0, 0), (3, 0, 0), (16, 0, 1), (8, 8, 0), (8, 0, 8)] {
        assert!(exchange_peer(w, r, phase).is_err());
    }
}

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
