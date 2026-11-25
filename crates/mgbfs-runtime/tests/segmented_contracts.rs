use mgbfs_runtime::segmented_history::{
    sum_rank_counts, ChunkRanges, SegmentGeometry, StateAddress, LOCAL_ROWS,
};

#[test]
fn owner_cuts_follow_actual_admitted_rows_and_keep_every_rank_addressable() {
    use mgbfs_runtime::segmented_history::weighted_owner_cuts;
    assert_eq!(
        weighted_owner_cuts(&[3, 1]).unwrap(),
        [0, 3 * LOCAL_ROWS / 4, LOCAL_ROWS]
    );
    assert_eq!(
        weighted_owner_cuts(&[1, u64::MAX, 1]).unwrap(),
        [0, 1, LOCAL_ROWS - 1, LOCAL_ROWS]
    );
    assert!(weighted_owner_cuts(&[1, 0]).is_err());
}

#[test]
fn local_reference_has_32_bit_rows_and_a_separate_segment_number() {
    for segment in [0, 1, 256, u32::MAX] {
        for bank in 0..3 {
            for row in [0, 1, LOCAL_ROWS - 1] {
                let loc = StateAddress::new(segment, bank, row).unwrap();
                assert_eq!(StateAddress::decode(loc.encode()).unwrap(), loc);
                assert_eq!(loc.local_payload(), ((bank as u64) << 32) | row);
            }
        }
    }
    assert!(StateAddress::new(0, 0, LOCAL_ROWS).is_err());
    assert!(StateAddress::new(0, 3, 0).is_err());
    assert!(StateAddress::decode(1u128 << 66).is_err());
    assert!(StateAddress::decode(3u128 << 32).is_err());
}

#[test]
fn independent_small_segments_exceed_the_old_single_u32_arena_limit() {
    let g = SegmentGeometry::new(1024, LOCAL_ROWS, 3).unwrap();
    assert_eq!(g.rows(), 1024u128 * 3 * u128::from(LOCAL_ROWS));
    assert!(g.rows() > u128::from(u32::MAX));
    assert_eq!(u128::from(g.arena_bytes(1).unwrap()), g.rows());
    assert!(SegmentGeometry::new(1, LOCAL_ROWS + 1, 3).is_err());
    assert!(SegmentGeometry::new(0, 8, 1).is_err());
    assert!(SegmentGeometry::new(1, 0, 1).is_err());
    assert!(SegmentGeometry::new(1, 8, 2).is_err());
    assert!(SegmentGeometry::new(u32::MAX, LOCAL_ROWS, 3)
        .unwrap()
        .arena_bytes(u32::MAX)
        .is_err());
}

#[test]
fn rank_totals_and_chunk_cursors_do_not_wrap_at_u32() {
    assert_eq!(
        sum_rank_counts(&[u64::from(u32::MAX), 99]).unwrap(),
        u64::from(u32::MAX) + 99
    );
    assert!(sum_rank_counts(&[u64::MAX, 1]).is_err());
    let ranges: Vec<_> = ChunkRanges::new(u64::from(u32::MAX) + 7, u32::MAX)
        .unwrap()
        .collect();
    assert_eq!(
        ranges,
        vec![
            0..u64::from(u32::MAX),
            u64::from(u32::MAX)..u64::from(u32::MAX) + 7
        ]
    );
    assert!(ChunkRanges::new(10, 0).is_err());
    assert_eq!(ChunkRanges::new(0, 4).unwrap().count(), 0);
}

#[test]
fn admission_counts_tables_history_and_communication_without_rank_wide_u32_truncation() {
    use mgbfs_runtime::segmented_history::SegmentLayout;
    let directed = SegmentLayout::new(14, 1, 1024, 1).unwrap();
    assert_eq!(directed.slots, 2048);
    assert_eq!(directed.state_bytes, 14 * 1024);
    assert_eq!(directed.position_bytes, 0);
    assert_eq!(directed.device_bytes, 30744);
    let rolling = SegmentLayout::new(14, 1, 1024, 3).unwrap();
    assert_eq!(rolling.slots, 8192);
    assert_eq!(rolling.position_bytes, 3 * 1024 * 8);
    assert_eq!(rolling.device_bytes, 133144);
    let reserve = 2u64 << 30;
    let fixed = 1u64 << 30;
    let admitted = rolling
        .admit(
            reserve + fixed + rolling.device_bytes * 5000000,
            reserve,
            fixed,
            None,
        )
        .unwrap();
    assert_eq!(admitted.segments, 5000000);
    assert_eq!(admitted.rows_per_bank, 5000000u64 * 1024);
    assert!(admitted.rows_per_bank > u32::MAX as u64);
    assert_eq!(
        admitted.device_bytes,
        fixed + rolling.device_bytes * 5000000
    );
    assert!(rolling
        .admit(
            reserve + fixed + rolling.device_bytes - 1,
            reserve,
            fixed,
            None
        )
        .is_err());
    assert_eq!(
        rolling
            .admit(
                reserve + fixed + rolling.device_bytes * 20,
                reserve,
                fixed,
                Some(2000)
            )
            .unwrap()
            .segments,
        20
    );
    assert!(SegmentLayout::new(0, 1, 1024, 1).is_err());
    assert!(SegmentLayout::new(2, 4, 1024, 1).is_err());
    assert!(SegmentLayout::new(2, 8, LOCAL_ROWS + 1, 1).is_err());
}

#[test]
fn queue_skew_retry_does_not_hide_independent_fatal_errors() {
    use mgbfs_runtime::segmented_history::classify_route_and_regeneration;
    assert_eq!(classify_route_and_regeneration(0, 0), 0);
    assert_eq!(classify_route_and_regeneration(1, 0), 1);
    assert_eq!(classify_route_and_regeneration(1, 4), 1);
    for (route, regen) in [(0, 4), (0, 2), (1, 2), (1, 6), (2, 0), (4, 0), (5, 4)] {
        assert!(
            classify_route_and_regeneration(route, regen) > 1,
            "route={route} regen={regen}"
        );
    }
    let rank0 = classify_route_and_regeneration(1, 4);
    let rank1 = classify_route_and_regeneration(0, 4);
    assert!(rank0.max(rank1) > 1);
}

#[test]
fn collective_vote_prioritizes_fatal_over_resource_and_retry() {
    use mgbfs_runtime::segmented_history::{
        decode_route_owner_vote, route_owner_vote, SegmentVote,
    };
    let source_fatal = 0x1_0004;
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(0, 0)),
        SegmentVote::Continue
    );
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(1, 0)),
        SegmentVote::Retry
    );
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(1, 10)),
        SegmentVote::Resource
    );
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(source_fatal, 10)),
        SegmentVote::SourceFatal(source_fatal)
    );
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(1, 107)),
        SegmentVote::OwnerFatal(107)
    );
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(source_fatal, 107)),
        SegmentVote::SourceFatal(source_fatal)
    );
    // A max vote between different ranks must obey the same priority.
    let resource = route_owner_vote(0, 10);
    let fatal = route_owner_vote(source_fatal, 0);
    let retry = route_owner_vote(1, 0);
    assert_eq!(
        decode_route_owner_vote(resource.max(fatal)),
        SegmentVote::SourceFatal(source_fatal)
    );
    assert_eq!(
        decode_route_owner_vote(retry.max(resource)),
        SegmentVote::Resource
    );
    let deadline = mgbfs_runtime::segmented_history::route_owner_stop_vote(2);
    assert_eq!(
        decode_route_owner_vote(retry.max(deadline)),
        SegmentVote::Stop(2)
    );
    assert_eq!(
        decode_route_owner_vote(resource.max(deadline)),
        SegmentVote::Resource
    );
    assert_eq!(
        decode_route_owner_vote(fatal.max(deadline)),
        SegmentVote::SourceFatal(source_fatal)
    );
    assert_eq!(
        decode_route_owner_vote(route_owner_vote(0, 107).max(resource)),
        SegmentVote::OwnerFatal(107)
    );
}

#[test]
fn full_width_segment_contract_supports_exactly_two_to_the_32_rows() {
    use mgbfs_runtime::segmented_history::SegmentLayout;
    assert_eq!(u64::from(LOCAL_ROWS), 1u64 << 32);
    let max_row = StateAddress::new(7, 2, (1u64 << 32) - 1).unwrap();
    assert_eq!(StateAddress::decode(max_row.encode()).unwrap(), max_row);
    assert!(StateAddress::new(0, 0, 1u64 << 32).is_err());
    let segment = SegmentLayout::new(14, 1, 1u64 << 32, 3).unwrap();
    assert!(segment.device_bytes > 180_000_000_000u64);
    assert!(SegmentLayout::new(14, 1, (1u64 << 32) + 1, 3).is_err());
}
