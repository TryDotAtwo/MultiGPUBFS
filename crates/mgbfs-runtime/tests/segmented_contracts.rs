use mgbfs_runtime::segmented_history::{ChunkRanges, SegmentGeometry, StateAddress, sum_rank_counts, LOCAL_ROWS};

#[test]
fn local_reference_has_24_bit_rows_and_a_separate_segment_number() {
    for segment in [0, 1, 256, u32::MAX] {
        for bank in 0..3 {
            for row in [0, 1, LOCAL_ROWS - 1] {
                let loc = StateAddress::new(segment, bank, row).unwrap();
                assert_eq!(StateAddress::decode(loc.encode()).unwrap(), loc);
                assert_eq!(loc.local_payload(), ((bank as u32) << 24) | row);
            }
        }
    }
    assert!(StateAddress::new(0, 0, LOCAL_ROWS).is_err());
    assert!(StateAddress::new(0, 3, 0).is_err());
    assert!(StateAddress::decode(1u64 << 26).is_err());
    assert!(StateAddress::decode(3u64 << 24).is_err());
}

#[test]
fn independent_small_segments_exceed_the_old_single_u32_arena_limit() {
    let g = SegmentGeometry::new(1024, LOCAL_ROWS, 3).unwrap();
    assert_eq!(g.rows(), 1024u64 * 3 * u64::from(LOCAL_ROWS));
    assert!(g.rows() > u64::from(u32::MAX));
    assert_eq!(g.arena_bytes(1).unwrap(), g.rows());
    assert!(SegmentGeometry::new(1, LOCAL_ROWS + 1, 3).is_err());
    assert!(SegmentGeometry::new(0, 8, 1).is_err());
    assert!(SegmentGeometry::new(1, 0, 1).is_err());
    assert!(SegmentGeometry::new(1, 8, 2).is_err());
    assert!(SegmentGeometry::new(u32::MAX, LOCAL_ROWS, 3).unwrap().arena_bytes(u32::MAX).is_err());
}

#[test]
fn rank_totals_and_chunk_cursors_do_not_wrap_at_u32() {
    assert_eq!(sum_rank_counts(&[u64::from(u32::MAX), 99]).unwrap(), u64::from(u32::MAX) + 99);
    assert!(sum_rank_counts(&[u64::MAX, 1]).is_err());
    let ranges: Vec<_> = ChunkRanges::new(u64::from(u32::MAX) + 7, u32::MAX).unwrap().collect();
    assert_eq!(ranges, vec![0..u64::from(u32::MAX), u64::from(u32::MAX)..u64::from(u32::MAX)+7]);
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
    assert_eq!(directed.device_bytes, 30736);
    let rolling = SegmentLayout::new(14, 1, 1024, 3).unwrap();
    assert_eq!(rolling.slots, 8192);
    assert_eq!(rolling.position_bytes, 3 * 1024 * 8);
    assert_eq!(rolling.device_bytes, 133136);
    let reserve = 2u64 << 30;
    let fixed = 1u64 << 30;
    let admitted = rolling.admit(reserve + fixed + rolling.device_bytes * 5000000, reserve, fixed, None).unwrap();
    assert_eq!(admitted.segments, 5000000);
    assert_eq!(admitted.rows_per_bank, 5000000u64 * 1024);
    assert!(admitted.rows_per_bank > u32::MAX as u64);
    assert_eq!(admitted.device_bytes, fixed + rolling.device_bytes * 5000000);
    assert!(rolling.admit(reserve + fixed + rolling.device_bytes - 1, reserve, fixed, None).is_err());
    assert_eq!(rolling.admit(reserve + fixed + rolling.device_bytes * 20, reserve, fixed, Some(2000)).unwrap().segments, 20);
    assert!(SegmentLayout::new(0, 1, 1024, 1).is_err());
    assert!(SegmentLayout::new(2, 4, 1024, 1).is_err());
    assert!(SegmentLayout::new(2, 8, (1<<24)+1, 1).is_err());
}

#[test]
fn queue_skew_retry_does_not_hide_independent_fatal_errors() {
    use mgbfs_runtime::segmented_history::classify_route_and_regeneration;
    assert_eq!(classify_route_and_regeneration(0,0),0);
    assert_eq!(classify_route_and_regeneration(1,0),1);
    assert_eq!(classify_route_and_regeneration(1,4),1);
    for (route,regen) in [(0,4),(0,2),(1,2),(1,6),(2,0),(4,0),(5,4)] {
        assert!(classify_route_and_regeneration(route,regen)>1, "route={route} regen={regen}");
    }
    let rank0=classify_route_and_regeneration(1,4);
    let rank1=classify_route_and_regeneration(0,4);
    assert!(rank0.max(rank1)>1);
}

#[test]
fn collective_vote_prioritizes_fatal_over_resource_and_retry() {
    use mgbfs_runtime::segmented_history::{route_owner_vote,decode_route_owner_vote,SegmentVote};
    let source_fatal=0x1_0004;
    assert_eq!(decode_route_owner_vote(route_owner_vote(0,0)),SegmentVote::Continue);
    assert_eq!(decode_route_owner_vote(route_owner_vote(1,0)),SegmentVote::Retry);
    assert_eq!(decode_route_owner_vote(route_owner_vote(1,10)),SegmentVote::Resource);
    assert_eq!(decode_route_owner_vote(route_owner_vote(source_fatal,10)),SegmentVote::SourceFatal(source_fatal));
    assert_eq!(decode_route_owner_vote(route_owner_vote(1,107)),SegmentVote::OwnerFatal(107));
    assert_eq!(decode_route_owner_vote(route_owner_vote(source_fatal,107)),SegmentVote::SourceFatal(source_fatal));
    // A max vote between different ranks must obey the same priority.
    let resource=route_owner_vote(0,10);let fatal=route_owner_vote(source_fatal,0);let retry=route_owner_vote(1,0);
    assert_eq!(decode_route_owner_vote(resource.max(fatal)),SegmentVote::SourceFatal(source_fatal));
    assert_eq!(decode_route_owner_vote(retry.max(resource)),SegmentVote::Resource);
    assert_eq!(decode_route_owner_vote(route_owner_vote(0,107).max(resource)),SegmentVote::OwnerFatal(107));
}
