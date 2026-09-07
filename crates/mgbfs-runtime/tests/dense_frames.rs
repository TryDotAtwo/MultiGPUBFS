use mgbfs_runtime::dense_frames::DenseFrames;

#[test]
fn equal_padded_sizes_keep_distinct_record_counts_and_ticket_identity() {
    use mgbfs_core::wire::{ExpectedFrame, FrameHeader, FrameKind};
    use mgbfs_runtime::{control_wire::Plane, scatter_admission::TicketKey};
    let mut plan = DenseFrames::new(&[1, 0], 16, 3, 1536).unwrap();
    plan.prepare(&[1, 2]).unwrap();
    assert_eq!(plan.sizes().unwrap(), &[768, 768]);
    let key = TicketKey {
        depth: 9,
        epoch: 99,
        source: 0,
        plane: Plane::Candidate,
        generation: 123,
    };
    let mut headers = [[0u8; 64]; 2];
    plan.encode_headers(key, 7, &mut headers).unwrap();
    for rank in 0..2 {
        let expected = ExpectedFrame {
            run_tag: 7,
            sequence: 99,
            batch: 123,
            depth: 9,
            source: 0,
            destination: rank,
            world: 2,
            kind: FrameKind::Dense,
            max_records: 3,
            max_payload: 768,
            state_stride: 16,
        };
        let decoded = FrameHeader::decode(&headers[rank as usize], &expected).unwrap();
        assert_eq!(decoded.count, if rank == 0 { 2 } else { 1 });
    }
    assert!(plan
        .encode_headers(TicketKey { source: 2, ..key }, 7, &mut headers)
        .is_err());
    assert!(plan
        .encode_headers(
            TicketKey {
                depth: u64::MAX,
                ..key
            },
            7,
            &mut headers
        )
        .is_err());
    assert!(plan
        .encode_headers(
            TicketKey {
                plane: Plane::Request,
                ..key
            },
            7,
            &mut headers
        )
        .is_err());
    assert!(plan.encode_headers(key, 7, &mut headers[..1]).is_err());
}

#[test]
fn physical_rank_frames_preserve_logical_sorted_ranges_and_empty_peers() {
    let mut plan = DenseFrames::new(&[2, 0, 1], 32, 18, 2304).unwrap();
    plan.prepare(&[1, 0, 17]).unwrap();
    assert_eq!(plan.sizes().unwrap(), &[0, 1536, 768]);
    let ranges: Vec<_> = plan
        .frames()
        .unwrap()
        .iter()
        .map(|f| (f.begin, f.count, f.offset, f.bytes))
        .collect();
    assert_eq!(ranges, [(1, 0, 0, 0), (1, 17, 0, 1536), (0, 1, 1536, 768)]);
    plan.prepare(&[0, 0, 0]).unwrap();
    assert_eq!(plan.sizes().unwrap(), &[0, 0, 0]);
    assert!(plan
        .frames()
        .unwrap()
        .iter()
        .all(|f| f.count == 0 && f.offset == 0));
}

#[test]
fn capacity_failure_never_exposes_partial_or_previous_frame_metadata() {
    let mut plan = DenseFrames::new(&[0, 1], 16, 3, 768).unwrap();
    plan.prepare(&[3, 0]).unwrap();
    assert!(plan.prepare(&[1, 2]).is_err()); // Two nonempty frames need 1536 bytes.
    assert!(plan.frames().is_err());
    assert!(plan.sizes().is_err());
    assert!(plan.prepare(&[0, 0]).is_err());
}

#[test]
fn invalid_mapping_stride_and_record_counts_are_rejected() {
    assert!(DenseFrames::new(&[0, 0], 16, 3, 1536).is_err());
    assert!(DenseFrames::new(&[0, 2], 16, 3, 1536).is_err());
    assert!(DenseFrames::new(&[], 16, 3, 1536).is_err());
    assert!(DenseFrames::new(&[0], 17, 3, 1536).is_err());
    let mut plan = DenseFrames::new(&[0, 1], 16, 3, 1536).unwrap();
    assert!(plan.frames().is_err());
    assert!(plan.prepare(&[2, 2]).is_err());
    let mut plan = DenseFrames::new(&[0, 1], 16, u32::MAX, u64::MAX).unwrap();
    assert!(plan.prepare(&[u32::MAX, 1]).is_err());
}
