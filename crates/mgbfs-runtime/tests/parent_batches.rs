use mgbfs_cuda::native_owner::Extent;
use mgbfs_runtime::parent_batches::ParentCursor;

#[test]
fn lookahead_preserves_physical_wrap_and_partial_tail_without_consuming() {
    let extents = [
        Extent {
            begin: 90,
            sequence: 190,
            count: 10,
            ..Extent::default()
        },
        Extent {
            begin: 0,
            sequence: 200,
            count: 5,
            ..Extent::default()
        },
    ];
    let mut cursor = ParentCursor::default();
    let mut actual = Vec::new();
    while let Some(batch) = cursor.peek(&extents, 4).unwrap() {
        assert_eq!(cursor.peek(&extents, 4).unwrap(), Some(batch));
        assert_eq!(cursor.take(&extents, 4).unwrap(), Some(batch));
        actual.push((
            batch.extent,
            batch.offset,
            batch.begin,
            batch.sequence,
            batch.count,
        ));
    }
    assert_eq!(
        actual,
        [
            (0, 0, 90, 190, 4),
            (0, 4, 94, 194, 4),
            (0, 8, 98, 198, 2),
            (1, 0, 0, 200, 4),
            (1, 4, 4, 204, 1)
        ]
    );
    assert_eq!(cursor.take(&extents, 4).unwrap(), None);
}

#[test]
fn empty_rank_is_empty_but_zero_batch_and_invalid_extents_are_errors() {
    let mut cursor = ParentCursor::default();
    assert_eq!(cursor.take(&[], 7).unwrap(), None);
    assert!(cursor.take(&[], 0).is_err());
    for extent in [
        Extent::default(),
        Extent {
            begin: u64::MAX,
            sequence: 0,
            count: 2,
            ..Extent::default()
        },
        Extent {
            begin: 0,
            sequence: u64::MAX,
            count: 2,
            ..Extent::default()
        },
    ] {
        assert!(ParentCursor::default().take(&[extent], 1).is_err());
    }
}

#[test]
fn fixed_depth_rounds_include_empty_rank_and_keep_physical_extents_separate() {
    let extents = [
        Extent {
            begin: 90,
            sequence: 190,
            count: 10,
            ..Extent::default()
        },
        Extent {
            begin: 0,
            sequence: 200,
            count: 5,
            ..Extent::default()
        },
    ];
    assert_eq!(ParentCursor::round_count(&[], 4).unwrap(), 1);
    assert_eq!(ParentCursor::round_count(&extents, 4).unwrap(), 5);
    assert_eq!(ParentCursor::round_count(&extents, 16).unwrap(), 2);
    assert!(ParentCursor::round_count(&extents, 0).is_err());
    assert!(ParentCursor::round_count(&[Extent::default()], 4).is_err());
}

#[test]
fn fixed_depth_rounds_reject_unrepresentable_schedule() {
    let extents = [
        Extent {
            begin: 0,
            sequence: 0,
            count: u64::from(u32::MAX),
            ..Extent::default()
        },
        Extent {
            begin: 0,
            sequence: 0,
            count: u64::from(u32::MAX),
            ..Extent::default()
        },
    ];
    assert!(ParentCursor::round_count(&extents, 1).is_err());
}
