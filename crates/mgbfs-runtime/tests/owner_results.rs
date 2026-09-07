use mgbfs_cuda::native_owner::Extent;
use mgbfs_runtime::{jobs::JobSpan, owner_results::append_dense_results};

fn record(begin: u64, sequence: u64, count: u64, descriptor: u64) -> Extent {
    Extent {
        begin,
        sequence,
        count,
        descriptor,
        granted_rows: count as u32,
        ready: 1,
        ..Extent::default()
    }
}

#[test]
fn only_consumed_first_descriptors_are_results_and_wrap_stays_separate() {
    // Slots 1 and 4 remain job descriptors, not published state extents.
    let records = [
        record(6, 6, 2, 10),
        Extent::default(),
        record(0, 8, 0, 0),
        record(0, 8, 3, 11),
        Extent::default(),
        record(3, 11, 2, 12),
    ];
    let spans: Vec<_> = [0, 2, 3, 5]
        .map(|first| JobSpan {
            first,
            ..JobSpan::default()
        })
        .into();
    let mut next = Vec::with_capacity(2);
    append_dense_results(&records, &spans, &mut next).unwrap();
    assert_eq!(next.len(), 2);
    assert_eq!(
        (
            next[0].begin,
            next[0].sequence,
            next[0].count,
            next[0].padding[1]
        ),
        (6, 6, 2, 10)
    );
    assert_eq!(
        (
            next[1].begin,
            next[1].sequence,
            next[1].count,
            next[1].padding[1]
        ),
        (0, 8, 5, 12)
    );
    assert_eq!(next[1].granted_rows, 5);
}

#[test]
fn malformed_or_unpublished_results_do_not_publish_frontier_metadata() {
    let spans = [
        JobSpan {
            first: 0,
            ..JobSpan::default()
        },
        JobSpan {
            first: 1,
            ..JobSpan::default()
        },
    ];
    for bad in [
        Extent::default(),
        Extent {
            granted_rows: 1,
            ..record(2, 2, 2, 1)
        },
        record(u64::MAX, 2, 2, 1),
    ] {
        let mut next = Vec::with_capacity(2);
        assert!(append_dense_results(&[record(0, 0, 2, 0), bad], &spans, &mut next).is_err());
        assert!(next.is_empty());
    }
    let mut next = Vec::with_capacity(1);
    assert!(append_dense_results(&[record(0, 0, 1, 0)], &spans, &mut next).is_err());
    assert!(next.is_empty());
}
