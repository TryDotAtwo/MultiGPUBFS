use mgbfs_core::owner_job::{BucketJob, Range};
use mgbfs_runtime::jobs::{split, JobSpan};
fn ranges(counts: &[u64]) -> Vec<Range> {
    let mut begin = 0;
    counts
        .iter()
        .map(|&count| {
            let r = Range { begin, count };
            begin += count;
            r
        })
        .collect()
}
#[test]
fn splits_large_bucket_and_never_crosses_shard_or_drops_rows() {
    let input = ranges(&[3, 12, 0, 2, 1, 3, 0, 0]);
    let old = ranges(&[1; 8]);
    let mut desc = [BucketJob::default(); 16];
    let mut spans = [JobSpan::default(); 16];
    let (nd, ns) = split(&input, &old, &old, 8, 2, 4, 7, &mut desc, &mut spans).unwrap();
    assert_eq!((nd, ns), (6, 4));
    assert_eq!(
        spans[..ns]
            .iter()
            .map(|s| (s.source_begin, s.rows, s.buckets))
            .collect::<Vec<_>>(),
        vec![(0, 8, 2), (8, 7, 1), (15, 2, 1), (17, 4, 2)]
    );
    for span in &spans[..ns] {
        let ds = &desc[span.first..span.first + span.buckets as usize];
        assert!(ds.iter().all(|d| d.bucket / 4 == ds[0].bucket / 4));
        assert_eq!(ds[0].incoming.begin, 0);
        assert_eq!(
            ds.iter().map(|d| d.incoming.count).sum::<u64>(),
            span.rows as u64
        );
        assert!(ds
            .iter()
            .all(|d| d.generation == 7 && d.accepted_count == 0));
    }
}
#[test]
fn empty_input_and_capacity_failure_are_explicit() {
    let r = ranges(&[0; 4]);
    let mut ds = [BucketJob::default(); 1];
    let mut spans = [JobSpan::default(); 1];
    assert_eq!(
        split(&r, &r, &r, 8, 2, 4, 0, &mut ds, &mut spans).unwrap(),
        (0, 0)
    );
    let r = ranges(&[10, 10, 0, 0]);
    assert!(split(&r, &r, &r, 8, 2, 4, 0, &mut ds, &mut spans).is_err());
    let mut bad = r;
    bad[1].begin = 9;
    assert!(split(&bad, &bad, &bad, 8, 2, 4, 0, &mut ds, &mut spans).is_err());
}
