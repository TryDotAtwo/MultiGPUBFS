use mgbfs_cuda::allocation::*;
#[test]
fn route_report_counts_shared_scratch_once_and_checks_query_identity() {
    assert_eq!(std::mem::size_of::<RouteBytes>(), 64);
    let q = RouteBytes {
        sorted: 272,
        refs: 136,
        indices: 68,
        selected: 68,
        flags: 17,
        scratch: 4096,
        sort_scratch: 4096,
        select_scratch: 2048,
    };
    let report = q.report(17).unwrap();
    assert_eq!(report.allocations.len(), 6);
    assert_eq!(
        report.allocations.iter().map(|p| p.bytes).sum::<u64>(),
        4657
    );
    assert!(report.source.contains("capacity=17"));
    assert!(q.report(18).is_err());
    assert!(RouteBytes { scratch: 2048, ..q }.report(17).is_err());
    assert!(RouteBytes::default().report(0).is_err());
}
#[test]
fn abi_query_sizes_become_named_rank_ledger_inputs_without_recounting() {
    assert_eq!(std::mem::size_of::<GenerateBytes>(), 48);
    assert_eq!(std::mem::size_of::<HashBytes>(), 40);
    let g = GenerateBytes {
        generators: 384,
        packed_parents: 128,
        products_s32: 768,
        workspace: 0,
        k: 16,
        stride: 16,
        rows: 24,
        columns: 8,
    };
    let q = g.report(2).unwrap();
    assert!(q.source.contains("variant=2"));
    let actual: Vec<_> = q
        .allocations
        .iter()
        .map(|a| (a.name.as_str(), a.bytes, a.alignment))
        .collect();
    assert_eq!(
        actual,
        vec![
            ("generators", 384, 256),
            ("packed_parents", 128, 256),
            ("products_s32", 768, 256),
            ("workspace", 0, 256)
        ]
    );
    let h = HashBytes {
        weights: 256,
        offsets: 16,
        partials_s32: 768,
        workspace: 0,
        stride: 16,
        reserved: 0,
    };
    let q = h.report().unwrap();
    let actual: Vec<_> = q
        .allocations
        .iter()
        .map(|a| (a.name.as_str(), a.bytes, a.alignment))
        .collect();
    assert_eq!(
        actual,
        vec![
            ("weights", 256, 256),
            ("offsets", 16, 256),
            ("partials_s32", 768, 256),
            ("workspace", 0, 256)
        ]
    );
    assert!(GenerateBytes::default().report(0).is_err());
    assert!(g.report(5).is_err());
    assert!(HashBytes { reserved: 1, ..h }.report().is_err());
    assert!(HashBytes::default().report().is_err());
}

#[test]
fn macro_stage_reports_preserve_every_queried_plane() {
    let materialize = MaterializeBytes {
        keys: 80,
        sorted: 80,
        indices: 40,
        order: 40,
        scratch: 256,
    };
    let report = materialize.report(10).unwrap();
    assert_eq!(report.allocations.len(), 5);
    assert_eq!(report.allocations.iter().map(|x| x.bytes).sum::<u64>(), 496);
    assert!(materialize.report(11).is_err());

    let future = FutureMergeBytes {
        merged: 160,
        unique: 96,
        tags: 80,
        unique_tags: 48,
        indices: 40,
        selected: 40,
        selected_count: 4,
        flags: 10,
        states: 384,
        state: 8,
        scratch: 256,
    };
    let report = future.report(64, 6, 4).unwrap();
    assert_eq!(report.allocations.len(), 11);
    assert_eq!(
        report.allocations.iter().map(|x| x.bytes).sum::<u64>(),
        1126
    );
    assert!(future.report(64, 6, 5).is_err());
    assert!(FutureMergeBytes::default().report(64, 6, 4).is_err());
}
