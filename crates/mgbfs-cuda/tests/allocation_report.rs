use mgbfs_cuda::allocation::*;
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
