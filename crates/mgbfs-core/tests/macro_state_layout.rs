use mgbfs_core::{macro_memory::MacroStateLayout, matrix::MatrixGroup};

#[test]
fn compact_macro_storage_uses_permutation_vectors_and_validates_generators() {
    let mut graph = MatrixGroup::symmetric_permutation_matrices(4).unwrap();
    graph.start = vec![0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    let compact = MacroStateLayout::derive(&graph, 5).unwrap();
    assert_eq!(compact.start, vec![1, 0, 2, 3]);
    assert_eq!((compact.width, compact.stride), (4, 16));
    let matrix = MacroStateLayout::derive(&graph, 1).unwrap();
    assert_eq!(matrix.start, graph.start);
    assert_eq!((matrix.width, matrix.stride), (16, 16));
    let large = MatrixGroup::symmetric_permutation_matrices(17).unwrap();
    let compact = MacroStateLayout::derive(&large, 5).unwrap();
    assert_eq!((compact.width, compact.stride), (17, 32));
    assert!(MacroStateLayout::derive(&graph, 6).is_err());
    assert!(MacroStateLayout::derive(&MatrixGroup::unitriangular(3, 3).unwrap(), 5).is_err());
}
