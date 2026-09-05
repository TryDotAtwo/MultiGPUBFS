use mgbfs_core::matrix::encode_permutation_matrix;

#[test]
fn one_hot_matrix_has_a_lossless_permutation_encoding() {
    let matrix = [0, 1, 0, 0, 0, 1, 1, 0, 0];
    assert_eq!(encode_permutation_matrix(&matrix, 3), Ok(vec![1, 2, 0]));
}

#[test]
fn archive_encoding_rejects_non_bijective_or_non_binary_matrices() {
    assert!(encode_permutation_matrix(&[1, 0, 1, 0], 2).is_err());
    assert!(encode_permutation_matrix(&[1, 0, 1, 0], 3).is_err());
    assert!(encode_permutation_matrix(&[2, 0, 0, 1], 2).is_err());
}
