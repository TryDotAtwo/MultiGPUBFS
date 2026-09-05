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

#[test]
fn compact_gemm_matches_matrix_composition_for_all_s3_pairs() {
    let permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for generator in permutations {
        for parent in permutations {
            let mut g = [0u8; 9];
            let mut p = [0u8; 9];
            for row in 0..3 {
                g[row * 3 + generator[row]] = 1;
                p[row * 3 + parent[row]] = 1;
            }
            let mut matrix_child = [0u8; 9];
            let mut compact_child = [0u8; 3];
            for row in 0..3 {
                for k in 0..3 {
                    compact_child[row] += g[row * 3 + k] * parent[k] as u8;
                    for col in 0..3 {
                        matrix_child[row * 3 + col] += g[row * 3 + k] * p[k * 3 + col];
                    }
                }
            }
            assert_eq!(encode_permutation_matrix(&matrix_child, 3).unwrap(), compact_child);
        }
    }
}
