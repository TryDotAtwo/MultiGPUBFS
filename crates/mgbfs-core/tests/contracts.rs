use mgbfs_core::{
    hash::{GemmHash, Hash128, PRIME},
    matrix::MatrixGroup,
};

#[test]
fn hash_matches_independent_hashlib_frozen_vectors() {
    let h = GemmHash::from_seed(16, [0; 16]).unwrap();
    assert_eq!(
        h.hash(&[0; 16]).unwrap(),
        Hash128([1710827310, 2245209978, 2416263789, 1202685372])
    );
    assert_eq!(
        h.hash(&(0u8..16).collect::<Vec<_>>()).unwrap(),
        Hash128([2244859959, 2401010834, 3228855414, 2263550226])
    );
}

#[test]
fn modular_hash_reconstructs_unsigned_limbs_without_signed_overflow() {
    let h = GemmHash {
        coefficients: vec![[4_294_967_290, 1, 256, 65_536], [7, 0, 2, 3]],
        offsets: [1, 2, 3, 4],
    };
    assert_eq!(
        h.hash(&[255, 2]).unwrap(),
        Hash128([4_294_967_051, 257, 65_287, 16_711_690])
    );
    let limbs = h.limbs();
    assert_eq!(limbs.len(), 32);
    let mut partials = [0i32; 16];
    for (i, x) in [255, 2].iter().enumerate() {
        for j in 0..16 {
            partials[j] += *x as i32 * limbs[i * 16 + j] as i32;
        }
    }
    assert_eq!(
        h.hash_from_partials(&partials).unwrap(),
        h.hash(&[255, 2]).unwrap()
    );
}

#[test]
fn seeded_hash_rejects_invalid_width_and_changes_with_seed() {
    assert!(GemmHash::from_seed(0, [0; 16]).is_err());
    assert!(GemmHash::from_seed(33026, [0; 16]).is_err());
    let a = GemmHash::from_seed(16, [0; 16]).unwrap();
    let b = GemmHash::from_seed(16, [1; 16]).unwrap();
    assert_ne!(a.hash(&[0; 16]).unwrap(), b.hash(&[0; 16]).unwrap());
    assert!(a.hash(&[0; 15]).is_err());
    assert!(a.coefficients.iter().flatten().all(|x| (*x as u64) < PRIME));
}

#[test]
fn matrix_action_is_left_multiplication_with_exact_modular_reduction() {
    let g = MatrixGroup::unitriangular(3, 5).unwrap();
    let state = vec![1, 2, 3, 0, 1, 4, 0, 0, 1];
    assert_eq!(
        g.successor(&state, 0).unwrap(),
        vec![1, 3, 2, 0, 1, 4, 0, 0, 1]
    );
    assert_eq!(
        g.successor(&state, 2).unwrap(),
        vec![1, 1, 4, 0, 1, 4, 0, 0, 1]
    );
}

#[test]
fn tiny_oracle_exhausts_literal_cyclic_layers_and_capacity_is_not_truncation() {
    let g = MatrixGroup::unitriangular(2, 5).unwrap();
    let layers = g.exact_layers(5).unwrap();
    assert_eq!(layers.iter().map(Vec::len).collect::<Vec<_>>(), [1, 2, 2]);
    assert_eq!(layers[1], vec![vec![1, 1, 0, 1], vec![1, 4, 0, 1]]);
    assert!(g.exact_layers(4).is_err());
}

#[test]
fn manifest_rejects_wrong_inverse_noncanonical_and_singular_start() {
    let mut g = MatrixGroup::unitriangular(2, 5).unwrap();
    g.inverse_map[0] = 0;
    assert!(g.validate().is_err());
    g.inverse_map[0] = 1;
    g.start[0] = 5;
    assert!(g.validate().is_err());
    g.start = vec![0; 4];
    assert!(g.validate().is_err());
}
