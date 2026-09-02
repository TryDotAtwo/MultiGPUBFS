use mgbfs_core::matrix::MatrixGroup;
#[test]
fn unitriangular_matrix_oracle_exhausts_moduli_two_through_six() {
    for m in 2..=6 {
        let g = MatrixGroup::unitriangular(4, m).unwrap();
        let total = (m as usize).pow(6);
        let layers = g.exact_layers(total).unwrap();
        assert_eq!(layers.iter().map(Vec::len).sum::<usize>(), total);
        if m == 5 {
            assert_eq!(layers.len() - 1, 10);
        }
        if m == 6 {
            assert_eq!(layers.len() - 1, 13);
        }
        // Inverse-closed Cayley edges cannot leave the three-layer window.
        let depths: std::collections::BTreeMap<_, _> = layers
            .iter()
            .enumerate()
            .flat_map(|(d, l)| l.iter().map(move |s| (s.clone(), d)))
            .collect();
        for (d, layer) in layers.iter().enumerate() {
            for state in layer {
                for mv in 0..g.generators.len() {
                    let child = g.successor(state, mv).unwrap();
                    assert!(depths[&child].abs_diff(d) <= 1);
                }
            }
        }
    }
}
