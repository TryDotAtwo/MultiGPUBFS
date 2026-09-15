use mgbfs_core::matrix::MatrixGroup;

#[test]
fn benchmark_names_select_the_intended_generators_and_group_order() {
    let (name, symmetric) = MatrixGroup::from_reference_label("s4").unwrap();
    assert_eq!(name, "s4");
    assert_eq!(symmetric.expected_max_unique_states, 24);
    assert_eq!(
        symmetric.generators,
        MatrixGroup::symmetric_permutation_matrices(4)
            .unwrap()
            .generators
    );
    let (name, unitriangular) = MatrixGroup::from_reference_label("u4m3").unwrap();
    assert_eq!(name, "u4m3");
    assert_eq!(unitriangular.expected_max_unique_states, 729);
    assert_eq!(unitriangular.modulus, 3);
    assert_eq!(
        unitriangular.generators,
        MatrixGroup::unitriangular(4, 3).unwrap().generators
    );
}

#[test]
fn malformed_or_unrepresentable_groups_never_select_another_graph() {
    for name in [
        "", "s", "S4", "s1", "s9999", "u4", "u4m1", "u4m257", "u4m3m2", "u0m2", "s4junk",
    ] {
        assert!(MatrixGroup::from_reference_label(name).is_err(), "{name}");
    }
}
