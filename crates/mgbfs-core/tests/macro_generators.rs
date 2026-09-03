use mgbfs_core::{config::RunConfigV1, macro_generators::MacroGeneratorSet, matrix::MatrixGroup};

fn hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn depth_one_is_exactly_the_original_generator_order() {
    let graph = MatrixGroup::unitriangular(4, 5).unwrap();
    let macros = MacroGeneratorSet::compile(&graph, 1).unwrap();
    assert_eq!(macros.requested_depth, 1);
    assert_eq!(macros.transitions.len(), graph.generators.len());
    for (id, transition) in macros.transitions.iter().enumerate() {
        assert_eq!(transition.matrix, graph.generators[id]);
        assert_eq!(transition.weight, 1);
        assert_eq!(transition.word, vec![id as u16]);
    }
}

#[test]
fn deeper_compilation_removes_identity_and_keeps_shortest_canonical_word() {
    let graph = MatrixGroup::unitriangular(3, 3).unwrap();
    let macros = MacroGeneratorSet::compile(&graph, 10).unwrap();
    assert!(macros
        .transitions
        .iter()
        .all(|t| t.weight >= 1 && t.weight <= 10));
    assert!(macros.transitions.iter().all(|t| t.matrix != graph.start));
    let mut matrices: Vec<_> = macros.transitions.iter().map(|t| &t.matrix).collect();
    matrices.sort();
    matrices.dedup();
    assert_eq!(matrices.len(), macros.transitions.len());
    for transition in &macros.transitions {
        assert_eq!(transition.word.len(), transition.weight as usize);
        let mut state = graph.start.clone();
        for &movement in &transition.word {
            state = graph.successor(&state, movement as usize).unwrap();
        }
        assert_eq!(state, transition.matrix);
    }
}

#[test]
fn weighted_macro_oracle_preserves_original_bfs_layers_through_depth_ten() {
    for modulus in 2..=4 {
        let graph = MatrixGroup::unitriangular(4, modulus).unwrap();
        let expected = graph
            .exact_layers(graph.expected_max_unique_states as usize)
            .unwrap();
        for macro_depth in [1, 2, 3, 10] {
            let macros = MacroGeneratorSet::compile(&graph, macro_depth).unwrap();
            let actual = graph
                .exact_layers_with_macros(&macros, graph.expected_max_unique_states as usize)
                .unwrap();
            assert_eq!(
                actual, expected,
                "modulus={modulus} macro_depth={macro_depth}"
            );
        }
    }
}

#[test]
fn zero_depth_and_route_slot_shortfall_fail_before_runtime_allocation() {
    let graph = MatrixGroup::unitriangular(4, 3).unwrap();
    assert_eq!(
        MacroGeneratorSet::compile(&graph, 0).unwrap_err(),
        "MACRO_DEPTH_ZERO"
    );

    let mut config = RunConfigV1::fixture(3).unwrap();
    config.macro_depth = 2;
    config.capacities.route_slot_records = config.parent_batch * graph.generators.len() as u64;
    assert_eq!(config.validate().unwrap_err(), "ROUTE_SLOT_CAPACITY");
}

#[test]
fn macro_depth_is_part_of_the_reproducible_config_digest() {
    let mut one = RunConfigV1::fixture(3).unwrap();
    let mut ten = one.clone();
    one.macro_depth = 1;
    ten.macro_depth = 10;
    ten.capacities.route_slot_records =
        ten.parent_batch * (ten.graph.expected_max_unique_states - 1);
    assert_ne!(one.digest().unwrap(), ten.digest().unwrap());
}

#[test]
fn symmetric_groups_are_non_toy_matrix_workloads_with_small_generator_sets() {
    let s8 = MatrixGroup::symmetric_permutation_matrices(8).unwrap();
    assert_eq!(s8.expected_max_unique_states, 40_320);
    assert_eq!(s8.generators.len(), 3);
    assert_eq!(
        s8.exact_layers(40_320)
            .unwrap()
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        40_320
    );

    let s12 = MatrixGroup::symmetric_permutation_matrices(12).unwrap();
    assert_eq!(s12.expected_max_unique_states, 479_001_600);
    assert_eq!(s12.generators.len(), 3);
    let macros = MacroGeneratorSet::compile(&s12, 10).unwrap();
    let counts = |set: &MacroGeneratorSet| {
        (1..=10)
            .map(|weight| {
                set.transitions
                    .iter()
                    .filter(|item| item.weight == weight)
                    .count()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        counts(&macros),
        vec![3, 6, 12, 24, 48, 90, 168, 314, 572, 1033]
    );
    assert_eq!(macros.transitions.len(), 2_270);
    assert_eq!(macros.effective_depth, 10);
    assert_eq!(
        hex(macros.digest_v1()),
        "17fb246bf0e30a8a8c33616a2f8f5563bc7cca707a2d5c740137daf035018141"
    );
    let s8_macros = MacroGeneratorSet::compile(&s8, 10).unwrap();
    assert_eq!(
        counts(&s8_macros),
        vec![3, 6, 12, 23, 44, 80, 142, 247, 411, 662]
    );
    assert_eq!(s8_macros.transitions.len(), 1_630);
    assert_eq!(
        hex(s8_macros.digest_v1()),
        "02220649b1f7c272ca1bbcf4bbb1eb24727913388fc58ea4814e506ad78320eb"
    );
}

#[test]
fn symmetric_group_constructor_rejects_unrepresentable_or_trivial_degrees() {
    assert_eq!(
        MatrixGroup::symmetric_permutation_matrices(1).unwrap_err(),
        "SYMMETRIC_DEGREE"
    );
    assert_eq!(
        MatrixGroup::symmetric_permutation_matrices(21).unwrap_err(),
        "SYMMETRIC_ORDER_OVERFLOW"
    );
}
