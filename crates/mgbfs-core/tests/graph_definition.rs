use mgbfs_core::graph_definition::{GraphAction, GraphDefinitionV2, MatrixGeneratorV2};
fn matrix(g: Vec<(Vec<i64>, u32)>, start: Vec<i64>, n: u32, m: u32) -> GraphDefinitionV2 {
    GraphDefinitionV2 {
        schema: 2,
        name: "test".into(),
        generator_names: (0..g.len()).map(|i| i.to_string()).collect(),
        start,
        expected_max_unique_states: None,
        action: GraphAction::Matrix {
            rows: n,
            cols: m,
            generators: g
                .into_iter()
                .map(|(matrix, modulo)| MatrixGeneratorV2 { matrix, modulo })
                .collect(),
        },
    }
}
#[test]
fn rectangular_and_wrapping_semantics() {
    let g = matrix(vec![(vec![1, 1, 0, 1], 257)], vec![256, 2], 2, 1);
    g.validate().unwrap();
    assert_eq!(g.successor(&g.start, 0).unwrap(), vec![1, 2]);
    let g = matrix(vec![(vec![2], 7)], vec![i64::MAX], 1, 1);
    assert_eq!(g.successor(&g.start, 0).unwrap(), vec![5]);
    let g = matrix(vec![(vec![2], 0)], vec![i64::MAX], 1, 1);
    assert_eq!(g.successor(&g.start, 0).unwrap(), vec![-2]);
}
#[test]
fn directed_repeated_labels_have_exact_all_visited_oracle() {
    let g = GraphDefinitionV2 {
        schema: 2,
        name: "lx".into(),
        generator_names: vec!["l".into(), "x".into()],
        start: vec![0, 0, 1, 2],
        expected_max_unique_states: None,
        action: GraphAction::Permutation {
            degree: 4,
            generators: vec![vec![1, 2, 3, 0], vec![1, 0, 2, 3]],
        },
    };
    g.validate().unwrap();
    assert!(!g.inverse_closed().unwrap());
    assert_eq!(
        g.exact_layers(100)
            .unwrap()
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        12
    );
    assert!(g.exact_layers(2).is_err());
}
#[test]
fn invalid_shapes_and_noncanonical_modular_inverse() {
    let bad = matrix(vec![(vec![1], 3)], vec![1, 2], 2, 1);
    assert!(bad.validate().is_err());
    let g = matrix(vec![(vec![1], 7)], vec![8], 1, 1);
    assert!(!g.inverse_closed().unwrap());
    let g = matrix(vec![(vec![1], 7)], vec![6], 1, 1);
    assert!(g.inverse_closed().unwrap());
    let g = matrix(vec![(vec![1], 7), (vec![1], 5)], vec![1], 1, 1);
    assert!(!g.inverse_closed().unwrap());
}
#[test]
fn serde_and_identity_exclude_names() {
    let mut g = matrix(vec![(vec![1], 3)], vec![1], 1, 1);
    let first = g.semantic_digest().unwrap();
    g.name = "changed".into();
    g.generator_names = vec!["changed".into()];
    assert_eq!(g.semantic_digest().unwrap(), first);
    let raw = serde_json::to_vec(&g).unwrap();
    let decoded: GraphDefinitionV2 = serde_json::from_slice(&raw).unwrap();
    assert_eq!(decoded.semantic_digest().unwrap(), first);
}

#[test]
fn wrapped_non_power_two_is_not_an_inverse_proof() {
    let g = matrix(vec![(vec![i64::MAX], 3)], vec![2], 1, 1);
    // a*a wraps to 1, but actual action maps both 1 and 2 to 1 modulo 3.
    assert_eq!(g.successor(&[2], 0).unwrap(), vec![1]);
    assert_eq!(g.successor(&[1], 0).unwrap(), vec![1]);
    assert!(!g.inverse_closed().unwrap());
    let g = matrix(vec![(vec![i64::MAX], 4)], vec![2], 1, 1);
    assert!(g.inverse_closed().unwrap());
    let g = matrix(vec![(vec![100], 101)], vec![2], 1, 1);
    assert!(g.inverse_closed().unwrap());
    let g = matrix(vec![(vec![i64::MAX], 0)], vec![2], 1, 1);
    assert!(g.inverse_closed().unwrap());
}

#[test]
fn reverse_paths_preserve_original_generator_labels_and_modular_domain() {
    let g = matrix(vec![(vec![2], 7), (vec![4], 7)], vec![3], 1, 1);
    let reverse = g.reversed_for_paths(&[vec![6]]).unwrap();
    assert_eq!(reverse.start, g.start);
    for label in 0..g.generator_count() {
        for x in 0..7 {
            let next = g.successor(&[x], label).unwrap();
            assert_eq!(reverse.successor(&next, label).unwrap(), [x]);
        }
    }
    assert!(g.reversed_for_paths(&[vec![7]]).is_err());
    let g = GraphDefinitionV2 {
        action: GraphAction::Permutation {
            degree: 3,
            generators: vec![vec![1, 2, 0]],
        },
        start: vec![-4, 2, 2],
        generator_names: vec!["rotate".into()],
        ..g
    };
    let next = g.successor(&g.start, 0).unwrap();
    assert_eq!(
        g.reversed_for_paths(&[next.clone()])
            .unwrap()
            .successor(&next, 0)
            .unwrap(),
        g.start
    );
}

#[test]
fn reverse_paths_invert_missing_generators_over_binary_rings() {
    for modulus in [0, 8] {
        let g = matrix(vec![(vec![2, 3, 1, 4], modulus)], vec![1, 6], 2, 1);
        assert!(!g.inverse_closed().unwrap());
        let reverse = g.reversed_for_paths(&[vec![2, 7]]).unwrap();
        let states = if modulus == 0 {
            vec![vec![i64::MIN, i64::MAX], vec![-3, -1], vec![1, 6]]
        } else {
            (0..8)
                .flat_map(|i| (0..8).map(move |j| vec![i, j]))
                .collect()
        };
        for state in states {
            let next = g.successor(&state, 0).unwrap();
            assert_eq!(reverse.successor(&next, 0).unwrap(), state);
        }
    }
    let singular = matrix(vec![(vec![2], 8)], vec![1], 1, 1);
    assert!(singular.reversed_for_paths(&[vec![2]]).is_err());
}

#[test]
fn reverse_paths_invert_composite_modulus_without_a_unit_in_first_column() {
    let g = matrix(vec![(vec![2, 3, 3, 4], 12)], vec![1, 6], 2, 1);
    assert!(!g.inverse_closed().unwrap());
    let reverse = g.reversed_for_paths(&[vec![3, 5]]).unwrap();
    for i in 0..12 {
        for j in 0..12 {
            let next = g.successor(&[i, j], 0).unwrap();
            assert_eq!(reverse.successor(&next, 0).unwrap(), [i, j]);
        }
    }
    let singular = matrix(vec![(vec![2, 0, 0, 1], 12)], vec![1, 6], 2, 1);
    assert!(singular.reversed_for_paths(&[vec![3, 5]]).is_err());
}
