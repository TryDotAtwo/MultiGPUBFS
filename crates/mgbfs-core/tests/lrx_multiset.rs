use mgbfs_core::lrx_multiset::LrxMultiset;

#[test]
fn four_equal_tail_has_the_requested_start_and_orbit_not_factorial() {
    let graph = LrxMultiset::new(15, 4).unwrap();
    assert_eq!(graph.start(), &[0,1,2,3,4,5,6,7,8,9,10,11,11,11,11]);
    assert_eq!(graph.order(), 54_486_432_000);
    assert_eq!(LrxMultiset::new(14, 4).unwrap().order(), 3_632_428_800);
}

#[test]
fn five_positions_four_equal_form_a_five_cycle() {
    let graph = LrxMultiset::new(5, 4).unwrap();
    let layers = graph.exact_layers(5).unwrap();
    assert_eq!(layers.iter().map(Vec::len).collect::<Vec<_>>(), [1,2,2]);
    assert!(graph.exact_layers(4).is_err());
}

#[test]
fn moves_act_on_positions_and_preserve_repetitions() {
    let graph = LrxMultiset::new(6, 4).unwrap();
    assert_eq!(graph.successor(graph.start(), 0).unwrap(), [1,2,2,2,2,0]);
    assert_eq!(graph.successor(graph.start(), 1).unwrap(), [2,0,1,2,2,2]);
    assert_eq!(graph.successor(graph.start(), 2).unwrap(), [1,0,2,2,2,2]);
    let layers = graph.exact_layers(30).unwrap();
    assert_eq!(layers.iter().map(Vec::len).sum::<usize>(), 30);
    assert!(graph.successor(&[0,1,2,3,4,5], 0).is_err());
    assert!(graph.successor(graph.start(), 3).is_err());
}

#[test]
fn malformed_degree_multiplicity_and_overflow_are_rejected() {
    for (n, repeated) in [(0,4), (3,4), (15,0), (21,1), (257,4)] {
        assert!(LrxMultiset::new(n, repeated).is_err());
    }
    assert_eq!(LrxMultiset::new(4,4).unwrap().exact_layers(1).unwrap().len(), 1);
}

#[test]
fn reference_label_and_generator_matrices_preserve_the_word_action() {
    let graph = LrxMultiset::from_label("lrx6r4").unwrap();
    assert_eq!(graph.label(), "lrx6r4");
    let matrices = graph.position_group().unwrap();
    matrices.validate().unwrap();
    for generator in 0..3 {
        let word = graph.start();
        let product: Vec<u8> = matrices.generators[generator].chunks(6)
            .map(|row| row.iter().zip(word).map(|(&a,&b)| a*b).sum()).collect();
        assert_eq!(product, graph.successor(word, generator).unwrap());
    }
    for bad in ["s15", "lrx15", "lrx15r0", "lrx15r4junk"] {
        assert!(LrxMultiset::from_label(bad).is_err());
    }
}
