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
    for (n, repeated) in [(0,4), (3,4), (15,0), (257,4)] {
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

#[test]
fn wide_orbit_is_exact_in_aligned_u64_words() {
    let graph = LrxMultiset::new(27,12).unwrap();
    let expected = (13u128..=27).product::<u128>();
    assert_eq!(graph.order_words(), &[expected as u64, (expected >> 64) as u64]);
    assert!(!graph.order_fits_u64());
    assert_eq!(graph.order(), u64::MAX);
    let easy = LrxMultiset::new(128,127).unwrap();
    assert_eq!(easy.order_words(), &[128]);
    assert_eq!(easy.exact_layers(128).unwrap().iter().map(Vec::len).sum::<usize>(),128);
    assert!(LrxMultiset::new(128,1).unwrap().order_words().len()>2);
}

#[test]
fn wide_word_orbit_does_not_require_full_symmetric_order() {
    let graph = LrxMultiset::new(32, 31).unwrap();
    assert_eq!(graph.order(), 32);
    assert!(graph.position_group().is_err()); // Full S_32 genuinely exceeds u64.
    let action = graph.position_action().unwrap();
    action.validate().unwrap();
    assert_eq!(action.expected_max_unique_states, 32);
    for generator in 0..3 {
        let product: Vec<u8> = action.generators[generator].chunks(32)
            .map(|row| row.iter().zip(graph.start()).map(|(&a,&b)| a*b).sum()).collect();
        assert_eq!(product, graph.successor(graph.start(), generator).unwrap());
    }
    assert_eq!(graph.exact_layers(32).unwrap().iter().map(Vec::len).sum::<usize>(), 32);
}
