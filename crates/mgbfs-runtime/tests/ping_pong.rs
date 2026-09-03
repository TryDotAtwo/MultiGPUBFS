#![cfg(feature = "cuda")]
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::dense_device::DenseDeviceStepper;

#[test]
fn generation_variants_preserve_full_layers() {
    check_generation_variants(6);
}
#[test]
fn generation_variants_small_feedback() {
    check_generation_variants(3);
}
fn check_generation_variants(max_modulus: u16) {
    for m in 2..=max_modulus {
        let g = MatrixGroup::unitriangular(4, m).unwrap();
        let oracle = g
            .exact_layers(g.expected_max_unique_states as usize)
            .unwrap();
        for variant in 1..=4 {
            for pre in [false, true] {
                let mut bfs = DenseDeviceStepper::new_pipelined_with_generation(
                    &g,
                    20260828u128.to_le_bytes(),
                    257,
                    g.expected_max_unique_states as u32,
                    pre,
                    variant,
                )
                .unwrap();
                for (d, expected) in oracle.iter().enumerate() {
                    let mut states = bfs.snapshot().unwrap();
                    states.sort();
                    assert_eq!(
                        &states, expected,
                        "m={m} variant={variant} depth={d} pre={pre}"
                    );
                    assert_eq!(bfs.advance().unwrap(), d + 1 < oracle.len());
                }
            }
        }
    }
}

#[test]
fn full_u4_pipelined_sweep() {
    for m in 2..=6 {
        let g = MatrixGroup::unitriangular(4, m).unwrap();
        let oracle = g
            .exact_layers(g.expected_max_unique_states as usize)
            .unwrap();
        for seed in [0u128, 1, 20260828] {
            for pre in [false, true] {
                let mut bfs = DenseDeviceStepper::new_pipelined(
                    &g,
                    seed.to_le_bytes(),
                    257,
                    g.expected_max_unique_states as u32,
                    pre,
                )
                .unwrap();
                for (d, expected) in oracle.iter().enumerate() {
                    let mut actual = bfs.snapshot().unwrap();
                    actual.sort();
                    assert_eq!(&actual, expected, "m={m} seed={seed} pre={pre} depth={d}");
                    assert_eq!(bfs.advance().unwrap(), d + 1 < oracle.len());
                }
            }
        }
    }
}

#[test]
fn reused_slots_and_partial_tails_preserve_every_layer() {
    for (n, m, batch) in [(3, 3, 1), (4, 2, 7), (4, 3, 64)] {
        let g = MatrixGroup::unitriangular(n, m).unwrap();
        let oracle = g
            .exact_layers(g.expected_max_unique_states as usize)
            .unwrap();
        for pre in [false, true] {
            let mut bfs = DenseDeviceStepper::new_pipelined(
                &g,
                [19; 16],
                batch,
                g.expected_max_unique_states as u32,
                pre,
            )
            .unwrap();
            for (depth, expected) in oracle.iter().enumerate() {
                let mut states = bfs.snapshot().unwrap();
                states.sort();
                assert_eq!(
                    &states, expected,
                    "n={n} m={m} batch={batch} depth={depth} pre={pre}"
                );
                assert_eq!(bfs.advance().unwrap(), depth + 1 < oracle.len());
            }
            assert!(bfs.snapshot().unwrap().is_empty());
        }
    }
}
#[test]
fn failure_with_both_slots_in_flight_is_sticky_and_drains_on_drop() {
    let g = MatrixGroup::unitriangular(4, 3).unwrap();
    let mut bfs = DenseDeviceStepper::new_pipelined(&g, [7; 16], 1, 6, false).unwrap();
    assert!(bfs.advance().unwrap());
    assert!(bfs.advance().is_err());
    assert!(bfs.advance().is_err());
    assert!(bfs.snapshot().is_err());
}
