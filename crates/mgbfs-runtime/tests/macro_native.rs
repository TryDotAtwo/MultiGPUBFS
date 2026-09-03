#![cfg(feature = "cuda")]
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::macro_native::{MacroNativeBfs, MacroNativeConfig};

#[test]
fn native_macro_layers_equal_full_state_oracle_for_k_and_partial_batches() {
    for (n, modulus) in [(3, 2), (3, 3), (4, 2)] {
        let graph = MatrixGroup::unitriangular(n, modulus).unwrap();
        let oracle = graph
            .exact_layers(graph.expected_max_unique_states as usize)
            .unwrap();
        for macro_depth in [1, 2, 3] {
            for prededup in [false, true] {
                let mut bfs = MacroNativeBfs::new(
                    &graph,
                    [macro_depth as u8; 16],
                    MacroNativeConfig {
                        macro_depth,
                        batch: 7,
                        layer_capacity: graph.expected_max_unique_states as u32,
                        future_capacity_per_depth: 16_384,
                        prededup,
                        generation_variant: 1,
                    },
                )
                .unwrap();
                let mut actual = vec![];
                loop {
                    let mut layer = bfs.snapshot().unwrap();
                    layer.sort();
                    actual.push(layer);
                    if !bfs.advance().unwrap() {
                        break;
                    }
                }
                assert_eq!(
                    actual, oracle,
                    "n={n} m={modulus} K={macro_depth} pre={prededup}"
                );
            }
        }
    }
}

#[test]
fn native_macro_runtime_capacity_failure_is_sticky() {
    let graph = MatrixGroup::unitriangular(3, 3).unwrap();
    let mut config = MacroNativeConfig {
        macro_depth: 3,
        batch: 8,
        layer_capacity: 27,
        future_capacity_per_depth: 128,
        prededup: false,
        generation_variant: 1,
    };
    config.future_capacity_per_depth = 1;
    let mut bfs = MacroNativeBfs::new(&graph, [0; 16], config).unwrap();
    assert!(bfs.advance().is_err());
    assert!(bfs.advance().is_err());
}
