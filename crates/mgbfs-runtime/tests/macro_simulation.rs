use mgbfs_core::{macro_generators::MacroGeneratorSet, matrix::MatrixGroup};
use mgbfs_runtime::macro_simulation::{run_macro, MacroSimulationConfig};

#[test]
fn macro_scheduler_preserves_original_layers_across_k_ranks_and_ready_orders() {
    for (n, modulus) in [(3, 2), (3, 3), (4, 2)] {
        let graph = MatrixGroup::unitriangular(n, modulus).unwrap();
        let oracle = graph
            .exact_layers(graph.expected_max_unique_states as usize)
            .unwrap();
        for macro_depth in [1, 2, 3, 10] {
            let macros = MacroGeneratorSet::compile(&graph, macro_depth).unwrap();
            for rank_map in [vec![0], vec![0, 1], vec![1, 0]] {
                for schedule in [0, 1, 7] {
                    for pre_dedup in [false, true] {
                        let result = run_macro(
                            &graph,
                            &macros,
                            &MacroSimulationConfig {
                                rank_map: rank_map.clone(),
                                buckets: 4,
                                future_capacity_per_bucket: 4096,
                                settled_capacity_per_bucket: 4096,
                                seed: [schedule as u8; 16],
                                schedule,
                                pre_dedup,
                            },
                        )
                        .unwrap();
                        assert_eq!(result.layers, oracle, "n={n} m={modulus} K={macro_depth}");
                        assert_eq!(
                            result.layers.iter().map(Vec::len).sum::<usize>() as u64,
                            graph.expected_max_unique_states
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn macro_scheduler_is_fail_fast_on_future_or_settled_capacity() {
    let graph = MatrixGroup::unitriangular(3, 3).unwrap();
    let macros = MacroGeneratorSet::compile(&graph, 3).unwrap();
    let base = MacroSimulationConfig {
        rank_map: vec![0],
        buckets: 1,
        future_capacity_per_bucket: 1,
        settled_capacity_per_bucket: 100,
        seed: [0; 16],
        schedule: 0,
        pre_dedup: false,
    };
    assert!(run_macro(&graph, &macros, &base).is_err());
    let mut settled = base;
    settled.future_capacity_per_bucket = 1000;
    settled.settled_capacity_per_bucket = 1;
    assert!(run_macro(&graph, &macros, &settled).is_err());
}
