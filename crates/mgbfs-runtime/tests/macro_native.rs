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
                        untouched_vram_reserve_bytes: 0,
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
fn native_macro_nonidentity_source_preserves_original_layers() {
    let mut graph = MatrixGroup::unitriangular(3, 3).unwrap();
    graph.start = graph.successor(&graph.start, 0).unwrap();
    graph.start = graph.successor(&graph.start, 1).unwrap();
    let expected = graph.exact_layers(27).unwrap();
    for macro_depth in [1, 2, 3, 10] {
        for prededup in [false, true] {
            let mut bfs = MacroNativeBfs::new(
                &graph,
                [macro_depth as u8; 16],
                MacroNativeConfig {
                    macro_depth,
                    batch: 7,
                    layer_capacity: 27,
                    future_capacity_per_depth: 1024,
                    prededup,
                    generation_variant: 1,
                    untouched_vram_reserve_bytes: 0,
                },
            )
            .unwrap();
            let mut actual = Vec::new();
            loop {
                let mut layer = bfs.snapshot().unwrap();
                layer.sort();
                actual.push(layer);
                if !bfs.advance().unwrap() {
                    break;
                }
            }
            assert_eq!(actual, expected, "K={macro_depth} pre={prededup}");
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
        untouched_vram_reserve_bytes: 0,
    };
    config.future_capacity_per_depth = 1;
    let mut bfs = MacroNativeBfs::new(&graph, [0; 16], config).unwrap();
    assert!(bfs.advance().is_err());
    assert!(bfs.advance().is_err());
}

#[test]
fn native_macro_vram_preflight_fails_before_runtime_buffers_are_allocated() {
    let graph = MatrixGroup::unitriangular(3, 2).unwrap();
    let result = MacroNativeBfs::new(
        &graph,
        [0; 16],
        MacroNativeConfig {
            macro_depth: 1,
            batch: 8,
            layer_capacity: 8,
            future_capacity_per_depth: 16,
            prededup: true,
            generation_variant: 1,
            untouched_vram_reserve_bytes: u64::MAX,
        },
    );
    match result {
        Err(error) => assert!(error.starts_with("VRAM_PREFLIGHT")),
        Ok(_) => panic!("preflight unexpectedly accepted an impossible reserve"),
    }
}

#[test]
fn native_macro_exposes_the_exact_two_bank_allocation_contract() {
    let graph = MatrixGroup::unitriangular(3, 2).unwrap();
    let bfs = MacroNativeBfs::new(
        &graph,
        [9; 16],
        MacroNativeConfig {
            macro_depth: 2,
            batch: 8,
            layer_capacity: 8,
            future_capacity_per_depth: 32,
            prededup: true,
            generation_variant: 1,
            untouched_vram_reserve_bytes: 0,
        },
    )
    .unwrap();
    let plan = bfs.memory_plan();
    assert_eq!(
        plan.shape.producer_state_bytes,
        2 * plan.shape.candidate_records * 16
    );
    assert_eq!(
        plan.shape.producer_hash_bytes,
        2 * plan.shape.candidate_records * 16
    );
    assert_eq!(
        plan.requested_device_bytes,
        plan.external_bytes + plan.library_bytes
    );
    assert_eq!(bfs.requested_device_bytes(), plan.requested_device_bytes);
}

#[test]
fn native_macro_archive_is_complete_and_verifiable() {
    use mgbfs_runtime::{
        archive::{verify, Extent},
        pinned_archive::PinnedArchive,
    };
    use std::sync::{Arc, Mutex};
    struct MemoryExtent(Arc<Mutex<Vec<u8>>>);
    impl Extent for MemoryExtent {
        fn reserve(&mut self, bytes: u64) -> std::io::Result<()> {
            self.0.lock().unwrap().resize(bytes as usize, 0);
            Ok(())
        }
        fn write_at(&mut self, offset: u64, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap()[offset as usize..offset as usize + bytes.len()]
                .copy_from_slice(bytes);
            Ok(bytes.len())
        }
        fn sync(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    for (graph, generation_variant) in [
        (MatrixGroup::unitriangular(4, 2).unwrap(), 1),
        (MatrixGroup::symmetric_permutation_matrices(4).unwrap(), 5),
    ] {
        let layout =
            mgbfs_core::macro_memory::MacroStateLayout::derive(&graph, generation_variant).unwrap();
        let expected: Vec<_> = graph
            .exact_layers(64)
            .unwrap()
            .into_iter()
            .map(|layer| {
                let mut states: Vec<_> = layer
                    .into_iter()
                    .map(|state| {
                        if generation_variant == 5 {
                            mgbfs_core::matrix::encode_permutation_matrix(&state, graph.rows)
                                .unwrap()
                        } else {
                            state
                        }
                    })
                    .collect();
                states.sort();
                states
            })
            .collect();
        let config = MacroNativeConfig {
            macro_depth: 3,
            batch: 7,
            layer_capacity: 64,
            future_capacity_per_depth: 256,
            prededup: true,
            generation_variant,
            untouched_vram_reserve_bytes: 0,
        };
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let mut archive = PinnedArchive::new(
            MemoryExtent(bytes.clone()),
            1_000_000,
            layout.width,
            [7; 32],
            7,
            32,
        )
        .unwrap();
        let mut bfs = MacroNativeBfs::new(&graph, [3; 16], config).unwrap();
        let mut actual = Vec::new();
        loop {
            let mut states = bfs.snapshot().unwrap();
            states.sort();
            actual.push(states);
            bfs.archive_current(&mut archive).unwrap();
            if !bfs.advance().unwrap() {
                break;
            }
        }
        archive.finish().unwrap();
        let data = bytes.lock().unwrap();
        verify(&data).unwrap();
        assert_eq!(actual, expected);
    }
}
