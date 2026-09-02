#![cfg(feature = "cuda")]
use mgbfs_core::matrix::MatrixGroup;
use mgbfs_runtime::dense_device::DenseDeviceStepper;

#[test]
fn gpu_feedback_exhausts_exact_layers_without_cpu_supplied_frontiers() {
    for modulus in 2..=6 {
        let group = MatrixGroup::unitriangular(4, modulus).unwrap();
        let oracle = group.exact_layers((modulus as usize).pow(6)).unwrap();
        let batches: &[u32] = if modulus <= 4 { &[7, 64] } else { &[257] };
        for &batch in batches {
            for seed in [0u128, 1, 20260828] {
                for prededup in [false, true] {
                    let mut gpu = DenseDeviceStepper::new(
                        &group,
                        seed.to_le_bytes(),
                        batch,
                        (modulus as u32).pow(6),
                        prededup,
                    )
                    .unwrap();
                    for (depth, expected) in oracle.iter().enumerate() {
                        let mut actual = gpu.snapshot().unwrap();
                        actual.sort();
                        assert_eq!(
                            &actual, expected,
                            "m={modulus} batch={batch} depth={depth} prededup={prededup}"
                        );
                        assert_eq!(gpu.advance().unwrap(), depth + 1 < oracle.len());
                    }
                    assert!(gpu.snapshot().unwrap().is_empty());
                    assert!(!gpu.advance().unwrap());
                }
            }
        }
    }
}

#[test]
fn capacity_failure_poisoning_does_not_publish_a_partial_layer() {
    let group = MatrixGroup::unitriangular(4, 3).unwrap();
    let mut gpu = DenseDeviceStepper::new(&group, [1; 16], 1, 1, false).unwrap();
    assert!(gpu.advance().is_err());
    assert!(gpu.advance().is_err());
    assert!(gpu.snapshot().is_err());
}
