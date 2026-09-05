#![cfg(feature = "cuda")]
use mgbfs_core::macro_generators::MacroGeneratorSet;
use mgbfs_core::{hash::GemmHash, matrix::MatrixGroup};
use mgbfs_cuda::ffi::*;
use std::ffi::{c_void, CStr};
#[test]
fn allocation_query_matches_frozen_geometry_and_c_abi() {
    let report = mgbfs_cuda::allocation::query_generation(4, 6, 256, 2, 0).unwrap();
    assert_eq!(report.allocations[2].bytes, 768);
    assert!(mgbfs_cuda::allocation::query_generation(0, 6, 256, 2, 0).is_err());
    assert_eq!(std::mem::size_of::<GenerateBytes>(), 48);
    for variant in 0..5 {
        let mut q = GenerateBytes::default();
        assert_eq!(
            unsafe { mgbfs_generate_query(4, 6, 256, 2, variant, &mut q) },
            0
        );
        assert_eq!(
            (q.generators, q.packed_parents, q.products_s32, q.workspace),
            (384, 128, 768, 0)
        );
        assert_eq!((q.k, q.stride, q.rows, q.columns), (16, 16, 24, 8));
        assert_ne!(
            unsafe { mgbfs_generate_query(4, 6, 256, u32::MAX, variant, &mut q) },
            0
        );
        assert_eq!((q.products_s32, q.rows), (0, 0));
    }
}

#[test]
fn materialize_and_future_merge_queries_cover_every_internal_allocation() {
    assert_eq!(std::mem::size_of::<MaterializeBytes>(), 40);
    let mut materialize = MaterializeBytes::default();
    assert_eq!(
        unsafe { mgbfs_materialize_query(64, 1024, 4096, &mut materialize) },
        0
    );
    assert_eq!((materialize.keys, materialize.sorted), (8192, 8192));
    assert_eq!((materialize.indices, materialize.order), (4096, 4096));
    assert!(materialize.scratch > 0);
    assert_ne!(
        unsafe { mgbfs_materialize_query(63, 1024, 4096, &mut materialize) },
        0
    );
    assert_eq!(materialize, MaterializeBytes::default());

    assert_eq!(std::mem::size_of::<FutureMergeBytes>(), 88);
    let mut future = FutureMergeBytes::default();
    assert_eq!(
        unsafe { mgbfs_future_merge_query(64, 2048, 1024, &mut future) },
        0
    );
    assert_eq!((future.merged, future.tags), (49_152, 24_576));
    assert_eq!((future.unique, future.unique_tags), (32_768, 16_384));
    assert_eq!(
        (future.indices, future.selected, future.flags),
        (12_288, 12_288, 3072)
    );
    assert_eq!(
        (future.selected_count, future.states, future.state),
        (4, 131_072, 8)
    );
    assert!(future.scratch > 0);
    assert_ne!(
        unsafe { mgbfs_future_merge_query(64, u32::MAX, 1, &mut future) },
        0
    );
    assert_eq!(future, FutureMergeBytes::default());
}
struct Buffer(*mut c_void);
#[test]
fn compact_permutation_generation_matches_gather() {
    for n in [3usize, 12, 17] {
        let stride = (n + 15) & !15;
        let mut generators = vec![0u8; 2 * n * n];
        for i in 0..n {
            generators[i * n + (i + 1) % n] = 1;
            generators[n * n + i * n + (i + n - 1) % n] = 1;
        }
        for count in [1usize, 7, 65] {
            let mut parents = vec![0u8; count * stride];
            for p in 0..count {
                for i in 0..n {
                    parents[p * stride + i] = ((i + p) % n) as u8;
                }
            }
            let input = Buffer::new(parents.len());
            let output = Buffer::new(count * 2 * stride);
            let mut handle = std::ptr::null_mut();
            let mut error = [0i8; 512];
            assert_eq!(
                unsafe {
                    mgbfs_generate_create_variant(
                        n as u32,
                        2,
                        2,
                        count as u32,
                        generators.as_ptr(),
                        5,
                        &mut handle,
                        error.as_mut_ptr(),
                        512,
                    )
                },
                0
            );
            let plan = Plan(handle);
            assert_eq!(
                unsafe { cudaMemcpy(input.0, parents.as_ptr().cast(), parents.len(), 1) },
                0
            );
            assert_eq!(
                unsafe {
                    mgbfs_generate_run(
                        plan.0,
                        input.0.cast(),
                        output.0.cast(),
                        count as u32,
                        std::ptr::null_mut(),
                    )
                },
                0
            );
            let mut actual = vec![255u8; count * 2 * stride];
            assert_eq!(
                unsafe { cudaMemcpy(actual.as_mut_ptr().cast(), output.0, actual.len(), 2) },
                0
            );
            for p in 0..count {
                for m in 0..2 {
                    for i in 0..stride {
                        let expected = if i < n {
                            parents[p * stride + (i + if m == 0 { 1 } else { n - 1 }) % n]
                        } else {
                            0
                        };
                        assert_eq!(
                            actual[(p * 2 + m) * stride + i],
                            expected,
                            "n={n} p={p} m={m} i={i}"
                        );
                    }
                }
            }
        }
    }
}
impl Buffer {
    fn new(bytes: usize) -> Self {
        let mut p = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, bytes) }, 0);
        Self(p)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.0);
        }
    }
}
struct Plan(*mut c_void);
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            mgbfs_generate_destroy(self.0);
        }
    }
}
struct HashPlan(*mut c_void);
impl Drop for HashPlan {
    fn drop(&mut self) {
        unsafe {
            mgbfs_hash_destroy(self.0);
        }
    }
}
#[test]
fn invalid_variant_and_legacy_grid_are_rejected_before_output_write() {
    let g = MatrixGroup::unitriangular(4, 16).unwrap();
    let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
    let mut handle = std::ptr::null_mut();
    let mut error = [0i8; 512];
    assert_ne!(
        unsafe {
            mgbfs_generate_create_variant(
                4,
                6,
                16,
                1,
                generators.as_ptr(),
                99,
                &mut handle,
                error.as_mut_ptr(),
                512,
            )
        },
        0
    );
    assert!(handle.is_null());
    let count = 524281u32;
    assert_eq!(
        unsafe {
            mgbfs_generate_create(
                4,
                6,
                16,
                count,
                generators.as_ptr(),
                &mut handle,
                error.as_mut_ptr(),
                512,
            )
        },
        0
    );
    let plan = Plan(handle);
    let input = Buffer::new(count as usize * 16);
    let output = Buffer::new(count as usize * 96);
    let sentinel = [73u8; 16];
    assert_eq!(
        unsafe { cudaMemcpy(output.0, sentinel.as_ptr().cast(), 16, 1) },
        0
    );
    assert_eq!(
        unsafe {
            mgbfs_generate_run(
                plan.0,
                input.0.cast(),
                output.0.cast(),
                count,
                std::ptr::null_mut(),
            )
        },
        7
    );
    let mut actual = [0u8; 16];
    assert_eq!(
        unsafe { cudaMemcpy(actual.as_mut_ptr().cast(), output.0, 16, 2) },
        0
    );
    assert_eq!(actual, sentinel);
}
#[test]
fn large_batch_crosses_old_grid_y_boundary() {
    for variant in 1..=4 {
        let g = MatrixGroup::unitriangular(4, 16).unwrap();
        let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
        let count = 1048576u32;
        let input = Buffer::new(count as usize * 16);
        let output = Buffer::new(count as usize * 6 * 16);
        let parents = g.start.repeat(count as usize);
        assert_eq!(
            unsafe { cudaMemcpy(input.0, parents.as_ptr().cast(), parents.len(), 1) },
            0
        );
        let mut handle = std::ptr::null_mut();
        let mut error = [0i8; 512];
        assert_eq!(
            unsafe {
                mgbfs_generate_create_variant(
                    4,
                    6,
                    16,
                    count,
                    generators.as_ptr(),
                    variant,
                    &mut handle,
                    error.as_mut_ptr(),
                    512,
                )
            },
            0
        );
        let plan = Plan(handle);
        assert_eq!(
            unsafe {
                mgbfs_generate_run(
                    plan.0,
                    input.0.cast(),
                    output.0.cast(),
                    count,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
        let mut actual = vec![0u8; count as usize * 6 * 16];
        assert_eq!(
            unsafe { cudaMemcpy(actual.as_mut_ptr().cast(), output.0, actual.len(), 2) },
            0
        );
        for row in actual.chunks_exact(96) {
            assert_eq!(row, generators.as_slice());
        }
    }
}
#[test]
fn tensor_generation_then_hash_matches_full_state_oracle_without_intermediate_host_sync() {
    for variant in 0..=4 {
        for (n, modulus) in [(2, 2), (3, 2), (3, 7), (4, 256), (5, 3)] {
            if variant == 4 && n != 4 {
                continue;
            }
            let mut g = MatrixGroup::unitriangular(n, modulus).unwrap();
            if (n, modulus) == (3, 2) {
                g.generators.truncate(1);
                g.inverse_map = vec![0];
                g.validate().unwrap();
            }
            let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
            let moves = g.generators.len();
            let width = n * n;
            let stride = (width + 15) & !15;
            let mut handle = std::ptr::null_mut();
            let mut error = [0i8; 512];
            let status = unsafe {
                mgbfs_generate_create_variant(
                    n as u32,
                    moves as u32,
                    modulus as u32,
                    67,
                    generators.as_ptr(),
                    variant,
                    &mut handle,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            assert_eq!(
                status,
                0,
                "{}",
                unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
            );
            let plan = Plan(handle);
            let hash = GemmHash::from_seed(width, [27; 16]).unwrap();
            let limbs = hash.limbs();
            let mut hp = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    mgbfs_hash_create(
                        width as u32,
                        (67 * moves) as u32,
                        limbs.as_ptr(),
                        hash.offsets.as_ptr(),
                        &mut hp,
                        error.as_mut_ptr(),
                        error.len(),
                    )
                },
                0
            );
            let hp = HashPlan(hp);
            // Dense canonical matrices test all arithmetic, not only sparse generators.
            let mut parents = vec![0u8; 67 * stride];
            for r in 0..67 {
                for c in 0..width {
                    parents[r * stride + c] = ((r * 173 + c * 97 + 255) % modulus as usize) as u8;
                }
            }
            let input = Buffer::new(parents.len());
            let children = Buffer::new(67 * moves * stride);
            let hashes = Buffer::new(67 * moves * 16);
            assert_eq!(
                unsafe { cudaMemcpy(input.0, parents.as_ptr().cast(), parents.len(), 1) },
                0
            );
            for count in [1u32, 3, 16, 67] {
                assert_eq!(
                    unsafe {
                        mgbfs_generate_run(
                            plan.0,
                            input.0.cast(),
                            children.0.cast(),
                            count,
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        mgbfs_hash_run(
                            hp.0,
                            children.0.cast(),
                            hashes.0.cast(),
                            count * moves as u32,
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
                let mut actual = vec![0u8; count as usize * moves * stride];
                let mut actual_hash = vec![0u32; count as usize * moves * 4];
                assert_eq!(
                    unsafe { cudaMemcpy(actual.as_mut_ptr().cast(), children.0, actual.len(), 2) },
                    0
                );
                assert_eq!(
                    unsafe {
                        cudaMemcpy(
                            actual_hash.as_mut_ptr().cast(),
                            hashes.0,
                            actual_hash.len() * 4,
                            2,
                        )
                    },
                    0
                );
                for parent in 0..count as usize {
                    for m in 0..moves {
                        let row = parent * moves + m;
                        let want = g
                            .successor(&parents[parent * stride..parent * stride + width], m)
                            .unwrap();
                        assert_eq!(
                            &actual[row * stride..row * stride + width],
                            want.as_slice(),
                            "n={n} modulus={modulus} parent={parent} move={m}"
                        );
                        assert!(actual[row * stride + width..(row + 1) * stride]
                            .iter()
                            .all(|&x| x == 0));
                        assert_eq!(
                            &actual_hash[row * 4..row * 4 + 4],
                            &hash.hash(&want).unwrap().0
                        );
                    }
                }
            }
            assert_ne!(
                unsafe {
                    mgbfs_generate_run(
                        plan.0,
                        input.0.cast(),
                        children.0.cast(),
                        68,
                        std::ptr::null_mut(),
                    )
                },
                0
            );
        }
    }
}

#[test]
fn one_macro_gemm_emits_weight_grouped_move_major_runs() {
    let graph = MatrixGroup::symmetric_permutation_matrices(8).unwrap();
    let macros = MacroGeneratorSet::compile(&graph, 3).unwrap();
    let generators: Vec<_> = macros
        .transitions
        .iter()
        .flat_map(|transition| transition.matrix.iter().copied())
        .collect();
    let weights: Vec<_> = macros
        .transitions
        .iter()
        .map(|transition| transition.weight)
        .collect();
    assert!(weights.windows(2).all(|pair| pair[0] <= pair[1]));
    let count = 2u32;
    let stride = 64usize;
    let parents = [
        graph.start.clone(),
        graph.successor(&graph.start, 2).unwrap(),
    ]
    .concat();
    let input = Buffer::new(parents.len());
    let output = Buffer::new(count as usize * macros.transitions.len() * stride);
    assert_eq!(
        unsafe { cudaMemcpy(input.0, parents.as_ptr().cast(), parents.len(), 1) },
        0
    );
    let mut handle = std::ptr::null_mut();
    let mut error = [0i8; 512];
    assert_eq!(
        unsafe {
            mgbfs_generate_create_macro_variant(
                8,
                macros.transitions.len() as u32,
                2,
                count,
                generators.as_ptr(),
                weights.as_ptr(),
                1,
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        },
        0,
        "{}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
    let plan = Plan(handle);
    assert_eq!(
        unsafe {
            mgbfs_generate_run(
                plan.0,
                input.0.cast(),
                output.0.cast(),
                count,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    let mut actual = vec![0u8; count as usize * macros.transitions.len() * stride];
    assert_eq!(
        unsafe { cudaMemcpy(actual.as_mut_ptr().cast(), output.0, actual.len(), 2) },
        0
    );
    for (movement, transition) in macros.transitions.iter().enumerate() {
        for parent in 0..count as usize {
            let row = movement * count as usize + parent;
            let want = super_multiply(
                &transition.matrix,
                &parents[parent * stride..(parent + 1) * stride],
                8,
                2,
            );
            assert_eq!(&actual[row * stride..(row + 1) * stride], want.as_slice());
        }
    }
}

fn super_multiply(a: &[u8], b: &[u8], n: usize, modulus: u16) -> Vec<u8> {
    let mut result = vec![0u8; n * n];
    for row in 0..n {
        for column in 0..n {
            let sum: u32 = (0..n)
                .map(|k| a[row * n + k] as u32 * b[k * n + column] as u32)
                .sum();
            result[row * n + column] = (sum % modulus as u32) as u8;
        }
    }
    result
}
