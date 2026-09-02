#![cfg(feature = "cuda")]
use mgbfs_core::{hash::GemmHash, matrix::MatrixGroup};
use mgbfs_cuda::ffi::*;
use std::ffi::{c_void, CStr};
struct Buffer(*mut c_void);
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
fn tensor_generation_then_hash_matches_full_state_oracle_without_intermediate_host_sync() {
    for (n, modulus) in [(2, 2), (3, 7), (4, 256), (5, 3)] {
        let g = MatrixGroup::unitriangular(n, modulus).unwrap();
        let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
        let moves = g.generators.len();
        let width = n * n;
        let stride = (width + 15) & !15;
        let mut handle = std::ptr::null_mut();
        let mut error = [0i8; 512];
        let status = unsafe {
            mgbfs_generate_create(
                n as u32,
                moves as u32,
                modulus as u32,
                67,
                generators.as_ptr(),
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
