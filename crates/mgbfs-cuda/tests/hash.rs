#![cfg(feature = "cuda")]
use mgbfs_core::hash::GemmHash;
use mgbfs_cuda::ffi::*;
use std::ffi::{c_void, CStr};
#[test]
fn allocation_query_matches_frozen_hash_buffers_and_c_abi() {
    let report = mgbfs_cuda::allocation::query_hash(9, 12).unwrap();
    assert_eq!(report.allocations[2].bytes, 768);
    assert!(mgbfs_cuda::allocation::query_hash(0, 12).is_err());
    assert_eq!(std::mem::size_of::<HashBytes>(), 40);
    let mut q = HashBytes::default();
    assert_eq!(unsafe { mgbfs_hash_query(9, 12, &mut q) }, 0);
    assert_eq!(
        (
            q.weights,
            q.offsets,
            q.partials_s32,
            q.workspace,
            q.stride,
            q.reserved
        ),
        (256, 16, 768, 0, 16, 0)
    );
    assert_ne!(unsafe { mgbfs_hash_query(33026, 12, &mut q) }, 0);
    assert_eq!((q.weights, q.partials_s32), (0, 0));
}

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
            mgbfs_hash_destroy(self.0);
        }
    }
}

#[test]
fn tensor_hash_matches_cpu_for_padding_unsigned_bytes_seeds_and_tail_counts() {
    for width in [1usize, 16, 17, 127] {
        for seed in [0u8, 1, 255] {
            let hash = GemmHash::from_seed(width, [seed; 16]).unwrap();
            let limbs = hash.limbs();
            let mut handle = std::ptr::null_mut();
            let mut err = [0i8; 512];
            let status = unsafe {
                mgbfs_hash_create(
                    width as u32,
                    257,
                    limbs.as_ptr(),
                    hash.offsets.as_ptr(),
                    &mut handle,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            assert_eq!(
                status,
                0,
                "{}",
                unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()
            );
            let plan = Plan(handle);
            let stride = (width + 15) & !15;
            let mut states = vec![0u8; 257 * stride];
            for r in 0..257 {
                for c in 0..width {
                    states[r * stride + c] = ((r * 173 + c * 97 + seed as usize) & 255) as u8;
                }
            }
            let input = Buffer::new(states.len());
            let output = Buffer::new(257 * 16);
            assert_eq!(
                unsafe { cudaMemcpy(input.0, states.as_ptr().cast(), states.len(), 1) },
                0
            );
            for count in [1u32, 7, 8, 63, 257] {
                assert_eq!(
                    unsafe {
                        mgbfs_hash_run(
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
                let mut actual = vec![0u32; count as usize * 4];
                assert_eq!(
                    unsafe {
                        cudaMemcpy(actual.as_mut_ptr().cast(), output.0, actual.len() * 4, 2)
                    },
                    0
                );
                for r in 0..count as usize {
                    assert_eq!(
                        &actual[r * 4..r * 4 + 4],
                        &hash
                            .hash(&states[r * stride..r * stride + width])
                            .unwrap()
                            .0,
                        "width={width} row={r}"
                    );
                }
            }
            assert_ne!(
                unsafe {
                    mgbfs_hash_run(
                        plan.0,
                        input.0.cast(),
                        output.0.cast(),
                        258,
                        std::ptr::null_mut(),
                    )
                },
                0
            );
        }
    }
}
