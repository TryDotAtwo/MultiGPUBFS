#![cfg(feature = "cuda")]
use mgbfs_core::hash::Hash128;
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
            mgbfs_route_destroy(self.0);
        }
    }
}
#[test]
fn cub_routes_all_128_bits_stably_and_optionally_deduplicates() {
    let cap = 4097usize;
    let mut ptr = std::ptr::null_mut();
    let mut err = [0i8; 512];
    assert_eq!(
        unsafe { mgbfs_route_create(cap as u32, &mut ptr, err.as_mut_ptr(), err.len()) },
        0,
        "{}",
        unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()
    );
    let plan = Plan(ptr);
    let mut input: Vec<Hash128> = (0..cap)
        .map(|i| {
            let j = (i % 257) as u32;
            Hash128([
                j.wrapping_mul(997),
                j % 7,
                j % 3,
                j.wrapping_mul(0x91471927),
            ])
        })
        .collect();
    input[0] = Hash128([0; 4]);
    input[1] = Hash128([u32::MAX; 4]);
    input[2] = Hash128([0, 0, 1, 0]);
    input[3] = Hash128([0, 1, 0, 0]);
    input[4] = Hash128([1, 0, 0, 0]);
    let refs: Vec<u64> = (0..cap as u64).collect();
    let keys = Buffer::new(cap * 16);
    let values = Buffer::new(cap * 8);
    let out = Buffer::new(cap * 16);
    let outrefs = Buffer::new(cap * 8);
    let count = Buffer::new(4);
    assert_eq!(
        unsafe { cudaMemcpy(keys.0, input.as_ptr().cast(), cap * 16, 1) },
        0
    );
    assert_eq!(
        unsafe { cudaMemcpy(values.0, refs.as_ptr().cast(), cap * 8, 1) },
        0
    );
    for len in [0usize, 1, 31, 256, 4097] {
        for dedup in [0, 1] {
            assert_eq!(
                unsafe {
                    mgbfs_route_run(
                        plan.0,
                        keys.0,
                        values.0.cast(),
                        out.0,
                        outrefs.0.cast(),
                        count.0.cast(),
                        len as u32,
                        dedup,
                        std::ptr::null_mut(),
                    )
                },
                0
            );
            assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
            let mut n = 0u32;
            assert_eq!(
                unsafe { cudaMemcpy((&mut n as *mut u32).cast(), count.0, 4, 2) },
                0
            );
            let mut expected: Vec<_> = input[..len]
                .iter()
                .copied()
                .zip(refs.iter().copied())
                .collect();
            expected.sort_by_key(|v| v.0);
            if dedup == 1 {
                expected.dedup_by_key(|v| v.0);
            }
            assert_eq!(n as usize, expected.len());
            let mut actual = vec![Hash128([0; 4]); n as usize];
            let mut actualrefs = vec![0u64; n as usize];
            if n > 0 {
                assert_eq!(
                    unsafe { cudaMemcpy(actual.as_mut_ptr().cast(), out.0, n as usize * 16, 2) },
                    0
                );
                assert_eq!(
                    unsafe {
                        cudaMemcpy(actualrefs.as_mut_ptr().cast(), outrefs.0, n as usize * 8, 2)
                    },
                    0
                );
            }
            assert_eq!(
                actual.into_iter().zip(actualrefs).collect::<Vec<_>>(),
                expected
            );
        }
    }
    assert_ne!(
        unsafe {
            mgbfs_route_run(
                plan.0,
                keys.0,
                values.0.cast(),
                out.0,
                outrefs.0.cast(),
                count.0.cast(),
                cap as u32 + 1,
                1,
                std::ptr::null_mut(),
            )
        },
        0
    );
}
