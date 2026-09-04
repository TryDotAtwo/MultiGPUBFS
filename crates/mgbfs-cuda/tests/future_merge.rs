#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;
use std::{
    ffi::{c_void, CStr},
    mem::{size_of, size_of_val},
};

struct Buffer(*mut c_void);
impl Buffer {
    fn new(bytes: usize) -> Self {
        let mut p = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, bytes.max(1)) }, 0);
        Self(p)
    }
    fn upload<T: Copy>(&self, x: &[T]) {
        assert_eq!(
            unsafe { cudaMemcpy(self.0, x.as_ptr().cast(), size_of_val(x), 1) },
            0
        );
    }
    fn read<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        let mut x = vec![T::default(); n];
        assert_eq!(
            unsafe { cudaMemcpy(x.as_mut_ptr().cast(), self.0, n * size_of::<T>(), 2) },
            0
        );
        x
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
            mgbfs_future_merge_destroy(self.0);
        }
    }
}
fn key(x: u32) -> [u32; 4] {
    [x, 0, 0, 0]
}

#[test]
fn bounded_merge_uses_live_bounds_and_preserves_old_duplicate_winner() {
    let mut raw = std::ptr::null_mut();
    let mut error = [0i8; 256];
    assert_eq!(
        unsafe { mgbfs_future_merge_create(16, 8, 8, &mut raw, error.as_mut_ptr(), error.len()) },
        0,
        "{}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
    let plan = Plan(raw);
    let future_states = Buffer::new(8 * 16);
    let future_hashes = Buffer::new(8 * 16);
    let state = Buffer::new(8);
    future_states.upload(&[[11u8; 16]]);
    future_hashes.upload(&[key(1)]);
    state.upload(&[FrontierState { count: 1, fatal: 0 }]);
    let source = Buffer::new(2 * 16);
    source.upload(&[[21u8; 16], [22u8; 16]]);
    let hashes = Buffer::new(2 * 16);
    hashes.upload(&[key(1), key(2)]);
    let refs = Buffer::new(2 * 8);
    refs.upload(&[0u64, 1]);
    let count = Buffer::new(4);
    count.upload(&[2u32]);
    assert_eq!(
        unsafe {
            mgbfs_future_merge_run_bounded(
                plan.0,
                future_states.0.cast(),
                future_hashes.0,
                state.0.cast(),
                1,
                source.0.cast(),
                2,
                hashes.0,
                refs.0.cast(),
                count.0.cast(),
                2,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    let settled = state.read::<FrontierState>(1)[0];
    assert_eq!((settled.count, settled.fatal), (2, 0));
    assert_eq!(future_hashes.read::<[u32; 4]>(2), vec![key(1), key(2)]);
    assert_eq!(
        future_states.read::<[u8; 16]>(2),
        vec![[11u8; 16], [22u8; 16]]
    );
    assert_eq!(
        unsafe {
            mgbfs_future_merge_run_bounded(
                plan.0,
                future_states.0.cast(),
                future_hashes.0,
                state.0.cast(),
                1,
                source.0.cast(),
                2,
                hashes.0,
                refs.0.cast(),
                count.0.cast(),
                2,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_ne!(state.read::<FrontierState>(1)[0].fatal, 0);
}
