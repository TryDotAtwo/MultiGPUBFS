#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;
use std::{
    ffi::{c_void, CStr},
    mem::size_of,
};

struct Buffer(*mut c_void);
impl Buffer {
    fn new(bytes: usize) -> Self {
        let mut p = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, bytes.max(1)) }, 0);
        Self(p)
    }
    fn upload<T: Copy>(&self, v: &[T]) {
        if !v.is_empty() {
            assert_eq!(
                unsafe { cudaMemcpy(self.0, v.as_ptr().cast(), size_of_val(v), 1) },
                0
            );
        }
    }
    fn read<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        if n > 0 {
            assert_eq!(
                unsafe { cudaMemcpy(v.as_mut_ptr().cast(), self.0, n * size_of::<T>(), 2) },
                0
            );
        }
        v
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
            mgbfs_macro_settle_destroy(self.0);
        }
    }
}
fn key(x: u32) -> [u32; 4] {
    [x, 0, 0, 0]
}

#[test]
fn settle_filters_two_k_history_and_same_future_duplicates_atomically() {
    let mut q = MacroSettleBytes::default();
    assert_eq!(unsafe { mgbfs_macro_settle_query(8, 4, 4, &mut q) }, 0);
    assert!(q.indices >= 8 * 4 && q.flags >= 8 && q.count == 4 && q.scratch > 0);
    let mut raw = std::ptr::null_mut();
    let mut error = [0i8; 256];
    assert_eq!(
        unsafe { mgbfs_macro_settle_create(8, 4, 4, &mut raw, error.as_mut_ptr(), error.len()) },
        0,
        "{}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
    let plan = Plan(raw);
    let future = Buffer::new(8 * 16);
    future.upload(&[key(1), key(2), key(2), key(4), key(7)]);
    let refs = Buffer::new(8 * 8);
    refs.upload(&[10u64, 20, 21, 40, 70]);
    let count = Buffer::new(4);
    count.upload(&[5u32]);
    let history = Buffer::new(4 * 4 * 16);
    history.upload(&[
        key(0),
        key(1),
        key(0),
        key(0),
        key(3),
        key(0),
        key(0),
        key(0),
        key(4),
        key(5),
        key(0),
        key(0),
        key(0),
        key(0),
        key(0),
        key(0),
    ]);
    let history_counts = Buffer::new(4 * 4);
    history_counts.upload(&[2u32, 1, 2, 0]);
    let out = Buffer::new(8 * 16);
    let outrefs = Buffer::new(8 * 8);
    let outcount = Buffer::new(4);
    let state = Buffer::new(size_of::<MacroSettleState>());
    state.upload(&[MacroSettleState::default()]);
    assert_eq!(
        unsafe {
            mgbfs_macro_settle_run(
                plan.0,
                future.0,
                refs.0.cast(),
                count.0.cast(),
                history.0,
                history_counts.0.cast(),
                out.0,
                outrefs.0.cast(),
                outcount.0.cast(),
                state.0.cast(),
                1,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_eq!(outcount.read::<u32>(1), vec![2]);
    assert_eq!(out.read::<[u32; 4]>(2), vec![key(2), key(7)]);
    assert_eq!(outrefs.read::<u64>(2), vec![20, 70]);
    assert_eq!(
        state.read::<MacroSettleState>(1)[0],
        MacroSettleState {
            last_epoch: 1,
            count: 2,
            fatal: 0
        }
    );
}

#[test]
fn malformed_sorted_runs_or_epoch_poison_without_publication() {
    let mut raw = std::ptr::null_mut();
    let mut error = [0i8; 128];
    assert_eq!(
        unsafe { mgbfs_macro_settle_create(4, 2, 2, &mut raw, error.as_mut_ptr(), error.len()) },
        0
    );
    let plan = Plan(raw);
    let future = Buffer::new(64);
    future.upload(&[key(2), key(1)]);
    let refs = Buffer::new(32);
    refs.upload(&[2u64, 1]);
    let count = Buffer::new(4);
    count.upload(&[2u32]);
    let history = Buffer::new(64);
    let hc = Buffer::new(8);
    hc.upload(&[0u32, 0]);
    let out = Buffer::new(64);
    let outrefs = Buffer::new(32);
    let outcount = Buffer::new(4);
    outcount.upload(&[99u32]);
    let state = Buffer::new(size_of::<MacroSettleState>());
    state.upload(&[MacroSettleState {
        last_epoch: 0,
        count: 99,
        fatal: 0,
    }]);
    assert_eq!(
        unsafe {
            mgbfs_macro_settle_run(
                plan.0,
                future.0,
                refs.0.cast(),
                count.0.cast(),
                history.0,
                hc.0.cast(),
                out.0,
                outrefs.0.cast(),
                outcount.0.cast(),
                state.0.cast(),
                1,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_eq!(outcount.read::<u32>(1), vec![0]);
    let failed = state.read::<MacroSettleState>(1)[0];
    assert_eq!((failed.fatal, failed.count), (3, 99));
}
