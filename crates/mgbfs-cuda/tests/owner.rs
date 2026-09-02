#![cfg(feature = "cuda")]
use mgbfs_core::hash::Hash128;
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
    fn upload<T: Copy>(&self, data: &[T]) {
        if !data.is_empty() {
            assert_eq!(
                unsafe { cudaMemcpy(self.0, data.as_ptr().cast(), std::mem::size_of_val(data), 1) },
                0
            );
        }
    }
    fn read<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        let mut result = vec![T::default(); n];
        if n > 0 {
            assert_eq!(
                unsafe { cudaMemcpy(result.as_mut_ptr().cast(), self.0, n * size_of::<T>(), 2) },
                0
            );
        }
        result
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
            mgbfs_owner_destroy(self.0);
        }
    }
}
fn key(x: u32) -> [u32; 4] {
    [x, 0, 0, 0]
}
#[test]
fn owner_merges_old_layers_and_cross_epoch_duplicates_then_poison_on_overflow() {
    let mut raw = std::ptr::null_mut();
    let mut err = [0i8; 512];
    assert_eq!(
        unsafe { mgbfs_owner_create(1025, 1024, &mut raw, err.as_mut_ptr(), 512) },
        0,
        "{}",
        unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy()
    );
    let plan = Plan(raw);
    let prev = Buffer::new(1024 * 16);
    let curr = Buffer::new(1024 * 16);
    let accepted = Buffer::new(1024 * 16);
    let state = Buffer::new(size_of::<OwnerState>());
    state.upload(&[OwnerState::default()]);
    prev.upload(&[key(1), key(3)]);
    curr.upload(&[key(5), key(7)]);
    let candidates = Buffer::new(1025 * 16);
    let refs = Buffer::new(1025 * 8);
    let count = Buffer::new(4);
    let survivors = Buffer::new(1025 * 16);
    let survivor_refs = Buffer::new(1025 * 8);
    let survivor_count = Buffer::new(4);
    let mut expected = std::collections::BTreeSet::new();
    for (epoch, input) in [
        vec![0, 1, 2, 2, 3, 4, 5, 6, 7, 8],
        (0..777).collect(),
        (700..1028).collect(),
    ]
    .into_iter()
    .enumerate()
    {
        let keys: Vec<_> = input.iter().copied().map(key).collect();
        let origins: Vec<_> = (0..keys.len() as u64)
            .map(|i| i + epoch as u64 * 10000)
            .collect();
        candidates.upload(&keys);
        refs.upload(&origins);
        count.upload(&[keys.len() as u32]);
        let new: std::collections::BTreeSet<_> = input
            .iter()
            .copied()
            .filter(|x| ![1, 3, 5, 7].contains(x) && !expected.contains(x))
            .collect();
        expected.extend(new.iter().copied());
        assert_eq!(
            unsafe {
                mgbfs_owner_run(
                    plan.0,
                    prev.0,
                    2,
                    curr.0,
                    2,
                    accepted.0,
                    state.0.cast(),
                    candidates.0,
                    refs.0.cast(),
                    count.0.cast(),
                    survivors.0,
                    survivor_refs.0.cast(),
                    survivor_count.0.cast(),
                    epoch as u64,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
        let s = state.read::<OwnerState>(1)[0];
        assert_eq!(s.fatal, 0);
        assert_eq!(s.count as usize, expected.len());
        assert_eq!(s.last_epoch, epoch as u64);
        assert_eq!(
            accepted.read::<[u32; 4]>(expected.len()),
            expected.iter().copied().map(key).collect::<Vec<_>>()
        );
        let actual_count = survivor_count.read::<u32>(1)[0] as usize;
        assert_eq!(actual_count, new.len());
        assert_eq!(
            survivors.read::<[u32; 4]>(actual_count),
            new.iter().copied().map(key).collect::<Vec<_>>()
        );
        let expected_refs: Vec<_> = new
            .iter()
            .map(|x| origins[input.iter().position(|i| i == x).unwrap()])
            .collect();
        assert_eq!(survivor_refs.read::<u64>(actual_count), expected_refs);
    }
    assert_eq!(expected.len(), 1024);
    let before = accepted.read::<[u32; 4]>(1024);
    candidates.upload(&[key(4096)]);
    refs.upload(&[9u64]);
    count.upload(&[1u32]);
    for epoch in [3, 4] {
        assert_eq!(
            unsafe {
                mgbfs_owner_run(
                    plan.0,
                    prev.0,
                    2,
                    curr.0,
                    2,
                    accepted.0,
                    state.0.cast(),
                    candidates.0,
                    refs.0.cast(),
                    count.0.cast(),
                    survivors.0,
                    survivor_refs.0.cast(),
                    survivor_count.0.cast(),
                    epoch,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
        assert_ne!(state.read::<OwnerState>(1)[0].fatal, 0);
        assert_eq!(survivor_count.read::<u32>(1)[0], 0);
        assert_eq!(accepted.read::<[u32; 4]>(1024), before);
    }
    assert_eq!(size_of::<Hash128>(), 16);
}
#[test]
fn owner_accepts_max_hash_once_without_host_sync_and_rejects_bad_epochs_and_runs() {
    let mut raw = std::ptr::null_mut();
    let mut err = [0i8; 256];
    assert_eq!(
        unsafe { mgbfs_owner_create(600, 600, &mut raw, err.as_mut_ptr(), 256) },
        0
    );
    let plan = Plan(raw);
    let accepted = Buffer::new(600 * 16);
    let state = Buffer::new(size_of::<OwnerState>());
    state.upload(&[OwnerState::default()]);
    let input = Buffer::new(600 * 16);
    input.upload(&[[u32::MAX; 4]; 600]);
    let refs = Buffer::new(600 * 8);
    refs.upload(&(0..600u64).collect::<Vec<_>>());
    let count = Buffer::new(4);
    count.upload(&[600u32]);
    let out = Buffer::new(600 * 16);
    let outrefs = Buffer::new(600 * 8);
    let outcount = Buffer::new(4);
    let run = |epoch| {
        assert_eq!(
            unsafe {
                mgbfs_owner_run(
                    plan.0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    accepted.0,
                    state.0.cast(),
                    input.0,
                    refs.0.cast(),
                    count.0.cast(),
                    out.0,
                    outrefs.0.cast(),
                    outcount.0.cast(),
                    epoch,
                    std::ptr::null_mut(),
                )
            },
            0
        );
    };
    run(0);
    run(1); // No host sync or device count readback between these epochs.
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    let s = state.read::<OwnerState>(1)[0];
    assert_eq!((s.fatal, s.count, s.last_epoch), (0, 1, 1));
    assert_eq!(accepted.read::<[u32; 4]>(1), vec![[u32::MAX; 4]]);
    assert_eq!(outcount.read::<u32>(1), vec![0]);
    count.upload(&[0u32]);
    run(2);
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_eq!(state.read::<OwnerState>(1)[0].last_epoch, 2);
    run(2);
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_eq!(state.read::<OwnerState>(1)[0].fatal, 2);
    assert_eq!(state.read::<OwnerState>(1)[0].count, 1);
    // A fresh bucket may reuse the scratch plan, but not the old bucket's state.
    state.upload(&[OwnerState::default()]);
    input.upload(&[key(1), key(3), key(2)]);
    count.upload(&[3u32]);
    run(0);
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_eq!(state.read::<OwnerState>(1)[0].fatal, 3);
    assert_eq!(state.read::<OwnerState>(1)[0].count, 0);
    state.upload(&[OwnerState::default()]);
    count.upload(&[601u32]);
    run(0);
    assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
    assert_eq!(state.read::<OwnerState>(1)[0].fatal, 1);
    assert_eq!(outcount.read::<u32>(1), vec![0]);
}
