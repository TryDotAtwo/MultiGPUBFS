#![cfg(feature = "cuda")]
use mgbfs_cuda::ffi::*;
use std::{ffi::c_void, mem::size_of};
struct Buffer(*mut c_void);
impl Buffer {
    fn new(n: usize) -> Self {
        let mut p = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, n.max(1)) }, 0);
        Self(p)
    }
    fn upload<T: Copy>(&self, v: &[T]) {
        assert_eq!(
            unsafe { cudaMemcpy(self.0, v.as_ptr().cast(), std::mem::size_of_val(v), 1) },
            0
        );
    }
    fn read<T: Copy + Default>(&self, n: usize) -> Vec<T> {
        let mut v = vec![T::default(); n];
        assert_eq!(
            unsafe { cudaMemcpy(v.as_mut_ptr().cast(), self.0, n * size_of::<T>(), 2) },
            0
        );
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
#[test]
fn appends_in_source_order_and_rejects_whole_invalid_batches() {
    unsafe {
        let mut p = std::ptr::null_mut();
        let mut err = [0i8; 512];
        assert_eq!(
            mgbfs_materialize_create(32, 4, 5, &mut p, err.as_mut_ptr(), 512),
            0
        );
        let source = Buffer::new(4 * 32);
        source.upload(&(0..128u8).collect::<Vec<_>>());
        let hashes = Buffer::new(64);
        hashes.upload(&[[30u32; 4], [10; 4], [20; 4], [99; 4]]);
        let refs = Buffer::new(32);
        refs.upload(&[3u64, 1, 2, 0]);
        let count = Buffer::new(4);
        count.upload(&[3u32]);
        let states = Buffer::new(5 * 32);
        states.upload(&[0xeeu8; 160]);
        let out = Buffer::new(5 * 16);
        out.upload(&[[0xeeu32; 4]; 5]);
        let meta = Buffer::new(8);
        meta.upload(&[FrontierState::default()]);
        let run = || {
            assert_eq!(
                mgbfs_materialize_run(
                    p,
                    source.0.cast(),
                    4,
                    hashes.0,
                    refs.0.cast(),
                    count.0.cast(),
                    states.0.cast(),
                    out.0,
                    meta.0.cast(),
                    std::ptr::null_mut()
                ),
                0
            );
        };
        run();
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(meta.read::<FrontierState>(1)[0].count, 3);
        assert_eq!(
            &states.read::<u8>(160)[..96],
            &(32..128u8).collect::<Vec<_>>()
        );
        assert_eq!(
            out.read::<[u32; 4]>(5),
            vec![[10; 4], [20; 4], [30; 4], [0xee; 4], [0xee; 4]]
        );
        count.upload(&[0u32]);
        run(); // zero batch followed by append, no sync
        count.upload(&[2u32]);
        run();
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(meta.read::<FrontierState>(1)[0].count, 5);
        let saved = states.read::<u8>(160);
        let saved_hashes = out.read::<[u32; 4]>(5);
        run();
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(meta.read::<FrontierState>(1)[0].fatal, 1);
        assert_eq!(meta.read::<FrontierState>(1)[0].count, 5);
        assert_eq!(states.read::<u8>(160), saved);
        assert_eq!(out.read::<[u32; 4]>(5), saved_hashes);
        for invalid in [4u64, u64::MAX] {
            meta.upload(&[FrontierState::default()]);
            refs.upload(&[1u64, invalid, 2, 0]);
            run();
            assert_eq!(cudaDeviceSynchronize(), 0);
            assert_eq!(meta.read::<FrontierState>(1)[0].fatal, 2);
            assert_eq!(meta.read::<FrontierState>(1)[0].count, 0);
            assert_eq!(states.read::<u8>(160), saved);
            refs.upload(&[0u64, 1, 2, 3]);
            run();
            assert_eq!(cudaDeviceSynchronize(), 0);
            assert_eq!(meta.read::<FrontierState>(1)[0].fatal, 2);
            assert_eq!(states.read::<u8>(160), saved);
        }
        meta.upload(&[FrontierState::default()]);
        count.upload(&[5u32]);
        run();
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(meta.read::<FrontierState>(1)[0].fatal, 1);
        assert_eq!(states.read::<u8>(160), saved);
        mgbfs_materialize_destroy(p);
    }
}
