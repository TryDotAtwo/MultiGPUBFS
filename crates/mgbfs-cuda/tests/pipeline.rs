#![cfg(feature = "cuda")]
//! Integration of device primitives, with CPU-supplied oracle frontiers.
//! This deliberately does not claim to be the native BFS scheduler.
use mgbfs_core::{
    hash::{GemmHash, Hash128},
    matrix::MatrixGroup,
};
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
        if !v.is_empty() {
            assert_eq!(
                unsafe { cudaMemcpy(self.0, v.as_ptr().cast(), std::mem::size_of_val(v), 1) },
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
struct Plan(*mut c_void, unsafe extern "C" fn(*mut c_void));
impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            self.1(self.0);
        }
    }
}
#[test]
fn generated_routed_owner_survivors_match_full_state_layers_for_both_prededup_modes() {
    for modulus in 2u16..=6 {
        let g = MatrixGroup::unitriangular(4, modulus).unwrap();
        let layers = g.exact_layers((modulus as usize).pow(6)).unwrap();
        let hash = GemmHash::from_seed(16, [93; 16]).unwrap();
        let limbs = hash.limbs();
        let generators: Vec<u8> = g.generators.iter().flatten().copied().collect();
        let mut p = std::ptr::null_mut();
        let mut err = [0i8; 512];
        assert_eq!(
            unsafe {
                mgbfs_generate_create(
                    4,
                    6,
                    modulus as u32,
                    6,
                    generators.as_ptr(),
                    &mut p,
                    err.as_mut_ptr(),
                    512,
                )
            },
            0
        );
        let generate = Plan(p, mgbfs_generate_destroy);
        assert_eq!(
            unsafe {
                mgbfs_hash_create(
                    16,
                    36,
                    limbs.as_ptr(),
                    hash.offsets.as_ptr(),
                    &mut p,
                    err.as_mut_ptr(),
                    512,
                )
            },
            0
        );
        let hashing = Plan(p, mgbfs_hash_destroy);
        assert_eq!(
            unsafe { mgbfs_route_create(36, &mut p, err.as_mut_ptr(), 512) },
            0
        );
        let route = Plan(p, mgbfs_route_destroy);
        assert_eq!(
            unsafe { mgbfs_owner_create(36, 64, &mut p, err.as_mut_ptr(), 512) },
            0
        );
        let owner = Plan(p, mgbfs_owner_destroy);
        let parents = Buffer::new(6 * 16);
        let children = Buffer::new(36 * 16);
        let hashes = Buffer::new(36 * 16);
        let origin = Buffer::new(36 * 8);
        origin.upload(&(0..36u64).collect::<Vec<_>>());
        let sorted = Buffer::new(36 * 16);
        let sorted_refs = Buffer::new(36 * 8);
        let route_count = Buffer::new(4);
        let prev = Buffer::new(64 * 16);
        let curr = Buffer::new(64 * 16);
        let accepted = Buffer::new(64 * 16);
        let state = Buffer::new(size_of::<OwnerState>());
        let survivors = Buffer::new(36 * 16);
        let refs = Buffer::new(36 * 8);
        let count = Buffer::new(4);
        for depth in 0..2 {
            for pre_dedup in [0, 1] {
                let parent_rows: Vec<u8> = layers[depth].iter().flatten().copied().collect();
                parents.upload(&parent_rows);
                state.upload(&[OwnerState::default()]);
                let mut prev_hashes: Vec<Hash128> = if depth == 0 {
                    vec![]
                } else {
                    layers[depth - 1]
                        .iter()
                        .map(|s| hash.hash(s).unwrap())
                        .collect()
                };
                prev_hashes.sort();
                prev.upload(&prev_hashes);
                let mut curr_hashes: Vec<Hash128> = layers[depth]
                    .iter()
                    .map(|s| hash.hash(s).unwrap())
                    .collect();
                curr_hashes.sort();
                curr.upload(&curr_hashes);
                let mut expected: Vec<Hash128> = layers[depth + 1]
                    .iter()
                    .map(|s| hash.hash(s).unwrap())
                    .collect();
                expected.sort();
                assert!(expected.windows(2).all(|v| v[0] != v[1]));
                let n = layers[depth].len() as u32;
                // Every call enqueues into the same stream. The route's device-side
                // count flows directly into owner; no host count synchronization.
                assert_eq!(
                    unsafe {
                        mgbfs_generate_run(
                            generate.0,
                            parents.0.cast(),
                            children.0.cast(),
                            n,
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        mgbfs_hash_run(
                            hashing.0,
                            children.0.cast(),
                            hashes.0.cast(),
                            n * 6,
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        mgbfs_route_run(
                            route.0,
                            hashes.0,
                            origin.0.cast(),
                            sorted.0,
                            sorted_refs.0.cast(),
                            route_count.0.cast(),
                            n * 6,
                            pre_dedup,
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                assert_eq!(
                    unsafe {
                        mgbfs_owner_run(
                            owner.0,
                            prev.0,
                            prev_hashes.len() as u32,
                            curr.0,
                            curr_hashes.len() as u32,
                            accepted.0,
                            state.0.cast(),
                            sorted.0,
                            sorted_refs.0.cast(),
                            route_count.0.cast(),
                            survivors.0,
                            refs.0.cast(),
                            count.0.cast(),
                            0,
                            std::ptr::null_mut(),
                        )
                    },
                    0
                );
                assert_eq!(unsafe { cudaDeviceSynchronize() }, 0);
                assert_eq!(state.read::<OwnerState>(1)[0].fatal, 0);
                assert_eq!(count.read::<u32>(1)[0] as usize, expected.len());
                let actual: Vec<Hash128> = survivors
                    .read::<[u32; 4]>(expected.len())
                    .into_iter()
                    .map(Hash128)
                    .collect();
                assert_eq!(
                    actual, expected,
                    "modulus={modulus} depth={depth} pre_dedup={pre_dedup}"
                );
                for (h, r) in actual.iter().zip(refs.read::<u64>(actual.len())) {
                    let child = g
                        .successor(&layers[depth][r as usize / 6], r as usize % 6)
                        .unwrap();
                    assert!(layers[depth + 1].contains(&child));
                    assert_eq!(*h, hash.hash(&child).unwrap());
                }
            }
        }
    }
}
