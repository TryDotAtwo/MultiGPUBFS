#![cfg(feature = "library-owner")]
//! Real GPU ABI test; run separately from CPU contracts on a visible T4.
use mgbfs_cuda::{ffi::*, library_owner::*};
use mgbfs_runtime::library_native::LibraryShard;
use std::{ffi::c_void, ptr};

unsafe fn allocate(bytes: usize) -> *mut c_void {
    let mut out = ptr::null_mut();
    assert_eq!(cudaMalloc(&mut out, bytes), 0);
    out
}

#[test]
fn rust_adapter_preserves_keys_and_publishes_only_after_completion() {
    unsafe {
        let mut stream = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
        let mut pool = ptr::null_mut();
        assert_eq!(
            mgbfs_library_pool_create_v1(64 << 20, 1 << 30, &mut pool),
            0
        );
        let input = allocate(48);
        let scratch = allocate(1280);
        // A private two-record destination is reserved for this fixture. This
        // validates the Rust owner ABI, not native StateRing or archive credits.
        let destination = allocate(32);
        let words: [u32; 12] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5];
        assert_eq!(cudaMemcpy(input, words.as_ptr().cast(), 48, 1), 0);
        let mut candidates = CandidatesV1 {
            keys: KeysV1 {
                words: [ptr::null(); 4],
                rows: 0,
                reserved: 0,
            },
            source_indices: ptr::null(),
        };
        assert_eq!(
            mgbfs_library_candidates_from_aos_v1(
                input,
                3,
                3,
                scratch,
                1280,
                stream,
                &mut candidates
            ),
            0
        );
        let history = KeysV1 {
            words: [ptr::null(); 4],
            rows: 0,
            reserved: 0,
        };
        let mut owner = LibraryShard::new_window(history, history, 2, stream).unwrap();
        let survivors = owner.compare(1, candidates).unwrap();
        assert_eq!(survivors.rows, 2);
        assert_eq!(owner.accepted(), 0);
        owner.commit(1, 2).unwrap();
        assert_eq!(owner.accepted(), 0);
        let keys = owner.export().unwrap();
        assert_eq!(
            mgbfs_library_keys_to_aos_v1(keys, destination, 2, stream),
            0
        );
        owner.record_completion(1).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !owner.poll_completion(1).unwrap() {
            assert_eq!(owner.accepted(), 0);
            assert!(
                std::time::Instant::now() < deadline,
                "GPU completion timeout"
            );
            std::thread::yield_now();
        }
        assert_eq!(owner.accepted(), 0);
        // This fixture has no native materializer/fatal control; CUDA completion
        // is checked above. A full runtime must inspect those controls here.
        owner.publish_completion(1).unwrap();
        assert_eq!(owner.accepted(), 2);
        let mut actual = [0u32; 8];
        assert_eq!(
            cudaMemcpy(actual.as_mut_ptr().cast(), destination, 32, 2),
            0
        );
        assert_eq!(actual, [1, 2, 3, 4, 1, 2, 3, 5]);
        assert_eq!(owner.compare(2, candidates).unwrap().rows, 0);
        owner.commit(2, 0).unwrap();
        owner.complete(2).unwrap();
        assert_eq!(owner.accepted(), 2);
        owner.seal().unwrap();
        let keys = owner.export().unwrap();
        assert_eq!(keys.rows, 2);
        assert_eq!(mgbfs_library_keys_to_aos_v1(keys, destination, 2, stream), 0);
        assert_eq!(cudaStreamSynchronize(stream), 0);
        actual.fill(0);
        assert_eq!(cudaMemcpy(actual.as_mut_ptr().cast(), destination, 32, 2), 0);
        assert_eq!(actual, [1, 2, 3, 4, 1, 2, 3, 5]);
        assert!(owner.compare(3, candidates).is_err());
        assert_eq!(owner.accepted(), 2);
        owner.close().unwrap();
        assert_eq!(mgbfs_library_pool_destroy_v1(pool), 0);
        for buffer in [input, scratch, destination] {
            assert_eq!(cudaFree(buffer), 0);
        }
        assert_eq!(cudaStreamDestroy(stream), 0);
    }
}
