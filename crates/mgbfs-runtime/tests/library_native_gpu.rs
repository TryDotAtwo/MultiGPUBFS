#![cfg(feature = "library-owner")]
//! Real GPU ABI test; run separately from CPU contracts on a visible T4.
use mgbfs_cuda::{ffi::*, library_owner::*};
use mgbfs_runtime::library_native::{finalize_shards, LibraryShard};
use std::{ffi::c_void, ptr};

unsafe fn allocate(bytes: usize) -> *mut c_void {
    let mut out = ptr::null_mut();
    assert_eq!(cudaMalloc(&mut out, bytes), 0);
    out
}

#[test]
fn shard_finalization_reuses_old_history_and_preserves_empty_shard_offsets() {
    unsafe {
        let mut stream = ptr::null_mut();
        assert_eq!(cudaStreamCreateWithFlags(&mut stream, 1), 0);
        let mut pool = ptr::null_mut();
        assert_eq!(mgbfs_library_pool_create_v1(64 << 20, 1 << 30, &mut pool), 0);
        let old_history = allocate(1024);
        let input = allocate(64);
        let scratch = allocate(1280);
        let output = allocate(64);
        let target = KeysV1 {
            words: std::array::from_fn(|p| old_history.cast::<u32>().add(p * 64).cast_const()),
            rows: 4,
            reserved: 0,
        };
        for (plane, data) in [[1, 1, 1, 99], [2, 2, 2, 99], [3, 3, 3, 99], [10, 11, 12, 99]].iter().enumerate() {
            assert_eq!(cudaMemcpy(target.words[plane].cast_mut().cast(), data.as_ptr().cast(), 16, 1), 0);
        }
        let empty = KeysV1 { words: [ptr::null(); 4], rows: 0, reserved: 0 };
        let mut owners = Vec::new();
        for shard in 0..3 {
            let previous = KeysV1 {
                words: target.words.map(|p| p.add(shard)), rows: 1, reserved: 0,
            };
            owners.push(LibraryShard::new_window(previous, empty, 2, stream).unwrap());
        }
        for (shard, values) in [&[10u32, 20, 20, 21][..], &[11][..], &[12, 30, 30][..]].iter().enumerate() {
            let words: Vec<u32> = values.iter().flat_map(|&w| [1, 2, 3, w]).collect();
            assert_eq!(cudaMemcpy(input, words.as_ptr().cast(), words.len() * 4, 1), 0);
            let mut candidates = CandidatesV1 { keys: empty, source_indices: ptr::null() };
            assert_eq!(mgbfs_library_candidates_from_aos_v1(input, values.len() as u32, 4, scratch, 1280, stream, &mut candidates), 0);
            let survivors = owners[shard].compare(1, candidates).unwrap();
            assert_eq!(survivors.rows, [2, 0, 1][shard]);
            owners[shard].commit(1, survivors.rows).unwrap();
            owners[shard].complete(1).unwrap();
        }
        let mut ranges = [99..99, 99..99, 99..99];
        assert_eq!(finalize_shards(&mut owners, target, &mut ranges, stream).unwrap(), 3);
        assert_eq!(ranges, [0..2, 2..2, 2..3]);
        assert_eq!(mgbfs_library_keys_to_aos_v1(KeysV1 { rows: 3, ..target }, output, 4, stream), 0);
        assert_eq!(cudaStreamSynchronize(stream), 0);
        let mut actual = [0u32; 12];
        assert_eq!(cudaMemcpy(actual.as_mut_ptr().cast(), output, 48, 2), 0);
        assert_eq!(actual, [1, 2, 3, 20, 1, 2, 3, 21, 1, 2, 3, 30]);
        for plane in target.words {
            let mut guard = 0u32;
            assert_eq!(cudaMemcpy((&mut guard as *mut u32).cast(), plane.add(3).cast(), 4, 2), 0);
            assert_eq!(guard, 99);
        }
        drop(owners);
        assert_eq!(mgbfs_library_pool_destroy_v1(pool), 0);
        for buffer in [old_history, input, scratch, output] { assert_eq!(cudaFree(buffer), 0); }
        assert_eq!(cudaStreamDestroy(stream), 0);
    }
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
